"""Full-v2 controls for F108's in-repository editable-mask consumer."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import threading
import zlib
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from kilix_f108_f115_contracts import contract_root
from PIL import Image, PngImagePlugin

from kilix_background_remover.contract_v2 import ContractRuntime, canonical_bytes
from kilix_background_remover.editable_mask import (
    EditableMaskDocument,
    consume_editable_mask_transcript,
    prepare_editable_mask_import,
)
from kilix_background_remover.errors import RemovalFailure


def _documents(
    output_dir: Path,
    request_factory: Callable[..., dict[str, object]],
    *,
    pixels: bytes | None = None,
) -> tuple[dict[str, Any], dict[str, Any], bytes]:
    request = request_factory(output_dir, output_kinds=["mask"])
    input_record = request["input"]
    assert isinstance(input_record, dict)
    width = int(input_record["width"])
    height = int(input_record["height"])
    request["edge"] = {
        "threshold_u8": 127,
        "feather_radius_px": 2,
        "matting_mode": "alpha",
        "preserve_source_alpha": True,
    }
    mask_path = Path(request["destinations"]["mask"])  # type: ignore[index]
    pixels = pixels or bytes((index * 17) % 256 for index in range(width * height))
    Image.frombytes("L", (width, height), pixels).save(mask_path, format="PNG", pnginfo=None)
    payload = mask_path.read_bytes()
    job = request["job"]
    model = request["model"]
    assert isinstance(job, dict)
    assert isinstance(model, dict)
    result: dict[str, Any] = {
        "schema": "kilix.background-removal.result/v2",
        "request_schema": "kilix.background-removal.request/v2",
        "job": {
            "schema": "kilix.media-job.result/v2",
            "request_id": job["request_id"],
            "sequence": 0,
            "state": "committed",
            "committed_at": "2026-08-29T22:30:00Z",
            "elapsed_ms": 1,
            "warnings": [],
            "diagnostic_reference": "diag-00000000000000000000000000000000",
        },
        "source": {
            "sha256": input_record["sha256"],
            "width": width,
            "height": height,
        },
        "mask": {
            "path": str(mask_path),
            "media_type": "image/png",
            "encoding": "gray8",
            "semantics": "foreground-alpha",
            "pixel_contract": "kilix.foreground-alpha-gray8/v2",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "width": width,
            "height": height,
        },
        "outputs": [],
        "model": copy.deepcopy(model),
        "backend": "onnxruntime-cpu",
        "settings": {
            "edge": copy.deepcopy(request["edge"]),
            "background": {"mode": "transparent"},
        },
    }
    return request, result, pixels


def _document(request: dict[str, Any]) -> EditableMaskDocument:
    source = request["input"]
    return EditableMaskDocument(
        source_sha256=source["sha256"],
        width=source["width"],
        height=source["height"],
    )


def test_full_v2_transcript_attaches_one_exact_full_geometry_mask(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, expected_pixels = _documents(tmp_path, request_factory)
    document = _document(request)
    source_identity = document.source_identity

    imported = consume_editable_mask_transcript(
        [canonical_bytes(request), canonical_bytes(result)],
        document,
    )

    assert imported is document.mask
    assert imported is not None
    assert imported.pixels == expected_pixels
    assert document.source_identity == source_identity
    assert document.revision == 1
    assert len(document.masks) == 1
    assert imported.provenance.width == request["input"]["width"]
    assert imported.provenance.height == request["input"]["height"]
    assert imported.provenance.edge_settings == request["edge"]
    assert imported.provenance.mask_samples_sha256 == hashlib.sha256(expected_pixels).hexdigest()
    assert (
        imported.provenance.candidate_manifest_sha256
        == "803a5661a708b366b1d26884a4cf52d45c71dac58926e8216eb69aa902cbd25c"
    )


def test_candidate_accepted_cancellation_commits_no_document_change() -> None:
    root = Path(str(contract_root()))
    fixture = json.loads((root / "fixtures/valid/cancel-accepted.json").read_bytes())
    messages = fixture["messages"]
    request = messages[0]
    document = _document(request)

    imported = consume_editable_mask_transcript(
        [canonical_bytes(message) for message in messages],
        document,
    )

    assert imported is None
    assert document.mask is None
    assert document.revision == 0


def test_local_cancellation_at_commit_is_atomic(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, _pixels = _documents(tmp_path, request_factory)
    plan = prepare_editable_mask_import(request, result)
    document = _document(request)
    cancellation = threading.Event()
    cancellation.set()

    with pytest.raises(RemovalFailure) as caught:
        document.commit(plan, cancel=cancellation)

    assert caught.value.code == "job.cancelled"
    assert document.mask is None
    assert document.revision == 0


def test_mask_import_is_exactly_once(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, _pixels = _documents(tmp_path, request_factory)
    plan = prepare_editable_mask_import(request, result)
    document = _document(request)
    document.commit(plan)

    with pytest.raises(RemovalFailure, match="already imported"):
        document.commit(plan)

    assert len(document.masks) == 1
    assert document.revision == 1


def test_ancillary_png_chunk_is_refused_even_when_digest_matches(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, pixels = _documents(tmp_path, request_factory)
    mask = result["mask"]
    path = Path(mask["path"])
    source = request["input"]
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("private", "must not cross the editor boundary")
    Image.frombytes("L", (source["width"], source["height"]), pixels).save(
        path,
        format="PNG",
        pnginfo=metadata,
    )
    payload = path.read_bytes()
    mask["bytes"] = len(payload)
    mask["sha256"] = hashlib.sha256(payload).hexdigest()

    with pytest.raises(RemovalFailure, match="forbidden chunk"):
        prepare_editable_mask_import(request, result)


@pytest.mark.parametrize(
    "join",
    ["source", "model", "settings", "geometry", "digest", "pixel-contract"],
)
def test_each_import_join_fails_closed(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
    join: str,
) -> None:
    request, result, _pixels = _documents(tmp_path, request_factory)
    if join == "source":
        result["source"]["sha256"] = "3" * 64
    elif join == "model":
        result["model"]["artifact_sha256"] = "3" * 64
    elif join == "settings":
        result["settings"]["edge"]["threshold_u8"] = 126
    elif join == "geometry":
        result["mask"]["width"] += 1
    elif join == "digest":
        result["mask"]["sha256"] = "3" * 64
    else:
        result["mask"]["pixel_contract"] = "kilix.foreground-alpha-gray8/v1"

    with pytest.raises(RemovalFailure):
        prepare_editable_mask_import(request, result)


def test_png_expansion_past_declared_geometry_is_refused_before_import(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, _pixels = _documents(tmp_path, request_factory)
    mask = result["mask"]
    path = Path(mask["path"])
    source = request["input"]
    width = source["width"]
    height = source["height"]
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    raw = b"\x00" * (height * (width + 1) + 1)
    payload = (
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", header)
        + _chunk(b"IDAT", zlib.compress(raw))
        + _chunk(b"IEND", b"")
    )
    path.write_bytes(payload)
    mask["bytes"] = len(payload)
    mask["sha256"] = hashlib.sha256(payload).hexdigest()

    with pytest.raises(RemovalFailure, match="expands beyond"):
        prepare_editable_mask_import(request, result)


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def test_reference_consumer_uses_the_installed_complete_runtime() -> None:
    runtime = ContractRuntime.load()

    assert len(runtime.manifest_entries) == 46
    assert len(runtime.documents) == 12
