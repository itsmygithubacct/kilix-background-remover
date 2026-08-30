"""Full-v2 controls for F108's in-repository editable-mask consumer."""

from __future__ import annotations

import copy
import hashlib
import json
import struct
import threading
import zlib
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
from kilix_f108_f115_contracts import contract_root
from PIL import Image, PngImagePlugin

from kilix_background_remover.contract_v2 import (
    ContractRuntime,
    canonical_bytes,
    process_foreground_plane,
)
from kilix_background_remover.editable_mask import (
    EditableLayerMask,
    EditableMaskDocument,
    EditableMaskImportPlan,
    consume_editable_mask_transcript,
    prepare_editable_mask_import,
    run_reference_editable_mask_operation,
)
from kilix_background_remover.errors import RemovalFailure
from kilix_background_remover.frontend import describe_image
from kilix_background_remover.worker import WorkerSupervisor


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


def test_verified_plan_cannot_be_forged_or_mutated_before_commit(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, _pixels = _documents(tmp_path, request_factory)
    plan = prepare_editable_mask_import(request, result)
    document = _document(request)
    changed = replace(plan, pixels=bytes(len(plan.pixels)))

    with pytest.raises(RemovalFailure, match="samples changed"):
        document.commit(changed)
    forged = EditableMaskImportPlan(plan.pixels, plan.provenance, object())
    with pytest.raises(RemovalFailure, match="not produced by the validator"):
        document.commit(forged)

    assert document.mask is None
    assert document.revision == 0


def test_committed_provenance_settings_are_immutable(
    tmp_path: Path,
    request_factory: Callable[..., dict[str, object]],
) -> None:
    request, result, _pixels = _documents(tmp_path, request_factory)
    document = _document(request)
    imported = document.commit(prepare_editable_mask_import(request, result))

    with pytest.raises(TypeError):
        imported.provenance.edge_settings["threshold_u8"] = 0  # type: ignore[index]
    with pytest.raises(TypeError):
        imported.provenance.background_settings["mode"] = "color"  # type: ignore[index]

    assert imported.provenance.edge_settings == request["edge"]


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


def test_reference_harness_submits_real_composited_layer_and_imports_provider_mask(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "composited-layer.png"
    width, height = 48, 32
    pixels = bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (x * 5 % 256, y * 7 % 256, (x + y) * 3 % 256, 255)
    )
    Image.frombytes("RGBA", (width, height), pixels).save(source_path, format="PNG")
    source_before = source_path.read_bytes()
    identity = describe_image(source_path)
    document = EditableMaskDocument(
        source_sha256=identity["sha256"],  # type: ignore[arg-type]
        width=width,
        height=height,
    )
    edge = {
        "threshold_u8": 31,
        "feather_radius_px": 1,
        "matting_mode": "alpha",
        "preserve_source_alpha": True,
    }

    with WorkerSupervisor(cancellation_database=tmp_path / "cancellation.sqlite3") as provider:
        imported = run_reference_editable_mask_operation(
            source_path,
            document,
            output_dir=tmp_path,
            provider=provider,
            edge_settings=edge,
        )

    assert imported is not None
    assert imported is document.mask
    assert len(imported.pixels) == width * height
    assert imported.provenance.edge_settings == edge
    assert imported.provenance.source_sha256 == identity["sha256"]
    assert source_path.read_bytes() == source_before
    assert document.revision == 1


def test_reference_harness_mid_operation_cancel_leaves_document_unmodified(
    tmp_path: Path,
) -> None:
    source_path = tmp_path / "cancelled-composited-layer.png"
    width, height = 128, 96
    pixels = bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (x * 2 % 256, y * 2 % 256, (x + y) % 256, 255)
    )
    Image.frombytes("RGBA", (width, height), pixels).save(source_path, format="PNG")
    identity = describe_image(source_path)
    document = EditableMaskDocument(
        source_sha256=identity["sha256"],  # type: ignore[arg-type]
        width=width,
        height=height,
    )
    cancellation = threading.Event()
    progress: list[dict[str, object]] = []

    def cancel_after_provider_started(message: dict[str, object]) -> None:
        progress.append(message)
        cancellation.set()

    with WorkerSupervisor(cancellation_database=tmp_path / "cancel-race.sqlite3") as provider:
        imported = run_reference_editable_mask_operation(
            source_path,
            document,
            output_dir=tmp_path,
            provider=provider,
            edge_settings={
                "threshold_u8": 0,
                "feather_radius_px": 0,
                "matting_mode": "alpha",
                "preserve_source_alpha": True,
            },
            cancel=cancellation,
            on_progress=cancel_after_provider_started,
        )

    assert progress
    assert imported is None
    assert document.mask is None
    assert document.revision == 0
    assert not list(tmp_path.glob("pane4-*.mask.png"))


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


def _reference_import(
    source_path: Path,
    output_dir: Path,
    provider: WorkerSupervisor,
    edge: Mapping[str, object],
) -> EditableLayerMask:
    identity = describe_image(source_path)
    document = EditableMaskDocument(
        source_sha256=identity["sha256"],  # type: ignore[arg-type]
        width=identity["width"],  # type: ignore[arg-type]
        height=identity["height"],  # type: ignore[arg-type]
    )
    output_dir.mkdir(parents=True, exist_ok=False)
    imported = run_reference_editable_mask_operation(
        source_path,
        document,
        output_dir=output_dir,
        provider=provider,
        edge_settings=edge,
    )
    assert imported is not None
    return imported


def test_reference_harness_reported_threshold_and_feather_describe_the_imported_samples(
    tmp_path: Path,
) -> None:
    """The reported edge settings are decidable against the samples, not decorative.

    A report that cannot be falsified is a notice. This control recovers the
    provider's own model plane through a neutral pane-4 operation, then requires
    the samples imported under the reported settings to equal the independent
    contract authority's recomputation from those *reported* settings. Mutations
    EM-1 (production threshold ignored) and EM-2 (production feather ignored)
    are killed 2/2 by the equality; the 2/2 null controls below prove the
    equality is sensitive to each reported field rather than trivially true.
    """

    source_path = tmp_path / "reported-edge-layer.png"
    width, height = 40, 24
    pixels = bytes(
        channel
        for y in range(height)
        for x in range(width)
        for channel in (
            (x * 11 + y) % 256,
            (y * 13 + x * 3) % 256,
            (x * 5 + y * 7) % 256,
            (17 + x * 6 + y * 9) % 256,
        )
    )
    Image.frombytes("RGBA", (width, height), pixels).save(source_path, format="PNG")
    with Image.open(source_path) as opened:
        source_alpha = list(opened.convert("RGBA").getchannel("A").tobytes())

    neutral = {
        "threshold_u8": 0,
        "feather_radius_px": 0,
        "matting_mode": "alpha",
        "preserve_source_alpha": False,
    }
    reported = {
        "threshold_u8": 31,
        "feather_radius_px": 1,
        "matting_mode": "alpha",
        "preserve_source_alpha": True,
    }

    with WorkerSupervisor(cancellation_database=tmp_path / "reported-edge.sqlite3") as provider:
        plane = _reference_import(source_path, tmp_path / "plane", provider, neutral)
        imported = _reference_import(source_path, tmp_path / "reported", provider, reported)

    model_plane = [sample / 255.0 for sample in plane.pixels]
    assert imported.provenance.edge_settings == reported
    assert imported.provenance.width == width
    assert imported.provenance.height == height
    assert len(imported.pixels) == width * height == len(model_plane)
    assert len(set(plane.pixels)) > 1

    def authority(edge: Mapping[str, object]) -> bytes:
        return bytes(
            process_foreground_plane(
                {
                    "width": width,
                    "height": height,
                    "model_plane": model_plane,
                    "source_alpha": source_alpha,
                    **dict(edge),
                }
            )
        )

    assert imported.pixels == authority(imported.provenance.edge_settings)
    assert imported.pixels != authority({**reported, "threshold_u8": 0})
    assert imported.pixels != authority({**reported, "feather_radius_px": 0})
