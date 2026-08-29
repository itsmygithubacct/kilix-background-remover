"""Installed-v2 reference consumer for an editable foreground-alpha mask."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import threading
import zlib
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, NoReturn, cast

from .contract_v2 import (
    ContractRefusal,
    ContractRuntime,
    canonical_bytes,
    validate_request_semantics,
    validate_result_semantics,
)
from .errors import RemovalFailure

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
ALLOWED_MASK_CHUNKS = {b"IHDR", b"IDAT", b"IEND"}
_LOCAL_ERROR_TEXT = {
    "job.cancelled": "Background removal was cancelled.",
    "background.deadline": "Background removal timed out.",
    "background.input-limit": "The image exceeds the background-removal limits.",
    "background.input-unreadable": "The image could not be read for background removal.",
    "background.invalid-request": "The background-removal request was invalid.",
    "background.artifact-invalid": "The installed model failed verification.",
    "background.backend-unavailable": "The background-removal backend is unavailable.",
    "background.profile-unavailable": "The selected model profile is unavailable.",
    "background.inference-failed": "Background removal could not process this image.",
    "background.output-failed": "The editable mask could not be written.",
    "background.internal": "Background removal failed internally.",
}


@dataclass(frozen=True, slots=True)
class EditableMaskProvenance:
    candidate_manifest_sha256: str
    request_id: str
    request_schema: str
    result_schema: str
    request_sha256: str
    result_sha256: str
    source_sha256: str
    width: int
    height: int
    mask_sha256: str
    mask_samples_sha256: str
    model_profile_id: str
    model_artifact_sha256: str
    backend: str
    edge_settings: Mapping[str, object]
    background_settings: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class EditableMaskImportPlan:
    """Verified final samples ready for one editor transaction."""

    pixels: bytes
    provenance: EditableMaskProvenance


@dataclass(frozen=True, slots=True)
class EditableLayerMask:
    pixels: bytes
    provenance: EditableMaskProvenance


class EditableMaskDocument:
    """Minimal composited-layer consumer with one atomic mask mutation."""

    def __init__(self, *, source_sha256: str, width: int, height: int) -> None:
        _require_digest(source_sha256, "source_sha256")
        if width <= 0 or height <= 0:
            raise ValueError("document geometry must be positive")
        self._source_sha256 = source_sha256
        self._width = width
        self._height = height
        self._lock = threading.Lock()
        self._mask: EditableLayerMask | None = None
        self._revision = 0

    @property
    def source_identity(self) -> tuple[str, int, int]:
        return self._source_sha256, self._width, self._height

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    @property
    def mask(self) -> EditableLayerMask | None:
        with self._lock:
            return self._mask

    @property
    def masks(self) -> tuple[EditableLayerMask, ...]:
        with self._lock:
            return () if self._mask is None else (self._mask,)

    def commit(
        self,
        plan: EditableMaskImportPlan,
        *,
        cancel: threading.Event | None = None,
    ) -> EditableLayerMask:
        """Attach exactly one verified mask without changing source pixels."""

        with self._lock:
            provenance = plan.provenance
            if provenance.source_sha256 != self._source_sha256 or (
                provenance.width,
                provenance.height,
            ) != (self._width, self._height):
                _invalid("The editable-mask target changed before commit.")
            if self._mask is not None:
                _invalid("The editable mask was already imported.")
            _check_cancel(cancel, "commit")
            imported = EditableLayerMask(plan.pixels, provenance)
            self._mask = imported
            self._revision += 1
            return imported


def consume_editable_mask_transcript(
    wires: Sequence[bytes],
    document: EditableMaskDocument,
    *,
    cancel: threading.Event | None = None,
    runtime: ContractRuntime | None = None,
) -> EditableLayerMask | None:
    """Validate all five v2 stages, then import or preserve cancellation."""

    authority = runtime or ContractRuntime.load()
    try:
        messages = [authority.accept_wire(wire) for wire in wires]
        authority.validate_transcript(messages)
    except ContractRefusal as exc:
        raise _failure("The provider transcript is not conformant.") from exc
    request = messages[0]
    results = [
        message
        for message in messages
        if message.get("schema") == "kilix.background-removal.result/v2"
    ]
    errors = [
        message
        for message in messages
        if message.get("schema") == "kilix.background-removal.error/v2"
    ]
    if results:
        if len(results) != 1 or errors:
            _invalid("The provider transcript has an ambiguous terminal.")
        plan = prepare_editable_mask_import(
            request,
            results[0],
            cancel=cancel,
            runtime=authority,
        )
        return document.commit(plan, cancel=cancel)
    if len(errors) != 1:
        _invalid("The provider transcript has no terminal.")
    job = _mapping(errors[0], "job")
    if job.get("state") == "cancelled" and job.get("code") == "job.cancelled":
        return None
    raise _wire_failure(errors[0])


def prepare_editable_mask_import(
    request: Mapping[str, object],
    result: Mapping[str, object],
    *,
    cancel: threading.Event | None = None,
    runtime: ContractRuntime | None = None,
) -> EditableMaskImportPlan:
    """Validate the complete v2 result join and bounded gray8 artifact."""

    authority = runtime or ContractRuntime.load()
    _check_cancel(cancel, "verify-output")
    try:
        authority.validate_message(cast(Mapping[str, Any], request))
        authority.validate_message(cast(Mapping[str, Any], result))
        validate_request_semantics(cast(Mapping[str, Any], request))
        validate_result_semantics(
            cast(Mapping[str, Any], request),
            cast(Mapping[str, Any], result),
        )
    except ContractRefusal as exc:
        raise _failure("The editable-mask request/result join is invalid.") from exc

    if request.get("schema") != "kilix.background-removal.request/v2":
        _invalid("The editable-mask request identity is invalid.")
    if result.get("schema") != "kilix.background-removal.result/v2":
        _invalid("The editable-mask result identity is invalid.")
    output_kinds = request.get("output_kinds")
    if output_kinds != ["mask"]:
        _invalid("The editable-mask action must request only a mask.")
    background = _mapping(request, "background")
    if background != {"mode": "transparent"}:
        _invalid("The editable-mask action must use a transparent background.")

    request_job = _mapping(request, "job")
    result_job = _mapping(result, "job")
    request_id = _text(request_job, "request_id")
    if _text(result_job, "request_id") != request_id or result_job.get("state") != "committed":
        _invalid("The editable-mask terminal does not join its request.")

    input_record = _mapping(request, "input")
    source_record = _mapping(result, "source")
    width = _positive_int(input_record, "width")
    height = _positive_int(input_record, "height")
    source_sha256 = _digest(input_record, "sha256")
    if source_record != {
        "sha256": source_sha256,
        "width": width,
        "height": height,
    }:
        _invalid("The mask result source projection does not match the request.")

    request_model = _mapping(request, "model")
    result_model = _mapping(result, "model")
    if result_model != request_model:
        _invalid("The mask result model identity does not match the request.")
    profile_id = _text(request_model, "profile_id")
    artifact_sha256 = _digest(request_model, "artifact_sha256")

    settings = _mapping(result, "settings")
    edge = _mapping(request, "edge")
    if settings.get("edge") != edge:
        _invalid("The effective edge settings do not match the request.")
    background_settings = _mapping(settings, "background")
    if background_settings != {"mode": "transparent"}:
        _invalid("The effective background settings are invalid.")

    mask = _mapping(result, "mask")
    destinations = _mapping(request, "destinations")
    mask_path = Path(_text(mask, "path"))
    if _text(destinations, "mask") != str(mask_path):
        _invalid("The mask path does not match the requested destination.")
    if (
        _text(mask, "media_type") != "image/png"
        or _text(mask, "encoding") != "gray8"
        or _text(mask, "semantics") != "foreground-alpha"
        or _text(mask, "pixel_contract") != "kilix.foreground-alpha-gray8/v2"
        or _positive_int(mask, "width") != width
        or _positive_int(mask, "height") != height
    ):
        _invalid("The editable mask profile or full-source geometry is invalid.")
    expected_bytes = _positive_int(mask, "bytes")
    expected_sha256 = _digest(mask, "sha256")

    limits = _mapping(request_job, "limits")
    max_pixels = _positive_int(limits, "max_decoded_pixels")
    max_output_bytes = _positive_int(limits, "max_output_bytes")
    if width * height > max_pixels or expected_bytes > max_output_bytes:
        _invalid("The editable mask exceeds its request limits.")

    encoded = _read_bound_file(mask_path, expected_bytes, max_output_bytes)
    if hashlib.sha256(encoded).hexdigest() != expected_sha256:
        _invalid("The editable mask digest does not match the result.")
    pixels = _decode_restrictive_gray8_png(encoded, width, height)
    _check_cancel(cancel, "verify-output")
    lock = authority.lock
    return EditableMaskImportPlan(
        pixels=pixels,
        provenance=EditableMaskProvenance(
            candidate_manifest_sha256=lock.manifest_sha256,
            request_id=request_id,
            request_schema=_text(request, "schema"),
            result_schema=_text(result, "schema"),
            request_sha256=hashlib.sha256(canonical_bytes(request)).hexdigest(),
            result_sha256=hashlib.sha256(canonical_bytes(result)).hexdigest(),
            source_sha256=source_sha256,
            width=width,
            height=height,
            mask_sha256=expected_sha256,
            mask_samples_sha256=hashlib.sha256(pixels).hexdigest(),
            model_profile_id=profile_id,
            model_artifact_sha256=artifact_sha256,
            backend=_text(result, "backend"),
            edge_settings=dict(edge),
            background_settings=dict(background_settings),
        ),
    )


def _read_bound_file(path: Path, expected: int, ceiling: int) -> bytes:
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as exc:
        raise _failure("The editable mask cannot be read.") from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode) or status.st_size != expected:
            _invalid("The editable mask byte identity does not match the result.")
        if not 1 <= status.st_size <= ceiling:
            _invalid("The editable mask exceeds its request limits.")
        payload = bytearray()
        while len(payload) <= ceiling:
            block = os.read(descriptor, min(1024 * 1024, ceiling + 1 - len(payload)))
            if not block:
                break
            payload.extend(block)
        after = os.fstat(descriptor)
        if (
            after.st_dev != status.st_dev
            or after.st_ino != status.st_ino
            or after.st_size != status.st_size
            or len(payload) != status.st_size
        ):
            _invalid("The editable mask changed during import.")
        return bytes(payload)
    finally:
        os.close(descriptor)


def _decode_restrictive_gray8_png(payload: bytes, width: int, height: int) -> bytes:
    chunks = _png_chunks(payload)
    if not chunks or chunks[0][0] != b"IHDR" or chunks[-1][0] != b"IEND":
        _invalid("The editable mask has an invalid PNG chunk order.")
    names = [name for name, _data in chunks]
    if (
        names.count(b"IHDR") != 1
        or names.count(b"IEND") != 1
        or names.count(b"IDAT") < 1
        or any(name not in ALLOWED_MASK_CHUNKS for name in names)
    ):
        _invalid("The editable mask PNG contains a forbidden chunk.")
    first_idat = names.index(b"IDAT")
    last_idat = len(names) - 1 - names[::-1].index(b"IDAT")
    if any(name != b"IDAT" for name in names[first_idat : last_idat + 1]):
        _invalid("The editable mask IDAT chunks are not contiguous.")
    ihdr = chunks[0][1]
    if len(ihdr) != 13:
        _invalid("The editable mask IHDR is invalid.")
    png_width, png_height, depth, color, compression, filtering, interlace = struct.unpack(
        ">IIBBBBB", ihdr
    )
    if (
        (png_width, png_height) != (width, height)
        or depth != 8
        or color != 0
        or compression != 0
        or filtering != 0
        or interlace != 0
    ):
        _invalid("The editable mask is not a non-interlaced gray8 PNG.")

    compressed = b"".join(data for name, data in chunks if name == b"IDAT")
    expected_raw = height * (width + 1)
    decoder = zlib.decompressobj()
    try:
        raw = decoder.decompress(compressed, expected_raw + 1)
        if len(raw) > expected_raw or decoder.unconsumed_tail:
            _invalid("The editable mask expands beyond its geometry.")
        tail = decoder.flush()
    except zlib.error as exc:
        raise _failure("The editable mask cannot be decoded.") from exc
    if not decoder.eof or decoder.unused_data or len(raw) + len(tail) != expected_raw:
        _invalid("The editable mask compressed payload is invalid.")
    raw += tail
    return _unfilter_rows(raw, width, height)


def _unfilter_rows(raw: bytes, width: int, height: int) -> bytes:
    prior = bytearray(width)
    output = bytearray()
    position = 0
    for _row in range(height):
        kind = raw[position]
        encoded = raw[position + 1 : position + 1 + width]
        position += width + 1
        decoded = bytearray(width)
        for column, value in enumerate(encoded):
            left = decoded[column - 1] if column else 0
            up = prior[column]
            upper_left = prior[column - 1] if column else 0
            if kind == 0:
                predictor = 0
            elif kind == 1:
                predictor = left
            elif kind == 2:
                predictor = up
            elif kind == 3:
                predictor = (left + up) // 2
            elif kind == 4:
                predictor = _paeth(left, up, upper_left)
            else:
                _invalid("The editable mask uses an invalid PNG row filter.")
            decoded[column] = (value + predictor) & 0xFF
        output.extend(decoded)
        prior = decoded
    return bytes(output)


def _paeth(left: int, up: int, upper_left: int) -> int:
    estimate = left + up - upper_left
    left_distance = abs(estimate - left)
    up_distance = abs(estimate - up)
    upper_left_distance = abs(estimate - upper_left)
    if left_distance <= up_distance and left_distance <= upper_left_distance:
        return left
    if up_distance <= upper_left_distance:
        return up
    return upper_left


def _png_chunks(payload: bytes) -> list[tuple[bytes, bytes]]:
    if not payload.startswith(PNG_SIGNATURE):
        _invalid("The editable mask does not have a PNG signature.")
    position = len(PNG_SIGNATURE)
    chunks: list[tuple[bytes, bytes]] = []
    while position < len(payload):
        if len(payload) - position < 12:
            _invalid("The editable mask PNG is truncated.")
        length = int.from_bytes(payload[position : position + 4], "big")
        name = payload[position + 4 : position + 8]
        end = position + 12 + length
        if end > len(payload):
            _invalid("The editable mask PNG is truncated.")
        data = payload[position + 8 : position + 8 + length]
        expected_crc = int.from_bytes(payload[position + 8 + length : end], "big")
        if zlib.crc32(name + data) & 0xFFFFFFFF != expected_crc:
            _invalid("The editable mask PNG has an invalid checksum.")
        chunks.append((name, data))
        position = end
        if name == b"IEND":
            break
    if position != len(payload):
        _invalid("The editable mask PNG has trailing bytes.")
    return chunks


def _wire_failure(error: Mapping[str, object]) -> RemovalFailure:
    job = _mapping(error, "job")
    code = _text(job, "code")
    category = _text(job, "category")
    retryable = job.get("retryable")
    if not isinstance(retryable, bool):
        _invalid("The provider error retryability is invalid.")
    message = _LOCAL_ERROR_TEXT.get(code, "The provider returned an unsupported response.")
    return RemovalFailure(code, message, category, _text(error, "phase"), retryable)


def _mapping(record: Mapping[str, object], key: str) -> dict[str, Any]:
    value = record.get(key)
    if not isinstance(value, dict) or not all(isinstance(name, str) for name in value):
        _invalid(f"The editable-mask {key} record is invalid.")
    return cast(dict[str, Any], value)


def _text(record: Mapping[str, object], key: str) -> str:
    value = record.get(key)
    if not isinstance(value, str) or not value:
        _invalid(f"The editable-mask {key} value is invalid.")
    return value


def _positive_int(record: Mapping[str, object], key: str) -> int:
    value = record.get(key)
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        _invalid(f"The editable-mask {key} value is invalid.")
    return value


def _digest(record: Mapping[str, object], key: str) -> str:
    value = _text(record, key)
    _require_digest(value, key)
    return value


def _require_digest(value: str, name: str) -> None:
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{name} digest is invalid")


def _check_cancel(cancel: threading.Event | None, phase: str) -> None:
    if cancel is not None and cancel.is_set():
        raise RemovalFailure(
            "job.cancelled",
            "Background removal was cancelled.",
            "cancellation",
            phase,
        )


def _invalid(message: str) -> NoReturn:
    raise _failure(message)


def _failure(message: str) -> RemovalFailure:
    return RemovalFailure(
        "background.output-failed",
        message,
        "output",
        "verify-output",
    )
