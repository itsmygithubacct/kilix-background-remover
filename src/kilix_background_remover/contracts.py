"""Strict product decoder for the OD-22-authorized candidate R5 request."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from datetime import datetime
from functools import lru_cache
from pathlib import Path, PurePosixPath
from typing import Any

from .contract_v2 import ContractRefusal, ContractRuntime, validate_request_semantics
from .errors import invalid_request

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
PROFILE_RE = re.compile(r"^f108-[a-z0-9]+(?:-[a-z0-9]+)*$")
OUTPUT_TO_DESTINATION = {
    "mask": "mask",
    "cutout-png": "cutout_png",
    "cutout-webp": "cutout_webp",
    "composite": "composite",
}
MEDIA_TYPES = {"image/jpeg", "image/png", "image/tiff", "image/webp"}


@dataclass(frozen=True, slots=True)
class Limits:
    deadline_ms: int
    max_decoded_pixels: int
    max_input_bytes: int
    max_output_bytes: int


@dataclass(frozen=True, slots=True)
class ImageInput:
    path: Path
    sha256: str
    bytes: int
    width: int
    height: int
    media_type: str
    alpha_mode: str
    color_space: str


@dataclass(frozen=True, slots=True)
class EdgeSettings:
    threshold_u8: int
    feather_radius_px: int
    matting_mode: str
    preserve_source_alpha: bool


@dataclass(frozen=True, slots=True)
class ModelIdentity:
    profile_id: str
    artifact_sha256: str


@dataclass(frozen=True, slots=True)
class RemovalRequest:
    request_id: str
    submitted_at: str
    limits: Limits
    input: ImageInput
    model: ModelIdentity
    output_kinds: tuple[str, ...]
    destinations: dict[str, Path]
    edge: EdgeSettings
    background: dict[str, object]
    background_image: ImageInput | None
    wire: dict[str, object]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise invalid_request(f"{name} must be an object.")
    return value


def _exact(mapping: dict[str, Any], required: set[str], allowed: set[str], name: str) -> None:
    keys = set(mapping)
    if not required <= keys or not keys <= allowed:
        raise invalid_request(f"{name} has missing or unknown fields.")


def _integer(value: Any, low: int, high: int, name: str) -> int:
    if type(value) is not int or not low <= value <= high:
        raise invalid_request(f"{name} is outside its frozen bound.")
    return value


def _number(value: Any, low: float, high: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise invalid_request(f"{name} must be numeric.")
    result = float(value)
    if not low <= result <= high:
        raise invalid_request(f"{name} is outside its frozen bound.")
    return result


def _string(value: Any, name: str) -> str:
    if not isinstance(value, str):
        raise invalid_request(f"{name} must be a string.")
    return value


def _sha(value: Any, name: str) -> str:
    result = _string(value, name)
    if not SHA256_RE.fullmatch(result):
        raise invalid_request(f"{name} must be a lowercase SHA-256 value.")
    return result


def _path(value: Any, name: str) -> Path:
    raw = _string(value, name)
    if "\0" in raw or len(raw) > 4096:
        raise invalid_request(f"{name} is not a safe absolute path.")
    pure = PurePosixPath(raw)
    if not pure.is_absolute() or any(part in {"", ".", ".."} for part in pure.parts[1:]):
        raise invalid_request(f"{name} is not a safe absolute path.")
    return Path(raw)


def _timestamp(value: Any, name: str) -> str:
    raw = _string(value, name)
    if not raw.endswith("Z"):
        raise invalid_request(f"{name} must be a UTC timestamp.")
    try:
        datetime.fromisoformat(raw[:-1] + "+00:00")
    except ValueError as exc:
        raise invalid_request(f"{name} must be an RFC 3339 timestamp.") from exc
    return raw


def _image_input(value: Any, name: str) -> ImageInput:
    image = _mapping(value, name)
    fields = {
        "alpha_mode",
        "bytes",
        "color_space",
        "height",
        "media_type",
        "orientation_applied",
        "path",
        "sha256",
        "width",
    }
    _exact(image, fields, fields, name)
    if image["orientation_applied"] is not True:
        raise invalid_request(f"{name}.orientation_applied must be true.")
    media_type = _string(image["media_type"], f"{name}.media_type")
    if media_type not in MEDIA_TYPES:
        raise invalid_request(f"{name}.media_type is not supported by v2.")
    alpha_mode = _string(image["alpha_mode"], f"{name}.alpha_mode")
    if alpha_mode not in {"opaque", "premultiplied", "straight"}:
        raise invalid_request(f"{name}.alpha_mode is not supported by v2.")
    color_space = _string(image["color_space"], f"{name}.color_space")
    if color_space not in {"display-p3", "linear-srgb", "rec2020-linear", "srgb"}:
        raise invalid_request(f"{name}.color_space is not supported by v2.")
    return ImageInput(
        path=_path(image["path"], f"{name}.path"),
        sha256=_sha(image["sha256"], f"{name}.sha256"),
        bytes=_integer(image["bytes"], 1, 2_147_483_648, f"{name}.bytes"),
        width=_integer(image["width"], 1, 100_000, f"{name}.width"),
        height=_integer(image["height"], 1, 100_000, f"{name}.height"),
        media_type=media_type,
        alpha_mode=alpha_mode,
        color_space=color_space,
    )


def parse_request(value: object) -> RemovalRequest:
    root = _mapping(value, "request")
    try:
        _runtime().validate_message(root)
        validate_request_semantics(root)
    except ContractRefusal as exc:
        raise invalid_request("The request does not satisfy the conditional R5 contract.") from exc
    fields = {
        "background",
        "destinations",
        "edge",
        "input",
        "job",
        "model",
        "output_kinds",
        "schema",
    }
    _exact(root, fields, fields, "request")
    if root["schema"] != "kilix.background-removal.request/v2":
        raise invalid_request("The request schema identity is not supported.")

    job = _mapping(root["job"], "job")
    job_fields = {"limits", "request_id", "schema", "submitted_at"}
    _exact(job, job_fields, job_fields, "job")
    if job["schema"] != "kilix.media-job.request/v2":
        raise invalid_request("The media-job schema identity is not supported.")
    request_id = _string(job["request_id"], "job.request_id")
    try:
        parsed_uuid = uuid.UUID(request_id)
    except ValueError as exc:
        raise invalid_request("job.request_id is not a canonical UUID.") from exc
    if str(parsed_uuid) != request_id or parsed_uuid.variant != uuid.RFC_4122:
        raise invalid_request("job.request_id is not a canonical UUID.")
    submitted_at = _timestamp(job["submitted_at"], "job.submitted_at")

    limits_wire = _mapping(job["limits"], "job.limits")
    limit_fields = {"deadline_ms", "max_decoded_pixels", "max_input_bytes", "max_output_bytes"}
    _exact(limits_wire, limit_fields, limit_fields, "job.limits")
    limits = Limits(
        deadline_ms=_integer(limits_wire["deadline_ms"], 1, 86_400_000, "deadline_ms"),
        max_decoded_pixels=_integer(
            limits_wire["max_decoded_pixels"], 1, 200_000_000, "max_decoded_pixels"
        ),
        max_input_bytes=_integer(
            limits_wire["max_input_bytes"], 1, 2_147_483_648, "max_input_bytes"
        ),
        max_output_bytes=_integer(
            limits_wire["max_output_bytes"], 1, 4_294_967_296, "max_output_bytes"
        ),
    )

    image_input = _image_input(root["input"], "input")

    model_wire = _mapping(root["model"], "model")
    model_fields = {"artifact_sha256", "profile_id"}
    _exact(model_wire, model_fields, model_fields, "model")
    profile_id = _string(model_wire["profile_id"], "model.profile_id")
    if not PROFILE_RE.fullmatch(profile_id):
        raise invalid_request("model.profile_id is not a frozen F108 profile identifier.")
    model = ModelIdentity(profile_id, _sha(model_wire["artifact_sha256"], "model.artifact_sha256"))

    kinds_wire = root["output_kinds"]
    if not isinstance(kinds_wire, list) or not 1 <= len(kinds_wire) <= 4:
        raise invalid_request("output_kinds must contain one to four values.")
    if not all(isinstance(kind, str) and kind in OUTPUT_TO_DESTINATION for kind in kinds_wire):
        raise invalid_request("output_kinds contains an unsupported value.")
    if len(set(kinds_wire)) != len(kinds_wire) or "mask" not in kinds_wire:
        raise invalid_request("output_kinds must be unique and include mask.")
    output_kinds = tuple(kinds_wire)

    destination_wire = _mapping(root["destinations"], "destinations")
    destination_allowed = set(OUTPUT_TO_DESTINATION.values())
    _exact(destination_wire, {"mask"}, destination_allowed, "destinations")
    destinations = {key: _path(raw, f"destinations.{key}") for key, raw in destination_wire.items()}
    expected_destinations = {OUTPUT_TO_DESTINATION[kind] for kind in output_kinds}
    if set(destinations) != expected_destinations:
        raise invalid_request("destinations must match output_kinds exactly.")
    destination_values = list(destinations.values())
    if (
        len(set(destination_values)) != len(destination_values)
        or image_input.path in destination_values
    ):
        raise invalid_request("input and output paths must be distinct.")

    edge_wire = _mapping(root["edge"], "edge")
    edge_fields = {
        "feather_radius_px",
        "matting_mode",
        "preserve_source_alpha",
        "threshold_u8",
    }
    _exact(edge_wire, edge_fields, edge_fields, "edge")
    matting = _string(edge_wire["matting_mode"], "edge.matting_mode")
    if matting not in {"alpha", "none"} or type(edge_wire["preserve_source_alpha"]) is not bool:
        raise invalid_request("edge contains an unsupported mode.")
    edge = EdgeSettings(
        threshold_u8=_integer(edge_wire["threshold_u8"], 0, 255, "edge.threshold_u8"),
        feather_radius_px=_integer(
            edge_wire["feather_radius_px"], 0, 4096, "edge.feather_radius_px"
        ),
        matting_mode=matting,
        preserve_source_alpha=edge_wire["preserve_source_alpha"],
    )

    background = _mapping(root["background"], "background")
    mode = background.get("mode")
    background_image: ImageInput | None = None
    if mode == "transparent":
        _exact(background, {"mode"}, {"mode"}, "background")
    elif mode == "color":
        _exact(background, {"mode", "rgba"}, {"mode", "rgba"}, "background")
        rgba = background["rgba"]
        if not isinstance(rgba, list) or len(rgba) != 4:
            raise invalid_request("background.rgba must contain four channels.")
        for index, channel in enumerate(rgba):
            _number(channel, 0.0, 1.0, f"background.rgba[{index}]")
    elif mode == "image":
        _exact(background, {"image", "mode"}, {"image", "mode"}, "background")
        background_image = _image_input(background["image"], "background.image")
        if background_image.path == image_input.path:
            raise invalid_request("The foreground and background paths must be distinct.")
        if background_image.path in destination_values:
            raise invalid_request("Background and output paths must be distinct.")
        if (background_image.width, background_image.height) != (
            image_input.width,
            image_input.height,
        ):
            raise invalid_request("A background image must match the foreground geometry.")
    else:
        raise invalid_request("background.mode is not supported by v2.")

    combined_bytes = image_input.bytes
    combined_pixels = image_input.width * image_input.height
    if background_image is not None:
        combined_bytes += background_image.bytes
        combined_pixels += background_image.width * background_image.height
    if combined_bytes > limits.max_input_bytes:
        raise invalid_request("Combined input bytes exceed the frozen request limit.")
    if combined_pixels > limits.max_decoded_pixels:
        raise invalid_request("Combined decoded pixels exceed the frozen request limit.")

    return RemovalRequest(
        request_id=request_id,
        submitted_at=submitted_at,
        limits=limits,
        input=image_input,
        model=model,
        output_kinds=output_kinds,
        destinations=destinations,
        edge=edge,
        background=dict(background),
        background_image=background_image,
        wire=dict(root),
    )


@lru_cache(maxsize=1)
def _runtime() -> ContractRuntime:
    return ContractRuntime.load()
