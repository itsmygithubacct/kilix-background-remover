"""Fixed local request construction shared by CLI and TUI front ends."""

from __future__ import annotations

import hashlib
import os
import stat
import uuid
import warnings
from datetime import UTC, datetime
from pathlib import Path

from PIL import Image

from .contract_v2 import ContractRefusal, ContractRuntime
from .contracts import sha256_file
from .errors import RemovalFailure
from .worker import reference_identity

MAX_JSON_BYTES = 2 * 1024 * 1024
MAX_INPUT_BYTES = 512 * 1024 * 1024
MAX_DECODED_PIXELS = 100_000_000
MAX_OUTPUT_BYTES = 1024 * 1024 * 1024
MEDIA_TYPES = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}


def load_json_document(path: Path) -> object:
    if path.is_symlink() or not path.is_file() or path.stat().st_size > MAX_JSON_BYTES:
        raise ValueError("JSON input must be a bounded regular file")
    try:
        return ContractRuntime.load().accept_wire(path.read_bytes())
    except ContractRefusal as exc:
        raise RemovalFailure(
            "background.invalid-request",
            "The request is not an accepted canonical v2 document.",
            "input",
            "accepted",
        ) from exc


def describe_image(path: Path) -> dict[str, object]:
    absolute = path.resolve(strict=True)
    status = absolute.lstat()
    if path.is_symlink() or not stat.S_ISREG(status.st_mode):
        raise RemovalFailure(
            "background.input-unreadable",
            "The input must be a regular file.",
            "input",
            "decode",
        )
    if not 1 <= status.st_size <= MAX_INPUT_BYTES:
        raise RemovalFailure(
            "background.input-limit",
            "The input exceeds the fixed byte limit.",
            "resource",
            "decode",
        )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", Image.DecompressionBombWarning)
        with Image.open(absolute) as opened:
            media_type = MEDIA_TYPES.get(opened.format or "")
            if media_type is None:
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The image format is not supported.",
                    "input",
                    "decode",
                )
            width, height = opened.size
            if width * height > 100_000_000:
                raise RemovalFailure(
                    "background.input-limit",
                    "The image exceeds the 100 megapixel input bound.",
                    "resource",
                    "decode",
                )
            if int(opened.getexif().get(274, 1)) != 1:
                raise RemovalFailure(
                    "background.invalid-request",
                    "Apply EXIF orientation before submitting the image.",
                    "input",
                    "decode",
                )
            alpha_mode = "straight" if "A" in opened.getbands() else "opaque"
    return {
        "path": str(absolute),
        "sha256": sha256_file(absolute),
        "bytes": status.st_size,
        "width": width,
        "height": height,
        "media_type": media_type,
        "orientation_applied": True,
        "alpha_mode": alpha_mode,
        "color_space": "srgb",
    }


def stable_output_key(relative_path: Path) -> str:
    digest = hashlib.sha256(os.fsencode(str(relative_path))).hexdigest()[:16]
    safe = "".join(
        character.lower() if character.isascii() and character.isalnum() else "-"
        for character in relative_path.name
    ).strip("-")
    safe = safe[:64] or "image"
    return f"{safe}-{digest}"


def make_request(
    image: dict[str, object],
    *,
    output_dir: Path,
    output_key: str,
    output_kinds: list[str],
    background: dict[str, object] | None = None,
    deadline_ms: int = 120_000,
) -> dict[str, object]:
    output_root = output_dir.resolve(strict=True)
    if output_dir.is_symlink() or not output_root.is_dir():
        raise ValueError("output directory must be an existing regular directory")
    if (
        not output_kinds
        or "mask" not in output_kinds
        or len(output_kinds) != len(set(output_kinds))
    ):
        raise ValueError("output kinds must be unique and include mask")
    suffixes = {
        "mask": "mask.png",
        "cutout-png": "cutout.png",
        "cutout-webp": "cutout.webp",
        "composite": "composite.png",
    }
    destination_keys = {
        "mask": "mask",
        "cutout-png": "cutout_png",
        "cutout-webp": "cutout_webp",
        "composite": "composite",
    }
    if any(kind not in suffixes for kind in output_kinds):
        raise ValueError("unsupported output kind")
    destinations = {
        destination_keys[kind]: str(output_root / f"{output_key}.{suffixes[kind]}")
        for kind in output_kinds
    }
    profile_id, artifact_sha256 = reference_identity()
    return {
        "schema": "kilix.background-removal.request/v2",
        "job": {
            "schema": "kilix.media-job.request/v2",
            "request_id": str(uuid.uuid4()),
            "submitted_at": datetime.now(UTC)
            .isoformat(timespec="microseconds")
            .replace("+00:00", "Z"),
            "limits": {
                "deadline_ms": deadline_ms,
                "max_decoded_pixels": MAX_DECODED_PIXELS,
                "max_input_bytes": MAX_INPUT_BYTES,
                "max_output_bytes": MAX_OUTPUT_BYTES,
            },
        },
        "input": image,
        "model": {
            "profile_id": profile_id,
            "artifact_sha256": artifact_sha256,
        },
        "output_kinds": output_kinds,
        "destinations": destinations,
        "edge": {
            "threshold_u8": 0,
            "feather_radius_px": 0,
            "matting_mode": "alpha",
            "preserve_source_alpha": True,
        },
        "background": background or {"mode": "transparent"},
    }
