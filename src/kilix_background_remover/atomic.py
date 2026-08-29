"""Verified same-directory staging and no-replace multi-output commit."""

from __future__ import annotations

import hashlib
import os
import struct
import tempfile
import zlib
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .contract_v2 import decode_gray8_png
from .errors import RemovalFailure

_DECODER_BOOKKEEPING = {
    "PNG": frozenset(),
    "WEBP": frozenset({"background", "duration", "loop", "timestamp"}),
}


@dataclass(slots=True)
class StagedImage:
    stage: Path
    destination: Path
    sha256: str
    bytes: int
    width: int
    height: int
    media_type: str
    kind: str


def stage_image(
    image: Image.Image,
    destination: Path,
    *,
    image_format: str,
    media_type: str,
    kind: str,
    max_output_bytes: int,
    staging_token: str,
) -> StagedImage:
    if destination.exists() or destination.is_symlink():
        raise RemovalFailure(
            "background.output-failed",
            "An output destination already exists.",
            "output",
            "write-output",
        )
    parent = destination.parent
    if not parent.is_dir() or parent.is_symlink():
        raise RemovalFailure(
            "background.output-failed",
            "An output directory is not available.",
            "output",
            "write-output",
        )
    save_args: dict[str, object] = {}
    if image_format == "PNG":
        save_args = {
            "compress_level": 9,
            "optimize": False,
            "icc_profile": None,
            "exif": b"",
            "pnginfo": None,
            "transparency": None,
        }
    elif image_format == "WEBP":
        save_args = {
            "lossless": True,
            "quality": 100,
            "method": 6,
            "exact": True,
            "icc_profile": b"",
            "exif": b"",
            "xmp": b"",
        }
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".kilix-f108-{staging_token}.", suffix=".stage", dir=parent
    )
    stage = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w+b") as stream:
            if image_format == "PNG" and kind == "mask":
                stream.write(_encode_gray8_png(image))
            else:
                image.save(stream, format=image_format, **save_args)
            stream.flush()
            os.fsync(stream.fileno())
        encoded_bytes = stage.stat().st_size
        if not 1 <= encoded_bytes <= max_output_bytes:
            raise RemovalFailure(
                "background.output-failed",
                "The encoded output exceeds its byte limit.",
                "resource",
                "write-output",
            )
        if kind == "mask":
            width, height, pixels = decode_gray8_png(stage.read_bytes())
            if (width, height) != image.size or bytes(pixels) != image.convert("L").tobytes():
                raise OSError("encoded mask pixels differ")
        with Image.open(stage) as check:
            check.load()
            if check.size != image.size:
                raise OSError("encoded geometry mismatch")
            unexpected_metadata = set(check.info) - _DECODER_BOOKKEEPING.get(
                image_format, frozenset()
            )
            if unexpected_metadata:
                names = ", ".join(sorted(unexpected_metadata))
                raise OSError(f"encoded output retained metadata: {names}")
    except Exception as exc:
        stage.unlink(missing_ok=True)
        if isinstance(exc, RemovalFailure):
            raise
        raise RemovalFailure(
            "background.output-failed",
            "The staged output could not be verified.",
            "output",
            "verify-output",
        ) from exc
    return StagedImage(
        stage=stage,
        destination=destination,
        sha256=_sha256(stage),
        bytes=encoded_bytes,
        width=image.width,
        height=image.height,
        media_type=media_type,
        kind=kind,
    )


def _encode_gray8_png(image: Image.Image) -> bytes:
    gray = image.convert("L")
    width, height = gray.size
    pixels = gray.tobytes()
    scanlines = b"".join(b"\x00" + pixels[row * width : (row + 1) * width] for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(scanlines, level=9))
        + _png_chunk(b"IEND", b"")
    )


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    checksum = zlib.crc32(kind + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + kind + payload + struct.pack(">I", checksum)


def discard_staged(items: list[StagedImage]) -> None:
    for item in items:
        item.stage.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def cleanup_staging_files(destinations: list[Path], staging_token: str) -> None:
    """Remove only supervisor-generated staging files for one untrusted job."""

    prefix = f".kilix-f108-{staging_token}."
    for parent in {destination.parent for destination in destinations}:
        try:
            entries = list(os.scandir(parent))
        except OSError:
            continue
        for entry in entries:
            if entry.name.startswith(prefix) and entry.name.endswith(".stage"):
                with suppress(OSError):
                    os.unlink(entry.path)


def commit_staged(items: list[StagedImage], *, fail_after_links: int | None = None) -> None:
    linked: list[StagedImage] = []
    committed = False
    try:
        for item in items:
            if item.destination.exists() or item.destination.is_symlink():
                raise FileExistsError(item.destination.name)
        for item in items:
            os.link(item.stage, item.destination, follow_symlinks=False)
            linked.append(item)
            if fail_after_links is not None and len(linked) >= fail_after_links:
                raise OSError("injected commit failure")
        parents = {item.destination.parent for item in items}
        for parent in parents:
            descriptor = os.open(parent, os.O_RDONLY | os.O_DIRECTORY)
            try:
                os.fsync(descriptor)
            finally:
                os.close(descriptor)
        committed = True
        for item in items:
            item.stage.unlink(missing_ok=True)
    except OSError as exc:
        if committed:
            return
        for item in linked:
            try:
                if item.destination.stat().st_ino == item.stage.stat().st_ino:
                    item.destination.unlink()
            except FileNotFoundError:
                pass
        discard_staged(items)
        raise RemovalFailure(
            "background.output-failed",
            "The output transaction could not be committed.",
            "output",
            "commit",
        ) from exc
