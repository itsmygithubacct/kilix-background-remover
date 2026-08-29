"""Verified same-directory staging and no-replace multi-output commit."""

from __future__ import annotations

import hashlib
import os
import stat
import struct
import tempfile
import zlib
from collections.abc import Callable, Sequence
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

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


@dataclass(slots=True)
class StagedFile:
    """A verified non-image artifact ready for the common atomic commit."""

    stage: Path
    destination: Path
    sha256: str
    bytes: int
    media_type: str
    kind: str


class StagedArtifact(Protocol):
    stage: Path
    destination: Path


@dataclass(frozen=True, slots=True)
class StagingPath:
    """A private sibling file whose inode must survive an external encoder."""

    stage: Path
    destination: Path
    device: int
    inode: int


def _check_destination(destination: Path) -> None:
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


def allocate_staging_path(destination: Path, *, staging_token: str) -> StagingPath:
    """Reserve a private same-directory output for a fixed-argv encoder."""

    _check_destination(destination)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".kilix-f108-{staging_token}.", suffix=".stage", dir=destination.parent
    )
    try:
        os.fchmod(descriptor, 0o600)
        status = os.fstat(descriptor)
    finally:
        os.close(descriptor)
    return StagingPath(Path(temporary), destination, status.st_dev, status.st_ino)


def finalize_staged_file(
    reserved: StagingPath,
    *,
    media_type: str,
    kind: str,
    max_output_bytes: int,
    verify: Callable[[Path], None],
) -> StagedFile:
    """Fsync and verify an encoder output without trusting its pathname."""

    descriptor = -1
    try:
        status = reserved.stage.lstat()
        if not _is_reserved_inode(status, reserved):
            raise OSError("the encoder replaced its reserved staging inode")
        if stat.S_IMODE(status.st_mode) != 0o600:
            raise OSError("the encoded staging file is not private")
        if not 1 <= status.st_size <= max_output_bytes:
            raise RemovalFailure(
                "background.output-failed",
                "The encoded output exceeds its byte limit.",
                "resource",
                "write-output",
            )
        descriptor = os.open(
            reserved.stage, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
        opened = os.fstat(descriptor)
        if not _same_snapshot(status, opened) or not _is_reserved_inode(opened, reserved):
            raise OSError("the staged output identity changed")
        os.fsync(descriptor)
        initial_digest = _sha256_descriptor(descriptor)
        verify(reserved.stage)
        after_path = reserved.stage.lstat()
        after_descriptor = os.fstat(descriptor)
        if not _same_snapshot(status, after_path) or not _same_snapshot(status, after_descriptor):
            raise OSError("the staged output changed during verification")
        digest = _sha256_descriptor(descriptor)
        if digest != initial_digest:
            raise OSError("the staged output content changed during verification")
        final_path = reserved.stage.lstat()
        final_descriptor = os.fstat(descriptor)
        if not _same_snapshot(after_path, final_path) or not _same_snapshot(
            after_descriptor, final_descriptor
        ):
            raise OSError("the staged output changed during hashing")
        return StagedFile(
            stage=reserved.stage,
            destination=reserved.destination,
            sha256=digest,
            bytes=status.st_size,
            media_type=media_type,
            kind=kind,
        )
    except Exception as exc:
        reserved.stage.unlink(missing_ok=True)
        if isinstance(exc, RemovalFailure):
            raise
        raise RemovalFailure(
            "background.output-failed",
            "The staged output could not be verified.",
            "output",
            "verify-output",
        ) from exc
    finally:
        if descriptor >= 0:
            os.close(descriptor)


def _is_reserved_inode(status: os.stat_result, reserved: StagingPath) -> bool:
    return (
        stat.S_ISREG(status.st_mode)
        and status.st_dev == reserved.device
        and status.st_ino == reserved.inode
    )


def _same_snapshot(left: os.stat_result, right: os.stat_result) -> bool:
    return (
        stat.S_ISREG(right.st_mode)
        and left.st_dev == right.st_dev
        and left.st_ino == right.st_ino
        and left.st_size == right.st_size
        and left.st_mtime_ns == right.st_mtime_ns
        and left.st_ctime_ns == right.st_ctime_ns
        and stat.S_IMODE(left.st_mode) == stat.S_IMODE(right.st_mode)
    )


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
    _check_destination(destination)
    parent = destination.parent
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


def discard_staged(items: Sequence[StagedArtifact]) -> None:
    for item in items:
        item.stage.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _sha256_descriptor(descriptor: int) -> str:
    digest = hashlib.sha256()
    os.lseek(descriptor, 0, os.SEEK_SET)
    for block in iter(lambda: os.read(descriptor, 1024 * 1024), b""):
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


def commit_staged(items: Sequence[StagedArtifact], *, fail_after_links: int | None = None) -> None:
    linked: list[StagedArtifact] = []
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
