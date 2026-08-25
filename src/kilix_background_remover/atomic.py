"""Verified same-directory staging and no-replace multi-output commit."""

from __future__ import annotations

import hashlib
import os
import tempfile
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from .errors import RemovalFailure


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
        save_args = {"compress_level": 9, "optimize": False}
    elif image_format == "WEBP":
        save_args = {"lossless": True, "quality": 100, "method": 6, "exact": True}
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".kilix-f108-{staging_token}.", suffix=".stage", dir=parent
    )
    stage = Path(temporary)
    try:
        os.fchmod(descriptor, 0o600)
        with os.fdopen(descriptor, "w+b") as stream:
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
        with Image.open(stage) as check:
            check.load()
            if check.size != image.size:
                raise OSError("encoded geometry mismatch")
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
