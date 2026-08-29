"""Bounded, orientation-explicit image decode."""

from __future__ import annotations

import hashlib
import io
import multiprocessing as mp
import os
import resource
import signal
import stat
import tempfile
import warnings
from contextlib import suppress
from dataclasses import dataclass
from multiprocessing.connection import Connection
from pathlib import Path
from typing import Any, cast

from PIL import Image, ImageCms, UnidentifiedImageError

from .contracts import ImageInput, Limits
from .errors import RemovalFailure

FORMAT_MEDIA_TYPE = {
    "JPEG": "image/jpeg",
    "PNG": "image/png",
    "TIFF": "image/tiff",
    "WEBP": "image/webp",
}
MAX_METADATA_BYTES = 1_048_576
# The image path produces one cutout from one frame.  TIFF and WebP are both
# accepted media types and both carry frames natively, so the media-type
# allowlist does not bound this.  Silently using frame 0 would be data loss on
# the user's input; offline video is F108's separate phase and will raise this
# deliberately rather than inherit it.
MAX_INPUT_FRAMES = 1


@dataclass(slots=True)
class DecodedImage:
    image: Image.Image
    source_alpha: Image.Image


@dataclass(frozen=True, slots=True)
class DecodeBudget:
    """Hard limits applied to the disposable untrusted-parser process."""

    wall_seconds: float = 30.0
    cpu_seconds: int = 30
    address_space_bytes: int = 2 * 1024 * 1024 * 1024


DEFAULT_DECODE_BUDGET = DecodeBudget()


def decode_image_bounded(
    source: ImageInput,
    limits: Limits,
    *,
    budget: DecodeBudget = DEFAULT_DECODE_BUDGET,
) -> DecodedImage:
    """Decode hostile bytes in a killable, resource-limited child process.

    Only a private, metadata-free RGBA PNG crosses back into the long-lived
    worker.  A parser stall, crash, or memory-limit exit therefore cannot leave
    the model session process executing hostile decoder state.
    """

    wall_seconds = min(budget.wall_seconds, limits.deadline_ms / 1000.0)
    if wall_seconds <= 0 or budget.cpu_seconds <= 0 or budget.address_space_bytes <= 0:
        raise ValueError("decode limits must be positive")
    context: Any = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="kilix-f108-decode-") as temporary:
        root = Path(temporary)
        os.chmod(root, 0o700)
        sanitized = root / "sanitized.png"
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_decode_child,
            args=(child, source, limits, budget, sanitized, os.getpid()),
            name="kilix-f108-image-decoder",
            daemon=True,
        )
        process.start()
        child.close()
        message: tuple[str, object] | None = None
        try:
            if parent.poll(wall_seconds):
                try:
                    received = parent.recv()
                except EOFError:
                    received = None
                if isinstance(received, tuple) and len(received) == 2:
                    message = received
            if message is None:
                if process.is_alive():
                    _terminate_process(process)
                    raise RemovalFailure(
                        "background.input-limit",
                        "The image decoder exceeded its time limit.",
                        "resource",
                        "decode",
                    )
                raise RemovalFailure(
                    "background.input-limit",
                    "The image decoder exceeded its resource limit.",
                    "resource",
                    "decode",
                )
            process.join(timeout=1.0)
            if process.is_alive():
                _terminate_process(process)
            kind, payload = message
            if kind == "failure" and isinstance(payload, tuple) and len(payload) == 5:
                code, safe_message, category, phase, retryable = payload
                if all(isinstance(value, str) for value in payload[:4]) and isinstance(
                    retryable, bool
                ):
                    raise RemovalFailure(
                        cast(str, code),
                        cast(str, safe_message),
                        cast(str, category),
                        cast(str, phase),
                        retryable,
                    )
            if kind == "limit" and payload is None:
                raise RemovalFailure(
                    "background.input-limit",
                    "The image decoder exceeded its resource limit.",
                    "resource",
                    "decode",
                )
            if kind != "ok" or payload is not None:
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The input image cannot be decoded.",
                    "input",
                    "decode",
                )
            return _load_sanitized(sanitized, source)
        finally:
            parent.close()
            if process.is_alive():
                _terminate_process(process)


def _terminate_process(process: Any) -> None:
    process.terminate()
    process.join(timeout=1.0)
    if process.is_alive():
        process.kill()
        process.join(timeout=1.0)


def _decode_child(
    connection: Connection,
    source: ImageInput,
    limits: Limits,
    budget: DecodeBudget,
    sanitized: Path,
    expected_parent: int,
) -> None:
    try:
        _arm_parent_death_signal(expected_parent)
        os.umask(0o077)
        _apply_decode_limits(source, budget)
        decoded = decode_image(source, limits)
        decoded.image.save(
            sanitized,
            format="PNG",
            compress_level=1,
            optimize=False,
            icc_profile=None,
            exif=b"",
            pnginfo=None,
        )
        descriptor = os.open(sanitized, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        connection.send(("ok", None))
    except RemovalFailure as failure:
        connection.send(
            (
                "failure",
                (
                    failure.code,
                    failure.safe_message,
                    failure.category,
                    failure.phase,
                    failure.retryable,
                ),
            )
        )
    except (MemoryError, OSError):
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("limit", None))
    except BaseException:
        with suppress(BrokenPipeError, EOFError, OSError):
            connection.send(("crash", None))
    finally:
        connection.close()


def _apply_decode_limits(source: ImageInput, budget: DecodeBudget) -> None:
    _set_limit(resource.RLIMIT_AS, budget.address_space_bytes)
    _set_limit(resource.RLIMIT_CPU, budget.cpu_seconds)
    maximum_file = min(
        budget.address_space_bytes,
        max(1024 * 1024, source.width * source.height * 5 + 1024 * 1024),
    )
    _set_limit(resource.RLIMIT_FSIZE, maximum_file)
    signal.signal(signal.SIGXFSZ, signal.SIG_DFL)


def _set_limit(kind: int, requested: int) -> None:
    _soft, hard = resource.getrlimit(kind)
    value = requested if hard == resource.RLIM_INFINITY else min(requested, hard)
    resource.setrlimit(kind, (value, value))


def _arm_parent_death_signal(expected_parent: int) -> None:
    """Linux parent-death binding; the PID check closes the prctl race."""

    try:
        import ctypes

        libc = ctypes.CDLL(None, use_errno=True)
        if libc.prctl(1, signal.SIGKILL, 0, 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "prctl(PR_SET_PDEATHSIG) failed")
        if os.getppid() != expected_parent:
            os.kill(os.getpid(), signal.SIGKILL)
    except ImportError:
        # This project targets Linux.  Keeping the child resource-bounded is a
        # safe fallback for import-only analysis on another POSIX platform.
        return


def _load_sanitized(path: Path, source: ImageInput) -> DecodedImage:
    try:
        with Image.open(path) as opened:
            if (
                opened.format != "PNG"
                or opened.mode != "RGBA"
                or opened.size != (source.width, source.height)
                or opened.info
            ):
                raise OSError("sanitized decoder output has an invalid profile")
            opened.load()
            rgba = opened.copy()
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise RemovalFailure(
            "background.input-unreadable",
            "The isolated image decoder returned an invalid result.",
            "input",
            "decode",
        ) from exc
    return DecodedImage(rgba, rgba.getchannel("A"))


def decode_image(source: ImageInput, limits: Limits) -> DecodedImage:
    try:
        # O_NONBLOCK so a FIFO or device path returns immediately instead of
        # blocking until a writer appears.  Without it the S_ISREG guard below
        # is unreachable for exactly the input class it exists to refuse, and a
        # hostile path stalls the worker until its deadline.
        descriptor = os.open(
            source.path, os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK
        )
    except OSError as exc:
        raise RemovalFailure(
            "background.input-unreadable", "The input image cannot be read.", "input", "decode"
        ) from exc
    try:
        with os.fdopen(descriptor, "rb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode):
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The input must be a regular file.",
                    "input",
                    "decode",
                )
            if status.st_size != source.bytes or status.st_size > limits.max_input_bytes:
                raise RemovalFailure(
                    "background.input-limit",
                    "The input byte limit or identity does not match.",
                    "input",
                    "decode",
                )
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != source.sha256:
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The input digest does not match the request.",
                    "input",
                    "decode",
                )
            pixels = source.width * source.height
            if pixels > limits.max_decoded_pixels or pixels > 100_000_000:
                raise RemovalFailure(
                    "background.input-limit",
                    "The decoded pixel limit is exceeded.",
                    "resource",
                    "decode",
                )
            stream.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(stream) as opened:
                    if opened.size != (source.width, source.height):
                        raise RemovalFailure(
                            "background.input-unreadable",
                            "The decoded geometry does not match the request.",
                            "input",
                            "decode",
                        )
                    frames = int(getattr(opened, "n_frames", 1))
                    if frames > MAX_INPUT_FRAMES:
                        raise RemovalFailure(
                            "background.input-limit",
                            "The input frame limit is exceeded.",
                            "input",
                            "decode",
                        )
                    actual_media = FORMAT_MEDIA_TYPE.get(opened.format or "")
                    if actual_media != source.media_type:
                        raise RemovalFailure(
                            "background.input-unreadable",
                            "The image bytes do not match the declared media type.",
                            "input",
                            "decode",
                        )
                    exif = opened.getexif()
                    if int(exif.get(274, 1)) != 1:
                        raise RemovalFailure(
                            "background.invalid-request",
                            "The submitted input has not had orientation applied.",
                            "input",
                            "decode",
                        )
                    opened.load()
                    metadata_bytes = len(exif.tobytes()) + _metadata_size(opened.info)
                    if metadata_bytes > MAX_METADATA_BYTES:
                        raise RemovalFailure(
                            "background.input-limit",
                            "The image metadata limit is exceeded.",
                            "resource",
                            "decode",
                        )
                    rgba = opened.convert("RGBA")
                    icc = opened.info.get("icc_profile")
                    if icc:
                        try:
                            source_profile = ImageCms.ImageCmsProfile(io.BytesIO(bytes(icc)))
                            target_profile = ImageCms.createProfile("sRGB")
                            rgb = ImageCms.profileToProfile(
                                rgba.convert("RGB"),
                                source_profile,
                                target_profile,
                                outputMode="RGB",
                            )
                            if rgb is None:
                                raise ImageCms.PyCMSError("ICC conversion returned no image")
                            rgb.putalpha(rgba.getchannel("A"))
                            rgba = rgb.convert("RGBA")
                        except (OSError, ImageCms.PyCMSError) as exc:
                            raise RemovalFailure(
                                "background.input-unreadable",
                                "The embedded color profile cannot be converted safely.",
                                "input",
                                "decode",
                            ) from exc
                    elif source.color_space != "srgb":
                        raise RemovalFailure(
                            "background.invalid-request",
                            "A non-sRGB input requires a bounded embedded ICC profile.",
                            "input",
                            "decode",
                        )
    except RemovalFailure:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise RemovalFailure(
            "background.input-unreadable", "The input image cannot be decoded.", "input", "decode"
        ) from exc
    return DecodedImage(rgba, rgba.getchannel("A"))


def _metadata_size(info: dict[str, object]) -> int:
    total = 0
    for key, value in info.items():
        if key == "exif":
            continue
        if isinstance(value, bytes):
            total += len(value)
        elif isinstance(value, str):
            total += len(value.encode("utf-8", errors="replace"))
        elif isinstance(value, tuple | list):
            total += sum(len(str(item).encode("utf-8", errors="replace")) for item in value)
        if total > MAX_METADATA_BYTES:
            return total
    return total
