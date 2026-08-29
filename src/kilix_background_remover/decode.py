"""Bounded, orientation-explicit image decode."""

from __future__ import annotations

import hashlib
import io
import json
import mmap
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
MAX_DECODE_STATUS_BYTES = 4_096
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
class InspectedImage:
    path: Path
    sha256: str
    bytes: int
    width: int
    height: int
    media_type: str
    alpha_mode: str
    color_space: str = "srgb"


@dataclass(frozen=True, slots=True)
class DecodeBudget:
    """Hard limits applied to the disposable untrusted-parser process."""

    wall_seconds: float = 30.0
    cpu_seconds: int = 30
    address_space_bytes: int = 2 * 1024 * 1024 * 1024


DEFAULT_DECODE_BUDGET = DecodeBudget()


def inspect_image_bounded(
    path: Path,
    *,
    max_input_bytes: int,
    max_decoded_pixels: int,
    budget: DecodeBudget = DEFAULT_DECODE_BUDGET,
) -> InspectedImage:
    """Identify request metadata without parsing image bytes in the caller."""

    if max_input_bytes <= 0 or max_decoded_pixels <= 0:
        raise ValueError("inspection limits must be positive")
    if budget.wall_seconds <= 0 or budget.cpu_seconds <= 0 or budget.address_space_bytes <= 0:
        raise ValueError("inspection budget must be positive")
    absolute, byte_count, digest = _identify_regular_image(path, max_input_bytes)
    context: Any = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="kilix-f108-inspect-") as temporary:
        root = Path(temporary)
        os.chmod(root, 0o700)
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_inspect_child,
            args=(
                child,
                absolute,
                byte_count,
                digest,
                max_decoded_pixels,
                budget,
                os.getpid(),
            ),
            name="kilix-f108-image-inspector",
            daemon=True,
        )
        process.start()
        child.close()
        try:
            message: dict[str, object] | None = None
            if parent.poll(budget.wall_seconds):
                try:
                    message = _parse_decode_status(parent.recv_bytes(MAX_DECODE_STATUS_BYTES))
                except (EOFError, OSError):
                    message = None
            if message is None:
                if process.is_alive():
                    _terminate_process(process)
                    raise RemovalFailure(
                        "background.input-limit",
                        "The image inspector exceeded its time limit.",
                        "resource",
                        "decode",
                    )
                raise RemovalFailure(
                    "background.input-limit",
                    "The image inspector exceeded its resource limit.",
                    "resource",
                    "decode",
                )
            process.join(timeout=1.0)
            if process.is_alive():
                _terminate_process(process)
            kind = message.get("kind")
            if kind == "failure":
                raise RemovalFailure(
                    cast(str, message["code"]),
                    cast(str, message["safe_message"]),
                    cast(str, message["category"]),
                    cast(str, message["phase"]),
                    cast(bool, message["retryable"]),
                )
            if kind == "limit":
                raise RemovalFailure(
                    "background.input-limit",
                    "The image inspector exceeded its resource limit.",
                    "resource",
                    "decode",
                )
            if (
                kind != "inspection"
                or message.get("bytes") != byte_count
                or message.get("sha256") != digest
            ):
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The isolated image inspector returned an invalid result.",
                    "input",
                    "decode",
                )
            return InspectedImage(
                path=absolute,
                sha256=digest,
                bytes=byte_count,
                width=cast(int, message["width"]),
                height=cast(int, message["height"]),
                media_type=cast(str, message["media_type"]),
                alpha_mode=cast(str, message["alpha_mode"]),
            )
        finally:
            parent.close()
            if process.is_alive():
                _terminate_process(process)


def _identify_regular_image(path: Path, maximum: int) -> tuple[Path, int, str]:
    absolute = path.absolute()
    try:
        descriptor = os.open(
            absolute,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as exc:
        raise RemovalFailure(
            "background.input-unreadable",
            "The input must be a regular file.",
            "input",
            "decode",
        ) from exc
    try:
        status = os.fstat(descriptor)
        if not stat.S_ISREG(status.st_mode):
            raise RemovalFailure(
                "background.input-unreadable",
                "The input must be a regular file.",
                "input",
                "decode",
            )
        if not 1 <= status.st_size <= maximum:
            raise RemovalFailure(
                "background.input-limit",
                "The input exceeds the fixed byte limit.",
                "resource",
                "decode",
            )
        digest = hashlib.sha256()
        while True:
            block = os.read(descriptor, 1024 * 1024)
            if not block:
                break
            digest.update(block)
        after = os.fstat(descriptor)
        if (
            after.st_dev != status.st_dev
            or after.st_ino != status.st_ino
            or after.st_size != status.st_size
        ):
            raise RemovalFailure(
                "background.input-unreadable",
                "The input changed while it was identified.",
                "input",
                "decode",
            )
        return absolute, status.st_size, digest.hexdigest()
    finally:
        os.close(descriptor)


def decode_image_bounded(
    source: ImageInput,
    limits: Limits,
    *,
    budget: DecodeBudget = DEFAULT_DECODE_BUDGET,
) -> DecodedImage:
    """Decode hostile bytes in a killable, resource-limited child process.

    Only a fixed-size, metadata-free RGBA raster crosses back into the
    long-lived worker.  The parent never unpickles child data or invokes an
    image parser.  A parser stall, crash, or memory-limit exit therefore cannot
    leave the model session process executing hostile decoder state.
    """

    wall_seconds = min(budget.wall_seconds, limits.deadline_ms / 1000.0)
    if wall_seconds <= 0 or budget.cpu_seconds <= 0 or budget.address_space_bytes <= 0:
        raise ValueError("decode limits must be positive")
    context: Any = mp.get_context("spawn")
    with tempfile.TemporaryDirectory(prefix="kilix-f108-decode-") as temporary:
        root = Path(temporary)
        os.chmod(root, 0o700)
        sanitized = root / "sanitized.rgba"
        parent, child = context.Pipe(duplex=False)
        process = context.Process(
            target=_decode_child,
            args=(child, source, limits, budget, sanitized, os.getpid()),
            name="kilix-f108-image-decoder",
            daemon=True,
        )
        process.start()
        child.close()
        message: dict[str, object] | None = None
        try:
            if parent.poll(wall_seconds):
                try:
                    received = parent.recv_bytes(MAX_DECODE_STATUS_BYTES)
                except (EOFError, OSError):
                    received = b""
                message = _parse_decode_status(received)
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
            kind = message.get("kind")
            if kind == "failure":
                raise RemovalFailure(
                    cast(str, message["code"]),
                    cast(str, message["safe_message"]),
                    cast(str, message["category"]),
                    cast(str, message["phase"]),
                    cast(bool, message["retryable"]),
                )
            if kind == "limit":
                raise RemovalFailure(
                    "background.input-limit",
                    "The image decoder exceeded its resource limit.",
                    "resource",
                    "decode",
                )
            if kind != "ok":
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The input image cannot be decoded.",
                    "input",
                    "decode",
                )
            return _load_sanitized_raster(sanitized, source)
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


def _parse_decode_status(payload: bytes) -> dict[str, object] | None:
    """Decode the child's one bounded, closed status record without pickle."""

    if not payload or len(payload) > MAX_DECODE_STATUS_BYTES:
        return None

    def reject_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
        record: dict[str, object] = {}
        for key, value in pairs:
            if key in record:
                raise ValueError("duplicate status key")
            record[key] = value
        return record

    try:
        decoded = json.loads(payload.decode("utf-8"), object_pairs_hook=reject_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
        return None
    if not isinstance(decoded, dict):
        return None
    kind = decoded.get("kind")
    if kind in {"ok", "limit", "crash"}:
        return decoded if set(decoded) == {"kind"} else None
    if kind == "inspection":
        if set(decoded) != {
            "alpha_mode",
            "bytes",
            "height",
            "kind",
            "media_type",
            "sha256",
            "width",
        }:
            return None
        if (
            not isinstance(decoded["bytes"], int)
            or isinstance(decoded["bytes"], bool)
            or decoded["bytes"] <= 0
            or not isinstance(decoded["width"], int)
            or isinstance(decoded["width"], bool)
            or decoded["width"] <= 0
            or not isinstance(decoded["height"], int)
            or isinstance(decoded["height"], bool)
            or decoded["height"] <= 0
            or decoded["media_type"] not in FORMAT_MEDIA_TYPE.values()
            or decoded["alpha_mode"] not in {"opaque", "straight"}
            or not isinstance(decoded["sha256"], str)
            or len(decoded["sha256"]) != 64
            or any(character not in "0123456789abcdef" for character in decoded["sha256"])
        ):
            return None
        return cast(dict[str, object], decoded)
    if kind != "failure" or set(decoded) != {
        "category",
        "code",
        "kind",
        "phase",
        "retryable",
        "safe_message",
    }:
        return None
    if not all(
        isinstance(decoded[field], str) and bool(decoded[field])
        for field in ("category", "code", "phase", "safe_message")
    ) or not isinstance(decoded["retryable"], bool):
        return None
    return cast(dict[str, object], decoded)


def _send_decode_status(connection: Connection, record: dict[str, object]) -> None:
    payload = json.dumps(
        record,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("ascii")
    if len(payload) > MAX_DECODE_STATUS_BYTES:
        raise OSError("decode status exceeds its fixed bound")
    connection.send_bytes(payload)


def _inspect_child(
    connection: Connection,
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    max_decoded_pixels: int,
    budget: DecodeBudget,
    expected_parent: int,
) -> None:
    try:
        _arm_parent_death_signal(expected_parent)
        os.umask(0o077)
        _set_limit(resource.RLIMIT_AS, budget.address_space_bytes)
        _set_limit(resource.RLIMIT_CPU, budget.cpu_seconds)
        _set_limit(resource.RLIMIT_FSIZE, 1024 * 1024)
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
        with os.fdopen(descriptor, "rb") as stream:
            status = os.fstat(stream.fileno())
            if not stat.S_ISREG(status.st_mode) or status.st_size != expected_bytes:
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The input changed before inspection.",
                    "input",
                    "decode",
                )
            digest = hashlib.sha256()
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
            if digest.hexdigest() != expected_sha256:
                raise RemovalFailure(
                    "background.input-unreadable",
                    "The input changed before inspection.",
                    "input",
                    "decode",
                )
            stream.seek(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", Image.DecompressionBombWarning)
                with Image.open(stream) as opened:
                    media_type = FORMAT_MEDIA_TYPE.get(opened.format or "")
                    if media_type is None:
                        raise RemovalFailure(
                            "background.input-unreadable",
                            "The image format is not supported.",
                            "input",
                            "decode",
                        )
                    width, height = opened.size
                    if (
                        width <= 0
                        or height <= 0
                        or width * height > max_decoded_pixels
                        or width * height > 100_000_000
                    ):
                        raise RemovalFailure(
                            "background.input-limit",
                            "The image exceeds the 100 megapixel input bound.",
                            "resource",
                            "decode",
                        )
                    if int(getattr(opened, "n_frames", 1)) > MAX_INPUT_FRAMES:
                        raise RemovalFailure(
                            "background.input-limit",
                            "The input frame limit is exceeded.",
                            "input",
                            "decode",
                        )
                    exif = opened.getexif()
                    if int(exif.get(274, 1)) != 1:
                        raise RemovalFailure(
                            "background.invalid-request",
                            "Apply EXIF orientation before submitting the image.",
                            "input",
                            "decode",
                        )
                    if len(exif.tobytes()) + _metadata_size(opened.info) > MAX_METADATA_BYTES:
                        raise RemovalFailure(
                            "background.input-limit",
                            "The image metadata limit is exceeded.",
                            "resource",
                            "decode",
                        )
                    alpha_mode = "straight" if "A" in opened.getbands() else "opaque"
        _send_decode_status(
            connection,
            {
                "kind": "inspection",
                "bytes": expected_bytes,
                "sha256": expected_sha256,
                "width": width,
                "height": height,
                "media_type": media_type,
                "alpha_mode": alpha_mode,
            },
        )
    except RemovalFailure as failure:
        with suppress(BrokenPipeError, EOFError, OSError):
            _send_decode_status(
                connection,
                {
                    "kind": "failure",
                    "code": failure.code,
                    "safe_message": failure.safe_message,
                    "category": failure.category,
                    "phase": failure.phase,
                    "retryable": failure.retryable,
                },
            )
    except MemoryError:
        with suppress(BrokenPipeError, EOFError, OSError):
            _send_decode_status(connection, {"kind": "limit"})
    except (OSError, UnidentifiedImageError, ValueError):
        with suppress(BrokenPipeError, EOFError, OSError):
            _send_decode_status(
                connection,
                {
                    "kind": "failure",
                    "code": "background.input-unreadable",
                    "safe_message": "The input image cannot be inspected.",
                    "category": "input",
                    "phase": "decode",
                    "retryable": False,
                },
            )
    except BaseException:
        with suppress(BrokenPipeError, EOFError, OSError):
            _send_decode_status(connection, {"kind": "crash"})
    finally:
        connection.close()


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
        raster = decoded.image.tobytes("raw", "RGBA")
        if len(raster) != source.width * source.height * 4:
            raise OSError("decoded raster has an invalid size")
        _write_sanitized_raster(sanitized, raster)
        _send_decode_status(connection, {"kind": "ok"})
    except RemovalFailure as failure:
        _send_decode_status(
            connection,
            {
                "kind": "failure",
                "code": failure.code,
                "safe_message": failure.safe_message,
                "category": failure.category,
                "phase": failure.phase,
                "retryable": failure.retryable,
            },
        )
    except (MemoryError, OSError):
        with suppress(BrokenPipeError, EOFError, OSError):
            _send_decode_status(connection, {"kind": "limit"})
    except BaseException:
        with suppress(BrokenPipeError, EOFError, OSError):
            _send_decode_status(connection, {"kind": "crash"})
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


def _write_sanitized_raster(destination: Path, payload: bytes) -> None:
    temporary = destination.with_name(destination.name + ".partial")
    descriptor = os.open(
        temporary,
        os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC | os.O_NOFOLLOW,
        0o600,
    )
    try:
        position = 0
        while position < len(payload):
            written = os.write(descriptor, payload[position : position + 1024 * 1024])
            if written <= 0:
                raise OSError("short sanitized raster write")
            position += written
        os.fsync(descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    finally:
        os.close(descriptor)
    os.replace(temporary, destination)
    directory = os.open(destination.parent, os.O_RDONLY | os.O_DIRECTORY | os.O_CLOEXEC)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def _load_sanitized_raster(path: Path, source: ImageInput) -> DecodedImage:
    expected = source.width * source.height * 4
    mapped: mmap.mmap | None = None
    borrowed: Image.Image | None = None
    try:
        descriptor = os.open(
            path,
            os.O_RDONLY | os.O_CLOEXEC | os.O_NOFOLLOW | os.O_NONBLOCK,
        )
    except OSError as exc:
        raise RemovalFailure(
            "background.input-unreadable",
            "The isolated image decoder returned an invalid result.",
            "input",
            "decode",
        ) from exc
    try:
        status = os.fstat(descriptor)
        if (
            not stat.S_ISREG(status.st_mode)
            or stat.S_IMODE(status.st_mode) != 0o600
            or status.st_size != expected
        ):
            raise OSError("sanitized raster has an invalid profile")
        mapped = mmap.mmap(descriptor, expected, access=mmap.ACCESS_READ)
        borrowed = Image.frombuffer(
            "RGBA",
            (source.width, source.height),
            cast(Any, mapped),
            "raw",
            "RGBA",
            0,
            1,
        )
        rgba = borrowed.copy()
        borrowed = None
        mapped.close()
        mapped = None
        after = os.fstat(descriptor)
        if (
            after.st_dev != status.st_dev
            or after.st_ino != status.st_ino
            or after.st_size != status.st_size
        ):
            raise OSError("sanitized raster changed during handoff")
    except MemoryError as exc:
        raise RemovalFailure(
            "background.input-limit",
            "The isolated image raster exceeds the worker memory limit.",
            "resource",
            "decode",
        ) from exc
    except (OSError, ValueError) as exc:
        raise RemovalFailure(
            "background.input-unreadable",
            "The isolated image decoder returned an invalid result.",
            "input",
            "decode",
        ) from exc
    finally:
        borrowed = None
        if mapped is not None:
            mapped.close()
        os.close(descriptor)
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
