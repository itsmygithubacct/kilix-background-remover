"""Bounded, orientation-explicit image decode."""

from __future__ import annotations

import hashlib
import io
import os
import stat
import warnings
from dataclasses import dataclass

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
