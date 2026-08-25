"""Frame-count refusal.

The technical plan requires rejecting inputs by frame count. Two of the four
accepted media types - TIFF and WebP - carry multiple frames natively, so the
requirement is not satisfied by the media-type allowlist.

Written before the guard: these controls must fail against the unguarded
decoder, which accepted a 30-frame TIFF and a 30-frame animated WebP and
discarded 29 frames of each without a word.

Mutation that must break the guard once added: remove the frame-count check and
these controls fail again.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
from PIL import Image

from kilix_background_remover.contracts import ImageInput, Limits
from kilix_background_remover.decode import decode_image
from kilix_background_remover.errors import RemovalFailure

LIMITS = Limits(
    deadline_ms=120_000,
    max_decoded_pixels=100_000_000,
    max_input_bytes=64 * 1024 * 1024,
    max_output_bytes=64 * 1024 * 1024,
)


def _multiframe(path: Path, image_format: str, frames: int, **save: object) -> ImageInput:
    pictures = [Image.new("RGB", (16, 16), (index * 8 % 256, 0, 0)) for index in range(frames)]
    pictures[0].save(path, format=image_format, save_all=True, append_images=pictures[1:], **save)
    media = {"TIFF": "image/tiff", "WEBP": "image/webp"}[image_format]
    return ImageInput(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        width=16,
        height=16,
        media_type=media,
        alpha_mode="opaque",
        color_space="srgb",
    )


def _single(path: Path, image_format: str) -> ImageInput:
    Image.new("RGB", (16, 16), (7, 8, 9)).save(path, format=image_format)
    media = {"TIFF": "image/tiff", "WEBP": "image/webp", "PNG": "image/png"}[image_format]
    return ImageInput(
        path=path,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        bytes=path.stat().st_size,
        width=16,
        height=16,
        media_type=media,
        alpha_mode="opaque",
        color_space="srgb",
    )


@pytest.mark.parametrize(
    ("image_format", "save"),
    [("TIFF", {}), ("WEBP", {"duration": 40})],
)
def test_multi_frame_input_is_refused_by_name(
    tmp_path: Path, image_format: str, save: dict[str, object]
) -> None:
    """Silently using frame 0 is data loss on the user's input.  The plan holds
    the audio path to 'never silent'; the image path is held to the same rule."""
    source = _multiframe(tmp_path / f"many.{image_format.lower()}", image_format, 30, **save)
    with pytest.raises(RemovalFailure) as caught:
        decode_image(source, LIMITS)
    assert caught.value.code == "background.input-limit"
    assert caught.value.safe_message == "The input frame limit is exceeded.", (
        f"refused by a different guard: {caught.value.safe_message}"
    )


@pytest.mark.parametrize("image_format", ["TIFF", "WEBP", "PNG"])
def test_single_frame_input_of_every_accepted_format_still_decodes(
    tmp_path: Path, image_format: str
) -> None:
    """The guard must not cost an ordinary single-frame input."""
    source = _single(tmp_path / f"one.{image_format.lower()}", image_format)
    decoded = decode_image(source, LIMITS)
    assert decoded.image.size == (16, 16)
