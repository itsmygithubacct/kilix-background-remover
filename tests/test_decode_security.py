"""Decode security battery.

**Population and how it was enumerated.** Two independent sources, so a gap in
either direction is visible:

1. Every ``raise RemovalFailure`` site in ``decode.py``, read from source in
   execution order. Twelve guards, G1-G12 below.
2. The Security bullets of ``0.2.1-KILIX-BACKGROUND-REMOVER.md``: compressed
   bytes, decoded pixels, dimensions, metadata size, **frame count**, **decode
   time**, untrusted parsers, and **networklessness**.

Cross-referencing the two is the point. Guards G1-G12 all have a doc
requirement. Two doc requirements had **no decode guard at all**: frame count,
since closed by G13 in tests/test_frame_count.py, and decode time, which
remains a recorded null.

**Mutation table, written before the controls.** Each row states what must break
the control. A control whose mutation cannot be shown to fire is worth nothing.

===  ================================================  =====================================
ID   Control                                           Mutation that must break it
===  ================================================  =====================================
G1   symlink input refused (``O_NOFOLLOW``)            drop ``O_NOFOLLOW`` from the open
G2   non-regular input refused (``S_ISREG``)           drop the ``S_ISREG`` check
G3   size identity and ``max_input_bytes``             drop the size comparison
G4   SHA-256 identity                                  drop the digest comparison
G5   pixel budget, incl. the hard 100 MP cap           drop the pixel-budget check
G6   declared geometry equals actual                   drop the geometry comparison
G7   declared media type equals actual format          drop the media-type comparison
G8   EXIF orientation must already be applied          drop the tag-274 check
G9   metadata byte budget                              raise the budget to infinity
G11  non-sRGB without an ICC profile refused           drop the colour-space check
G12  malformed bytes refused, not crashed              (covered by the catch-all)
===  ================================================  =====================================

Nulls are asserted as nulls at the end of the file, not left implicit.
"""

from __future__ import annotations

import hashlib
import os
import socket
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


def _write_png(path: Path, size: tuple[int, int] = (8, 8), **save: object) -> Path:
    Image.new("RGB", size, (10, 20, 30)).save(path, format="PNG", **save)
    return path


def _describe(path: Path, **override: object) -> ImageInput:
    data = path.read_bytes()
    with Image.open(path) as opened:
        width, height = opened.size
    fields: dict[str, object] = {
        "path": path,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "width": width,
        "height": height,
        "media_type": "image/png",
        "alpha_mode": "opaque",
        "color_space": "srgb",
    }
    fields.update(override)
    return ImageInput(**fields)  # type: ignore[arg-type]


def _refusal(source: ImageInput) -> RemovalFailure:
    with pytest.raises(RemovalFailure) as caught:
        decode_image(source, LIMITS)
    return caught.value


# --- G1 --------------------------------------------------------------------
def test_g1_symlinked_input_is_refused(tmp_path: Path) -> None:
    real = _write_png(tmp_path / "real.png")
    link = tmp_path / "link.png"
    link.symlink_to(real)
    described = _describe(real)
    through_link = ImageInput(
        path=link,
        sha256=described.sha256,
        bytes=described.bytes,
        width=described.width,
        height=described.height,
        media_type=described.media_type,
        alpha_mode=described.alpha_mode,
        color_space=described.color_space,
    )
    failure = _refusal(through_link)
    assert failure.code == "background.input-unreadable"


# --- G2 --------------------------------------------------------------------
def test_g2_non_regular_input_is_refused(tmp_path: Path) -> None:
    fifo = tmp_path / "pipe.png"
    os.mkfifo(fifo)
    descriptor = os.open(fifo, os.O_RDONLY | os.O_NONBLOCK)
    try:
        failure = _refusal(
            ImageInput(
                path=fifo,
                sha256="0" * 64,
                bytes=0,
                width=8,
                height=8,
                media_type="image/png",
                alpha_mode="opaque",
                color_space="srgb",
            )
        )
        # The code alone cannot discriminate: the catch-all returns the same
        # code when the FIFO read fails with EAGAIN.  Name the check.
        assert failure.code == "background.input-unreadable"
        assert failure.safe_message == "The input must be a regular file.", (
            f"refused by a different guard: {failure.safe_message}"
        )
    finally:
        os.close(descriptor)


# --- G3 --------------------------------------------------------------------
def test_g3_declared_size_must_match_the_file(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "a.png")
    failure = _refusal(_describe(image, bytes=image.stat().st_size + 1))
    assert failure.code == "background.input-limit"


def test_g3_input_over_the_byte_budget_is_refused(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "b.png")
    tiny = Limits(
        deadline_ms=LIMITS.deadline_ms,
        max_decoded_pixels=LIMITS.max_decoded_pixels,
        max_input_bytes=8,
        max_output_bytes=LIMITS.max_output_bytes,
    )
    with pytest.raises(RemovalFailure) as caught:
        decode_image(_describe(image), tiny)
    assert caught.value.code == "background.input-limit"


# --- G4 --------------------------------------------------------------------
def test_g4_digest_mismatch_is_refused(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "c.png")
    failure = _refusal(_describe(image, sha256="f" * 64))
    assert failure.code == "background.input-unreadable"


# --- G5 --------------------------------------------------------------------
def test_g5_decompression_bomb_is_refused_before_decode() -> None:
    """The corpus 100 MP fixture is a genuine bomb by ratio: ~435 KB of PNG
    expands to ~400 MB of RGBA.

    Pillow's own DecompressionBombWarning is deliberately suppressed in
    decode.py, so this guard - not Pillow - is what stands between the process
    and that allocation. The refusal must arrive before any decode.
    """
    bomb = Path(__file__).resolve().parents[1] / "tests/fixtures/corpus/large-100mp.png"
    compressed = bomb.stat().st_size
    ratio = (10_000 * 10_000 * 4) / compressed
    assert ratio > 500, f"expansion ratio {ratio:.0f}x is not bomb-like"
    source = ImageInput(
        path=bomb,
        sha256=hashlib.sha256(bomb.read_bytes()).hexdigest(),
        bytes=compressed,
        width=10_000,
        height=10_000,
        media_type="image/png",
        alpha_mode="straight",
        color_space="srgb",
    )
    budget = Limits(
        deadline_ms=LIMITS.deadline_ms,
        max_decoded_pixels=1_000_000,
        max_input_bytes=LIMITS.max_input_bytes,
        max_output_bytes=LIMITS.max_output_bytes,
    )
    with pytest.raises(RemovalFailure) as caught:
        decode_image(source, budget)
    assert caught.value.code == "background.input-limit"


def test_g5_declared_pixels_over_the_hard_cap_are_refused(tmp_path: Path) -> None:
    """The 100 MP cap is checked against declared dimensions before the file is
    opened, so an oversized declaration cannot reach the decoder at all."""
    image = _write_png(tmp_path / "cap.png")
    failure = _refusal(_describe(image, width=100_001, height=1_001))
    assert failure.code == "background.input-limit"


def test_g5_pixel_budget_below_the_hard_cap_is_honoured(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "d.png", size=(64, 64))
    small = Limits(
        deadline_ms=LIMITS.deadline_ms,
        max_decoded_pixels=16,
        max_input_bytes=LIMITS.max_input_bytes,
        max_output_bytes=LIMITS.max_output_bytes,
    )
    with pytest.raises(RemovalFailure) as caught:
        decode_image(_describe(image), small)
    assert caught.value.code == "background.input-limit"


# --- G6 --------------------------------------------------------------------
def test_g6_declared_geometry_must_match_the_bytes(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "e.png", size=(8, 8))
    failure = _refusal(_describe(image, width=9))
    assert failure.code == "background.input-unreadable"


# --- G7 --------------------------------------------------------------------
def test_g7_media_type_confusion_is_refused(tmp_path: Path) -> None:
    """PNG bytes declared as JPEG. A parser selected by declaration rather than
    content is how a decoder is pointed at the wrong parser."""
    image = _write_png(tmp_path / "f.png")
    failure = _refusal(_describe(image, media_type="image/jpeg"))
    assert failure.code == "background.input-unreadable"


# --- G8 --------------------------------------------------------------------
def test_g8_unapplied_exif_orientation_is_refused(tmp_path: Path) -> None:
    image = tmp_path / "g.jpg"
    picture = Image.new("RGB", (8, 8), (1, 2, 3))
    exif = picture.getexif()
    exif[274] = 6
    picture.save(image, format="JPEG", exif=exif)
    failure = _refusal(_describe(image, media_type="image/jpeg"))
    assert failure.code == "background.invalid-request"


# --- G9 --------------------------------------------------------------------
def test_g9_metadata_bomb_is_refused(tmp_path: Path) -> None:
    image = tmp_path / "h.png"
    from PIL import PngImagePlugin

    info = PngImagePlugin.PngInfo()
    info.add_text("bulk", "A" * (2 * 1024 * 1024))
    Image.new("RGB", (8, 8), (4, 5, 6)).save(image, format="PNG", pnginfo=info)
    failure = _refusal(_describe(image))
    assert failure.code == "background.input-limit"


# --- G11 -------------------------------------------------------------------
def test_g11_non_srgb_without_icc_is_refused(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "i.png")
    failure = _refusal(_describe(image, color_space="display-p3"))
    assert failure.code == "background.invalid-request"


# --- G12 -------------------------------------------------------------------
def test_g12_truncated_image_is_refused_not_crashed(tmp_path: Path) -> None:
    image = _write_png(tmp_path / "j.png", size=(64, 64))
    data = image.read_bytes()
    truncated = tmp_path / "k.png"
    truncated.write_bytes(data[: len(data) // 2])
    source = ImageInput(
        path=truncated,
        sha256=hashlib.sha256(truncated.read_bytes()).hexdigest(),
        bytes=truncated.stat().st_size,
        width=64,
        height=64,
        media_type="image/png",
        alpha_mode="opaque",
        color_space="srgb",
    )
    failure = _refusal(source)
    assert failure.code == "background.input-unreadable"


def test_g12_non_image_bytes_are_refused(tmp_path: Path) -> None:
    junk = tmp_path / "l.png"
    junk.write_bytes(os.urandom(4096))
    source = ImageInput(
        path=junk,
        sha256=hashlib.sha256(junk.read_bytes()).hexdigest(),
        bytes=junk.stat().st_size,
        width=8,
        height=8,
        media_type="image/png",
        alpha_mode="opaque",
        color_space="srgb",
    )
    assert _refusal(source).code == "background.input-unreadable"


# --- networklessness -------------------------------------------------------
def test_decode_opens_no_socket(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Absence of an import is not proof. Make socket creation fatal and decode."""
    opened: list[str] = []

    def refuse(*args: object, **kwargs: object) -> None:
        opened.append("socket")
        raise AssertionError("decode attempted network access")

    monkeypatch.setattr(socket, "socket", refuse)
    monkeypatch.setattr(socket, "create_connection", refuse)
    image = _write_png(tmp_path / "m.png")
    decode_image(_describe(image), LIMITS)
    assert opened == []


# --- recorded nulls --------------------------------------------------------
def test_frame_count_null_is_now_closed(tmp_path: Path) -> None:
    """This was a recorded NULL and is now a guard - see tests/test_frame_count.py.

    The null as first written understated the exposure. It observed that a GIF
    is refused by media type and inferred the gap was bounded. It is not: TIFF
    and WebP are accepted media types and both carry frames natively, so a
    30-frame TIFF decoded as frame 0 with 29 frames discarded silently. The
    correction is recorded rather than quietly replaced.
    """
    animated = tmp_path / "n.gif"
    frames = []
    for index in range(24):
        frame = Image.new("RGB", (8, 8), (index * 10 % 256, 0, 0)).convert("P")
        frames.append(frame)
    frames[0].save(animated, format="GIF", save_all=True, append_images=frames[1:], duration=10)
    with Image.open(animated) as opened:
        assert getattr(opened, "n_frames", 1) == 24
    # The frame guard now runs before the media-type comparison, so an animated
    # GIF is refused as a frame-limit violation rather than an unknown media
    # type. That ordering is deliberate: frame count is a property of the bytes,
    # and it names the actual reason the input cannot be used.
    failure = _refusal(_describe(animated, media_type="image/gif"))
    assert failure.code == "background.input-limit"
    assert failure.safe_message == "The input frame limit is exceeded."
