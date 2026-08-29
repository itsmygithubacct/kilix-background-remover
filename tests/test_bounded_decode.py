"""Disposable-parser controls for the production image decode path."""

from __future__ import annotations

import hashlib
import tempfile
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image

from kilix_background_remover.contracts import ImageInput, Limits
from kilix_background_remover.decode import DecodeBudget, decode_image_bounded
from kilix_background_remover.errors import RemovalFailure


def _source(path: Path) -> ImageInput:
    payload = path.read_bytes()
    with Image.open(path) as opened:
        width, height = opened.size
    return ImageInput(
        path=path,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
        width=width,
        height=height,
        media_type="image/png",
        alpha_mode="straight",
        color_space="srgb",
    )


def _limits() -> Limits:
    return Limits(
        deadline_ms=10_000,
        max_decoded_pixels=1_000_000,
        max_input_bytes=1_000_000,
        max_output_bytes=10_000_000,
    )


def test_bounded_decoder_returns_only_sanitized_pixels(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    image = Image.new("RGBA", (17, 13), (11, 22, 33, 44))
    image.save(path, pnginfo=None)

    decoded = decode_image_bounded(_source(path), _limits())

    assert decoded.image.mode == "RGBA"
    assert decoded.image.size == (17, 13)
    assert decoded.image.info == {}
    assert decoded.source_alpha.getextrema() == (44, 44)


def test_bounded_decoder_kills_a_wall_time_overrun_and_cleans_scratch(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (64, 64), (1, 2, 3)).save(path)
    monkeypatch.setattr(tempfile, "tempdir", str(tmp_path))

    with pytest.raises(RemovalFailure) as caught:
        decode_image_bounded(
            _source(path),
            _limits(),
            budget=DecodeBudget(wall_seconds=0.001, cpu_seconds=1),
        )

    assert caught.value.code == "background.input-limit"
    assert caught.value.safe_message == "The image decoder exceeded its time limit."
    assert not list(tmp_path.glob("kilix-f108-decode-*"))


def test_bounded_decoder_refuses_a_memory_ceiling_breach(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    Image.effect_noise((2048, 2048), 64).convert("RGBA").save(path)
    large_limits = Limits(
        deadline_ms=10_000,
        max_decoded_pixels=5_000_000,
        max_input_bytes=32 * 1024 * 1024,
        max_output_bytes=32 * 1024 * 1024,
    )

    with pytest.raises(RemovalFailure) as caught:
        decode_image_bounded(
            _source(path),
            large_limits,
            budget=DecodeBudget(
                wall_seconds=10,
                cpu_seconds=10,
                address_space_bytes=16 * 1024 * 1024,
            ),
        )

    assert caught.value.code == "background.input-limit"
    assert caught.value.safe_message == "The image decoder exceeded its resource limit."


def test_bounded_decoder_preserves_a_typed_identity_refusal(tmp_path: Path) -> None:
    path = tmp_path / "source.png"
    Image.new("RGB", (8, 8), (1, 2, 3)).save(path)
    source = replace(_source(path), sha256="f" * 64)

    with pytest.raises(RemovalFailure) as caught:
        decode_image_bounded(source, _limits())

    assert caught.value.code == "background.input-unreadable"
    assert caught.value.safe_message == "The input digest does not match the request."
