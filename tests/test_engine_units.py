from __future__ import annotations

import math
from dataclasses import replace
from pathlib import Path

import pytest
from PIL import Image, PngImagePlugin

from kilix_background_remover.atomic import commit_staged, stage_image
from kilix_background_remover.contracts import (
    EdgeSettings,
    ImageInput,
    Limits,
    parse_request,
    sha256_file,
)
from kilix_background_remover.decode import decode_image
from kilix_background_remover.errors import RemovalFailure
from kilix_background_remover.postprocess import (
    apply_edge_policy,
    prediction_to_mask,
    render_color_composite,
    render_cutout,
    render_image_composite,
)


def test_decode_verifies_identity_alpha_and_limits(
    request_factory: object, corpus: Path, tmp_path: Path
) -> None:
    request = request_factory(  # type: ignore[operator]
        tmp_path, source=corpus / "existing-alpha.png"
    )
    parsed = parse_request(request)
    decoded = decode_image(parsed.input, parsed.limits)
    assert decoded.image.mode == "RGBA"
    assert decoded.image.size == (128, 96)
    assert decoded.source_alpha.getextrema() != (255, 255)

    with pytest.raises(RemovalFailure, match="digest"):
        decode_image(replace(parsed.input, sha256="0" * 64), parsed.limits)
    with pytest.raises(RemovalFailure, match="pixel limit"):
        decode_image(parsed.input, replace(parsed.limits, max_decoded_pixels=100))
    with pytest.raises(RemovalFailure, match="embedded ICC"):
        decode_image(replace(parsed.input, color_space="linear-srgb"), parsed.limits)


def test_decode_rejects_symlink_and_corrupt_file(corpus: Path, tmp_path: Path) -> None:
    source = corpus / "portrait-hair.png"
    link = tmp_path / "input.png"
    link.symlink_to(source)
    linked = ImageInput(
        path=link,
        sha256=sha256_file(source),
        bytes=source.stat().st_size,
        width=128,
        height=96,
        media_type="image/png",
        alpha_mode="straight",
        color_space="srgb",
    )
    limits = Limits(1000, 100_000_000, 268_435_456, 536_870_912)
    with pytest.raises(RemovalFailure, match="cannot be read"):
        decode_image(linked, limits)

    corrupt_path = corpus / "corrupt-truncated.png"
    corrupt = replace(
        linked,
        path=corrupt_path,
        sha256=sha256_file(corrupt_path),
        bytes=corrupt_path.stat().st_size,
    )
    with pytest.raises(RemovalFailure, match="cannot be decoded"):
        decode_image(corrupt, limits)


def test_decode_rejects_orientation_huge_metadata_and_invalid_icc(tmp_path: Path) -> None:
    limits = Limits(1000, 100_000_000, 268_435_456, 536_870_912)

    oriented_path = tmp_path / "oriented.jpg"
    exif = Image.Exif()
    exif[274] = 6
    Image.new("RGB", (4, 3), "red").save(oriented_path, exif=exif)
    oriented = ImageInput(
        oriented_path,
        sha256_file(oriented_path),
        oriented_path.stat().st_size,
        4,
        3,
        "image/jpeg",
        "opaque",
        "srgb",
    )
    with pytest.raises(RemovalFailure, match="orientation"):
        decode_image(oriented, limits)

    metadata_path = tmp_path / "metadata.png"
    metadata = PngImagePlugin.PngInfo()
    metadata.add_text("oversized", "x" * 1_048_577)
    Image.new("RGB", (4, 3), "red").save(metadata_path, pnginfo=metadata)
    metadata_input = replace(
        oriented,
        path=metadata_path,
        sha256=sha256_file(metadata_path),
        bytes=metadata_path.stat().st_size,
        media_type="image/png",
    )
    with pytest.raises(RemovalFailure, match="metadata limit"):
        decode_image(metadata_input, limits)

    icc_path = tmp_path / "invalid-icc.png"
    Image.new("RGB", (4, 3), "red").save(icc_path, icc_profile=b"not-an-icc-profile")
    icc_input = replace(
        oriented,
        path=icc_path,
        sha256=sha256_file(icc_path),
        bytes=icc_path.stat().st_size,
        media_type="image/png",
    )
    with pytest.raises(RemovalFailure, match="color profile"):
        decode_image(icc_input, limits)


def test_mask_guards_edge_policy_and_compositing() -> None:
    mask, warnings = prediction_to_mask([0.0, 0.25, 0.75, 1.0], 2, 2)
    assert mask.getpixel((0, 0)) == 0
    assert warnings == []
    empty, warnings = prediction_to_mask([0.0, 0.0, 0.0, 0.0], 2, 2)
    assert empty.getextrema() == (0, 0)
    assert warnings[0]["code"] == "background.no-salient-object"
    with pytest.raises(RemovalFailure, match="constant nonzero"):
        prediction_to_mask([0.5] * 4, 2, 2)
    with pytest.raises(RemovalFailure, match="non-finite"):
        prediction_to_mask([0.0, 1.0, math.nan, 0.5], 2, 2)

    source = Image.new("RGBA", (2, 2), (255, 0, 0, 128))
    edge = EdgeSettings(0.0, 0.0, "alpha", True)
    multiplied = apply_edge_policy(Image.new("L", (2, 2), 255), source.getchannel("A"), edge)
    assert multiplied.getextrema() == (128, 128)
    assert render_cutout(source, multiplied).getchannel("A").getextrema() == (128, 128)
    assert render_color_composite(source, multiplied, [0.0, 0.0, 1.0, 1.0]).size == (
        2,
        2,
    )
    assert render_image_composite(source, multiplied, Image.new("RGB", (2, 2))).size == (
        2,
        2,
    )


def test_atomic_multi_output_rollback_and_commit(tmp_path: Path) -> None:
    first = stage_image(
        Image.new("L", (8, 8), 128),
        tmp_path / "mask.png",
        image_format="PNG",
        media_type="image/png",
        kind="mask",
        max_output_bytes=1024 * 1024,
        staging_token="a" * 32,
    )
    second = stage_image(
        Image.new("RGBA", (8, 8), (1, 2, 3, 4)),
        tmp_path / "cutout.png",
        image_format="PNG",
        media_type="image/png",
        kind="cutout-png",
        max_output_bytes=1024 * 1024,
        staging_token="a" * 32,
    )
    with pytest.raises(RemovalFailure, match="could not be committed"):
        commit_staged([first, second], fail_after_links=1)
    assert not first.destination.exists()
    assert not second.destination.exists()
    assert not first.stage.exists()
    assert not second.stage.exists()

    committed = stage_image(
        Image.new("L", (8, 8), 255),
        tmp_path / "committed.png",
        image_format="PNG",
        media_type="image/png",
        kind="mask",
        max_output_bytes=1024 * 1024,
        staging_token="b" * 32,
    )
    commit_staged([committed])
    assert committed.destination.is_file()
    assert sha256_file(committed.destination) == committed.sha256
    assert not committed.stage.exists()
