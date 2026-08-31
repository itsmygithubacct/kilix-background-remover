"""Causal controls for exact-cover, fixed-pixel inference tiling.

T-1 fails if a tile can exceed its fixed working-set pixel cap.
T-2 fails if two-axis subdivision loses, repeats, or reorders source pixels.
T-3 drives the production inference path and fails if a wide row bypasses T-1.
"""

from __future__ import annotations

import threading
from typing import Any

import pytest
from PIL import Image

from kilix_background_remover import worker as worker_module
from kilix_background_remover.tiling import iter_inference_tiles


@pytest.mark.parametrize(
    ("width", "height", "budget"),
    [(1, 1, 1), (11, 7, 5), (13, 17, 31), (19, 3, 7), (5, 23, 64)],
)
def test_t1_t2_tiles_are_bounded_and_cover_every_pixel_once(
    width: int,
    height: int,
    budget: int,
) -> None:
    tiles = tuple(iter_inference_tiles(width, height, max_pixels=budget))
    assert [tile.index for tile in tiles] == list(range(len(tiles)))
    assert all(0 < tile.pixels <= budget for tile in tiles)

    coverage = bytearray(width * height)
    for tile in tiles:
        for y in range(tile.top, tile.bottom):
            for x in range(tile.left, tile.right):
                coverage[y * width + x] += 1
    assert coverage == bytearray([1]) * (width * height)


def test_t1_wide_geometry_is_split_across_both_axes_without_materializing_pixels() -> None:
    width = 2_500_003
    height = 2
    budget = 1_048_576
    tiles = tuple(iter_inference_tiles(width, height, max_pixels=budget))
    assert len(tiles) == 6
    assert sum(tile.pixels for tile in tiles) == width * height
    assert max(tile.pixels for tile in tiles) == budget
    assert all(tile.height == 1 for tile in tiles)


@pytest.mark.parametrize(
    ("width", "height", "budget"),
    [(0, 1, 1), (1, 0, 1), (1, 1, 0), (-1, 1, 1), (True, 1, 1)],
)
def test_t1_invalid_geometry_is_refused(width: int, height: int, budget: int) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        tuple(iter_inference_tiles(width, height, max_pixels=budget))


def test_t3_production_inference_subdivides_a_row_wider_than_the_cap(
    monkeypatch: Any,
) -> None:
    class RedChannelSession:
        def __init__(self) -> None:
            self.shapes: list[tuple[int, ...]] = []

        def run(self, outputs: list[str], feeds: dict[str, Any]) -> list[Any]:
            assert outputs == ["mask"]
            tensor = feeds["image"]
            self.shapes.append(tuple(int(value) for value in tensor.shape))
            return [tensor[:, :1, :, :]]

    monkeypatch.setattr(worker_module, "INFERENCE_TILE_PIXELS", 7)
    source = Image.new("RGB", (11, 3))
    red = [(x + y * 11) % 256 for y in range(3) for x in range(11)]
    source.putdata([(value, 0, 0) for value in red])
    session = RedChannelSession()

    mask, warnings = worker_module._run_onnx_mask(session, source, threading.Event())

    assert mask.tobytes() == bytes(red)
    assert warnings == []
    assert session.shapes == [
        (1, 3, 1, 7),
        (1, 3, 1, 4),
        (1, 3, 1, 7),
        (1, 3, 1, 4),
        (1, 3, 1, 7),
        (1, 3, 1, 4),
    ]
    assert all(shape[2] * shape[3] <= 7 for shape in session.shapes)
