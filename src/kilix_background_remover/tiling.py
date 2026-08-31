"""Deterministic tile geometry for the bounded inference working set."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class InferenceTile:
    """One non-overlapping core tile in row-major inference order."""

    index: int
    left: int
    top: int
    right: int
    bottom: int

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def box(self) -> tuple[int, int, int, int]:
        return self.left, self.top, self.right, self.bottom


def _positive_integer(value: int, field: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def iter_inference_tiles(
    width: int,
    height: int,
    *,
    max_pixels: int,
) -> Iterator[InferenceTile]:
    """Yield exact-cover tiles whose individual pixel counts never exceed the cap.

    Full-width horizontal strips are retained whenever one source row fits the
    working-set cap. Wider hostile geometries are subdivided across both axes.
    The cores neither overlap nor leave gaps; seam and overlap policy therefore
    remain outside this resource-control primitive.
    """

    checked_width = _positive_integer(width, "width")
    checked_height = _positive_integer(height, "height")
    checked_max = _positive_integer(max_pixels, "max_pixels")

    tile_width = min(checked_width, checked_max)
    tile_height = min(checked_height, max(1, checked_max // tile_width))
    index = 0
    for top in range(0, checked_height, tile_height):
        bottom = min(checked_height, top + tile_height)
        for left in range(0, checked_width, tile_width):
            right = min(checked_width, left + tile_width)
            yield InferenceTile(index, left, top, right, bottom)
            index += 1
