"""Mask validation, edge policy and image rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image

from .contracts import EdgeSettings
from .errors import RemovalFailure


def prediction_to_mask(
    values: Sequence[float], width: int, height: int
) -> tuple[Image.Image, list[dict[str, str]]]:
    if len(values) != width * height:
        raise RemovalFailure(
            "background.inference-failed",
            "The model returned an invalid mask geometry.",
            "provider",
            "infer",
        )
    if any(not math.isfinite(value) for value in values):
        raise RemovalFailure(
            "background.inference-failed",
            "The model returned a non-finite mask.",
            "provider",
            "infer",
        )
    low = min(values)
    high = max(values)
    warnings: list[dict[str, str]] = []
    if high == low:
        if high <= 0.0:
            warnings.append(
                {
                    "code": "background.no-salient-object",
                    "safe_message": "No foreground was detected.",
                }
            )
            return Image.new("L", (width, height), 0), warnings
        raise RemovalFailure(
            "background.inference-failed",
            "The model returned a constant nonzero mask.",
            "provider",
            "infer",
        )
    payload = bytes(math.floor(max(0.0, min(1.0, value)) * 255.0 + 0.5) for value in values)
    return Image.frombytes("L", (width, height), payload), warnings


def apply_edge_policy(
    mask: Image.Image, source_alpha: Image.Image, edge: EdgeSettings
) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")
    threshold = edge.threshold_u8
    if edge.matting_mode == "none":
        mask = mask.point(lambda value: 255 if value >= threshold else 0)
    else:
        mask = mask.point(lambda value: value if value >= threshold else 0)
    if edge.feather_radius_px > 0:
        mask = _square_mean(mask, edge.feather_radius_px)
    if edge.preserve_source_alpha:
        alpha = source_alpha.convert("L").tobytes()
        payload = bytes(
            min(value, source) for value, source in zip(mask.tobytes(), alpha, strict=True)
        )
        mask = Image.frombytes("L", mask.size, payload)
    return mask


def _square_mean(mask: Image.Image, radius: int) -> Image.Image:
    width, height = mask.size
    source = mask.tobytes()
    span = radius * 2 + 1
    horizontal = [0] * (width * height)
    for y in range(height):
        row = source[y * width : (y + 1) * width]
        prefix = [0]
        for value in row:
            prefix.append(prefix[-1] + value)
        for x in range(width):
            low = max(0, x - radius)
            high = min(width - 1, x + radius)
            total = prefix[high + 1] - prefix[low]
            total += max(0, radius - x) * row[0]
            total += max(0, x + radius - width + 1) * row[-1]
            horizontal[y * width + x] = total
    divisor = span * span
    rounding = divisor // 2
    result = bytearray(width * height)
    for x in range(width):
        column = [horizontal[y * width + x] for y in range(height)]
        prefix = [0]
        for value in column:
            prefix.append(prefix[-1] + value)
        for y in range(height):
            low = max(0, y - radius)
            high = min(height - 1, y + radius)
            total = prefix[high + 1] - prefix[low]
            total += max(0, radius - y) * column[0]
            total += max(0, y + radius - height + 1) * column[-1]
            result[y * width + x] = (total + rounding) // divisor
    return Image.frombytes("L", mask.size, bytes(result))


def render_cutout(source: Image.Image, mask: Image.Image) -> Image.Image:
    result = source.copy().convert("RGBA")
    result.putalpha(mask)
    return result


def render_color_composite(
    source: Image.Image, mask: Image.Image, rgba: list[float]
) -> Image.Image:
    channels = tuple(round(channel * 255.0) for channel in rgba)
    background = Image.new("RGBA", source.size, channels)
    return Image.composite(source.convert("RGBA"), background, mask)


def render_image_composite(
    source: Image.Image, mask: Image.Image, background: Image.Image
) -> Image.Image:
    if background.size != source.size:
        raise RemovalFailure(
            "background.invalid-request",
            "A background image must match the foreground geometry.",
            "input",
            "postprocess",
        )
    return Image.composite(source.convert("RGBA"), background.convert("RGBA"), mask)
