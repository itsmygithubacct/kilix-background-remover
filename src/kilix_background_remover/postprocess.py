"""Mask validation, edge policy and image rendering."""

from __future__ import annotations

import math
from collections.abc import Sequence

from PIL import Image, ImageChops, ImageFilter

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
    payload = bytes(round(max(0.0, min(1.0, value)) * 255.0) for value in values)
    return Image.frombytes("L", (width, height), payload), warnings


def apply_edge_policy(
    mask: Image.Image, source_alpha: Image.Image, edge: EdgeSettings
) -> Image.Image:
    if mask.mode != "L":
        mask = mask.convert("L")
    threshold = round(edge.threshold * 255.0)
    if edge.matting_mode == "none":
        mask = mask.point(lambda value: 255 if value >= threshold else 0)
    elif threshold > 0:
        denominator = max(1, 255 - threshold)
        mask = mask.point(lambda value: round(max(0, value - threshold) * 255 / denominator))
    if edge.feather_radius_px > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(edge.feather_radius_px))
    if edge.preserve_source_alpha:
        mask = ImageChops.multiply(mask, source_alpha.convert("L"))
    return mask


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
