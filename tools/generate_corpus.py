#!/usr/bin/env python3
"""Generate the wholly owned, deterministic F108 image/mask corpus."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import zlib
from collections.abc import Callable, Iterable
from pathlib import Path

RGBA = tuple[int, int, int, int]
PixelFn = Callable[[int, int], tuple[RGBA, int]]


def _chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload))
    )


def write_png(path: Path, width: int, height: int, channels: int, rows: Iterable[bytes]) -> None:
    color_type = {1: 0, 4: 6}[channels]
    compressor = zlib.compressobj(level=9)
    compressed: list[bytes] = []
    count = 0
    for row in rows:
        if len(row) != width * channels:
            raise ValueError("row length does not match PNG geometry")
        count += 1
        part = compressor.compress(b"\x00" + row)
        if part:
            compressed.append(part)
    if count != height:
        raise ValueError("row count does not match PNG geometry")
    compressed.append(compressor.flush())
    ihdr = struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)
    path.write_bytes(
        b"\x89PNG\r\n\x1a\n"
        + _chunk(b"IHDR", ihdr)
        + _chunk(b"IDAT", b"".join(compressed))
        + _chunk(b"IEND", b"")
    )


def render_pair(
    root: Path, name: str, width: int, height: int, pixel: PixelFn
) -> tuple[Path, Path]:
    image_path = root / f"{name}.png"
    mask_path = root / f"{name}.mask.png"

    def image_rows() -> Iterable[bytes]:
        for y in range(height):
            row = bytearray()
            for x in range(width):
                rgba, _ = pixel(x, y)
                row.extend(rgba)
            yield bytes(row)

    def mask_rows() -> Iterable[bytes]:
        for y in range(height):
            yield bytes(pixel(x, y)[1] for x in range(width))

    write_png(image_path, width, height, 4, image_rows())
    write_png(mask_path, width, height, 1, mask_rows())
    return image_path, mask_path


def small_pixel(kind: str, width: int, height: int) -> PixelFn:
    cx, cy = width / 2.0, height / 2.0

    def fn(x: int, y: int) -> tuple[RGBA, int]:
        checker = 38 + 18 * ((x // 8 + y // 8) % 2)
        bg: RGBA = (checker, checker + 5, checker + 9, 255)
        dx, dy = x - cx, y - cy
        radius = math.hypot(dx, dy)
        mask = 0
        color: RGBA = (216, 92, 48, 255)
        if kind == "hair":
            mask = 255 if radius < 25 or (18 < radius < 42 and (x * 7 + y * 11) % 19 < 2) else 0
        elif kind == "fur":
            edge = 30 + ((x * 17 + y * 13) % 11) - 5
            mask = 255 if radius < edge else 0
        elif kind == "translucent":
            mask = 128 if abs(dx) < 34 and abs(dy) < 24 else 0
            color = (80, 190, 210, 150)
        elif kind == "product":
            mask = 255 if abs(dx) < 28 and abs(dy) < 34 else 0
            color = (210, 214, 224, 255)
        elif kind == "plant":
            stem = abs(dx) < 2 and -30 < dy < 34
            leaves = ((dx + 15) ** 2 + (dy + 8) ** 2 < 13**2) or (
                (dx - 15) ** 2 + (dy - 8) ** 2 < 13**2
            )
            mask = 255 if stem or leaves else 0
            color = (52, 170, 73, 255)
        elif kind == "vehicle":
            body = abs(dx) < 38 and -12 < dy < 18
            cabin = abs(dx) < 20 and -28 < dy <= -12
            wheels = (dx + 24) ** 2 + (dy - 19) ** 2 < 8**2 or (dx - 24) ** 2 + (
                dy - 19
            ) ** 2 < 8**2
            mask = 255 if body or cabin or wheels else 0
            color = (42, 115, 220, 255)
        elif kind == "illustration":
            mask = 255 if abs(dx) + abs(dy) < 39 else 0
            color = (238, 194, 38, 255)
        elif kind == "low-contrast":
            mask = 255 if radius < 31 else 0
            bg = (120, 124, 126, 255)
            color = (132, 136, 138, 255)
        elif kind == "clutter":
            bg = ((x * 37 + y * 17) % 256, (x * 11 + y * 29) % 256, (x * 23 + y * 7) % 256, 255)
            mask = 255 if radius < 24 else 0
        elif kind == "thin-structure":
            mask = 255 if abs(dx) < 1.5 or abs(dy) < 1.5 or abs(dx - dy) < 1.5 else 0
            color = (240, 240, 245, 255)
        elif kind == "holes":
            mask = 255 if 17 < radius < 34 else 0
            color = (185, 92, 205, 255)
        elif kind == "existing-alpha":
            mask = 255 if radius < 31 else 0
            color = (226, 72, 100, 128)
        elif kind == "no-salient":
            mask = 0
            bg = (90, 90, 90, 255)
        else:
            raise ValueError(kind)
        return (color if mask else bg), mask

    return fn


def digest_record(
    path: Path, root: Path, width: int | None, height: int | None
) -> dict[str, object]:
    data = path.read_bytes()
    record: dict[str, object] = {
        "path": path.relative_to(root).as_posix(),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "media_type": "image/png",
    }
    if width is not None and height is not None:
        record.update(width=width, height=height)
    return record


def generate(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict[str, object]] = []
    cases = [
        ("portrait-hair", "hair", ["portrait", "hair", "thin-boundary"]),
        ("ragged-fur", "fur", ["fur", "difficult-edge"]),
        ("translucent-object", "translucent", ["translucent", "soft-alpha"]),
        ("reflective-product", "product", ["product", "reflective"]),
        ("branching-plant", "plant", ["plant", "thin-structure"]),
        ("vehicle", "vehicle", ["vehicle"]),
        ("illustration", "illustration", ["illustration"]),
        ("low-contrast", "low-contrast", ["low-contrast"]),
        ("clutter", "clutter", ["clutter"]),
        ("thin-structure", "thin-structure", ["thin-structure"]),
        ("holes", "holes", ["holes"]),
        ("existing-alpha", "existing-alpha", ["existing-alpha"]),
        ("no-salient-object", "no-salient", ["no-salient-object", "negative"]),
    ]
    for name, kind, categories in cases:
        width, height = 128, 96
        image, mask = render_pair(root, name, width, height, small_pixel(kind, width, height))
        items.append(
            {
                "id": name,
                "categories": categories,
                "source": "deterministic-procedural-generator",
                "license": "Apache-2.0",
                "input": digest_record(image, root, width, height),
                "ground_truth_mask": digest_record(mask, root, width, height),
                "expected": "success",
            }
        )

    large_w = large_h = 10_000

    def large_rows() -> Iterable[bytes]:
        bg = b"\x40\x45\x4a\xff"
        fg = b"\xb0\x68\x38\xff"
        for y in range(large_h):
            if 2_500 <= y < 7_500:
                yield bg * 2_500 + fg * 5_000 + bg * 2_500
            else:
                yield bg * large_w

    def large_mask_rows() -> Iterable[bytes]:
        for y in range(large_h):
            yield (
                (b"\x00" * 2_500 + b"\xff" * 5_000 + b"\x00" * 2_500)
                if 2_500 <= y < 7_500
                else b"\x00" * large_w
            )

    large_image = root / "large-100mp.png"
    large_mask = root / "large-100mp.mask.png"
    write_png(large_image, large_w, large_h, 4, large_rows())
    write_png(large_mask, large_w, large_h, 1, large_mask_rows())
    items.append(
        {
            "id": "large-100mp",
            "categories": ["large", "100-megapixel", "bounded-tiling"],
            "source": "deterministic-procedural-generator",
            "license": "Apache-2.0",
            "input": digest_record(large_image, root, large_w, large_h),
            "ground_truth_mask": digest_record(large_mask, root, large_w, large_h),
            "expected": "explicit-large-image-probe-only",
        }
    )

    corrupt = root / "corrupt-truncated.png"
    corrupt.write_bytes(b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00")
    items.append(
        {
            "id": "corrupt-truncated",
            "categories": ["corrupt", "negative"],
            "source": "deterministic-procedural-generator",
            "license": "Apache-2.0",
            "input": digest_record(corrupt, root, None, None),
            "ground_truth_mask": None,
            "expected": "typed-decode-error",
        }
    )

    generator_path = Path(__file__).resolve()
    manifest = {
        "schema": "kilix.background-removal.corpus/v1",
        "frozen_at": "2026-08-25",
        "owner": "itsmygithubacct",
        "license": "Apache-2.0",
        "generator": {
            "path": "tools/generate_corpus.py",
            "sha256": hashlib.sha256(generator_path.read_bytes()).hexdigest(),
        },
        "quality_disposition": "safety-and-contract-fixtures-only; no model quality claim",
        "items": items,
    }
    manifest_path = root / "corpus-manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    paths = sorted(path for path in root.iterdir() if path.is_file() and path.name != "SHA256SUMS")
    sums = "".join(
        f"{hashlib.sha256(path.read_bytes()).hexdigest()}  {path.name}\n" for path in paths
    )
    (root / "SHA256SUMS").write_text(sums, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    generate(args.root)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
