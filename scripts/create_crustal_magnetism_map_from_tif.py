#!/usr/bin/env python3
"""Create a crustal magnetism map from local TIFF with legend."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


INPUT_TIF = Path("data/magnetism/PIA02819.tif")
OUTPUT_DIR = Path("outputs")
OUTPUT_MAP = OUTPUT_DIR / "mars_crustal_magnetism_from_tif_with_legend.png"


def build_proxy_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """
    Build a signed, unitless proxy from RGB channels.

    The NASA TIFF uses red/blue tones to indicate opposite magnetic polarity.
    We use color dominance and colorfulness to derive a relative index in
    [-1, 1], where negative is blue-dominant and positive is red-dominant.
    """
    arr = rgb.astype(np.float32)
    red = arr[..., 0]
    green = arr[..., 1]
    blue = arr[..., 2]

    # Signed polarity proxy (red minus blue).
    signed = (red - blue) / 255.0

    # Colorfulness proxy to suppress grayscale topographic background.
    chroma = (np.maximum(np.maximum(red, green), blue) - np.minimum(np.minimum(red, green), blue)) / 255.0
    proxy = signed * chroma
    return np.clip(proxy, -1.0, 1.0)


def colorize_proxy(proxy: np.ndarray) -> np.ndarray:
    """Map proxy values to blue-white-red RGB for display."""
    p = ((proxy + 1.0) * 0.5).astype(np.float32)  # 0..1
    out = np.empty((proxy.shape[0], proxy.shape[1], 3), dtype=np.uint8)

    left = p <= 0.5
    right = ~left

    # -1..0 : blue -> white
    t_left = (p[left] / 0.5)
    out[..., 0][left] = (255.0 * t_left).astype(np.uint8)
    out[..., 1][left] = (255.0 * t_left).astype(np.uint8)
    out[..., 2][left] = 255

    # 0..1 : white -> red
    t_right = ((p[right] - 0.5) / 0.5)
    out[..., 0][right] = 255
    out[..., 1][right] = (255.0 * (1.0 - t_right)).astype(np.uint8)
    out[..., 2][right] = (255.0 * (1.0 - t_right)).astype(np.uint8)
    return out


def add_legend(canvas: Image.Image, x0: int, y0: int, width: int, height: int) -> None:
    """Draw vertical legend bar with labels."""
    draw = ImageDraw.Draw(canvas)
    for i in range(height):
        t = i / max(1, height - 1)  # 0 top, 1 bottom
        # top = +1 (red), middle = 0 (white), bottom = -1 (blue)
        if t < 0.5:
            u = t / 0.5
            r, g, b = 255, int(255 * (1.0 - u)), int(255 * (1.0 - u))
        else:
            u = (t - 0.5) / 0.5
            r, g, b = int(255 * (1.0 - u)), int(255 * (1.0 - u)), 255
        draw.line([(x0, y0 + i), (x0 + width, y0 + i)], fill=(r, g, b))

    draw.rectangle((x0, y0, x0 + width, y0 + height), outline=(0, 0, 0), width=2)
    draw.text((x0 + width + 12, y0 - 2), "+1.0", fill=(0, 0, 0))
    draw.text((x0 + width + 12, y0 + height // 2 - 7), "0.0", fill=(0, 0, 0))
    draw.text((x0 + width + 12, y0 + height - 14), "-1.0", fill=(0, 0, 0))
    draw.text((x0 - 2, y0 + height + 10), "Relative crustal magnetism index (unitless)", fill=(0, 0, 0))


def main() -> None:
    if not INPUT_TIF.exists():
        raise FileNotFoundError(f"Missing local TIFF: {INPUT_TIF}")

    src = Image.open(INPUT_TIF).convert("RGB")
    rgb = np.asarray(src)

    proxy = build_proxy_from_rgb(rgb)
    mapped = colorize_proxy(proxy)
    map_img = Image.fromarray(mapped, mode="RGB")

    # Add white panel at bottom for legend and caption.
    legend_pad = 130
    canvas = Image.new("RGB", (map_img.width, map_img.height + legend_pad), color=(255, 255, 255))
    canvas.paste(map_img, (0, 0))

    add_legend(canvas, x0=60, y0=map_img.height + 12, width=42, height=95)
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (420, map_img.height + 18),
        "Source: local data/magnetism/PIA02819.tif (Mars Crustal Magnetic Field Remnants)",
        fill=(0, 0, 0),
    )
    draw.text(
        (420, map_img.height + 42),
        "Blue and red represent opposite polarity signs; intensity reflects stronger color contrast in TIFF.",
        fill=(0, 0, 0),
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    canvas.save(OUTPUT_MAP)
    print(f"Saved map: {OUTPUT_MAP}")


if __name__ == "__main__":
    main()
