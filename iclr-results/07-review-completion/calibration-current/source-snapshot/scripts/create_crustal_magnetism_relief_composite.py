#!/usr/bin/env python3
"""Create a more realistic Mars relief + crustal magnetism composite map."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageEnhance, ImageFilter


RELIEF_PATH = Path("data/mars_mgs_mola_dem_mosaic_global_1024.jpg")
MAG_PATH = Path("data/magnetism/PIA02819.tif")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "mars_crustal_magnetism_relief_composite.png"


def magnetism_proxy_from_tif(mag_rgb: np.ndarray) -> np.ndarray:
    """
    Build signed magnetism proxy in [-1, 1] from TIFF colors.

    Red/blue indicate opposite polarities in the source rendering.
    """
    arr = mag_rgb.astype(np.float32)
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    signed = (r - b) / 255.0
    chroma = (np.maximum(np.maximum(r, g), b) - np.minimum(np.minimum(r, g), b)) / 255.0
    proxy = np.clip(signed * chroma, -1.0, 1.0)

    # Ignore black background outside the mapped globe in PIA02819.
    non_black = (r + g + b) > 18.0
    proxy = np.where(non_black, proxy, 0.0)
    return proxy


def proxy_to_blue_white_red(proxy: np.ndarray) -> np.ndarray:
    """Map signed proxy to RGB (blue -> white -> red)."""
    p = ((proxy + 1.0) * 0.5).astype(np.float32)
    out = np.empty((proxy.shape[0], proxy.shape[1], 3), dtype=np.uint8)

    left = p <= 0.5
    right = ~left

    t_left = p[left] / 0.5
    out[..., 0][left] = (255.0 * t_left).astype(np.uint8)
    out[..., 1][left] = (255.0 * t_left).astype(np.uint8)
    out[..., 2][left] = 255

    t_right = (p[right] - 0.5) / 0.5
    out[..., 0][right] = 255
    out[..., 1][right] = (255.0 * (1.0 - t_right)).astype(np.uint8)
    out[..., 2][right] = (255.0 * (1.0 - t_right)).astype(np.uint8)
    return out


def add_legend_and_text(base: Image.Image) -> Image.Image:
    """Append legend panel and labels."""
    panel_h = 140
    canvas = Image.new("RGB", (base.width, base.height + panel_h), (250, 250, 250))
    canvas.paste(base, (0, 0))

    draw = ImageDraw.Draw(canvas)
    draw.rectangle((0, base.height, canvas.width - 1, canvas.height - 1), outline=(160, 160, 160), width=1)

    # Horizontal legend.
    x0, y0, w, h = 40, base.height + 40, 360, 24
    for x in range(w):
        t = x / max(1, (w - 1))  # 0..1
        if t <= 0.5:
            u = t / 0.5
            r, g, b = int(255 * u), int(255 * u), 255
        else:
            u = (t - 0.5) / 0.5
            r, g, b = 255, int(255 * (1.0 - u)), int(255 * (1.0 - u))
        draw.line([(x0 + x, y0), (x0 + x, y0 + h)], fill=(r, g, b))
    draw.rectangle((x0, y0, x0 + w, y0 + h), outline=(0, 0, 0), width=1)

    draw.text((x0, y0 - 22), "Crustal magnetism proxy (unitless, local-data color reconstruction)", fill=(0, 0, 0))
    draw.text((x0 - 8, y0 + h + 6), "-1 (blue polarity)", fill=(0, 0, 0))
    draw.text((x0 + w // 2 - 26, y0 + h + 6), "0", fill=(0, 0, 0))
    draw.text((x0 + w - 10, y0 + h + 6), "+1 (red polarity)", fill=(0, 0, 0))

    draw.text(
        (460, base.height + 24),
        "Basemap: Mars MOLA relief mosaic (local JPG)\nOverlay: magnetic polarity/intensity inferred from local PIA02819 TIFF colors",
        fill=(0, 0, 0),
    )
    return canvas


def main() -> None:
    if not RELIEF_PATH.exists():
        raise FileNotFoundError(f"Missing relief image: {RELIEF_PATH}")
    if not MAG_PATH.exists():
        raise FileNotFoundError(f"Missing magnetism TIFF: {MAG_PATH}")

    relief = Image.open(RELIEF_PATH).convert("RGB")
    relief = ImageEnhance.Contrast(relief).enhance(1.20)
    relief = ImageEnhance.Color(relief).enhance(0.85)

    mag_img = Image.open(MAG_PATH).convert("RGB")
    mag_proxy = magnetism_proxy_from_tif(np.asarray(mag_img))
    mag_proxy_img = Image.fromarray(((mag_proxy + 1.0) * 127.5).astype(np.uint8), mode="L")
    mag_proxy_img = mag_proxy_img.resize(relief.size, Image.BICUBIC)

    proxy = (np.asarray(mag_proxy_img).astype(np.float32) / 127.5) - 1.0
    proxy = np.clip(proxy, -1.0, 1.0)

    # Slight smoothing for cleaner regional bands.
    proxy_u8 = ((proxy + 1.0) * 127.5).astype(np.uint8)
    proxy_u8 = np.asarray(Image.fromarray(proxy_u8, mode="L").filter(ImageFilter.GaussianBlur(radius=1.2)))
    proxy = (proxy_u8.astype(np.float32) / 127.5) - 1.0

    mag_color = proxy_to_blue_white_red(proxy).astype(np.float32)
    relief_arr = np.asarray(relief).astype(np.float32)

    # Overlay alpha grows with absolute proxy magnitude.
    alpha = np.clip(np.abs(proxy) ** 0.8, 0.0, 1.0) * 0.65
    comp = relief_arr * (1.0 - alpha[..., None]) + mag_color * alpha[..., None]
    comp = np.clip(comp, 0, 255).astype(np.uint8)

    comp_img = Image.fromarray(comp, mode="RGB")
    final_img = add_legend_and_text(comp_img)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    final_img.save(OUTPUT_PATH)
    print(f"Saved composite map: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
