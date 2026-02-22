#!/usr/bin/env python3
"""Create a Mars south-pole melting map from local MGS/MOLA DEM."""

from __future__ import annotations

from pathlib import Path
from PIL import Image, ImageDraw, ImageOps


DEM_PATH = Path("data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")
OUTPUT_DIR = Path("outputs")
OUTPUT_PATH = OUTPUT_DIR / "mars_south_pole_melt_map.png"

# Latitude extent used for the south polar region.
POLAR_NORTH_EDGE_DEG = -60.0


def lat_to_row(lat_deg: float, image_height: int) -> int:
    """Convert latitude to raster row for simple cylindrical global raster."""
    return int(round(((90.0 - lat_deg) / 180.0) * (image_height - 1)))


def build_latitude_gradient(width: int, height: int, lat_top: float, lat_bottom: float) -> Image.Image:
    """Create a grayscale image where brighter values represent warmer latitudes."""
    grad = Image.new("L", (width, height))
    pixels = []
    lat_range = max(1e-6, lat_top - lat_bottom)
    for y in range(height):
        lat_here = lat_top - (y / max(1, height - 1)) * lat_range
        # 0 at -90 deg, 255 at -60 deg (warmer edge of south cap).
        warm_factor = max(0.0, min(1.0, (lat_here + 90.0) / 30.0))
        value = int(round(255.0 * warm_factor))
        pixels.extend([value] * width)
    grad.putdata(pixels)
    return grad


def draw_latitude_grid(draw: ImageDraw.ImageDraw, width: int, height: int, lat_top: float, lat_bottom: float) -> None:
    """Draw simple latitude guide lines for readability."""
    for lat in (-60, -70, -80, -90):
        frac = (lat_top - lat) / max(1e-6, (lat_top - lat_bottom))
        y = int(round(frac * (height - 1)))
        draw.line([(0, y), (width, y)], fill=(240, 240, 240), width=1)
        label = f"{abs(lat)}S"
        draw.text((8, max(0, y - 12)), label, fill=(245, 245, 245))


def main() -> None:
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")

    # Large DEM safety: allow loading very large source raster.
    Image.MAX_IMAGE_PIXELS = None

    with Image.open(DEM_PATH) as dem_raw:
        # Keep processing lightweight while preserving south-pole structure.
        if hasattr(Image, "Resampling"):
            resample_filter = Image.Resampling.BILINEAR
        else:
            resample_filter = Image.BILINEAR
        dem_small = dem_raw.resize((4096, 2048), resample=resample_filter)

    start_row = lat_to_row(POLAR_NORTH_EDGE_DEG, dem_small.height)
    south_cap = dem_small.crop((0, start_row, dem_small.width, dem_small.height))

    # Normalize elevation for visual contrast (bright = high terrain).
    dem8 = ImageOps.autocontrast(south_cap.convert("L"))

    # Lowland factor supports potential local melt ponding.
    lowlands = ImageOps.invert(dem8)
    lat_gradient = build_latitude_gradient(dem8.width, dem8.height, -60.0, -90.0)

    # Heuristic "melt tendency" index:
    #  - warmer near cap edge (-60 deg) and
    #  - higher in topographically low terrain.
    melt_index = Image.blend(lowlands, lat_gradient, alpha=0.65)

    # Build a soft mask so strongest red highlights are highest melt tendency.
    melt_mask = melt_index.point(lambda p: 0 if p < 120 else min(255, (p - 120) * 2))

    # Terrain base map colors.
    base_rgb = ImageOps.colorize(
        dem8,
        black="#0b1f3a",
        mid="#5f7892",
        white="#f0f0ea",
    ).convert("RGB")

    overlay = Image.new("RGB", base_rgb.size, (255, 60, 60))
    base_rgb.paste(overlay, mask=melt_mask)

    draw = ImageDraw.Draw(base_rgb)
    draw_latitude_grid(draw, base_rgb.width, base_rgb.height, -60.0, -90.0)
    draw.text((12, 12), "Mars South Polar Cap Melt Tendency (MGS MOLA DEM)", fill=(255, 255, 255))
    draw.text((12, 30), "Red = higher relative melt tendency (heuristic)", fill=(255, 220, 220))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    base_rgb.save(OUTPUT_PATH)
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
