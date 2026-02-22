#!/usr/bin/env python3
"""Create map of south-pole meltwater fit within a wider Hellas boundary."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile


DEM_PATH = Path("data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")
OUTPUT_DIR = Path("outputs")
GLOBAL_MAP_PATH = OUTPUT_DIR / "mars_wider_hellas_melt_fit_global_map.png"
ZOOM_MAP_PATH = OUTPUT_DIR / "mars_wider_hellas_melt_fit_zoom_map.png"
REPORT_PATH = OUTPUT_DIR / "mars_wider_hellas_melt_fit_report.md"

# Wider Hellas boundary used in prior calculation (0..360 lon convention).
HELLAS_LAT_MIN = -60.0
HELLAS_LAT_MAX = -20.0
HELLAS_LON_MIN = 220.0
HELLAS_LON_MAX = 320.0

MARS_RADIUS_M = 3_389_500.0


def row_lats(height: int) -> np.ndarray:
    row = np.arange(height, dtype=np.float64)
    return 90.0 - ((row + 0.5) * 180.0 / height)


def col_lons(width: int) -> np.ndarray:
    col = np.arange(width, dtype=np.float64)
    return (col + 0.5) * 360.0 / width


def cell_area_by_row(lat_deg: np.ndarray, dlon_deg: float) -> np.ndarray:
    """Cell area per row using actual latitude centers for those rows."""
    if lat_deg.size > 1:
        dlat = np.deg2rad(abs(float(lat_deg[1] - lat_deg[0])))
    else:
        dlat = np.deg2rad(180.0)
    dlon = np.deg2rad(dlon_deg)
    return (MARS_RADIUS_M**2) * dlat * dlon * np.cos(np.deg2rad(lat_deg))


def water_volume_m3(elev_m: np.ndarray, level_m: float, area_row_m2: np.ndarray) -> float:
    depth = np.maximum(level_m - elev_m, 0.0)
    return float(np.sum(depth * area_row_m2[:, None]))


def solve_level_for_volume(elev_m: np.ndarray, area_row_m2: np.ndarray, target_vol_m3: float) -> float:
    low = float(np.min(elev_m) - 1000.0)
    high = float(np.max(elev_m) + 1000.0)
    for _ in range(64):
        mid = 0.5 * (low + high)
        if water_volume_m3(elev_m, mid, area_row_m2) < target_vol_m3:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def to_u8(elev_m: np.ndarray) -> np.ndarray:
    lo = float(np.percentile(elev_m, 1))
    hi = float(np.percentile(elev_m, 99))
    if hi <= lo:
        hi = lo + 1.0
    return (np.clip((elev_m - lo) / (hi - lo), 0, 1) * 255.0).astype(np.uint8)


def lat_to_row(lat_deg: float, height: int) -> int:
    return int(round(((90.0 - lat_deg) / 180.0) * (height - 1)))


def lon_to_col(lon_deg: float, width: int) -> int:
    return int(round((lon_deg / 360.0) * (width - 1)))


def main() -> None:
    if not DEM_PATH.exists():
        raise FileNotFoundError(f"Missing DEM: {DEM_PATH}")

    # Use reduced resolution for speed while preserving basin-scale geometry.
    stride = 16
    elev_full = tifffile.memmap(DEM_PATH)
    elev = np.asarray(elev_full[::stride, ::stride], dtype=np.float64)
    h, w = elev.shape
    lat = row_lats(h)
    lon = col_lons(w)

    lat_mask = (lat >= HELLAS_LAT_MIN) & (lat <= HELLAS_LAT_MAX)
    lon_mask = (lon >= HELLAS_LON_MIN) & (lon <= HELLAS_LON_MAX)
    elev_h = elev[np.ix_(lat_mask, lon_mask)]
    lat_h = lat[lat_mask]
    if lon.size > 1:
        dlon_deg = abs(float(lon[1] - lon[0]))
    else:
        dlon_deg = 360.0
    area_row_h = cell_area_by_row(lat_h, dlon_deg)

    mars_area_m2 = 4.0 * math.pi * (MARS_RADIUS_M**2)
    melt_volume_m3 = 11.0 * mars_area_m2
    fit_level_m = solve_level_for_volume(elev_h, area_row_h, melt_volume_m3)

    # Flood mask only inside wider Hellas window.
    flooded_h = elev_h <= fit_level_m
    flooded_global = np.zeros_like(elev, dtype=bool)
    flooded_global[np.ix_(lat_mask, lon_mask)] = flooded_h

    # Build global map.
    shade = to_u8(elev)
    rgb = np.dstack([shade, shade, shade]).astype(np.uint8)
    rgb[flooded_global] = np.array([40, 120, 255], dtype=np.uint8)
    img = Image.fromarray(rgb, mode="RGB")
    draw = ImageDraw.Draw(img)

    # Draw wider Hellas boundary box.
    top = lat_to_row(HELLAS_LAT_MAX, h)
    bottom = lat_to_row(HELLAS_LAT_MIN, h)
    left = lon_to_col(HELLAS_LON_MIN, w)
    right = lon_to_col(HELLAS_LON_MAX, w)
    draw.rectangle((left, top, right, bottom), outline=(255, 220, 80), width=2)

    draw.rectangle((0, 0, w - 1, 54), fill=(0, 0, 0))
    draw.text((10, 8), "Wider Hellas meltwater fit from south-pole melt volume (11 m GEL)", fill=(255, 255, 255))
    draw.text((10, 28), f"Blue=flooded in Hellas, Yellow box=Hellas window, Level={fit_level_m:.1f} m", fill=(210, 230, 255))
    draw.text((10, 44), f"Volume={melt_volume_m3:.1f} m^3", fill=(210, 230, 255))

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(GLOBAL_MAP_PATH)

    # Build zoom map for Hellas region only.
    shade_h = to_u8(elev_h)
    rgb_h = np.dstack([shade_h, shade_h, shade_h]).astype(np.uint8)
    rgb_h[flooded_h] = np.array([40, 120, 255], dtype=np.uint8)
    zoom = Image.fromarray(rgb_h, mode="RGB")
    zdraw = ImageDraw.Draw(zoom)
    zdraw.rectangle((0, 0, zoom.width - 1, 42), fill=(0, 0, 0))
    zdraw.text((10, 8), "Hellas zoom: flooded area from 11 m GEL meltwater", fill=(255, 255, 255))
    zdraw.text((10, 24), f"Level={fit_level_m:.1f} m", fill=(210, 230, 255))
    zoom.save(ZOOM_MAP_PATH)

    lowest = float(np.min(elev_h))
    lowest_idx = np.unravel_index(np.argmin(elev_h), elev_h.shape)
    lat_vals = lat_h
    lon_vals = lon[lon_mask]
    lowest_lat = float(lat_vals[lowest_idx[0]])
    lowest_lon = float(lon_vals[lowest_idx[1]])

    report = [
        "# Wider Hellas melt fit map",
        "",
        f"- Source DEM: `{DEM_PATH}`",
        f"- Hellas boundary: lat {HELLAS_LAT_MIN}..{HELLAS_LAT_MAX}, lon {HELLAS_LON_MIN}..{HELLAS_LON_MAX}",
        "- Note: map and volume fit are computed on a stride-16 downsampled grid for speed.",
        f"- South-pole meltwater volume (11 m GEL): `{melt_volume_m3}` m^3",
        f"- Solved fill level in Hellas window: `{fit_level_m}` m",
        f"- Lowest point in this window: `{lowest}` m at lat `{lowest_lat}`, lon `{lowest_lon}`",
        f"- Output global map: `{GLOBAL_MAP_PATH}`",
        f"- Output zoom map: `{ZOOM_MAP_PATH}`",
    ]
    REPORT_PATH.write_text("\n".join(report), encoding="utf-8")

    print(f"Saved: {GLOBAL_MAP_PATH}")
    print(f"Saved: {ZOOM_MAP_PATH}")
    print(f"Saved: {REPORT_PATH}")


if __name__ == "__main__":
    main()
