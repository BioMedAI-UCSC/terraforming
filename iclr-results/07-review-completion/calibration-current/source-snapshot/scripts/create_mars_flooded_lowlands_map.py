#!/usr/bin/env python3
"""Map flooded Mars lowlands from south-polar meltwater using local MOLA DEM."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
import tifffile


DEM_PATH = Path("data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif")
OUTPUT_DIR = Path("outputs")
MAP_OUTPUT = OUTPUT_DIR / "mars_south_pole_melt_flooded_lowlands_map.png"
REPORT_OUTPUT = OUTPUT_DIR / "mars_south_pole_melt_flooded_lowlands_report.md"

MARS_RADIUS_M = 3_389_500.0
MARS_SURFACE_AREA_M2 = 4.0 * np.pi * (MARS_RADIUS_M**2)


def row_latitudes_deg(height: int) -> np.ndarray:
    """Latitude at row centers for equirectangular global raster."""
    row = np.arange(height, dtype=np.float64)
    return 90.0 - ((row + 0.5) * 180.0 / height)


def cell_area_by_row_m2(height: int, width: int) -> np.ndarray:
    """Area of one raster cell for each row."""
    lat_deg = row_latitudes_deg(height)
    dlat = np.deg2rad(180.0 / height)
    dlon = np.deg2rad(360.0 / width)
    return (MARS_RADIUS_M**2) * dlat * dlon * np.cos(np.deg2rad(lat_deg))


def water_volume_m3(elev_m: np.ndarray, level_m: float, area_row_m2: np.ndarray) -> float:
    """Compute global water volume needed to fill terrain to a given level."""
    depth = np.maximum(level_m - elev_m, 0.0)
    return float(np.sum(depth * area_row_m2[:, None]))


def solve_level_for_target_volume(elev_m: np.ndarray, area_row_m2: np.ndarray, target_volume_m3: float) -> float:
    """Find water level (m) whose integrated volume matches target."""
    low = float(np.min(elev_m) - 500.0)
    high = float(np.max(elev_m) + 500.0)
    for _ in range(48):
        mid = 0.5 * (low + high)
        vol = water_volume_m3(elev_m, mid, area_row_m2)
        if vol < target_volume_m3:
            low = mid
        else:
            high = mid
    return 0.5 * (low + high)


def normalize_to_8bit(elev_m: np.ndarray) -> np.ndarray:
    """Scale elevations to 0..255 for quick map rendering."""
    lo = float(np.percentile(elev_m, 1.0))
    hi = float(np.percentile(elev_m, 99.0))
    if hi <= lo:
        hi = lo + 1.0
    clipped = np.clip(elev_m, lo, hi)
    return ((clipped - lo) / (hi - lo) * 255.0).astype(np.uint8)


def save_map_image(elev_m: np.ndarray, inundated: np.ndarray, level_m: float) -> None:
    """Save RGB flood map image with overlays."""
    shade = normalize_to_8bit(elev_m)
    base = np.dstack([shade, shade, shade]).astype(np.uint8)

    # Inundated lowlands after meltwater redistribution.
    base[inundated] = np.array([45, 120, 255], dtype=np.uint8)

    img = Image.fromarray(base, mode="RGB")
    draw = ImageDraw.Draw(img)
    draw.rectangle((0, 0, img.width - 1, 48), fill=(0, 0, 0))
    draw.text((10, 8), "Mars flooded lowlands from south-polar meltwater (MGS/MOLA DEM)", fill=(255, 255, 255))
    draw.text((10, 26), f"Blue: lowlands below solved water level, Level ~ {level_m:.0f} m", fill=(210, 230, 255))
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    img.save(MAP_OUTPUT)


def lon_centers_deg(width: int) -> np.ndarray:
    """Longitude at column centers on [0, 360)."""
    col = np.arange(width, dtype=np.float64)
    return (col + 0.5) * 360.0 / width


def lon_in_range(lon: np.ndarray, start: float, end: float) -> np.ndarray:
    """Longitude mask on [0,360), supporting wrapped ranges."""
    if start <= end:
        return (lon >= start) & (lon <= end)
    return (lon >= start) | (lon <= end)


def region_area_km2(
    mask: np.ndarray,
    area_row_m2: np.ndarray,
    lat_deg: np.ndarray,
    lon_deg: np.ndarray,
    lat_min: float,
    lat_max: float,
    lon_start: float,
    lon_end: float,
) -> float:
    """Area of inundation inside a simple lat-lon region box."""
    lat_mask = (lat_deg >= lat_min) & (lat_deg <= lat_max)
    lon_mask = lon_in_range(lon_deg, lon_start, lon_end)
    region = mask[np.ix_(lat_mask, lon_mask)]
    areas = area_row_m2[lat_mask]
    return float(np.sum(region * areas[:, None]) / 1e6)


def write_report(gel_m: float, level_m: float, inundated: np.ndarray, lat_deg: np.ndarray, area_row_m2: np.ndarray) -> None:
    """Write coverage metrics in markdown."""
    inundated_area_m2 = float(np.sum(area_row_m2[:, None] * inundated))
    inundated_pct = 100.0 * inundated_area_m2 / MARS_SURFACE_AREA_M2

    north_mask = lat_deg >= 0.0
    south_mask = lat_deg < 0.0
    tropics_mask = (lat_deg >= -30.0) & (lat_deg <= 30.0)
    north_area_m2 = float(np.sum(area_row_m2[north_mask, None] * inundated[north_mask]))
    south_area_m2 = float(np.sum(area_row_m2[south_mask, None] * inundated[south_mask]))
    tropics_area_m2 = float(np.sum(area_row_m2[tropics_mask, None] * inundated[tropics_mask]))

    lon_deg = lon_centers_deg(inundated.shape[1])
    utopia_km2 = region_area_km2(inundated, area_row_m2, lat_deg, lon_deg, 20.0, 60.0, 70.0, 130.0)
    acidalia_km2 = region_area_km2(inundated, area_row_m2, lat_deg, lon_deg, 20.0, 55.0, 300.0, 30.0)
    hellas_km2 = region_area_km2(inundated, area_row_m2, lat_deg, lon_deg, -55.0, -30.0, 40.0, 90.0)
    argyre_km2 = region_area_km2(inundated, area_row_m2, lat_deg, lon_deg, -60.0, -35.0, 300.0, 340.0)

    lines = [
        "# Mars flooded lowlands coverage",
        "",
        "Computed from local `Mars_MGS_MOLA_DEM_mosaic_global_463m.tif` (downsampled).",
        "",
        f"- Assumed south-polar meltwater global-equivalent layer (GEL): **{gel_m:.1f} m**",
        f"- Solved global water level relative to MOLA datum: **{level_m:.1f} m**",
        f"- Inundated lowland area: **{inundated_area_m2 / 1e6:,.0f} km^2** ({inundated_pct:.2f}% of Mars)",
        f"- Inundated area in northern hemisphere: **{north_area_m2 / 1e6:,.0f} km^2**",
        f"- Inundated area in southern hemisphere: **{south_area_m2 / 1e6:,.0f} km^2**",
        f"- Inundated area in tropical band (-30..30): **{tropics_area_m2 / 1e6:,.0f} km^2**",
        "",
        "Approximate regional coverage (simple lat/lon boxes):",
        f"- Utopia Planitia box: **{utopia_km2:,.0f} km^2**",
        f"- Acidalia Planitia box: **{acidalia_km2:,.0f} km^2**",
        f"- Hellas Planitia box: **{hellas_km2:,.0f} km^2**",
        f"- Argyre Planitia box: **{argyre_km2:,.0f} km^2**",
        "",
        "Interpretation:",
        "- Assumes south-polar meltwater is redistributed planet-wide and ponds in topographic lows.",
        "- Does not model time evolution, evaporation, infiltration, ice refreezing, or detailed routing.",
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_OUTPUT.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gel-m",
        type=float,
        default=11.0,
        help="South-polar meltwater global equivalent layer in meters (default: 11.0).",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=16,
        help="DEM downsampling stride for speed (default: 16).",
    )
    args = parser.parse_args()

    if not DEM_PATH.exists():
        raise FileNotFoundError(f"DEM not found: {DEM_PATH}")
    if args.gel_m <= 0.0:
        raise ValueError("--gel-m must be > 0")
    if args.stride < 1:
        raise ValueError("--stride must be >= 1")

    elev_full = tifffile.memmap(DEM_PATH)
    elev_m = np.asarray(elev_full[:: args.stride, :: args.stride], dtype=np.float64)

    lat_deg = row_latitudes_deg(elev_m.shape[0])
    area_row_m2 = cell_area_by_row_m2(elev_m.shape[0], elev_m.shape[1])

    target_volume_m3 = args.gel_m * MARS_SURFACE_AREA_M2
    level_m = solve_level_for_target_volume(elev_m, area_row_m2, target_volume_m3)

    inundated = elev_m <= level_m
    save_map_image(elev_m=elev_m, inundated=inundated, level_m=level_m)
    write_report(gel_m=args.gel_m, level_m=level_m, inundated=inundated, lat_deg=lat_deg, area_row_m2=area_row_m2)

    print(f"Saved map: {MAP_OUTPUT}")
    print(f"Saved report: {REPORT_OUTPUT}")


if __name__ == "__main__":
    main()
