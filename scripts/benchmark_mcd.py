#!/usr/bin/env python3
"""Fetch MCD v6.1 maps and compare them with a gcm3d MOLA map NetCDF.

The MCD web service only supports two free dimensions.  A global diurnal mean
is therefore assembled from its 12 native fixed-local-time samples rather than
requesting ``localtime=all`` alongside global latitude and longitude.

Example
-------
UV_CACHE_DIR=/private/tmp/terraforming-uv-cache uv run python \
  scripts/benchmark_mcd.py outputs/gcm3d_maps/current-p0_maps.nc \
  --output-dir outputs/gcm3d_maps/current-p0_mcd
"""

from __future__ import annotations

import argparse
import html
import http.client
import json
import math
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

MCD_CGI = "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py"
MCD_ROOT = "https://www-mars.lmd.jussieu.fr/mcd_python/"
MARS_GRAVITY_M_S2 = 3.72076
FIELDS = {
    "temperature": ("Surface temperature", "K", 1.0),
    "surface_pressure": ("Surface pressure", "Pa", 1.0),
    "wind_speed": ("Horizontal wind speed", "m/s", 1.0),
    # MCD supplies kg/m2; gcm3d stores the surface reservoir as pressure-equivalent.
    "co2_ice": ("Monthly mean surface CO2 ice layer", "Pa-equivalent", MARS_GRAVITY_M_S2),
}


def mcd_query(
    ls: float,
    local_time: float,
    dust: int = 1,
    high_res: bool = True,
    altitude_m: float = 10.0,
) -> str:
    """Return an official MCD map URL matched to the model surface fields."""
    params = {
        "var1": "tsurf", "var2": "ps", "var3": "wind", "var4": "co2ice",
        "datekeyhtml": "1", "ls": f"{ls:g}", "localtime": f"{local_time:g}",
        "latitude": "all", "longitude": "all", "altitude": f"{altitude_m:g}", "zkey": "3",
        "dust": str(dust), "hrkey": "1" if high_res else "0",
        "averaging": "off", "isfixedlt": "on", "animation": "off",
        "animframes": "12", "dpi": "160", "islog": "off", "colorm": "jet",
        "proj": "cyl", "iswind": "off",
    }
    return f"{MCD_CGI}?{urllib.parse.urlencode(params)}"


def download(url: str, timeout: float = 180.0, attempts: int = 3) -> bytes:
    """Download with bounded retries for the MCD server's occasional truncation."""
    request = urllib.request.Request(url, headers={"User-Agent": "terraforming-gcm3d-validation/1"})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except (http.client.IncompleteRead, TimeoutError, urllib.error.URLError):
            if attempt == attempts:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def fetch_ascii(
    ls: float,
    local_time: float,
    timeout: float = 180.0,
    *,
    dust: int = 1,
    high_res: bool = True,
    altitude_m: float = 10.0,
) -> tuple[str, str]:
    """Submit an MCD request and return (ASCII text, source URL)."""
    page = download(mcd_query(ls, local_time, dust, high_res, altitude_m), timeout).decode(
        "utf-8", errors="replace"
    )
    match = re.search(r"href=['\"]([^'\"]+\.txt)['\"]", page)
    if not match:
        message = re.sub(r"<[^>]+>", " ", page)
        raise RuntimeError(f"MCD did not return a data link: {' '.join(message.split())[:400]}")
    data_url = urllib.parse.urljoin(MCD_ROOT + "cgi-bin/", html.unescape(match.group(1)))
    return download(data_url, timeout).decode("utf-8", errors="replace"), data_url


def parse_mcd_ascii(text: str) -> xr.Dataset:
    """Parse the four 2-D blocks emitted by the MCD v6.1 web interface."""
    blocks = re.split(r"(?=#{10,}\n### MCD_v)", text)
    arrays: dict[str, xr.DataArray] = {}
    lookup = {description.lower(): name for name, (description, _, _) in FIELDS.items()}
    for block in blocks:
        column = re.search(r"### Columns 2\+ are (.+?)\s*$", block, re.MULTILINE)
        header = re.search(r"^---- \|\|\s+(.+)$", block, re.MULTILINE)
        if not column or not header:
            continue
        description = column.group(1).strip()
        name = next((value for key, value in lookup.items() if description.lower().startswith(key)), None)
        if name is None:
            continue
        lat = np.fromstring(header.group(1), sep=" ")
        lon_values, rows = [], []
        for line in block[header.end():].splitlines():
            if "||" not in line or line.startswith("-") and line.startswith("---"):
                continue
            left, right = line.split("||", 1)
            try:
                lon = float(left.strip())
            except ValueError:
                continue
            values = np.fromstring(right, sep=" ")
            if values.size == lat.size:
                lon_values.append(lon)
                rows.append(values)
        if not rows:
            raise ValueError(f"No numeric rows found for MCD field {description!r}")
        data = np.asarray(rows, dtype=float).T * FIELDS[name][2]
        arrays[name] = xr.DataArray(
            data, dims=("lat", "lon"), coords={"lat": lat, "lon": lon_values},
            attrs={"units": FIELDS[name][1], "mcd_description": description},
        )
    missing = set(FIELDS) - set(arrays)
    if missing:
        raise ValueError(f"MCD response is missing fields: {sorted(missing)}")
    return xr.Dataset(arrays)


def normalize_longitude(ds: xr.Dataset) -> xr.Dataset:
    normalized = ds.assign_coords(lon=np.mod(ds.lon, 360.0)).sortby("lon")
    # MCD includes both -180 and +180; collapse the duplicate after normalization.
    return normalized.groupby("lon").mean()


def interpolate_periodic(reference: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    reference = normalize_longitude(reference)
    left = reference.isel(lon=-1).assign_coords(lon=float(reference.lon[-1]) - 360.0)
    right = reference.isel(lon=0).assign_coords(lon=float(reference.lon[0]) + 360.0)
    extended = xr.concat([left, reference, right], dim="lon")
    return extended.interp(lat=target.lat, lon=target.lon)


def weighted_metrics(model: np.ndarray, reference: np.ndarray, lat: np.ndarray) -> dict[str, float]:
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], model.shape)
    valid = np.isfinite(model) & np.isfinite(reference) & (weights > 0)
    x, y, w = model[valid], reference[valid], weights[valid]
    w = w / w.sum()
    error = x - y
    x_mean, y_mean = np.sum(w * x), np.sum(w * y)
    covariance = np.sum(w * (x - x_mean) * (y - y_mean))
    denominator = math.sqrt(np.sum(w * (x - x_mean) ** 2) * np.sum(w * (y - y_mean) ** 2))
    return {
        "bias": float(np.sum(w * error)),
        "mae": float(np.sum(w * np.abs(error))),
        "rmse": float(math.sqrt(np.sum(w * error**2))),
        "spatial_correlation": float(covariance / denominator) if denominator else float("nan"),
        "model_area_mean": float(x_mean),
        "mcd_area_mean": float(y_mean),
    }


def plot_comparison(model: xr.Dataset, mcd: xr.Dataset, output: Path) -> None:
    fig, axes = plt.subplots(len(FIELDS), 3, figsize=(15, 13), constrained_layout=True)
    for row, (name, (_, units, _)) in enumerate(FIELDS.items()):
        values = [model[name], mcd[name], model[name] - mcd[name]]
        titles = [f"gcm3d {name}", f"MCD {name}", "gcm3d − MCD"]
        for column, (field, title) in enumerate(zip(values, titles, strict=True)):
            cmap = "RdBu_r" if column == 2 else ("Blues" if name == "co2_ice" else "viridis")
            image = axes[row, column].pcolormesh(model.lon, model.lat, field, shading="auto", cmap=cmap)
            axes[row, column].set(title=title, xlabel="longitude (°E)", ylabel="latitude (°N)")
            fig.colorbar(image, ax=axes[row, column], label=units, shrink=0.8)
    fig.suptitle("gcm3d transient vs MCD v6.1 climatology (diagnostic, not validation)")
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path, help="gcm3d map NetCDF")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ls", type=float, default=0.0)
    parser.add_argument("--local-times", default="0,2,4,6,8,10,12,14,16,18,20,22")
    parser.add_argument("--dust", type=int, default=1, help="MCD scenario 1 = climatology/average EUV")
    parser.add_argument("--altitude-m", type=float, help="MCD wind height; defaults to the model NetCDF diagnostic height")
    parser.add_argument("--no-high-res", action="store_true")
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--from-cache", action="store_true", help="do not contact MCD; use cached ASCII files")
    parser.add_argument("--cache-dir", type=Path, help="read raw ASCII caches from another benchmark directory")
    parser.add_argument("--refresh", action="store_true", help="redownload even when a cached response exists")
    args = parser.parse_args()

    with xr.open_dataset(args.model) as metadata:
        if args.altitude_m is None:
            args.altitude_m = float(metadata.attrs.get("approximate_wind_height_m", 10.0))

    local_times = [float(value) for value in args.local_times.split(",")]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples, sources = [], []
    for local_time in local_times:
        cache_root = args.cache_dir or args.output_dir
        cache = cache_root / f"mcd_ls{args.ls:g}_lt{local_time:g}_z{args.altitude_m:g}m.txt"
        if cache.exists() and not args.refresh:
            text = cache.read_text()
            source = "cache"
        elif args.from_cache:
            if not cache.exists():
                raise FileNotFoundError(f"missing cached MCD response: {cache}")
            text = cache.read_text()
            source = "cache"
        else:
            print(f"Fetching MCD Ls={args.ls:g}°, LT={local_time:g}h ...", flush=True)
            text, source = fetch_ascii(
                args.ls,
                local_time,
                args.timeout,
                dust=args.dust,
                high_res=not args.no_high_res,
                altitude_m=args.altitude_m,
            )
            cache.write_text(text)
        samples.append(parse_mcd_ascii(text))
        sources.append(source)

    mcd_native = xr.concat(samples, dim=xr.IndexVariable("local_time", local_times)).mean("local_time")
    mcd_native.attrs.update(
        title="MCD v6.1 diurnal mean assembled from fixed-local-time maps",
        ls_deg=args.ls, local_times_hours=",".join(f"{x:g}" for x in local_times),
        altitude_m_above_surface=args.altitude_m,
        dust_scenario=args.dust, high_resolution_topography=not args.no_high_res,
    )
    native_path = args.output_dir / "mcd_diurnal_mean_native.nc"
    mcd_native.to_netcdf(native_path)

    with xr.open_dataset(args.model) as opened:
        model = opened[list(FIELDS)].load()
        model_attrs = dict(opened.attrs)
    mcd = interpolate_periodic(mcd_native, model)
    metrics = {name: weighted_metrics(model[name].values, mcd[name].values, model.lat.values) for name in FIELDS}
    report = {
        "status": "diagnostic_only_transient_vs_climatology",
        "warning": model_attrs.get("fidelity", "Model averaging state is unknown"),
        "model": str(args.model), "mcd_version": "6.1", "ls_deg": args.ls,
        "local_times_hours": local_times, "dust_scenario": args.dust,
        "mcd_wind_altitude_m_above_surface": args.altitude_m,
        "high_resolution_topography": not args.no_high_res,
        "co2_ice_conversion": f"MCD kg/m2 multiplied by Mars gravity {MARS_GRAVITY_M_S2} m/s2",
        "metrics": metrics, "source_urls": sources,
    }
    report_path = args.output_dir / "benchmark.json"
    report_path.write_text(json.dumps(report, indent=2, allow_nan=True) + "\n")
    comparison_path = args.output_dir / "comparison.png"
    plot_comparison(model, mcd, comparison_path)
    print(json.dumps(metrics, indent=2, allow_nan=True))
    print(f"Wrote {native_path}, {report_path}, and {comparison_path}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        raise SystemExit(1) from error
