#!/usr/bin/env python3
"""Stream a bounded ARCO-MACDA subset and prepare daily Dinosaur-grid states.

The source archive is a 216 GB logical Zarr v3 dataset. This tool reads only the
time chunks needed for the requested Mars year and sol range, computes daily
means, linearly interpolates the 35 native sigma levels to Dinosaur's 12 sigma
midpoints, and horizontally interpolates to its T21 Gaussian grid.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path

import fsspec
import numpy as np
import xarray as xr

from src.framework.gcm.coordinates import coordinate_system


DEFAULT_URL = (
    "https://huggingface.co/datasets/ananyo01/ARCO-MACDA/resolve/main/"
    "macda_combined.zarr"
)
DEFAULT_VARIABLES = (
    "temp",
    "uwind",
    "vwind",
    "psurf",
    "tsurf",
    "coldust",
    "co2ice",
)
SAMPLES_PER_SOL = 12


def _scalar_at(array: xr.DataArray, index: int) -> float:
    return float(array.isel(time=index).compute().item())


def _lower_bound(array: xr.DataArray, value: float, *, right: bool = False) -> int:
    """Binary-search a monotonic remote array using O(log n) chunk reads."""
    lo, hi = 0, int(array.sizes["time"])
    while lo < hi:
        mid = (lo + hi) // 2
        item = _scalar_at(array, mid)
        before = item <= value if right else item < value
        if before:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _mars_year_bounds(dataset: xr.Dataset, mars_year: int) -> tuple[int, int]:
    years = dataset["MY_Ls"]
    start = _lower_bound(years, float(mars_year))
    stop = _lower_bound(years, float(mars_year), right=True)
    if start == stop:
        first = _scalar_at(years, 0)
        last = _scalar_at(years, int(years.sizes["time"]) - 1)
        raise ValueError(
            f"Mars year {mars_year} is absent; source range is MY {first:g}-{last:g}"
        )
    return start, stop


def _nearest_contiguous_window(
    dataset: xr.Dataset, year_start: int, year_stop: int, center_ls: float, sols: int
) -> tuple[int, int]:
    """Return the nearest complete daily-aligned window, avoiding source gaps."""
    count = sols * SAMPLES_PER_SOL
    time = np.asarray(
        dataset.time.isel(time=slice(year_start, year_stop)).compute(),
        dtype=np.float64,
    )
    ls = np.asarray(
        dataset.Ls.isel(time=slice(year_start, year_stop)).compute(),
        dtype=np.float64,
    )
    expected = 1.0 / SAMPLES_PER_SOL
    candidates = []
    for start in range(0, len(time) - count + 1, SAMPLES_PER_SOL):
        stop = start + count
        if np.allclose(np.diff(time[start:stop]), expected, rtol=0.0, atol=1e-9):
            midpoint = start + count // 2
            distance = abs((ls[midpoint] - center_ls + 180.0) % 360.0 - 180.0)
            candidates.append((distance, start, stop))
    if not candidates:
        raise ValueError(
            f"MY selection contains no contiguous {sols}-sol window"
        )
    _, start, stop = min(candidates)
    return year_start + start, year_start + stop


def _daily_mean(sample: xr.Dataset) -> xr.Dataset:
    if sample.sizes["time"] % SAMPLES_PER_SOL:
        raise ValueError("sample length must contain complete 12-state sols")
    time = np.asarray(sample.time, dtype=np.float64)
    spacing = np.diff(time)
    expected = 1.0 / SAMPLES_PER_SOL
    if spacing.size and not np.allclose(spacing, expected, rtol=0.0, atol=1e-9):
        raise ValueError(
            "ARCO-MACDA time coordinate is not uniformly spaced at 1/12 sol: "
            f"min={spacing.min():.12g}, max={spacing.max():.12g}"
        )

    physical = sample.drop_vars(["Ls", "MY_Ls"])
    daily = physical.coarsen(time=SAMPLES_PER_SOL, boundary="exact").mean(
        keep_attrs=True
    )
    ls_rad = np.deg2rad(sample["Ls"])
    sin_ls = np.sin(ls_rad).coarsen(time=SAMPLES_PER_SOL, boundary="exact").mean()
    cos_ls = np.cos(ls_rad).coarsen(time=SAMPLES_PER_SOL, boundary="exact").mean()
    daily["Ls"] = np.mod(np.rad2deg(np.arctan2(sin_ls, cos_ls)), 360.0)
    daily["Ls"].attrs.update(sample["Ls"].attrs)
    daily["MY_Ls"] = sample["MY_Ls"].coarsen(
        time=SAMPLES_PER_SOL, boundary="exact"
    ).mean()
    daily["MY_Ls"].attrs.update(sample["MY_Ls"].attrs)
    return daily


def _periodic_longitude(dataset: xr.Dataset) -> xr.Dataset:
    base = dataset.assign_coords(lon=np.mod(dataset.lon, 360.0)).sortby("lon")
    left = base.isel(lon=[-1]).assign_coords(lon=base.lon.isel(lon=[-1]) - 360.0)
    right = base.isel(lon=[0]).assign_coords(lon=base.lon.isel(lon=[0]) + 360.0)
    return xr.concat(
        [left, base, right],
        dim="lon",
        data_vars="all",
        coords="minimal",
        compat="override",
    )


def _to_dinosaur_grid(dataset: xr.Dataset, truncation: str, layers: int) -> xr.Dataset:
    coords = coordinate_system(truncation, layers)
    target_lon = np.mod(np.degrees(np.asarray(coords.horizontal.longitudes)), 360.0)
    target_lat = np.degrees(np.asarray(coords.horizontal.latitudes))
    target_lev = np.asarray(coords.vertical.centers)

    temporal = dataset[["Ls", "MY_Ls"]]
    spatial = dataset.drop_vars(["Ls", "MY_Ls"])
    ordered = spatial.sortby("lat")
    if "lev" in ordered.coords:
        ordered = ordered.sortby("lev")
    periodic = _periodic_longitude(ordered)
    targets = {
        "lat": xr.DataArray(
            target_lat, dims="lat", attrs={"units": "degrees_north"}
        ),
        "lon": xr.DataArray(
            target_lon, dims="lon", attrs={"units": "degrees_east"}
        ),
    }
    if "lev" in periodic.coords:
        targets["lev"] = xr.DataArray(
            target_lev, dims="lev", attrs={"units": "1"}
        )
    result = periodic.interp(targets, method="linear")
    if "lev" in result.coords:
        result["lev"].attrs.update(
            standard_name="atmosphere_sigma_coordinate",
            long_name="Dinosaur sigma-layer midpoint",
            axis="Z",
        )
    result["Ls"] = temporal["Ls"]
    result["MY_Ls"] = temporal["MY_Ls"]
    # Assigning the temporal variables also merges their source time coordinate,
    # so replace its attributes and encoding only after those assignments. The
    # remote coordinate uses CF date encoding (``days since 0001-01-01``,
    # ``calendar=none``), but these values are Martian elapsed sols rather than
    # datetimes. Keeping that encoding makes normal xarray reads require cftime.
    result["time"].attrs = {
        "long_name": "Martian sols since MY24 start (daily midpoint)",
        "units": "sol",
        "axis": "T",
    }
    result["time"].encoding = {}
    return result


def _variable_stats(dataset: xr.Dataset) -> dict[str, dict[str, float | int | str]]:
    result: dict[str, dict[str, float | int | str]] = {}
    for name, variable in dataset.data_vars.items():
        values = np.asarray(variable)
        finite = np.isfinite(values)
        stats: dict[str, float | int | str] = {
            "units": str(variable.attrs.get("units", "")),
            "count": int(values.size),
            "finite_count": int(finite.sum()),
            "finite_fraction": float(finite.mean()),
        }
        if finite.any():
            finite_values = values[finite]
            stats.update(
                minimum=float(finite_values.min()),
                maximum=float(finite_values.max()),
                mean=float(finite_values.mean()),
            )
        result[name] = stats
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-url", default=DEFAULT_URL)
    parser.add_argument("--mars-year", type=int, default=24)
    parser.add_argument("--start-sol", type=int, default=0)
    parser.add_argument(
        "--center-ls", type=float,
        help="choose the sol window centered nearest this solar longitude",
    )
    parser.add_argument("--sols", type=int, default=10)
    parser.add_argument("--truncation", default="T21")
    parser.add_argument("--layers", type=int, default=12)
    parser.add_argument("--variables", nargs="+", default=list(DEFAULT_VARIABLES))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/arco_macda/macda_my24_sols0000-0009_daily_t21_l12.nc"),
    )
    args = parser.parse_args()
    if args.start_sol < 0 or args.sols < 1 or args.layers < 1:
        parser.error("--start-sol must be non-negative; --sols and --layers must be positive")

    mapper = fsspec.get_mapper(args.source_url)
    source = xr.open_zarr(mapper, consolidated=True, decode_times=False)
    missing = sorted(set(args.variables) - set(source.data_vars))
    if missing:
        parser.error(f"variables absent from ARCO-MACDA: {missing}")

    year_start, year_stop = _mars_year_bounds(source, args.mars_year)
    if args.center_ls is not None:
        target_ls = args.center_ls % 360.0
        start, stop = _nearest_contiguous_window(
            source, year_start, year_stop, target_ls, args.sols
        )
        first_time = _scalar_at(source.time, year_start)
        args.start_sol = int(round(_scalar_at(source.time, start) - first_time))
    else:
        start = year_start + args.start_sol * SAMPLES_PER_SOL
        stop = start + args.sols * SAMPLES_PER_SOL
    if stop > year_stop:
        available_sols = (year_stop - year_start) // SAMPLES_PER_SOL
        parser.error(
            f"selection ends beyond MY {args.mars_year}; available complete sols: "
            f"{available_sols}"
        )

    selected_names = [*args.variables, "Ls", "MY_Ls"]
    sample = source[selected_names].isel(time=slice(start, stop))
    daily_native = _daily_mean(sample)
    prepared = _to_dinosaur_grid(daily_native, args.truncation, args.layers).load()
    if not all(np.isfinite(np.asarray(prepared[name])).all() for name in args.variables):
        raise ValueError("prepared ARCO-MACDA sample contains non-finite training fields")

    prepared.attrs = {
        "title": "ARCO-MACDA daily means on the Dinosaur grid",
        "source_url": args.source_url,
        "source_doi": "10.57967/hf/8771",
        "underlying_macda_doi": "10.5285/cd037a9ea387438fabf4d674dbe53088",
        "source_license": "CC BY 4.0; underlying MACDA v2.0 OGL v3.0",
        "mars_year": args.mars_year,
        "source_start_index": start,
        "source_stop_index_exclusive": stop,
        "source_samples_per_sol": SAMPLES_PER_SOL,
        "temporal_reduction": "arithmetic daily mean; circular mean for solar longitude",
        "vertical_regridding": "linear interpolation in sigma",
        "horizontal_regridding": "periodic bilinear latitude-longitude interpolation",
        "target_truncation": args.truncation,
        "target_layers": args.layers,
        "retrieved_utc": dt.datetime.now(dt.UTC).isoformat(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    encoding = {
        name: {"zlib": True, "complevel": 4, "shuffle": True}
        for name in prepared.data_vars
    }
    prepared.to_netcdf(args.output, engine="h5netcdf", encoding=encoding)

    metadata = {
        "artifact": str(args.output),
        "artifact_sha256": _sha256(args.output),
        "artifact_bytes": args.output.stat().st_size,
        "retrieved_utc": prepared.attrs["retrieved_utc"],
        "source_url": args.source_url,
        "source_logical_shape": {name: int(size) for name, size in source.sizes.items()},
        "source_logical_bytes": int(source.nbytes),
        "source_time_spacing_sols": 1.0 / SAMPLES_PER_SOL,
        "source_samples_per_sol": SAMPLES_PER_SOL,
        "source_mars_year_range": [
            int(_scalar_at(source["MY_Ls"], 0)),
            int(_scalar_at(source["MY_Ls"], source.sizes["time"] - 1)),
        ],
        "selection": {
            "mars_year": args.mars_year,
            "start_sol": args.start_sol,
            "sols": args.sols,
            "requested_center_ls_deg": args.center_ls,
            "source_index_range": [start, stop],
        },
        "target": {
            "truncation": args.truncation,
            "layers": args.layers,
            "shape": {name: int(size) for name, size in prepared.sizes.items()},
        },
        "variable_stats": _variable_stats(prepared),
        "citations": {
            "arco_macda_doi": "10.57967/hf/8771",
            "macda_v2_doi": "10.5285/cd037a9ea387438fabf4d674dbe53088",
        },
        "licenses": ["CC BY 4.0", "Open Government Licence v3.0"],
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
