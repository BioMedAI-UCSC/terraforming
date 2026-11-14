"""Reusable MCD v6.1 web client and map-alignment utilities."""

from __future__ import annotations

import html
import http.client
import math
import re
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np
import xarray as xr

MCD_CGI = "https://www-mars.lmd.jussieu.fr/mcd_python/cgi-bin/mcdcgi.py"
MCD_ROOT = "https://www-mars.lmd.jussieu.fr/mcd_python/"
MARS_GRAVITY_M_S2 = 3.72076
FIELDS = {
    "temperature": ("Surface temperature", "K", 1.0),
    "surface_pressure": ("Surface pressure", "Pa", 1.0),
    "wind_speed": ("Horizontal wind speed", "m/s", 1.0),
    "co2_ice": ("Monthly mean surface CO2 ice layer", "Pa-equivalent", MARS_GRAVITY_M_S2),
}


def query_url(ls: float, local_time: float, dust: int = 1,
              high_res: bool = True, altitude_m: float = 10.0) -> str:
    params = {
        "var1": "tsurf", "var2": "ps", "var3": "wind", "var4": "co2ice",
        "datekeyhtml": "1", "ls": f"{ls:g}", "localtime": f"{local_time:g}",
        "latitude": "all", "longitude": "all", "altitude": f"{altitude_m:g}",
        "zkey": "3", "dust": str(dust), "hrkey": "1" if high_res else "0",
        "averaging": "off", "isfixedlt": "on", "animation": "off",
        "animframes": "12", "dpi": "160", "islog": "off", "colorm": "jet",
        "proj": "cyl", "iswind": "off",
    }
    return f"{MCD_CGI}?{urllib.parse.urlencode(params)}"


def _download(url: str, timeout: float = 180.0, attempts: int = 3) -> bytes:
    request = urllib.request.Request(
        url, headers={"User-Agent": "terraforming-gcm3d-validation/1"}
    )
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
                return response.read()
        except (http.client.IncompleteRead, TimeoutError, urllib.error.URLError):
            if attempt == attempts:
                raise
            time.sleep(float(attempt))
    raise AssertionError("unreachable")


def fetch_ascii(ls: float, local_time: float, timeout: float = 180.0, *,
                dust: int = 1, high_res: bool = True,
                altitude_m: float = 10.0) -> tuple[str, str]:
    page = _download(
        query_url(ls, local_time, dust, high_res, altitude_m), timeout
    ).decode("utf-8", errors="replace")
    match = re.search(r"href=['\"]([^'\"]+\.txt)['\"]", page)
    if not match:
        message = re.sub(r"<[^>]+>", " ", page)
        raise RuntimeError(
            f"MCD did not return a data link: {' '.join(message.split())[:400]}"
        )
    data_url = urllib.parse.urljoin(
        MCD_ROOT + "cgi-bin/", html.unescape(match.group(1))
    )
    return _download(data_url, timeout).decode("utf-8", errors="replace"), data_url


def parse_ascii(text: str) -> xr.Dataset:
    blocks = re.split(r"(?=#{10,}\n### MCD_v)", text)
    arrays: dict[str, xr.DataArray] = {}
    lookup = {description.lower(): name for name, (description, _, _) in FIELDS.items()}
    for block in blocks:
        column = re.search(r"### Columns 2\+ are (.+?)\s*$", block, re.MULTILINE)
        header = re.search(r"^---- \|\|\s+(.+)$", block, re.MULTILINE)
        if not column or not header:
            continue
        description = column.group(1).strip()
        name = next(
            (value for key, value in lookup.items()
             if description.lower().startswith(key)), None
        )
        if name is None:
            continue
        lat = np.fromstring(header.group(1), sep=" ")
        lon_values, rows = [], []
        for line in block[header.end():].splitlines():
            if "||" not in line:
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
    return normalized.groupby("lon").mean()


def interpolate_periodic(reference: xr.Dataset, target: xr.Dataset) -> xr.Dataset:
    reference = normalize_longitude(reference)
    left = reference.isel(lon=-1).assign_coords(lon=float(reference.lon[-1]) - 360.0)
    right = reference.isel(lon=0).assign_coords(lon=float(reference.lon[0]) + 360.0)
    extended = xr.concat([left, reference, right], dim="lon")
    return extended.interp(lat=target.lat, lon=target.lon)


def weighted_metrics(model: np.ndarray, reference: np.ndarray,
                     lat: np.ndarray) -> dict[str, float]:
    weights = np.broadcast_to(np.cos(np.deg2rad(lat))[:, None], model.shape)
    valid = np.isfinite(model) & np.isfinite(reference) & (weights > 0)
    x, y, w = model[valid], reference[valid], weights[valid]
    w = w / w.sum()
    error = x - y
    x_mean, y_mean = np.sum(w * x), np.sum(w * y)
    covariance = np.sum(w * (x - x_mean) * (y - y_mean))
    denominator = math.sqrt(
        np.sum(w * (x - x_mean) ** 2) * np.sum(w * (y - y_mean) ** 2)
    )
    return {
        "bias": float(np.sum(w * error)),
        "mae": float(np.sum(w * np.abs(error))),
        "rmse": float(math.sqrt(np.sum(w * error**2))),
        "spatial_correlation": float(covariance / denominator) if denominator else float("nan"),
        "model_area_mean": float(x_mean),
        "mcd_area_mean": float(y_mean),
    }
