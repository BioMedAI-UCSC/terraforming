"""Load REMS (Curiosity rover) ground-truth data from PDS files.

Data sources (already downloaded to data/rems/):
  sol224_modrdr.TAB   - per-second REMS MODRDR6, Sol 224, Gale Crater
  mars-weather.csv    - The Pudding dataset: daily min/max temp + pressure per sol
"""
from __future__ import annotations

import csv
import re

import numpy as np

# Gale Crater (Curiosity landing site)
REMS_LAT = -4.589
REMS_LON = 137.441

MARS_SOL_HOURS = 88_775.244 / 3600.0          # 24.659 h per sol
MODRDR_PATH   = "data/rems/sol224_modrdr.TAB"
DAILY_PATH    = "data/rems/mars-weather.csv"


def _parse_val(s: str) -> float:
    """Parse a fixed-width numeric field; return NaN on UNK/NULL/empty."""
    s = s.strip().strip('"').strip()
    try:
        return float(s)
    except ValueError:
        return np.nan


def _parse_lmst(s: str) -> float:
    """Parse LMST field 'SSSSSM HH:MM:SS.sss' → hours (float)."""
    s = s.strip().strip('"').strip()
    m = re.search(r'M(\d{2}):(\d{2}):(\d+(?:\.\d+)?)', s)
    if m:
        h, mn, sc = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h + mn / 60.0 + sc / 3600.0
    return np.nan


def load_modrdr(filepath: str = MODRDR_PATH) -> dict:
    """Parse a REMS MODRDR6 TAB file into arrays aligned to LMST.

    Column layout (0-indexed, comma-separated):
      0  TIMESTAMP
      1  LMST   (Local Mean Solar Time)
      7  BRIGHTNESS_TEMP  (ground/surface temp, K; often UNK in some sols)
      11 BOOM1_LOCAL_AIR_TEMP  (air temp, K)
      37 PRESSURE  (Pa)

    Returns
    -------
    dict with keys:
      lmst_hours   float[N]  – hours since sol midnight (sorted)
      rotation_deg float[N]  – lmst_hours / MARS_SOL_HOURS * 360
      ground_temp  float[N]  – BRIGHTNESS_TEMP in K  (NaN where missing)
      air_temp     float[N]  – BOOM1_LOCAL_AIR_TEMP in K  (NaN where missing)
      pressure     float[N]  – PRESSURE in Pa  (NaN where missing)
    """
    lmst_hours, ground_temp, air_temp, pressure = [], [], [], []

    with open(filepath, "r") as f:
        for line in f:
            parts = line.split(",")
            if len(parts) < 38:
                continue
            lmst_h = _parse_lmst(parts[1])
            if np.isnan(lmst_h):
                continue
            lmst_hours.append(lmst_h)
            ground_temp.append(_parse_val(parts[7]))
            air_temp.append(_parse_val(parts[11]))
            pressure.append(_parse_val(parts[37]))

    lmst_hours  = np.array(lmst_hours)
    ground_temp = np.array(ground_temp)
    air_temp    = np.array(air_temp)
    pressure    = np.array(pressure)

    idx = np.argsort(lmst_hours)
    return {
        "lmst_hours":   lmst_hours[idx],
        "rotation_deg": lmst_hours[idx] / MARS_SOL_HOURS * 360.0,
        "ground_temp":  ground_temp[idx],
        "air_temp":     air_temp[idx],
        "pressure":     pressure[idx],
    }


def load_daily(filepath: str = DAILY_PATH) -> dict:
    """Parse The Pudding mars-weather.csv.

    Temperature in CSV is °C → converted to K.

    Returns
    -------
    dict (sorted by Solar Longitude) with keys:
      sol        int[N]
      ls         float[N]  – Solar Longitude (degrees)
      min_temp_K float[N]
      max_temp_K float[N]
      avg_temp_K float[N]
      pressure   float[N]  – Pa
    """
    sol, ls, min_t, max_t, pres = [], [], [], [], []

    with open(filepath, "r", newline="") as f:
        for row in csv.DictReader(f):
            try:
                s  = int(row["sol"])
                l  = float(row["ls"])
                mn = float(row["min_temp"]) + 273.15
                mx = float(row["max_temp"]) + 273.15
                pr = float(row["pressure"])
            except (ValueError, KeyError):
                continue
            sol.append(s);  ls.append(l)
            min_t.append(mn); max_t.append(mx); pres.append(pr)

    sol   = np.array(sol,   dtype=int)
    ls    = np.array(ls)
    min_t = np.array(min_t)
    max_t = np.array(max_t)
    pres  = np.array(pres)

    idx = np.argsort(ls)
    return {
        "sol":        sol[idx],
        "ls":         ls[idx],
        "min_temp_K": min_t[idx],
        "max_temp_K": max_t[idx],
        "avg_temp_K": (min_t[idx] + max_t[idx]) / 2.0,
        "pressure":   pres[idx],
    }
