"""Standalone output — CSV serialisation and matplotlib plots.

No dependency on experiments/.  All serialisation and plotting logic lives here.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from cli.runner import RunResult, MULTI_POINTS
from src.celestials import MARS_ROTATION_PERIOD

_SOL_SECONDS = float(MARS_ROTATION_PERIOD.item())
_SOL_HOURS   = _SOL_SECONDS / 3600.0


def _v(t) -> float:
    return float(t.item())


def _tag(out_cfg: dict) -> str:
    name = out_cfg.get("out_dir")
    return name if name else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _ensure(path: str) -> str:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


# ── CSV ───────────────────────────────────────────────────────────────────────

def save_csv(result: RunResult, filepath: str) -> None:
    """Write a RunResult's history to CSV."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_hours", "temperature_k", "pressure_pa",
                    "ice_mass_kg", "solar_flux_wm2", "orbital_angle_rad"])
        for s in result.history:
            w.writerow([
                _v(s.time) / 3600.0,
                _v(s.surface_temperature),
                _v(s.surface_pressure),
                _v(s.ice_mass),
                _v(s.solar_flux),
                _v(s.orbital_angle),
            ])
    print(f"  CSV  → {filepath}")


# ── Diurnal / multi-sol plot ──────────────────────────────────────────────────

def plot_diurnal(result: RunResult, filepath: str, title: str) -> None:
    """Three-panel plot: temperature, pressure, ice mass vs time."""
    history = result.history
    max_t   = max(_v(s.time) for s in history)
    use_sols = max_t > 3 * _SOL_SECONDS

    if use_sols:
        xs     = [_v(s.time) / _SOL_SECONDS for s in history]
        xlabel = "Time (sols)"
    else:
        xs     = [_v(s.time) / _SOL_SECONDS * 360.0 for s in history]
        xlabel = "Rotation angle (°)"

    temps  = [_v(s.surface_temperature) for s in history]
    press  = [_v(s.surface_pressure)    for s in history]
    ice    = [_v(s.ice_mass)            for s in history]

    lw = max(0.8, 2.0 - 1.2 * min(len(xs) / 2000, 1.0))

    panels = [
        (temps, "Temperature (K)",  "#d62728", "temp"),
        (press, "Pressure (Pa)",    "#1f77b4", "pressure"),
        (ice,   "Ice Mass (kg)",    "#2ca02c", "ice"),
    ]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for ys, ylabel, color, tag in panels:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, color=color, linewidth=lw)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title} — {ylabel.split(' (')[0]}", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.4, linestyle="--")
        plt.tight_layout()
        out = filepath.replace(".png", f"_{tag}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Plot → {out}")


# ── Multi-coordinate temperature overlay ─────────────────────────────────────

def plot_multi_coord(results: list[RunResult], filepath: str, title: str) -> None:
    """Single figure overlaying temperature traces for all coordinates."""
    colors = {
        "North":               "#1f77b4",
        "Equator":             "#d62728",
        "Southern Hemisphere": "#2ca02c",
    }

    fig, ax = plt.subplots(figsize=(12, 6))

    for res in results:
        times_h = np.array([_v(s.time) / 3600.0 for s in res.history])
        temps   = np.array([_v(s.surface_temperature) for s in res.history])
        lw      = max(1.0, 2.0 - 1.0 * min(len(times_h) / 2000, 1.0))
        ns      = "N" if res.lat >= 0 else "S"
        label   = f"{res.name}  ({abs(res.lat):.0f}°{ns}, {res.lon:.0f}°E)"
        ax.plot(times_h, temps, label=label,
                color=colors.get(res.name, "#555555"), linewidth=lw)

    # Vertical sol boundaries
    max_h = max(_v(r.history[-1].time) for r in results) / 3600.0
    for i in range(1, int(max_h / _SOL_HOURS) + 1):
        ax.axvline(i * _SOL_HOURS, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot → {filepath}")


# ── Seasonal temperature plot (year runs) ─────────────────────────────────────

def plot_seasonal(result: RunResult, filepath: str, title: str,
                  spin_up_sols: int = 10) -> None:
    """Daily min/max/avg temperature vs Solar Longitude (Ls)."""
    history    = result.history
    spin_up_s  = spin_up_sols * _SOL_SECONDS

    times = np.array([_v(s.time) for s in history])
    temps = np.array([_v(s.surface_temperature) for s in history])
    ls_rad = np.array([_v(s.orbital_angle) for s in history]) + 251.0 * np.pi / 180.0
    ls = (ls_rad * 180.0 / np.pi) % 360.0
    sols = times // _SOL_SECONDS

    counts = np.array([np.sum(sols == s) for s in np.unique(sols)])
    min_steps = max(10, int(0.8 * int(np.median(counts))))

    daily_ls, daily_max, daily_min, daily_avg = [], [], [], []
    for sol in np.unique(sols[times > spin_up_s]):
        mask = (sols == sol) & (times > spin_up_s)
        if np.sum(mask) < min_steps:
            continue
        t_day = temps[mask]
        ls_d  = ls[mask]
        if ls_d.max() > 350 and ls_d.min() < 10:
            ls_d = np.where(ls_d < 180, ls_d + 360, ls_d)
        daily_ls.append(float(np.mean(ls_d) % 360))
        daily_max.append(float(np.max(t_day)))
        daily_min.append(float(np.min(t_day)))
        daily_avg.append(float(np.mean(t_day)))

    idx = np.argsort(daily_ls)
    ls_arr  = np.array(daily_ls)[idx]
    mx_arr  = np.array(daily_max)[idx]
    mn_arr  = np.array(daily_min)[idx]
    avg_arr = np.array(daily_avg)[idx]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ls_arr, mx_arr,  label="Daily Max", color="#d62728", linewidth=2)
    plt.plot(ls_arr, mn_arr,  label="Daily Min", color="#1f77b4", linewidth=2, linestyle="--")
    plt.plot(ls_arr, avg_arr, label="Daily Avg", color="#2ca02c", linewidth=2)
    plt.fill_between(ls_arr, mn_arr, mx_arr, alpha=0.08, color="#888888")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Solar Longitude Ls (°)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot → {filepath}")


# ── Dispatch ──────────────────────────────────────────────────────────────────

def dispatch(results: list[RunResult], cfg: dict, exp_type: str) -> None:
    """Save all outputs for a completed simulation."""
    out_cfg = cfg["output"]
    if not out_cfg["save_csv"] and not out_cfg["save_plot"]:
        return

    if out_cfg.get("output_path"):
        out_dir = out_cfg["output_path"]
    else:
        out_dir = os.path.join("outputs", _tag(out_cfg))

    p   = cfg["planet"]
    ns  = "N" if p["latitude"] >= 0 else "S"
    loc = f"{abs(p['latitude']):.1f}°{ns}, {p['longitude']:.1f}°E"

    if exp_type == "sol":
        r     = results[0]
        sols  = cfg["experiment"]["sols"]
        title = f"Mars — {sols:.0f} sol{'s' if sols!=1 else ''}  ({loc})"
        if out_cfg["save_csv"]:
            save_csv(r, os.path.join(out_dir, "mars_sol.csv"))
        if out_cfg["save_plot"]:
            plot_diurnal(r, os.path.join(out_dir, "mars_sol.png"), title)

    elif exp_type == "year":
        r     = results[0]
        title = f"Mars — 1 year  ({loc})"
        if out_cfg["save_csv"]:
            save_csv(r, os.path.join(out_dir, "mars_year.csv"))
        if out_cfg["save_plot"]:
            plot_diurnal(r, os.path.join(out_dir, "mars_year.png"), title)
            plot_seasonal(r, os.path.join(out_dir, "mars_seasonal.png"),
                          f"Mars Seasonal Temperatures — 1 Year")

    elif exp_type == "multi":
        sols  = cfg["experiment"]["sols"]
        title = f"Mars Multi-Coord — {sols:.0f} sol{'s' if sols!=1 else ''}  (137°E)"
        for r in results:
            safe = r.name.lower().replace(" ", "_")
            if out_cfg["save_csv"]:
                save_csv(r, os.path.join(out_dir, f"mars_{safe}.csv"))
            if out_cfg["save_plot"]:
                ns2   = "N" if r.lat >= 0 else "S"
                rtitle = f"Mars {r.name} ({abs(r.lat):.0f}°{ns2})"
                plot_diurnal(r, os.path.join(out_dir, f"mars_{safe}.png"), rtitle)
        if out_cfg["save_plot"]:
            plot_multi_coord(results, os.path.join(out_dir, "mars_multi_coord.png"), title)
