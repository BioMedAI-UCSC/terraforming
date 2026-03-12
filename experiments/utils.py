"""Shared utilities for Mars terraforming experiments."""
from __future__ import annotations

import csv
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np

from src.celestials import MARS_ROTATION_PERIOD

MARS_SOL_HOURS = 88_775.244 / 3600.0  # 24.659 h

REMS_NOTE = "REMS Curiosity — Gale Crater (4.6°S, 137.4°E)"


def _v(t) -> float:
    """torch.Tensor → Python float."""
    return float(t.item())


def save_history_to_csv(history, filename: str):
    """Write integration steps to a CSV file."""
    filepath = os.path.join("outputs", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["time_hours", "temperature_k", "pressure_pa", "ice_mass_kg", "solar_flux_wm2", "orbital_angle_rad"])
        for s in history:
            writer.writerow([
                _v(s.time) / 3600.0,
                _v(s.surface_temperature),
                _v(s.surface_pressure),
                _v(s.ice_mass),
                _v(s.solar_flux),
                _v(s.orbital_angle),
            ])
    print(f"  Data saved to {filepath}")


def plot_history(history, filename: str, title: str, ground: Optional[dict] = None):
    """Plot temperature, pressure, and ice mass evolution.

    Parameters
    ----------
    ground : dict, optional
        Ground-truth overlay from ``rems_loader.load_modrdr()``.
        Expected keys: ``rotation_deg``, ``ground_temp``, ``air_temp``, ``pressure``.
        For multi-sol runs the pattern is tiled automatically.
    """
    max_time_s = max([_v(s.time) for s in history])
    use_sols = max_time_s > 3 * _v(MARS_ROTATION_PERIOD)
    n_sols = max_time_s / _v(MARS_ROTATION_PERIOD)

    if use_sols:
        times = [_v(s.time) / _v(MARS_ROTATION_PERIOD) for s in history]
        xlabel = "Time (Sols)"
    else:
        times = [(_v(s.time) / _v(MARS_ROTATION_PERIOD)) * 360.0 for s in history]
        xlabel = "Rotation Angle (°)"

    temps      = [_v(s.surface_temperature) for s in history]
    pressures  = [_v(s.surface_pressure) for s in history]
    ice_masses = [_v(s.ice_mass) for s in history]

    n  = len(times)
    lw = max(0.8, 2.0 - 1.2 * min(n / 2000, 1.0))

    # Build per-series ground-truth (x, y) pairs — each series uses its own mask
    # gt_series[suffix] = list of (x_arr, y_arr, label, color, linestyle)
    gt_series: dict = {"_temp.png": [], "_pressure.png": [], "_ice.png": []}

    if ground is not None:
        rot   = ground["rotation_deg"]   # 0-360° for one sol
        g_air = ground["air_temp"]
        g_gnd = ground["ground_temp"]
        g_prs = ground["pressure"]

        def _tile(raw_x, raw_y):
            """Return tiled (x, y) across all simulated sols, NaNs removed."""
            mask = ~np.isnan(raw_y)
            if not mask.any():
                return None, None
            if use_sols:
                xs, ys = [], []
                for i in range(int(np.ceil(n_sols))):
                    xs.append(raw_x[mask] / 360.0 + i)
                    ys.append(raw_y[mask])
                return np.concatenate(xs), np.concatenate(ys)
            return raw_x[mask], raw_y[mask]

        for raw_y, lbl, clr, ls, sfx in [
            (g_air, "REMS Air Temp",    "#ff7f0e", "-",  "_temp.png"),
            (g_gnd, "REMS Ground Temp", "#9467bd", "--", "_temp.png"),
            (g_prs, "REMS Pressure",    "#ff7f0e", "-",  "_pressure.png"),
        ]:
            x, y = _tile(rot, raw_y)
            if x is not None:
                gt_series[sfx].append((x, y, lbl, clr, ls))

    for data, label, color, suffix, ylabel in [
        (temps,      "Temperature", "#d62728", "_temp.png",     "Temperature (K)"),
        (pressures,  "Pressure",    "#1f77b4", "_pressure.png", "Pressure (Pa)"),
        (ice_masses, "Ice Mass",    "#2ca02c", "_ice.png",      "Ice Mass (kg)"),
    ]:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(times, data, label=label, color=color, linewidth=lw, zorder=3)

        # Ground-truth overlay
        series = gt_series.get(suffix, [])
        if series:
            gt_lw = max(0.6, lw * 0.7)
            for gt_x, gt_y, gt_label, gt_color, gt_ls in series:
                plt.plot(gt_x, gt_y, label=gt_label, color=gt_color,
                         linewidth=gt_lw, linestyle=gt_ls, alpha=0.85, zorder=2)
            plt.text(0.01, 0.02, REMS_NOTE, transform=plt.gca().transAxes,
                     fontsize=8, color="#555555", va="bottom")

        plt.ylabel(ylabel, fontsize=12)
        plt.xlabel(xlabel, fontsize=12)
        plt.title(f"{title} — {label}", fontsize=13, fontweight="bold")
        plt.legend(fontsize=10, framealpha=0.9)
        plt.grid(True, alpha=0.4, linestyle="--")
        plt.tight_layout()
        out = os.path.join("outputs", filename.replace(".png", suffix))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out, dpi=150)
        print(f"  Plot saved to {out}")
        plt.close(fig)


def plot_seasonal_temps(history, filename: str, title: str,
                        spin_up_sols: int = 0, ground: Optional[dict] = None):
    """Plot daily min, max, and avg temperatures over Solar Longitude (Ls).

    Parameters
    ----------
    spin_up_sols : int
        Number of initial sols to discard as thermal spin-up.
    ground : dict, optional
        Ground-truth overlay from ``rems_loader.load_daily()``.
        Expected keys: ``ls``, ``min_temp_K``, ``max_temp_K``, ``avg_temp_K``.
    """
    sol_period = _v(MARS_ROTATION_PERIOD)
    spin_up_seconds = spin_up_sols * sol_period

    times = np.array([_v(s.time) for s in history])
    temps = np.array([_v(s.surface_temperature) for s in history])

    ls_rad = np.array([_v(s.orbital_angle) for s in history]) + 251.0 * np.pi / 180.0
    ls = (ls_rad * 180 / np.pi) % 360.0

    sols = times // sol_period

    all_counts = np.array([np.sum(sols == s) for s in np.unique(sols)])
    expected_steps = int(np.median(all_counts))
    min_steps = max(10, int(0.8 * expected_steps))

    unique_sols = np.unique(sols[times > spin_up_seconds])

    daily_ls, daily_max, daily_min, daily_avg = [], [], [], []

    for sol in unique_sols:
        mask = (sols == sol) & (times > spin_up_seconds)
        if np.sum(mask) < min_steps:
            continue
        t_day = temps[mask]
        ls_day_array = ls[mask]
        if np.max(ls_day_array) > 350 and np.min(ls_day_array) < 10:
            ls_day_array = np.where(ls_day_array < 180, ls_day_array + 360, ls_day_array)
            ls_day = np.mean(ls_day_array) % 360.0
        else:
            ls_day = np.mean(ls_day_array)
        daily_max.append(np.max(t_day))
        daily_min.append(np.min(t_day))
        daily_avg.append(np.mean(t_day))
        daily_ls.append(ls_day)

    idx = np.argsort(daily_ls)
    daily_ls  = np.array(daily_ls)[idx]
    daily_max = np.array(daily_max)[idx]
    daily_min = np.array(daily_min)[idx]
    daily_avg = np.array(daily_avg)[idx]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(daily_ls, daily_max, label="Model Daily Max", color="#d62728", linewidth=2.0)
    plt.plot(daily_ls, daily_min, label="Model Daily Min", color="#1f77b4", linewidth=2.0, linestyle="--")
    plt.plot(daily_ls, daily_avg, label="Model Daily Avg", color="#2ca02c", linewidth=2.0)

    if ground is not None:
        plt.fill_between(ground["ls"], ground["min_temp_K"], ground["max_temp_K"],
                         alpha=0.18, color="#ff7f0e", label="REMS Min–Max band")
        plt.plot(ground["ls"], ground["avg_temp_K"],
                 color="#ff7f0e", linewidth=1.5, linestyle="-.",
                 label="REMS Daily Avg")
        plt.text(0.01, 0.02, REMS_NOTE, transform=plt.gca().transAxes,
                 fontsize=8, color="#555555", va="bottom")

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Solar Longitude Ls (°)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.legend(fontsize=10, framealpha=0.9)

    filepath = os.path.join("outputs", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    print(f"  Seasonal plot saved to {filepath}")
    plt.close(fig)
