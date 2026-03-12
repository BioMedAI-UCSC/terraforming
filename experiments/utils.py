"""Shared utilities for Mars terraforming experiments."""
from __future__ import annotations

import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from src.celestials import MARS_ROTATION_PERIOD


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


def plot_history(history, filename: str, title: str):
    """Plot temperature, pressure, and ice mass evolution."""
    max_time_s = max([_v(s.time) for s in history])
    use_sols = max_time_s > 3 * _v(MARS_ROTATION_PERIOD)

    if use_sols:
        times = [_v(s.time) / _v(MARS_ROTATION_PERIOD) for s in history]
        xlabel = "Time (Sols)"
    else:
        times = [(_v(s.time) / _v(MARS_ROTATION_PERIOD)) * 360.0 for s in history]
        xlabel = "Rotation Angle (°)"

    temps = [_v(s.surface_temperature) for s in history]
    pressures = [_v(s.surface_pressure) for s in history]
    ice_masses = [_v(s.ice_mass) for s in history]

    lw = 0.15 if len(times) > 1000 else 1.5
    al = 0.7 if len(times) > 1000 else 1.0

    for data, label, color, suffix, ylabel in [
        (temps,      "Temperature", "tab:red",  "_temp.png",     "Temperature (K)"),
        (pressures,  "Pressure",    "tab:blue", "_pressure.png", "Pressure (Pa)"),
        (ice_masses, "Ice Mass",    "tab:cyan", "_ice.png",      "Ice Mass (kg)"),
    ]:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(times, data, label=label, color=color, linewidth=lw, alpha=al)
        plt.ylabel(ylabel)
        plt.xlabel(xlabel)
        plt.title(f"{title} - {label}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        out = os.path.join("outputs", filename.replace(".png", suffix))
        os.makedirs(os.path.dirname(out), exist_ok=True)
        plt.savefig(out)
        print(f"  Plot saved to {out}")
        plt.close(fig)


def plot_seasonal_temps(history, filename: str, title: str, spin_up_sols: int = 0):
    """Plot daily min, max, and avg temperatures over Solar Longitude (Ls).

    spin_up_sols: number of initial sols to discard as thermal spin-up.
    """
    sol_period = _v(MARS_ROTATION_PERIOD)
    spin_up_seconds = spin_up_sols * sol_period

    times = np.array([_v(s.time) for s in history])
    temps = np.array([_v(s.surface_temperature) for s in history]) - 273.15

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
    plt.plot(daily_ls, daily_max, label="Daily Max", color="cyan")
    plt.plot(daily_ls, daily_min, label="Daily Min", color="red", linestyle="--")
    plt.plot(daily_ls, daily_avg, label="Daily Avg", color="green")
    plt.title(title)
    plt.xlabel("Solar Longitude Ls (°)")
    plt.ylabel("Temperature (°C)")
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, alpha=0.3)
    plt.legend()

    filepath = os.path.join("outputs", filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath)
    print(f"  Seasonal plot saved to {filepath}")
    plt.close(fig)
