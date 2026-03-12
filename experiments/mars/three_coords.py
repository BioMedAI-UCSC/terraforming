"""Experiment: seasonal temperature evolution at three latitudes.

Runs spin-up + one full year at North (45°N), Equator (0°), and
Southern Hemisphere (-40°N), all at 137°E longitude.

Usage (from project root):
    python experiments/three_coords.py
    python experiments/three_coords.py --accuracy accurate --dt 900
    python experiments/three_coords.py --no-plot
"""
from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
from src.engine import Accuracy, TimeController

from experiments.utils import _v, save_history_to_csv, plot_history, plot_seasonal_temps

POINTS = [
    {"name": "North",               "lat":  45.0, "lon": 137.0},
    {"name": "Equator",             "lat":   0.0, "lon": 137.0},
    {"name": "Southern Hemisphere", "lat": -40.0, "lon": 137.0},
]


def plot_three_temperatures(histories: dict, filename: str, title: str):
    """Plot daily-average seasonal temperature for three coordinates."""
    fig = plt.figure(figsize=(10, 6))
    colors = {"North": "tab:blue", "Equator": "tab:red", "Southern Hemisphere": "tab:green"}

    spin_up_seconds = 10 * _v(MARS_ROTATION_PERIOD)

    for name, history in histories.items():
        history_filtered = [s for s in history if _v(s.time) > spin_up_seconds]
        if not history_filtered:
            continue

        times = np.array([_v(s.time) for s in history_filtered])
        temps = np.array([_v(s.surface_temperature) for s in history_filtered]) - 273.15

        ls_rad = np.array([_v(s.orbital_angle) for s in history_filtered]) + 251.0 * np.pi / 180.0
        ls = (ls_rad * 180 / np.pi) % 360.0

        sols = times // _v(MARS_ROTATION_PERIOD)
        unique_sols = np.unique(sols)

        daily_ls, daily_avg = [], []

        for sol in unique_sols:
            mask = (sols == sol)
            if np.sum(mask) < 10:
                continue
            t_day = temps[mask]
            ls_day_array = ls[mask]
            if np.max(ls_day_array) > 350 and np.min(ls_day_array) < 10:
                ls_day_array = np.where(ls_day_array < 180, ls_day_array + 360, ls_day_array)
                ls_day = np.mean(ls_day_array) % 360.0
            else:
                ls_day = np.mean(ls_day_array)
            daily_avg.append(np.mean(t_day))
            daily_ls.append(ls_day)

        idx = np.argsort(daily_ls)
        daily_ls  = np.array(daily_ls)[idx]
        daily_avg = np.array(daily_avg)[idx]

        plt.plot(daily_ls, daily_avg, label=f"{name} Avg",
                 color=colors.get(name, "black"), linewidth=2)

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
    print(f"  Three-coordinate plot saved to {filepath}")
    plt.close(fig)


def run_three_coordinates(
    accuracy: Accuracy = Accuracy.ACCURATE,
    dt: float = 600.0,
    save: bool = True,
    plot: bool = True,
):
    """Run simulations for three latitudes with spin-up and seasonal sampling."""
    spin_up = 10 * _v(MARS_ROTATION_PERIOD)
    n_complete_sols = int(_v(MARS_ORBITAL_PERIOD) / _v(MARS_ROTATION_PERIOD))
    total_duration = spin_up + n_complete_sols * _v(MARS_ROTATION_PERIOD)

    histories = {}
    for pt in POINTS:
        print(f"\n{'─' * 60}")
        print(f"  Running {pt['name']} ({pt['lat']}°N, {pt['lon']}°E)  —  {accuracy.value} (dt={dt}s)")
        print(f"{'─' * 60}")

        mars = Mars(latitude=pt["lat"], longitude=pt["lon"])
        tc = TimeController(mars, dt=dt, accuracy=accuracy)

        step_count = [0]

        def progress(state, t, _dur=total_duration):
            step_count[0] += 1
            if step_count[0] % 10000 == 0:
                pct = _v(t) / _dur * 100
                print(f"    ... {pct:5.1f}%  T={_v(state.thermal.surface_temperature):.2f} K")

        history = tc.run(duration=total_duration, callback=progress)
        histories[pt["name"]] = history

        safe_name = pt["name"].lower().replace(" ", "_")
        if save:
            save_history_to_csv(history, f"668_sols/mars_{safe_name}_evolution.csv")
        if plot:
            plot_history(history, f"668_sols/mars_{safe_name}_evolution.png",
                         f"Mars Evolution: {pt['name']}")
            plot_seasonal_temps(history, f"668_sols/mars_{safe_name}_seasonal.png",
                                f"Mars Seasonal: {pt['name']}", spin_up_sols=10)

    if plot:
        plot_three_temperatures(
            histories,
            "668_sols/mars_three_coords_temps.png",
            "Temperature across Seasons at Three Coordinates (137°E)",
        )

    return histories


def main():
    parser = argparse.ArgumentParser(description="Three-coordinate seasonal Mars experiment")
    parser.add_argument("--accuracy", choices=["fast", "accurate"],
                        help="Integration accuracy mode")
    parser.add_argument("--dt", type=float, default=600.0,
                        help="Timestep in seconds (default: 600)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSVs")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    accuracy = Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE
    os.makedirs("outputs/668_sols", exist_ok=True)

    run_three_coordinates(
        accuracy=accuracy,
        dt=args.dt,
        save=not args.no_save,
        plot=not args.no_plot,
    )


if __name__ == "__main__":
    main()
