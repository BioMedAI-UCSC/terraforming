"""Experiment: evolve Mars for one Martian year (~668 sols).

Usage (from project root):
    python experiments/one_year.py
    python experiments/one_year.py --accuracy fast --dt 1800
    python experiments/one_year.py --no-plot
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
from src.engine import Accuracy, TimeController

from experiments.utils import _v, save_history_to_csv, plot_history, plot_seasonal_temps


def run_one_year(accuracy: Accuracy = Accuracy.FAST, dt: float = 3600.0):
    """Evolve Mars for one Martian year and print summary statistics."""
    mars = Mars()
    tc = TimeController(mars, dt=dt, accuracy=accuracy)

    # Round to a whole number of sols so there is no partial last day.
    n_complete_sols = int(_v(MARS_ORBITAL_PERIOD) / _v(MARS_ROTATION_PERIOD))
    year_seconds = n_complete_sols * _v(MARS_ROTATION_PERIOD)
    print(f"\n{'─' * 60}")
    print(f"  One Year  ({n_complete_sols} sols)  —  {accuracy.value} mode, dt={dt:.0f} s")
    print(f"{'─' * 60}")

    step_count = [0]

    def progress(state, t):
        step_count[0] += 1
        if step_count[0] % 5000 == 0:
            pct = _v(t) / year_seconds * 100
            print(f"    ... {pct:5.1f}%  T={_v(state.thermal.surface_temperature):.2f} K  "
                  f"P={_v(state.atmosphere.surface_pressure):.2f} Pa")

    history = tc.run(duration=year_seconds, callback=progress)

    temps  = [_v(s.surface_temperature) for s in history]
    press  = [_v(s.surface_pressure)    for s in history]
    fluxes = [_v(s.solar_flux)          for s in history]

    print(f"  Steps recorded    : {len(history)}")
    print(f"  Temperature (K)   : {min(temps):.2f}  →  {max(temps):.2f}   (ΔT = {max(temps) - min(temps):.2f})")
    print(f"  Pressure    (Pa)  : {min(press):.2f}  →  {max(press):.2f}   (ΔP = {max(press) - min(press):.4f})")
    print(f"  Solar flux  (W/m²): {min(fluxes):.2f}  →  {max(fluxes):.2f}")
    print(f"  Final ice mass    : {_v(history[-1].ice_mass):.4e} kg")
    print(f"  Orbital angle     : {_v(history[-1].orbital_angle):.6f} rad")
    return history


def main():
    parser = argparse.ArgumentParser(description="One-year Mars evolution experiment")
    parser.add_argument("--accuracy", choices=["fast", "accurate"], default="accurate",
                        help="Integration accuracy mode (default: accurate)")
    parser.add_argument("--dt", type=float, default=3600.0,
                        help="Timestep in seconds (default: 3600)")
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="Custom experiment name (default: UTC timestamp)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSV")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    parser.add_argument("--no-seasonal", action="store_true", help="Skip seasonal temperature plot")
    args = parser.parse_args()

    accuracy = Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE
    tag = args.name if args.name else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("668_sols", f"exp_{tag}")

    history = run_one_year(accuracy=accuracy, dt=args.dt)

    if not args.no_save:
        save_history_to_csv(history, f"{out_dir}/mars_year_evolution.csv")
    if not args.no_plot:
        plot_history(history, f"{out_dir}/mars_year_evolution.png", "Mars Evolution (1 Year)")
    if not args.no_seasonal:
        plot_seasonal_temps(history, f"{out_dir}/mars_seasonal_temps.png",
                            "Mars Temps: ~90°C Swing Summer to Winter")


if __name__ == "__main__":
    main()
