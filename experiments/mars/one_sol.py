"""Experiment: evolve Mars for one sol.

Usage (from project root):
    python experiments/one_sol.py
    python experiments/one_sol.py --accuracy accurate --dt 900
"""
from __future__ import annotations

import argparse
import os

from src.celestials import Mars, MARS_ROTATION_PERIOD
from src.engine import Accuracy, TimeController

from experiments.utils import _v, save_history_to_csv, plot_history


def run_one_sol(accuracy: Accuracy = Accuracy.FAST, dt: float = 3600.0):
    """Evolve Mars for one sol and print summary statistics."""
    mars = Mars()
    tc = TimeController(mars, dt=dt, accuracy=accuracy)

    sol_seconds = _v(MARS_ROTATION_PERIOD)
    print(f"\n{'─' * 60}")
    print(f"  One Sol  ({sol_seconds / 3600:.1f} h)  —  {accuracy.value} mode, dt={dt:.0f} s")
    print(f"{'─' * 60}")

    history = tc.run(duration=MARS_ROTATION_PERIOD)

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
    parser = argparse.ArgumentParser(description="One-sol Mars evolution experiment")
    parser.add_argument("--accuracy", choices=["fast", "accurate"], default="accurate",
                        help="Integration accuracy mode (default: accurate)")
    parser.add_argument("--dt", type=float, default=3600.0,
                        help="Timestep in seconds (default: 3600)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSV")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    accuracy = Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE
    os.makedirs("outputs/sol", exist_ok=True)

    history = run_one_sol(accuracy=accuracy, dt=args.dt)

    if not args.no_save:
        save_history_to_csv(history, "sol/mars_sol_evolution.csv")
    if not args.no_plot:
        plot_history(history, "sol/mars_sol_evolution.png", "Mars Evolution (1 Sol)")


if __name__ == "__main__":
    main()
