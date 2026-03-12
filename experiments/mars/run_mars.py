#!/usr/bin/env python
"""Run Mars terraforming experiments.

Each experiment can be run independently or combined via --run flags.

Usage (from project root):
    python experiments/run_mars.py                        # run all
    python experiments/run_mars.py --run sol              # one sol only
    python experiments/run_mars.py --run year             # one year only
    python experiments/run_mars.py --run coords           # three coordinates only
    python experiments/run_mars.py --run sol year         # sol + year
    python experiments/run_mars.py --run year --no-plot   # year, skip plots
    python experiments/run_mars.py --run year --no-save   # year, skip CSV

Individual experiment scripts can also be run directly:
    python experiments/one_sol.py
    python experiments/one_year.py
    python experiments/three_coords.py
"""
from __future__ import annotations

import argparse
import os

from src.engine import Accuracy

from experiments.mars.one_sol import run_one_sol
from experiments.mars.one_year import run_one_year
from experiments.mars.three_coords import run_three_coordinates
from experiments.utils import save_history_to_csv, plot_history, plot_seasonal_temps


def main():
    parser = argparse.ArgumentParser(
        description="Mars terraforming experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--run",
        nargs="+",
        choices=["sol", "year", "coords"],
        default=["sol", "year", "coords"],
        metavar="EXPERIMENT",
        help="Experiments to run: sol, year, coords (default: all)",
    )
    parser.add_argument(
        "--accuracy", choices=["fast", "accurate"], default=None,
        help="Override accuracy mode for all experiments",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSVs")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    print("=" * 60)
    print("  Terraforming Mars — Evolution Experiments")
    print(f"  Running: {', '.join(args.run)}")
    print("=" * 60)

    os.makedirs("outputs/sol",      exist_ok=True)
    os.makedirs("outputs/668_sols", exist_ok=True)

    if "sol" in args.run:
        accuracy = (Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE) \
                   if args.accuracy else Accuracy.ACCURATE
        # Warm-up pass in FAST mode (quick sanity check, no output saved)
        run_one_sol(Accuracy.FAST, dt=3600.0)
        history_sol = run_one_sol(accuracy, dt=3600.0)
        if not args.no_save:
            save_history_to_csv(history_sol, "sol/mars_sol_evolution.csv")
        if not args.no_plot:
            plot_history(history_sol, "sol/mars_sol_evolution.png", "Mars Evolution (1 Sol)")

    if "year" in args.run:
        accuracy = (Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE) \
                   if args.accuracy else Accuracy.ACCURATE
        history_year = run_one_year(accuracy, dt=3600.0)
        if not args.no_save:
            save_history_to_csv(history_year, "668_sols/mars_year_evolution.csv")
        if not args.no_plot:
            plot_history(history_year, "668_sols/mars_year_evolution.png", "Mars Evolution (1 Year)")
            plot_seasonal_temps(history_year, "668_sols/mars_seasonal_temps.png",
                                "Mars Temps: ~90°C Swing Summer to Winter")

    if "coords" in args.run:
        accuracy = (Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE) \
                   if args.accuracy else Accuracy.FAST
        run_three_coordinates(
            accuracy=accuracy,
            dt=600.0,
            save=not args.no_save,
            plot=not args.no_plot,
        )

    print(f"\n{'=' * 60}")
    print("  Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
