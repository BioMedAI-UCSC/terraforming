#!/usr/bin/env python
"""Run Mars terraforming experiments.

Each experiment can be run independently or combined via --run flags.
Outputs are saved under a timestamped (or named) folder so successive
runs never overwrite each other.  All outputs/ are git-ignored by default;
manually commit only validated experiments.

Usage (from project root):
    python experiments/mars/run_mars.py                             # run all, timestamped
    python experiments/mars/run_mars.py --name baseline            # named folder
    python experiments/mars/run_mars.py --run sol                  # one sol only
    python experiments/mars/run_mars.py --run year                 # one year only
    python experiments/mars/run_mars.py --run coords               # three coordinates only
    python experiments/mars/run_mars.py --run sol year             # sol + year
    python experiments/mars/run_mars.py --run year --no-plot       # year, skip plots
    python experiments/mars/run_mars.py --run year --no-save       # year, skip CSV

Output layout:
    outputs/sol/exp_<tag>/mars_sol_evolution.{csv,png}
    outputs/668_sols/exp_<tag>/mars_year_evolution.{csv,png}
    outputs/668_sols/exp_<tag>/mars_seasonal_temps.png
    outputs/coords/exp_<tag>/...

where <tag> is either --name value or a UTC timestamp (YYYYMMDD_HHMMSS).
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

from src.engine import Accuracy

from experiments.mars.one_sol import run_one_sol
from experiments.mars.one_year import run_one_year
from experiments.mars.three_coords import run_three_coordinates
from experiments.utils import save_history_to_csv, plot_history, plot_seasonal_temps


def _exp_tag(name: str | None) -> str:
    """Return experiment folder tag: custom name or UTC timestamp."""
    if name:
        return name
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _out(base: str, tag: str, filename: str) -> str:
    """Return path relative to outputs/ — utils functions prepend outputs/ automatically."""
    return os.path.join(base, f"exp_{tag}", filename)


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
        "--name",
        default=None,
        metavar="NAME",
        help="Custom experiment name (used as folder suffix instead of timestamp)",
    )
    parser.add_argument(
        "--accuracy", choices=["fast", "accurate"], default=None,
        help="Override accuracy mode for all experiments",
    )
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSVs")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    args = parser.parse_args()

    tag = _exp_tag(args.name)

    print("=" * 60)
    print("  Terraforming Mars — Evolution Experiments")
    print(f"  Running:  {', '.join(args.run)}")
    print(f"  Exp tag:  exp_{tag}")
    print("=" * 60)

    if "sol" in args.run:
        accuracy = (Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE) \
                   if args.accuracy else Accuracy.ACCURATE
        # Warm-up pass in FAST mode (quick sanity check, no output saved)
        run_one_sol(Accuracy.FAST, dt=3600.0)
        history_sol = run_one_sol(accuracy, dt=3600.0)
        if not args.no_save:
            save_history_to_csv(history_sol, _out("sol", tag, "mars_sol_evolution.csv"))
        if not args.no_plot:
            plot_history(history_sol, _out("sol", tag, "mars_sol_evolution.png"), "Mars Evolution (1 Sol)")

    if "year" in args.run:
        accuracy = (Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE) \
                   if args.accuracy else Accuracy.ACCURATE
        history_year = run_one_year(accuracy, dt=3600.0)
        if not args.no_save:
            save_history_to_csv(history_year, _out("668_sols", tag, "mars_year_evolution.csv"))
        if not args.no_plot:
            plot_history(history_year, _out("668_sols", tag, "mars_year_evolution.png"), "Mars Evolution (1 Year)")
            plot_seasonal_temps(history_year, _out("668_sols", tag, "mars_seasonal_temps.png"),
                                "Mars Temps: ~90°C Swing Summer to Winter")

    if "coords" in args.run:
        accuracy = (Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE) \
                   if args.accuracy else Accuracy.FAST
        run_three_coordinates(
            accuracy=accuracy,
            dt=600.0,
            save=not args.no_save,
            plot=not args.no_plot,
            out_dir=os.path.join("coords", f"exp_{tag}"),
        )

    print(f"\n{'=' * 60}")
    print(f"  Done.  Outputs in outputs/*/exp_{tag}/")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
