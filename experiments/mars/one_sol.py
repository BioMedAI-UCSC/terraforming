"""Experiment: evolve Mars for N sols.

Usage (from project root):
    python experiments/mars/one_sol.py
    python experiments/mars/one_sol.py --sols 5 --accuracy accurate --dt 900
    python experiments/mars/one_sol.py --sols 3 --multi-coord
    python experiments/mars/one_sol.py --sols 2 --lat 60 --lon 0
    python experiments/mars/one_sol.py --no-plot
    python experiments/mars/one_sol.py --ground
"""
from __future__ import annotations

import argparse
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from src.celestials import Mars, MARS_ROTATION_PERIOD
from src.engine import Accuracy, TimeController

from experiments.utils import _v, save_history_to_csv, plot_history
from experiments.rems_loader import load_modrdr


MULTI_POINTS = [
    {"name": "North",               "lat":  45.0, "lon": 137.0},
    {"name": "Equator",             "lat":   0.0, "lon": 137.0},
    {"name": "Southern Hemisphere", "lat": -40.0, "lon": 137.0},
]


def run_sols(
    accuracy: Accuracy = Accuracy.FAST,
    dt: float = 300.0,
    n_sols: float = 1.0,
    lat: float = 22.0,
    lon: float = 137.0,
    elevation_m: float = 0.0,
    initial_ls_deg: float = 251.0,
    ice_mass: float = 5.0e15,
) -> list:
    """Evolve Mars for n_sols and print summary statistics."""
    mars = Mars(latitude=lat, longitude=lon, elevation_m=elevation_m,
                initial_ls_deg=initial_ls_deg, ice_mass=ice_mass)
    tc = TimeController(mars, dt=dt, accuracy=accuracy)

    duration = n_sols * _v(MARS_ROTATION_PERIOD)
    sol_label = f"{n_sols:.0f} sol{'s' if n_sols != 1 else ''}"
    print(f"\n{'─' * 60}")
    print(f"  {sol_label}  ({lat}°N, {lon}°E)  —  {accuracy.value} mode, dt={dt:.0f} s")
    print(f"{'─' * 60}")

    history = tc.run(duration=duration)

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


def plot_multi_coord_sols(histories: dict, filename: str, title: str,
                          out_dir: str, ground: dict = None):
    """Plot temperature vs time (hours) for multiple coordinates on a single figure."""
    from experiments.utils import REMS_NOTE
    sol_seconds = _v(MARS_ROTATION_PERIOD)
    sol_hours   = sol_seconds / 3600.0
    colors = {"North": "#1f77b4", "Equator": "#d62728", "Southern Hemisphere": "#2ca02c"}

    fig, ax = plt.subplots(figsize=(12, 6))

    coord_info = {pt["name"]: pt for pt in MULTI_POINTS}
    n_steps = 0
    for name, history in histories.items():
        times_h = np.array([_v(s.time) / 3600.0 for s in history])
        temps   = np.array([_v(s.surface_temperature) for s in history])
        n_steps = max(n_steps, len(times_h))
        lw = max(1.0, 2.0 - 1.0 * min(len(times_h) / 2000, 1.0))
        pt = coord_info.get(name, {})
        lat_v = pt.get("lat", 0.0)
        lon_v = pt.get("lon", 0.0)
        ns = "N" if lat_v >= 0 else "S"
        legend_label = f"{name}  ({abs(lat_v):.0f}°{ns}, {lon_v:.0f}°E)"
        ax.plot(times_h, temps, label=legend_label, color=colors.get(name, "#333333"), linewidth=lw)

    # Ground-truth overlay: tile REMS air-temp for each sol
    if ground is not None:
        n_sols_gt = int(np.ceil(times_h[-1] / sol_hours))
        gt_lw = max(0.8, 1.2)
        mask = ~np.isnan(ground["air_temp"])
        if mask.any():
            for sol_i in range(n_sols_gt):
                x_tile = ground["lmst_hours"][mask] + sol_i * sol_hours
                y_tile = ground["air_temp"][mask]
                label = "REMS Air Temp (Sol 224)" if sol_i == 0 else "_nolegend_"
                ax.plot(x_tile, y_tile, label=label, color="#ff7f0e",
                        linewidth=gt_lw, linestyle="-.", alpha=0.85, zorder=2)
        ax.text(0.01, 0.02, REMS_NOTE, transform=ax.transAxes,
                fontsize=8, color="#555555", va="bottom")

    # Draw vertical sol boundaries
    n_sols_total = times_h[-1] / sol_hours
    for sol_i in range(1, int(n_sols_total) + 1):
        ax.axvline(sol_i * sol_hours, color="#555555", linewidth=0.8,
                   linestyle="--", alpha=0.6, label="_nolegend_")

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()

    filepath = os.path.join("outputs", out_dir, filename)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    print(f"  Multi-coord plot saved to {filepath}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="N-sol Mars evolution experiment")
    parser.add_argument("--sols", type=float, default=1.0,
                        help="Number of sols to simulate (default: 1)")
    parser.add_argument("--accuracy", choices=["fast", "accurate"], default="accurate",
                        help="Integration accuracy mode (default: accurate)")
    parser.add_argument("--dt", type=float, default=300.0,
                        help="Timestep in seconds (default: 300)")
    parser.add_argument("--lat", type=float, default=22.0,
                        help="Latitude in degrees N (default: 22.0; ignored with --multi-coord)")
    parser.add_argument("--lon", type=float, default=137.0,
                        help="Longitude in degrees E (default: 137.0; ignored with --multi-coord)")
    parser.add_argument("--elevation", type=float, default=0.0,
                        help="Surface elevation in metres relative to Mars datum (negative = below datum; e.g. Gale Crater = -4500)")
    parser.add_argument("--ls", type=float, default=251.0,
                        help="Initial Solar Longitude in degrees (default: 251 = perihelion; REMS Sol 224 = Ls 287)")
    parser.add_argument("--ice-mass", type=float, default=5.0e15,
                        help="Initial total CO₂ ice mass in kg (default: 5e15; increase to slow seasonal pressure changes)")
    parser.add_argument("--multi-coord", action="store_true",
                        help="Run at North (45°N), Equator (0°), and S. Hemisphere (-40°N) at 137°E")
    parser.add_argument("--name", default=None, metavar="NAME",
                        help="Custom experiment name (default: UTC timestamp)")
    parser.add_argument("--no-save", action="store_true", help="Skip saving CSVs")
    parser.add_argument("--no-plot", action="store_true", help="Skip saving plots")
    parser.add_argument("--ground", action="store_true",
                        help="Overlay REMS Curiosity ground-truth data (Sol 224, Gale Crater)")
    args = parser.parse_args()

    accuracy = Accuracy.FAST if args.accuracy == "fast" else Accuracy.ACCURATE
    tag = args.name if args.name else datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join("sol", f"exp_{tag}")
    sol_label = f"{args.sols:.0f}sol"

    ground = load_modrdr() if args.ground else None

    if args.multi_coord:
        histories = {}
        for pt in MULTI_POINTS:
            history = run_sols(
                accuracy=accuracy, dt=args.dt, n_sols=args.sols,
                lat=pt["lat"], lon=pt["lon"],
            )
            histories[pt["name"]] = history
            safe_name = pt["name"].lower().replace(" ", "_")
            if not args.no_save:
                save_history_to_csv(history, f"{out_dir}/mars_{safe_name}_{sol_label}.csv")
            if not args.no_plot:
                sol_str = f"{args.sols:.0f} Sol{'s' if args.sols != 1 else ''}"
                lat_s = f"{pt['lat']:+.1f}°" if pt['lat'] >= 0 else f"{pt['lat']:.1f}°"
                plot_history(history, f"{out_dir}/mars_{safe_name}_{sol_label}.png",
                             f"Mars {pt['name']} — {sol_str}  ({lat_s}N, {pt['lon']:.1f}°E)",
                             ground=ground)

        if not args.no_plot:
            sol_str = f"{args.sols:.0f} Sol{'s' if args.sols != 1 else ''}"
            plot_multi_coord_sols(
                histories,
                f"mars_multi_coord_{sol_label}.png",
                f"Surface Temperature — {sol_str}  (137°E)",
                out_dir=out_dir,
                ground=ground,
            )
    else:
        history = run_sols(
            accuracy=accuracy, dt=args.dt, n_sols=args.sols,
            lat=args.lat, lon=args.lon, elevation_m=args.elevation,
            initial_ls_deg=args.ls, ice_mass=args.ice_mass,
        )
        if not args.no_save:
            save_history_to_csv(history, f"{out_dir}/mars_sol_evolution.csv")
        if not args.no_plot:
            sol_str = f"{args.sols:.0f} Sol{'s' if args.sols != 1 else ''}"
            ns = "N" if args.lat >= 0 else "S"
            plot_history(history, f"{out_dir}/mars_sol_evolution.png",
                         f"Mars Evolution — {sol_str}  ({abs(args.lat):.1f}°{ns}, {args.lon:.1f}°E)",
                         ground=ground)


if __name__ == "__main__":
    main()
