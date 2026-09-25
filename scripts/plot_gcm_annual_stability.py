#!/usr/bin/env python3
"""Plot annual GCM stability, conservation, seasonal cycle, and throughput."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) < 2:
        raise ValueError("diagnostics must contain initialization and later checkpoints")
    return {name: np.asarray([float(row[name]) for row in rows]) for name in rows[0]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path,
                        help="output stem; defaults to RUN_DIR/annual_stability")
    args = parser.parse_args()
    run = args.run_dir
    diagnostics = run / "convection" / "checkpoint_diagnostics.csv"
    manifest = json.loads((run / "manifest.json").read_text())
    data = read_csv(diagnostics)
    if not all(np.isfinite(values).all() for values in data.values()):
        raise ValueError("checkpoint diagnostics contain nonfinite values")

    elapsed = data["elapsed_sols"]
    ls = data["solar_longitude_deg"]
    pressure = data["mean_surface_pressure_pa"]
    ice = data["mean_co2_ice_pa"]
    mass = data["atmosphere_plus_surface_reservoir_mass_kg"]
    drift = np.abs(mass / mass[0] - 1.0)
    timing = manifest["runs"]["convection"]
    wall_minutes = timing["invocation_elapsed_seconds"] / 60.0
    years_per_day = timing["simulated_sols_per_wall_hour"] * 24.0 / manifest["sols"]

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.titlesize": 14})
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(ls, pressure, color="#2962a3", lw=2, label="Atmospheric pressure")
    ax.plot(ls, ice, color="#d36b32", lw=2, label="Surface CO₂ frost")
    ax.plot(ls, pressure + ice, color="black", lw=1.25, ls="--", label="Pressure + frost")
    ax.set(title="Seasonal CO₂ exchange", xlabel="Solar longitude Ls (°)", ylabel="Global mean (Pa)", xlim=(0, 360))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)

    ax = axes[0, 1]
    ax.fill_between(ls, data["min_surface_temperature_k"], data["max_surface_temperature_k"],
                    color="#df7b33", alpha=.22, label="Surface range")
    ax.plot(ls, data["mean_surface_temperature_k"], color="#c44e20", lw=2,
            label="Mean surface")
    ax.fill_between(ls, data["min_air_temperature_k"], data["max_air_temperature_k"],
                    color="#4777a8", alpha=.18, label="Full column air range")
    ax.set(title="Finite temperature envelope", xlabel="Solar longitude Ls (°)", ylabel="Temperature (K)", xlim=(0, 360))
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2)

    ax = axes[1, 0]
    floor = np.finfo(float).eps
    ax.semilogy(elapsed, np.maximum(drift, floor), color="#238b6b", lw=2)
    ax.axhline(1e-5, color="#b43c39", ls="--", lw=1.2, label="Numerical gate: $10^{-5}$")
    ax.set(title="Atmosphere + surface CO₂ conservation", xlabel="Elapsed sols",
           ylabel="Absolute relative drift")
    ax.legend(frameon=False, fontsize=8)
    ax.grid(alpha=.2, which="both")

    ax = axes[1, 1]
    ax.plot(ls, data["max_wind_speed_ms"], color="#7651a8", lw=1.8, label="Maximum full column wind")
    ax2 = ax.twinx()
    ax2.plot(ls, data["mean_deep_soil_temperature_k"] - data["mean_deep_soil_temperature_k"][0],
             color="#8a6d1d", lw=1.8, label="Deep soil change")
    ax.set(title="Bounded dynamics and slow reservoir", xlabel="Solar longitude Ls (°)",
           ylabel="Maximum wind (m s⁻¹)", xlim=(0, 360))
    ax2.set_ylabel("Mean deep soil ΔT (K)")
    lines = ax.lines + ax2.lines
    ax.legend(lines, [line.get_label() for line in lines], frameon=False, fontsize=8, loc="upper left")
    ax.grid(alpha=.2)

    max_drift = float(np.max(drift))
    fig.suptitle(
        "AEGIS: one Mars year of stable differentiable simulation\n"
        f"T21/L12 · dt=450 s · float64 · 131,900 steps · {wall_minutes:.1f} min on one GPU · "
        f"{years_per_day:.2f} Mars years/day · max CO₂ drift {max_drift:.2e}"
    )
    stem = args.output or (run / "annual_stability")
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    summary = {
        "status": "complete_finite_and_conservative_not_equilibrium_validation",
        "elapsed_sols": float(elapsed[-1]), "final_ls_deg": float(ls[-1]),
        "steps": int(data["step"][-1]), "wall_minutes": wall_minutes,
        "simulated_sols_per_wall_hour": timing["simulated_sols_per_wall_hour"],
        "simulated_mars_years_per_wall_day": years_per_day,
        "maximum_relative_co2_reservoir_drift": max_drift,
        "ames_speedup": None,
        "ames_speedup_reason": "No matched Ames wall time and hardware measurement is present in the archive.",
    }
    stem.with_name(stem.name + "_summary").with_suffix(".json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    # Contextual public figure from the official NASA Ames Legacy GCM tutorial.
    # It is deliberately labeled indicative because the tutorial omits hardware.
    ames_hours = manifest["sols"] / 20.0
    timing_rows = [
        {"model": "AEGIS", "mars_year_hours": wall_minutes / 60.0,
         "status": "measured", "configuration": "T21 64x32 L12 dt450 float64, one GPU"},
        {"model": "NASA Ames Legacy GCM tutorial", "mars_year_hours": ames_hours,
         "status": "extrapolated from approximately 20 sols/hour",
         "configuration": "60x36 L24 dt120, tutorial hardware unspecified"},
    ]
    comparison_csv = stem.with_name("ames_timing_comparison").with_suffix(".csv")
    with comparison_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(timing_rows[0]))
        writer.writeheader(); writer.writerows(timing_rows)
    comparison_stem = stem.with_name("ames_timing_comparison")
    fig, ax = plt.subplots(figsize=(7.5, 4.2), constrained_layout=True)
    bars = ax.bar([row["model"] for row in timing_rows], [row["mars_year_hours"] for row in timing_rows],
                  color=["#2962a3", "#999999"])
    ax.bar_label(bars, labels=[f"{row['mars_year_hours']:.2f} h" for row in timing_rows], padding=4)
    ax.set(ylabel="Wall hours per 668.6-sol Mars year",
           title=f"Annual throughput context: AEGIS is {ames_hours/(wall_minutes/60):.1f}× faster")
    ax.text(.5, -.19,
            "Indicative only: Ames hardware is unspecified and configurations differ in layers, timestep, physics, and output.",
            transform=ax.transAxes, ha="center", fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(comparison_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    fig.savefig(comparison_stem.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(json.dumps({"png": str(stem.with_suffix('.png')), "pdf": str(stem.with_suffix('.pdf'))}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
