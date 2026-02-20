#!/usr/bin/env python
"""Run a basic Mars evolution — one sol and one year.

This script imports terraforming as a dependency and demonstrates
both the FAST (reduced-order) and ACCURATE (RK4) integration modes.

Usage (from project root):
    python experiments/run_mars.py
"""

from __future__ import annotations

from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
from src.engine import Accuracy, TimeController
import matplotlib.pyplot as plt
import os
import csv


def _v(t) -> float:
    """torch.Tensor → Python float."""
    return float(t.item())


def save_history_to_csv(history, filename: str):
    """Write integration steps to a CSV file."""
    filepath = os.path.join("outputs", filename)
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
    print(f"  💾 Data saved to {filepath}")


def plot_history(history, filename: str, title: str):
    """Plot temperature, pressure, and ice mass evolution."""
    max_time_s = max([_v(s.time) for s in history])
    use_sols = max_time_s > 3 * _v(MARS_ROTATION_PERIOD)
    
    if use_sols:
        times = [_v(s.time) / _v(MARS_ROTATION_PERIOD) for s in history]
        xlabel = "Time (Sols)"
    else:
        times = [_v(s.time) / 3600.0 for s in history]
        xlabel = "Time (hours)"

    temps = [_v(s.surface_temperature) for s in history]
    pressures = [_v(s.surface_pressure) for s in history]
    ice_masses = [_v(s.ice_mass) for s in history]

    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)

    # Temperature
    axes[0].plot(times, temps, label="Temperature", color="tab:red")
    axes[0].set_ylabel("Temperature (K)")
    axes[0].set_title(f"{title} - Surface Temperature")
    axes[0].grid(True, alpha=0.3)

    # Pressure
    axes[1].plot(times, pressures, label="Pressure", color="tab:blue")
    axes[1].set_ylabel("Pressure (Pa)")
    axes[1].set_title(f"{title} - Surface Pressure")
    axes[1].grid(True, alpha=0.3)

    # Ice Mass
    axes[2].plot(times, ice_masses, label="Ice Mass", color="tab:cyan")
    axes[2].set_ylabel("Ice Mass (kg)")
    axes[2].set_xlabel(xlabel)
    axes[2].set_title(f"{title} - Ice Mass")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    filepath = os.path.join("outputs", filename)
    plt.savefig(filepath)
    print(f"  📈 Plot saved to {filepath}")
    plt.close(fig)


def run_one_sol(accuracy: Accuracy = Accuracy.FAST, dt: float = 900.0):
    """Evolve Mars for one sol and print summary statistics."""
    mars = Mars()
    tc = TimeController(mars, dt=dt, accuracy=accuracy)

    sol_seconds = _v(MARS_ROTATION_PERIOD)
    print(f"\n{'─' * 60}")
    print(f"  One Sol  ({sol_seconds / 3600:.1f} h)  —  {accuracy.value} mode, dt={dt:.0f} s")
    print(f"{'─' * 60}")

    history = tc.run(duration=MARS_ROTATION_PERIOD)

    temps = [_v(s.surface_temperature) for s in history]
    press = [_v(s.surface_pressure) for s in history]
    fluxes = [_v(s.solar_flux) for s in history]

    print(f"  Steps recorded    : {len(history)}")
    print(f"  Temperature (K)   : {min(temps):.2f}  →  {max(temps):.2f}   "
          f"(ΔT = {max(temps) - min(temps):.2f})")
    print(f"  Pressure    (Pa)  : {min(press):.2f}  →  {max(press):.2f}   "
          f"(ΔP = {max(press) - min(press):.4f})")
    print(f"  Solar flux  (W/m²): {min(fluxes):.2f}  →  {max(fluxes):.2f}")
    print(f"  Final ice mass    : {_v(history[-1].ice_mass):.4e} kg")
    print(f"  Orbital angle     : {_v(history[-1].orbital_angle):.6f} rad")
    return history


def run_one_year(accuracy: Accuracy = Accuracy.FAST, dt: float = 3600.0):
    """Evolve Mars for one Martian year and print summary statistics."""
    mars = Mars()
    tc = TimeController(mars, dt=dt, accuracy=accuracy)

    year_seconds = _v(MARS_ORBITAL_PERIOD)
    n_sols = year_seconds / _v(MARS_ROTATION_PERIOD)
    print(f"\n{'─' * 60}")
    print(f"  One Year  ({n_sols:.1f} sols)  —  {accuracy.value} mode, dt={dt:.0f} s")
    print(f"{'─' * 60}")

    step_count = [0]

    def progress(state, t):
        step_count[0] += 1
        if step_count[0] % 5000 == 0:
            pct = _v(t) / year_seconds * 100
            print(f"    ... {pct:5.1f}%  T={_v(state.surface_temperature):.2f} K  "
                  f"P={_v(state.surface_pressure):.2f} Pa")

    history = tc.run(duration=MARS_ORBITAL_PERIOD, callback=progress)

    temps = [_v(s.surface_temperature) for s in history]
    press = [_v(s.surface_pressure) for s in history]
    fluxes = [_v(s.solar_flux) for s in history]

    print(f"  Steps recorded    : {len(history)}")
    print(f"  Temperature (K)   : {min(temps):.2f}  →  {max(temps):.2f}   "
          f"(ΔT = {max(temps) - min(temps):.2f})")
    print(f"  Pressure    (Pa)  : {min(press):.2f}  →  {max(press):.2f}   "
          f"(ΔP = {max(press) - min(press):.4f})")
    print(f"  Solar flux  (W/m²): {min(fluxes):.2f}  →  {max(fluxes):.2f}")
    print(f"  Final ice mass    : {_v(history[-1].ice_mass):.4e} kg")
    print(f"  Orbital angle     : {_v(history[-1].orbital_angle):.6f} rad")
    return history


def main():
    print("=" * 60)
    print("  🔴  Terraforming Mars — Basic Evolution Experiment")
    print("=" * 60)

    os.makedirs("outputs", exist_ok=True)

    # ── One Sol ───────────────────────────────────────────────
    run_one_sol(Accuracy.FAST, dt=3600.0)
    history_sol = run_one_sol(Accuracy.ACCURATE, dt=3600.0) # 1 hour dt
    save_history_to_csv(history_sol, "mars_sol_evolution.csv")
    plot_history(history_sol, "mars_sol_evolution.png", "Mars Evolution (1 Sol)")

    # ── One Year (fast only — accurate takes ~10 min) ─────────
    history_year = run_one_year(Accuracy.FAST, dt=3600.0) # 1 hour dt
    save_history_to_csv(history_year, "mars_year_evolution.csv")
    plot_history(history_year, "mars_year_evolution.png", "Mars Evolution (1 Year)")

    print(f"\n{'=' * 60}")
    print("  ✅  Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
