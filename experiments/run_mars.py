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


def _v(t) -> float:
    """tf.Tensor → Python float."""
    return float(t.numpy())


def plot_history(history, filename: str, title: str):
    """Plot temperature, pressure, and ice mass evolution."""
    # Convert time to hours for readable X-axis
    times = [_v(s.time) / 3600.0 for s in history]
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
    axes[2].set_xlabel("Time (hours)")
    axes[2].set_title(f"{title} - Ice Mass")
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"  📈 Plot saved to {filename}")
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

    # ── One Sol ───────────────────────────────────────────────
    run_one_sol(Accuracy.FAST)
    history_sol = run_one_sol(Accuracy.ACCURATE)
    plot_history(history_sol, "mars_sol_evolution.png", "Mars Evolution (1 Sol)")

    # ── One Year (fast only — accurate takes ~10 min) ─────────
    history_year = run_one_year(Accuracy.FAST, dt=3600.0)
    plot_history(history_year, "mars_year_evolution.png", "Mars Evolution (1 Year)")

    print(f"\n{'=' * 60}")
    print("  ✅  Done.")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
