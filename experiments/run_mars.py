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
import numpy as np

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
        # Convert time into the planet's rotation angle (0 to 360 degrees for 1 Sol)
        times = [(_v(s.time) / _v(MARS_ROTATION_PERIOD)) * 360.0 for s in history]
        xlabel = "Rotation Angle (°)"

    temps = [_v(s.surface_temperature) for s in history]
    pressures = [_v(s.surface_pressure) for s in history]
    ice_masses = [_v(s.ice_mass) for s in history]

    lw = 0.15 if len(times) > 1000 else 1.5
    al = 0.7 if len(times) > 1000 else 1.0

    # Temperature
    fig_temp = plt.figure(figsize=(10, 4))
    plt.plot(times, temps, label="Temperature", color="tab:red", linewidth=lw, alpha=al)
    plt.ylabel("Temperature (K)")
    plt.xlabel(xlabel)
    plt.title(f"{title} - Surface Temperature")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    temp_file = os.path.join("outputs", filename.replace(".png", "_temp.png"))
    plt.savefig(temp_file)
    print(f"  📈 Plot saved to {temp_file}")
    plt.close(fig_temp)

    # Pressure
    fig_press = plt.figure(figsize=(10, 4))
    plt.plot(times, pressures, label="Pressure", color="tab:blue", linewidth=1.5, alpha=1.0)
    plt.ylabel("Pressure (Pa)")
    plt.xlabel(xlabel)
    plt.title(f"{title} - Surface Pressure")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    press_file = os.path.join("outputs", filename.replace(".png", "_pressure.png"))
    plt.savefig(press_file)
    print(f"  📈 Plot saved to {press_file}")
    plt.close(fig_press)

    # Ice Mass
    fig_ice = plt.figure(figsize=(10, 4))
    plt.plot(times, ice_masses, label="Ice Mass", color="tab:cyan", linewidth=1.5, alpha=1.0)
    plt.ylabel("Ice Mass (kg)")
    plt.xlabel(xlabel)
    plt.title(f"{title} - Ice Mass")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    ice_file = os.path.join("outputs", filename.replace(".png", "_ice.png"))
    plt.savefig(ice_file)
    print(f"  📈 Plot saved to {ice_file}")
    plt.close(fig_ice)


def plot_seasonal_temps(history, filename: str, title: str, spin_up_sols: int = 0):
    """Plot daily min, max, and avg temperatures over Solar Longitude (Ls).

    spin_up_sols: number of initial sols to discard as thermal spin-up.
    These early sols share the same Ls as the end of the orbit but have
    out-of-equilibrium temperatures, which would create sorting artifacts.
    """
    import numpy as np

    sol_period = _v(MARS_ROTATION_PERIOD)
    spin_up_seconds = spin_up_sols * sol_period

    times = np.array([_v(s.time) for s in history])
    temps = np.array([_v(s.surface_temperature) for s in history]) - 273.15

    ls_rad = np.array([_v(s.orbital_angle) for s in history]) + 251.0 * np.pi / 180.0
    ls = (ls_rad * 180 / np.pi) % 360.0

    sols = times // sol_period

    # Estimate expected timesteps per complete sol; skip any partial last sol
    # (occurs when the simulation duration is not a multiple of the rotation period).
    all_counts = np.array([np.sum(sols == s) for s in np.unique(sols)])
    expected_steps = int(np.median(all_counts))
    min_steps = max(10, int(0.8 * expected_steps))

    unique_sols = np.unique(sols[times > spin_up_seconds])

    daily_ls = []
    daily_max = []
    daily_min = []
    daily_avg = []

    for sol in unique_sols:
        mask = (sols == sol) & (times > spin_up_seconds)
        if np.sum(mask) < min_steps: continue
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
    daily_ls = np.array(daily_ls)[idx]
    daily_max = np.array(daily_max)[idx]
    daily_min = np.array(daily_min)[idx]
    daily_avg = np.array(daily_avg)[idx]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(daily_ls, daily_max, label='Daily Max', color='cyan')
    plt.plot(daily_ls, daily_min, label='Daily Min', color='red', linestyle='--')
    plt.plot(daily_ls, daily_avg, label='Daily Avg', color='green')

    plt.title(title)
    plt.xlabel("Solar Longitude Ls (°)")
    plt.ylabel("Temperature (°C)")
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filepath = os.path.join("outputs", filename)
    plt.savefig(filepath)
    print(f"  📈 Seasonal Plot saved to {filepath}")
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

    # Round to a whole number of sols so there is no partial last day.
    # MARS_ORBITAL_PERIOD = 668.62 sols; a fractional last sol would produce a
    # biased daily average that creates a spike in the seasonal temperature plot.
    n_complete_sols = int(_v(MARS_ORBITAL_PERIOD) / _v(MARS_ROTATION_PERIOD))
    year_seconds = n_complete_sols * _v(MARS_ROTATION_PERIOD)
    n_sols = n_complete_sols
    print(f"\n{'─' * 60}")
    print(f"  One Year  ({n_sols} sols)  —  {accuracy.value} mode, dt={dt:.0f} s")
    print(f"{'─' * 60}")

    step_count = [0]

    def progress(state, t):
        step_count[0] += 1
        if step_count[0] % 5000 == 0:
            pct = _v(t) / year_seconds * 100
            print(f"    ... {pct:5.1f}%  T={_v(state.thermal.surface_temperature):.2f} K  "
                  f"P={_v(state.atmosphere.surface_pressure):.2f} Pa")

    history = tc.run(duration=year_seconds, callback=progress)

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


def plot_three_temperatures(histories, filename: str, title: str):
    """Plot seasonal temperatures for three coordinates with spin-up removal."""
    import numpy as np
    
    fig = plt.figure(figsize=(10, 6))
    colors = {"North": "tab:blue", "Equator": "tab:red", "Southern Hemisphere": "tab:green"}
    
    # We strip the first few days of spin-up to avoid initial transients
    spin_up_sols = 10
    spin_up_seconds = spin_up_sols * _v(MARS_ROTATION_PERIOD)

    for name, history in histories.items():
        # Only take points after spin-up
        history_filtered = [s for s in history if _v(s.time) > spin_up_seconds]
        if not history_filtered: continue

        times = np.array([_v(s.time) for s in history_filtered])
        temps = np.array([_v(s.surface_temperature) for s in history_filtered]) - 273.15
        
        ls_rad = np.array([_v(s.orbital_angle) for s in history_filtered]) + 251.0 * np.pi / 180.0
        ls = (ls_rad * 180 / np.pi) % 360.0

        sols = times // _v(MARS_ROTATION_PERIOD)
        unique_sols = np.unique(sols)

        daily_ls = []
        daily_avg = []

        for sol in unique_sols:
            mask = (sols == sol)
            if np.sum(mask) < 10: continue # Need enough points for a representative average
            
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
        daily_ls = np.array(daily_ls)[idx]
        daily_avg = np.array(daily_avg)[idx]

        plt.plot(daily_ls, daily_avg, label=f'{name} Avg', color=colors.get(name, "black"), linewidth=2)

    plt.title(title)
    plt.xlabel("Solar Longitude Ls (°)")
    plt.ylabel("Temperature (°C)")
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    filepath = os.path.join("outputs", filename)
    plt.savefig(filepath)
    print(f"  📈 Three-coordinate plot (refined) saved to {filepath}")
    plt.close(fig)


def run_three_coordinates(accuracy: Accuracy = Accuracy.FAST, dt: float = 600.0):
    """Run simulations for three specific coordinates with higher resolution and spin-up."""
    points = [
        {"name": "North", "lat": 45.0, "lon": 137.0},
        {"name": "Equator", "lat": 0.0, "lon": 137.0},
        {"name": "Southern Hemisphere", "lat": -40.0, "lon": 137.0},
    ]

    histories = {}
    # Run for spin-up + an integer number of complete sols so the last daily
    # average is never computed from a partial day.  MARS_ORBITAL_PERIOD is
    # 668.62 sols; truncating to 668 complete sols avoids a biased partial-sol
    # average that otherwise creates a temperature spike near Ls ≈ 256°.
    spin_up = 10 * _v(MARS_ROTATION_PERIOD)
    n_complete_sols = int(_v(MARS_ORBITAL_PERIOD) / _v(MARS_ROTATION_PERIOD))
    total_duration = spin_up + n_complete_sols * _v(MARS_ROTATION_PERIOD)
    
    for pt in points:
        print(f"\n{'─' * 60}")
        print(f"  Running {pt['name']} ({pt['lat']}°N, {pt['lon']}°E)  —  {accuracy.value} (dt={dt}s)")
        print(f"{'─' * 60}")
        
        mars = Mars(latitude=pt["lat"], longitude=pt["lon"])
        tc = TimeController(mars, dt=dt, accuracy=accuracy)
        
        step_count = [0]
        def progress(state, t):
            step_count[0] += 1
            if step_count[0] % 10000 == 0:
                pct = _v(t) / total_duration * 100
                print(f"    ... {pct:5.1f}%  T={_v(state.thermal.surface_temperature):.2f} K")
        
        history = tc.run(duration=total_duration, callback=progress)
        histories[pt["name"]] = history

        # Save individual data for each point
        safe_name = pt["name"].lower().replace(" ", "_")
        save_history_to_csv(history, f"668_sols/mars_{safe_name}_evolution.csv")
        plot_history(history, f"668_sols/mars_{safe_name}_evolution.png", f"Mars Evolution: {pt['name']}")
        plot_seasonal_temps(history, f"668_sols/mars_{safe_name}_seasonal.png", f"Mars Seasonal: {pt['name']}", spin_up_sols=10)
        
    plot_three_temperatures(histories, "668_sols/mars_three_coords_temps.png", "Temperature across Seasons at Three Coordinates (137°E)")
    return histories


def main():
    print("=" * 60)
    print("  🔴  Terraforming Mars — Basic Evolution Experiment")
    print("=" * 60)

    os.makedirs("outputs/sol", exist_ok=True)
    os.makedirs("outputs/668_sols", exist_ok=True)

    # ── One Sol ───────────────────────────────────────────────
    run_one_sol(Accuracy.FAST, dt=3600.0)
    history_sol = run_one_sol(Accuracy.ACCURATE, dt=3600.0) # 1 hour dt
    save_history_to_csv(history_sol, "sol/mars_sol_evolution.csv")
    plot_history(history_sol, "sol/mars_sol_evolution.png", "Mars Evolution (1 Sol)")

    # ── One Year (accurate mode to capture exact ODE) ─────────
    history_year = run_one_year(Accuracy.ACCURATE, dt=3600.0) # 1 hour dt
    save_history_to_csv(history_year, "668_sols/mars_year_evolution.csv")
    plot_history(history_year, "668_sols/mars_year_evolution.png", "Mars Evolution (1 Year)")
    plot_seasonal_temps(history_year, "668_sols/mars_seasonal_temps.png", "Mars Temps: ~90°C Swing Summer to Winter")

    # ── Three Coordinates Experiment ───────────────────────────
    # Run with Accuracy.FAST and 10 min timestep for better seasonal sampling coverage
    run_three_coordinates(Accuracy.FAST, dt=600.0)

    print(f"\n{'=' * 60}")
    print("  ✅  Done.")
    print(f"{'=' * 60}\n")

if __name__ == "__main__":
    main()
