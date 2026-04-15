"""Standalone output — CSV serialisation and matplotlib plots.

Receives SimConfig and RunResult objects; no dependency on experiments/.
"""

from __future__ import annotations

import csv
import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np

from cli.models import ExpType, SimConfig
from cli.runner import RunResult
from src.celestials import MARS_ROTATION_PERIOD

_SOL_SECONDS = float(MARS_ROTATION_PERIOD.item())
_SOL_HOURS   = _SOL_SECONDS / 3600.0


def _v(t) -> float:
    return float(t.item())


def _out_dir(cfg: SimConfig) -> str:
    if cfg.output.output_path:
        return cfg.output.output_path
    tag = cfg.output.out_dir or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    return os.path.join("outputs", tag)


# ── CSV ────────────────────────────────────────────────────────────────────────

def save_csv(result: RunResult, filepath: str) -> None:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["time_hours", "temperature_k", "pressure_pa",
                    "ice_mass_kg", "solar_flux_wm2", "orbital_angle_rad"])
        for s in result.history:
            w.writerow([
                _v(s.time) / 3600.0,
                _v(s.surface_temperature),
                _v(s.surface_pressure),
                _v(s.ice_mass),
                _v(s.solar_flux),
                _v(s.orbital_angle),
            ])
    print(f"  CSV  → {filepath}")


# ── Diurnal / multi-sol plot ───────────────────────────────────────────────────

def plot_diurnal(result: RunResult, filepath: str, title: str) -> None:
    history  = result.history
    max_t    = max(_v(s.time) for s in history)
    use_sols = max_t > 3 * _SOL_SECONDS

    times = np.array([_v(s.time)                for s in history])
    temps = np.array([_v(s.surface_temperature) for s in history])
    press = np.array([_v(s.surface_pressure)    for s in history])
    ice   = np.array([_v(s.ice_mass)            for s in history])

    n_sols        = max_t / _SOL_SECONDS
    steps_per_sol = len(history) / max(n_sols, 1)

    # When there are multiple steps per sol on a long run (>5 sols), aggregate
    # to daily means so that diurnal oscillations do not alias into a thick
    # filled band.  Short runs (≤5 sols) keep raw timesteps to show the diurnal
    # cycle.
    if steps_per_sol > 1.5 and n_sols > 5:
        sol_idx     = (times / _SOL_SECONDS).astype(int)
        unique_sols = np.unique(sol_idx)
        agg_times, agg_T, agg_P, agg_I = [], [], [], []
        for s in unique_sols:
            m = sol_idx == s
            agg_times.append(float(s) * _SOL_SECONDS + 0.5 * _SOL_SECONDS)
            agg_T.append(float(np.mean(temps[m])))
            agg_P.append(float(np.mean(press[m])))
            agg_I.append(float(np.mean(ice[m])))
        times = np.array(agg_times)
        temps = np.array(agg_T)
        press = np.array(agg_P)
        ice   = np.array(agg_I)

    xs     = (times / _SOL_SECONDS          if use_sols
              else times / _SOL_SECONDS * 360.0)
    xlabel = "Time (sols)" if use_sols else "Rotation angle (°)"
    lw     = max(0.8, 2.0 - 1.2 * min(len(xs) / 5000, 1.0))

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    for ys, ylabel, color, tag in [
        (temps, "Temperature (K)",  "#d62728", "temp"),
        (press, "Pressure (Pa)",    "#1f77b4", "pressure"),
        (ice,   "Ice Mass (kg)",    "#2ca02c", "ice"),
    ]:
        fig = plt.figure(figsize=(10, 4))
        plt.plot(xs, ys, color=color, linewidth=lw)
        plt.xlabel(xlabel, fontsize=12)
        plt.ylabel(ylabel, fontsize=12)
        plt.title(f"{title} — {ylabel.split(' (')[0]}", fontsize=13, fontweight="bold")
        plt.grid(True, alpha=0.4, linestyle="--")
        plt.tight_layout()
        out = filepath.replace(".png", f"_{tag}.png")
        plt.savefig(out, dpi=150)
        plt.close(fig)
        print(f"  Plot → {out}")


# ── Multi-coordinate overlay ───────────────────────────────────────────────────

def plot_multi_coord(results: list[RunResult], filepath: str, title: str) -> None:
    colors = {
        "North":               "#1f77b4",
        "Equator":             "#d62728",
        "Southern Hemisphere": "#2ca02c",
    }
    fig, ax = plt.subplots(figsize=(12, 6))
    for res in results:
        times_h = np.array([_v(s.time) / 3600.0          for s in res.history])
        temps   = np.array([_v(s.surface_temperature)     for s in res.history])
        lw      = max(1.0, 2.0 - 1.0 * min(len(times_h) / 2000, 1.0))
        ns      = "N" if res.lat >= 0 else "S"
        label   = f"{res.name}  ({abs(res.lat):.0f}°{ns}, {res.lon:.0f}°E)"
        ax.plot(times_h, temps, label=label,
                color=colors.get(res.name, "#555555"), linewidth=lw)

    max_h = max(_v(r.history[-1].time) for r in results) / 3600.0
    for i in range(1, int(max_h / _SOL_HOURS) + 1):
        ax.axvline(i * _SOL_HOURS, color="#888", linewidth=0.7, linestyle="--", alpha=0.5)

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xlabel("Time (hours)", fontsize=12)
    ax.set_ylabel("Temperature (K)", fontsize=12)
    ax.grid(True, alpha=0.4, linestyle="--")
    ax.legend(fontsize=11, framealpha=0.9)
    fig.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot → {filepath}")


# ── Seasonal plot (year runs) ──────────────────────────────────────────────────

def plot_seasonal(result: RunResult, filepath: str, title: str,
                  spin_up_sols: int = 10) -> None:
    history   = result.history
    spin_up_s = spin_up_sols * _SOL_SECONDS

    times  = np.array([_v(s.time)             for s in history])
    temps  = np.array([_v(s.surface_temperature) for s in history])
    ls_rad = np.array([_v(s.orbital_angle)     for s in history]) + 251.0 * np.pi / 180.0
    ls     = (ls_rad * 180.0 / np.pi) % 360.0
    sols   = times // _SOL_SECONDS

    counts    = np.array([np.sum(sols == s) for s in np.unique(sols)])
    min_steps = max(10, int(0.8 * int(np.median(counts))))

    daily_ls, daily_max, daily_min, daily_avg = [], [], [], []
    for sol in np.unique(sols[times > spin_up_s]):
        mask = (sols == sol) & (times > spin_up_s)
        if np.sum(mask) < min_steps:
            continue
        t_day = temps[mask]
        ls_d  = ls[mask]
        if ls_d.max() > 350 and ls_d.min() < 10:
            ls_d = np.where(ls_d < 180, ls_d + 360, ls_d)
        daily_ls.append(float(np.mean(ls_d) % 360))
        daily_max.append(float(np.max(t_day)))
        daily_min.append(float(np.min(t_day)))
        daily_avg.append(float(np.mean(t_day)))

    idx     = np.argsort(daily_ls)
    ls_arr  = np.array(daily_ls)[idx]
    mx_arr  = np.array(daily_max)[idx]
    mn_arr  = np.array(daily_min)[idx]
    avg_arr = np.array(daily_avg)[idx]

    fig = plt.figure(figsize=(10, 6))
    plt.plot(ls_arr, mx_arr,  label="Daily Max", color="#d62728", linewidth=2)
    plt.plot(ls_arr, mn_arr,  label="Daily Min", color="#1f77b4", linewidth=2, linestyle="--")
    plt.plot(ls_arr, avg_arr, label="Daily Avg", color="#2ca02c", linewidth=2)
    plt.fill_between(ls_arr, mn_arr, mx_arr, alpha=0.08, color="#888888")
    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Solar Longitude Ls (°)", fontsize=12)
    plt.ylabel("Temperature (K)", fontsize=12)
    plt.xlim(0, 360)
    plt.xticks(np.arange(0, 361, 30))
    plt.grid(True, alpha=0.4, linestyle="--")
    plt.legend(fontsize=10, framealpha=0.9)
    plt.tight_layout()

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    plt.savefig(filepath, dpi=150)
    plt.close(fig)
    print(f"  Plot → {filepath}")


# ── Intervention plot ─────────────────────────────────────────────────────────

def plot_intervention(result: RunResult, filepath: str, title: str) -> None:
    """Temperature, pressure, GHF and ΔF over N Mars years of GHG injection."""
    history = result.history
    years   = [s.year for s in history]

    temps = [_v(s.surface_temperature) for s in history]
    press = [_v(s.surface_pressure)    for s in history]
    dFs   = [_v(s.delta_F)             for s in history]
    ghfs  = [_v(s.greenhouse_factor)   for s in history]

    os.makedirs(os.path.dirname(filepath), exist_ok=True)

    # ── Temperature ──────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(years, temps, color="#d62728", linewidth=1.5)
    ax.axhline(273.15, color="#aaaaaa", linestyle="--", linewidth=1.0, label="0°C")
    ax.set_xlabel("Mars Years", fontsize=12)
    ax.set_ylabel("Annual Mean Temperature (K)", fontsize=12)
    ax.set_title(f"{title} — Temperature", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.4, linestyle="--")
    fig.tight_layout()
    out_T = filepath.replace(".png", "_temp.png")
    fig.savefig(out_T, dpi=150); plt.close(fig)
    print(f"  Plot → {out_T}")

    # ── Pressure ─────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(years, press, color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("Mars Years", fontsize=12)
    ax.set_ylabel("Annual Mean Pressure (Pa)", fontsize=12)
    ax.set_title(f"{title} — Pressure", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.4, linestyle="--")
    fig.tight_layout()
    out_P = filepath.replace(".png", "_pressure.png")
    fig.savefig(out_P, dpi=150); plt.close(fig)
    print(f"  Plot → {out_P}")

    # ── Radiative forcing + GHF (twin axes) ───────────────────────────────────
    fig, ax1 = plt.subplots(figsize=(11, 4))
    ax2 = ax1.twinx()
    ax1.plot(years, dFs,  color="#ff7f0e", linewidth=1.5, label="ΔF (W/m²)")
    ax2.plot(years, ghfs, color="#9467bd", linewidth=1.5, linestyle="--",
             label="GHF")
    ax1.set_xlabel("Mars Years", fontsize=12)
    ax1.set_ylabel("Radiative Forcing ΔF (W/m²)", fontsize=12, color="#ff7f0e")
    ax2.set_ylabel("Greenhouse Factor", fontsize=12, color="#9467bd")
    ax1.tick_params(axis='y', labelcolor="#ff7f0e")
    ax2.tick_params(axis='y', labelcolor="#9467bd")
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=10)
    ax1.set_title(f"{title} — Forcing & Greenhouse Factor", fontsize=13, fontweight="bold")
    ax1.grid(True, alpha=0.4, linestyle="--")
    fig.tight_layout()
    out_F = filepath.replace(".png", "_forcing.png")
    fig.savefig(out_F, dpi=150); plt.close(fig)
    print(f"  Plot → {out_F}")

    # ── GHG masses (stacked per compound) ────────────────────────────────────
    compounds = list(history[0].ghg_masses_kg.keys()) if hasattr(history[0], "ghg_masses_kg") else []
    if compounds:
        fig, ax = plt.subplots(figsize=(11, 4))
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2"]
        for i, cmp in enumerate(compounds):
            masses = [_v(s.ghg_masses_kg[cmp]) / 1e9 for s in history]   # Gt
            ax.plot(years, masses, color=colors[i % len(colors)], linewidth=1.5, label=cmp)
        ax.set_xlabel("Mars Years", fontsize=12)
        ax.set_ylabel("Atmospheric Mass (Gt)", fontsize=12)
        ax.set_title(f"{title} — Atmospheric GHG Inventory", fontsize=13, fontweight="bold")
        ax.legend(fontsize=10, framealpha=0.9)
        ax.grid(True, alpha=0.4, linestyle="--")
        fig.tight_layout()
        out_M = filepath.replace(".png", "_ghg_mass.png")
        fig.savefig(out_M, dpi=150); plt.close(fig)
        print(f"  Plot → {out_M}")


# ── Dispatch ───────────────────────────────────────────────────────────────────

def dispatch(results: list[RunResult], cfg: SimConfig) -> None:
    """Save all outputs for a completed simulation."""
    o = cfg.output
    if not o.save_csv and not o.save_plot:
        return

    out_dir = _out_dir(cfg)
    p       = cfg.planet
    ns      = "N" if p.latitude >= 0 else "S"
    loc     = f"{abs(p.latitude):.1f}°{ns}, {p.longitude:.1f}°E"
    exp     = cfg.experiment.type

    if exp == ExpType.sol:
        r     = results[0]
        sols  = cfg.experiment.sols
        title = f"Mars — {sols:.0f} sol{'s' if sols != 1 else ''}  ({loc})"
        if o.save_csv:
            save_csv(r, os.path.join(out_dir, "mars_sol.csv"))
        if o.save_plot:
            plot_diurnal(r, os.path.join(out_dir, "mars_sol.png"), title)

    elif exp == ExpType.year:
        r     = results[0]
        title = f"Mars — 1 year  ({loc})"
        if o.save_csv:
            save_csv(r, os.path.join(out_dir, "mars_year.csv"))
        if o.save_plot:
            plot_diurnal(r, os.path.join(out_dir, "mars_year.png"), title)
            plot_seasonal(r, os.path.join(out_dir, "mars_seasonal.png"),
                          "Mars Seasonal Temperatures — 1 Year")

    elif exp == ExpType.intervention:
        r      = results[0]
        n_yrs  = cfg.intervention.n_years
        cmpds  = ", ".join(cfg.intervention.injection.keys()) or "baseline"
        title  = f"Mars Intervention — {n_yrs} yr  ({loc})  [{cmpds}]"
        if o.save_csv:
            save_csv(r, os.path.join(out_dir, "mars_intervention.csv"))
        if o.save_plot:
            plot_intervention(r, os.path.join(out_dir, "mars_intervention.png"), title)

    elif exp == ExpType.multi:
        sols  = cfg.experiment.sols
        title = f"Mars Multi-Coord — {sols:.0f} sol{'s' if sols != 1 else ''}  (137°E)"
        for r in results:
            safe = r.name.lower().replace(" ", "_")
            if o.save_csv:
                save_csv(r, os.path.join(out_dir, f"mars_{safe}.csv"))
            if o.save_plot:
                ns2    = "N" if r.lat >= 0 else "S"
                rtitle = f"Mars {r.name} ({abs(r.lat):.0f}°{ns2})"
                plot_diurnal(r, os.path.join(out_dir, f"mars_{safe}.png"), rtitle)
        if o.save_plot:
            plot_multi_coord(results, os.path.join(out_dir, "mars_multi_coord.png"), title)
