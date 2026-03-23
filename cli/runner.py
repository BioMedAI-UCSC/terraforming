"""Simulation runner — translates a config dict into Mars + TimeController.

This module is the only bridge between the CLI and the physics framework.
It imports exclusively from `src.*` (the terraforming package).
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import Callable

import click

from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
from src.engine import Accuracy, TimeController

# Three canonical latitudes used in multi-coordinate runs
MULTI_POINTS = [
    {"name": "North",               "lat":  45.0, "lon": 137.0},
    {"name": "Equator",             "lat":   0.0, "lon": 137.0},
    {"name": "Southern Hemisphere", "lat": -40.0, "lon": 137.0},
]

# ── Color helpers ─────────────────────────────────────────────────────────────

def _temp_color(k: float) -> str:
    """Return a click fg color reflecting the temperature."""
    if k < 150:   return "bright_blue"
    if k < 200:   return "blue"
    if k < 240:   return "cyan"
    if k < 273:   return "bright_yellow"
    if k < 310:   return "yellow"
    return "bright_red"


def _styled_temp(k: float) -> str:
    return click.style(f"{k:.1f} K", fg=_temp_color(k), bold=True)


def _styled_pressure(pa: float) -> str:
    return click.style(f"{pa:.1f} Pa", fg="cyan", bold=True)


def _styled_ice(kg: float) -> str:
    return click.style(f"{kg:.3e} kg", fg="bright_white")


def _styled_flux(wm2: float) -> str:
    return click.style(f"{wm2:.1f} W/m²", fg="yellow")


def _divider(char: str = "─", width: int = 62) -> str:
    return click.style(char * width, fg="bright_red", dim=True)


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Output of a single simulation run."""
    name:    str
    history: list
    lat:     float
    lon:     float


def _v(t) -> float:
    return float(t.item())


def _accuracy(s: str) -> Accuracy:
    return Accuracy.FAST if s == "fast" else Accuracy.ACCURATE


def _build_mars(planet_cfg: dict) -> Mars:
    kw: dict = {
        "surface_temperature": planet_cfg["surface_temperature"],
        "surface_pressure":    planet_cfg["surface_pressure"],
        "albedo":              planet_cfg["albedo"],
        "greenhouse_factor":   planet_cfg["greenhouse_factor"],
        "ice_mass":            planet_cfg["ice_mass"],
        "latitude":            planet_cfg["latitude"],
        "longitude":           planet_cfg["longitude"],
        "elevation_m":         planet_cfg["elevation_m"],
        "initial_ls_deg":      planet_cfg["initial_ls_deg"],
    }
    if planet_cfg.get("composition") is not None:
        kw["composition"] = planet_cfg["composition"]
    return Mars(**kw)


# ── Progress bar ──────────────────────────────────────────────────────────────

def _year_progress(year_seconds: float) -> Callable:
    """Return a TimeController callback that renders an in-place progress bar."""
    step        = [0]
    bar_width   = 28

    def cb(state, t):
        step[0] += 1
        if step[0] % 2000 == 0:
            pct   = _v(t) / year_seconds
            filled = int(bar_width * pct)
            bar   = (click.style("█" * filled,       fg="bright_red")
                   + click.style("░" * (bar_width - filled), fg="bright_black"))
            T     = _v(state.thermal.surface_temperature)
            P     = _v(state.atmosphere.surface_pressure)
            line  = (f"  [{bar}] "
                   + click.style(f"{pct*100:5.1f}%", fg="bright_yellow")
                   + "  T=" + _styled_temp(T)
                   + "  P=" + _styled_pressure(P))
            click.echo(f"\r{line}    ", nl=False)
            if pct >= 1.0:
                click.echo()

    return cb


# ── Public runners ────────────────────────────────────────────────────────────

def run_sol(cfg: dict) -> list[RunResult]:
    p, e, x = cfg["planet"], cfg["engine"], cfg["experiment"]
    mars     = _build_mars(p)
    tc       = TimeController(mars, dt=e["dt"], accuracy=_accuracy(e["accuracy"]))
    duration = x["sols"] * _v(MARS_ROTATION_PERIOD)

    _print_run_header(p["latitude"], p["longitude"], x["sols"], e)
    history = tc.run(duration=duration)
    _print_summary(history)

    return [RunResult(name="simulation", history=history,
                      lat=p["latitude"], lon=p["longitude"])]


def run_year(cfg: dict) -> list[RunResult]:
    p, e     = cfg["planet"], cfg["engine"]
    mars     = _build_mars(p)
    tc       = TimeController(mars, dt=e["dt"], accuracy=_accuracy(e["accuracy"]))

    n_sols       = int(_v(MARS_ORBITAL_PERIOD) / _v(MARS_ROTATION_PERIOD))
    year_seconds = n_sols * _v(MARS_ROTATION_PERIOD)

    click.echo()
    click.echo(_divider())
    click.echo(
        "  " + click.style("○  ", fg="bright_red")
        + click.style(f"1 Martian Year  ({n_sols} sols)", fg="bright_white", bold=True)
        + click.style(f"  ·  {e['accuracy']} mode  dt={e['dt']:.0f} s", fg="bright_black")
    )
    click.echo(_divider())

    history = tc.run(duration=year_seconds, callback=_year_progress(year_seconds))
    click.echo()
    _print_summary(history)

    return [RunResult(name="year", history=history,
                      lat=p["latitude"], lon=p["longitude"])]


def run_multi(cfg: dict) -> list[RunResult]:
    p, e, x  = cfg["planet"], cfg["engine"], cfg["experiment"]
    results: list[RunResult] = []

    for pt in MULTI_POINTS:
        pt_planet = {**p, "latitude": pt["lat"], "longitude": pt["lon"]}
        mars = _build_mars(pt_planet)
        tc   = TimeController(mars, dt=e["dt"], accuracy=_accuracy(e["accuracy"]))
        duration = x["sols"] * _v(MARS_ROTATION_PERIOD)

        _print_run_header(pt["lat"], pt["lon"], x["sols"], e, label=pt["name"])
        history = tc.run(duration=duration)
        _print_summary(history)
        results.append(RunResult(name=pt["name"], history=history,
                                 lat=pt["lat"], lon=pt["lon"]))

    return results


# ── Output helpers ────────────────────────────────────────────────────────────

def _print_run_header(lat: float, lon: float, sols: float,
                      engine: dict, label: str | None = None) -> None:
    ns    = "N" if lat >= 0 else "S"
    loc   = click.style(f"{abs(lat):.1f}°{ns}, {lon:.1f}°E", fg="bright_white")
    sol_s = click.style(f"{sols:.0f} sol{'s' if sols != 1 else ''}", fg="bright_yellow")
    mode  = click.style(f"{engine['accuracy']} mode  dt={engine['dt']:.0f} s",
                        fg="bright_black")
    tag   = (click.style(f"  {label}  ", fg="bright_red", bold=True)
             + f"({loc})" if label else f"  ({loc})")
    click.echo()
    click.echo(_divider())
    click.echo(f"{tag}  ·  {sol_s}  ·  {mode}")
    click.echo(_divider())


def _print_summary(history: list) -> None:
    temps  = [_v(s.surface_temperature) for s in history]
    press  = [_v(s.surface_pressure)    for s in history]
    fluxes = [_v(s.solar_flux)          for s in history]
    ice_f  = _v(history[-1].ice_mass)

    lbl = lambda s: click.style(s, fg="bright_black")

    click.echo(
        f"  {lbl('steps')}      {click.style(str(len(history)), fg='bright_white')}\n"
        f"  {lbl('T (K)')}      {_styled_temp(min(temps))}  →  {_styled_temp(max(temps))}"
        + click.style(f"   ΔT={max(temps)-min(temps):.1f}", fg="bright_black") + "\n"
        f"  {lbl('P (Pa)')}     {_styled_pressure(min(press))}  →  {_styled_pressure(max(press))}"
        + click.style(f"   ΔP={max(press)-min(press):.1f}", fg="bright_black") + "\n"
        f"  {lbl('flux')}       {_styled_flux(min(fluxes))}  →  {_styled_flux(max(fluxes))}\n"
        f"  {lbl('final ice')}  {_styled_ice(ice_f)}"
    )
