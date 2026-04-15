"""Simulation runner — translates a SimConfig into Mars + TimeController.

This module is the only bridge between the CLI and the physics framework.
It imports exclusively from `src.*` (the terraforming package).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import click

from typing import Any

from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
from src.engine import Accuracy as SrcAccuracy, TimeController, BatchedTimeController
from src.interventions import InterventionController

from cli.models import Accuracy, SimConfig


# Three canonical latitudes used in multi-coordinate runs
MULTI_POINTS = [
    {"name": "North",               "lat":  45.0, "lon": 137.0},
    {"name": "Equator",             "lat":   0.0, "lon": 137.0},
    {"name": "Southern Hemisphere", "lat": -40.0, "lon": 137.0},
]


# ── Color helpers ──────────────────────────────────────────────────────────────

def _temp_color(k: float) -> str:
    if k < 150:
        return "bright_blue"
    if k < 200:
        return "blue"
    if k < 240:
        return "cyan"
    if k < 273:
        return "bright_yellow"
    if k < 310:
        return "yellow"
    return "bright_red"


def _styled_temp(k: float) -> str:
    return click.style(f"{k:.1f} K", fg=_temp_color(k), bold=True)


def _styled_pressure(pa: float) -> str:
    return click.style(f"{pa:.1f} Pa", fg="cyan", bold=True)


def _styled_ice(kg: float) -> str:
    return click.style(f"{kg:.3e} kg", fg="bright_white")


def _styled_flux(wm2: float) -> str:
    return click.style(f"{wm2:.1f} W/m²", fg="yellow")


def _divider(width: int = 62) -> str:
    return click.style("─" * width, fg="bright_red", dim=True)


# ── Data ──────────────────────────────────────────────────────────────────────

@dataclass
class RunResult:
    """Output of a single simulation run."""
    name:    str
    history: list
    lat:     float
    lon:     float


# ── Internal helpers ──────────────────────────────────────────────────────────

def _v(t) -> float:
    return float(t.item())


def _src_accuracy(acc: Accuracy) -> SrcAccuracy:
    return SrcAccuracy.FAST if acc == Accuracy.fast else SrcAccuracy.ACCURATE


def _build_mars(cfg: SimConfig) -> Mars:
    p = cfg.planet
    kw: dict[str, Any] = {
        "surface_temperature": p.surface_temperature,
        "surface_pressure":    p.surface_pressure,
        "albedo":              p.albedo,
        "greenhouse_factor":   p.greenhouse_factor,
        "ice_mass":            p.ice_mass,
        "latitude":            p.latitude,
        "longitude":           p.longitude,
        "elevation_m":         p.elevation_m,
        "initial_ls_deg":      p.initial_ls_deg,
    }
    if p.composition is not None:
        kw["composition"] = p.composition
    return Mars(**kw)  # device auto-detected (CUDA if available)


def _build_mars_at(cfg: SimConfig, lat: float, lon: float) -> Mars:
    """Build a Mars instance with overridden lat/lon (used in multi-coord runs)."""
    p = cfg.planet
    kw: dict[str, Any] = {
        "surface_temperature": p.surface_temperature,
        "surface_pressure":    p.surface_pressure,
        "albedo":              p.albedo,
        "greenhouse_factor":   p.greenhouse_factor,
        "ice_mass":            p.ice_mass,
        "latitude":            lat,
        "longitude":           lon,
        "elevation_m":         p.elevation_m,
        "initial_ls_deg":      p.initial_ls_deg,
    }
    if p.composition is not None:
        kw["composition"] = p.composition
    return Mars(**kw)  # device auto-detected (CUDA if available)


# ── Progress bar ──────────────────────────────────────────────────────────────

def _year_progress(year_seconds: float) -> Callable:
    step      = [0]
    bar_width = 28

    def cb(state, t):
        step[0] += 1
        if step[0] % 2000 == 0:
            pct    = _v(t) / year_seconds
            filled = int(bar_width * pct)
            bar    = (click.style("█" * filled,              fg="bright_red")
                    + click.style("░" * (bar_width - filled), fg="bright_black"))
            T  = _v(state.thermal.surface_temperature)
            P  = _v(state.atmosphere.surface_pressure)
            line = (f"  [{bar}] "
                  + click.style(f"{pct*100:5.1f}%", fg="bright_yellow")
                  + "  T=" + _styled_temp(T)
                  + "  P=" + _styled_pressure(P))
            click.echo(f"\r{line}    ", nl=False)
            if pct >= 1.0:
                click.echo()

    return cb


# ── Public runners ─────────────────────────────────────────────────────────────

def run_sol(cfg: SimConfig) -> list[RunResult]:
    mars     = _build_mars(cfg)
    tc       = TimeController(mars, dt=cfg.engine.dt,
                              accuracy=_src_accuracy(cfg.engine.accuracy))
    duration = cfg.experiment.sols * _v(MARS_ROTATION_PERIOD)

    _print_run_header(cfg.planet.latitude, cfg.planet.longitude,
                      cfg.experiment.sols, cfg.engine)
    history = tc.run(duration=duration)
    _print_summary(history)

    return [RunResult(name="simulation", history=history,
                      lat=cfg.planet.latitude, lon=cfg.planet.longitude)]


def run_year(cfg: SimConfig) -> list[RunResult]:
    mars         = _build_mars(cfg)
    tc           = TimeController(mars, dt=cfg.engine.dt,
                                  accuracy=_src_accuracy(cfg.engine.accuracy))
    n_sols       = int(_v(MARS_ORBITAL_PERIOD) / _v(MARS_ROTATION_PERIOD))
    year_seconds = n_sols * _v(MARS_ROTATION_PERIOD)

    click.echo()
    click.echo(_divider())
    click.echo(
        "  " + click.style("○  ", fg="bright_red")
        + click.style(f"1 Martian Year  ({n_sols} sols)", fg="bright_white", bold=True)
        + click.style(f"  ·  {cfg.engine.accuracy.value} mode  dt={cfg.engine.dt:.0f} s",
                      fg="bright_black")
    )
    click.echo(_divider())

    history = tc.run(duration=year_seconds, callback=_year_progress(year_seconds))
    click.echo()
    _print_summary(history)

    return [RunResult(name="year", history=history,
                      lat=cfg.planet.latitude, lon=cfg.planet.longitude)]


def run_multi(cfg: SimConfig) -> list[RunResult]:
    """Run all MULTI_POINTS locations in parallel via BatchedTimeController.

    On CUDA this executes all B=3 simulations as a single batched kernel per
    timestep, achieving much higher GPU utilisation than sequential runs.
    """
    duration  = cfg.experiment.sols * _v(MARS_ROTATION_PERIOD)

    # Mars() auto-detects CUDA; all instances land on the same device
    mars_list = [
        _build_mars_at(cfg, lat=pt["lat"], lon=pt["lon"])
        for pt in MULTI_POINTS
    ]

    # Print headers before launching (cosmetic only)
    for pt in MULTI_POINTS:
        _print_run_header(pt["lat"], pt["lon"], cfg.experiment.sols,
                          cfg.engine, label=pt["name"])

    btc = BatchedTimeController(
        mars_list,
        dt=cfg.engine.dt,
        accuracy=_src_accuracy(cfg.engine.accuracy),
        compile=(mars_list[0]._device.type == 'cuda'),  # fuse kernels on GPU
    )
    all_histories = btc.run(duration=duration)

    results: list[RunResult] = []
    for pt, history in zip(MULTI_POINTS, all_histories):
        _print_summary(history)
        results.append(RunResult(name=pt["name"], history=history,
                                 lat=pt["lat"], lon=pt["lon"]))
    return results


def run_intervention(cfg: SimConfig) -> list[RunResult]:
    """Run a GHG intervention experiment over N Mars years.

    Injects the configured compounds at fixed annual rates, updating the
    greenhouse factor each year and simulating the resulting climate evolution.
    """
    iv   = cfg.intervention
    mars = _build_mars(cfg)

    if not iv.injection:
        click.echo(click.style(
            "  ⚠  No --inject compounds specified — running baseline (no injection).",
            fg="bright_yellow"
        ))

    ic = InterventionController(
        mars,
        injection_schedule_kg_yr = iv.injection,
        dt       = cfg.engine.dt,
        accuracy = _src_accuracy(cfg.engine.accuracy),
        compile  = (mars._device.type == 'cuda'),
    )

    click.echo()
    click.echo(_divider())
    click.echo(
        "  " + click.style("◉  ", fg="bright_red")
        + click.style(f"Intervention  {iv.n_years} Mars years", fg="bright_white", bold=True)
        + click.style(f"  ·  {cfg.engine.accuracy.value} mode  dt={cfg.engine.dt:.0f} s",
                      fg="bright_black")
    )
    if iv.injection:
        for compound, rate_kg in iv.injection.items():
            click.echo(
                "  " + click.style(f"  {compound:<8}", fg="bright_cyan", bold=True)
                + click.style(f"  {rate_kg:.2e} kg/yr", fg="bright_white")
            )
    click.echo(_divider())

    year_bar = [0]

    def _progress(snap) -> None:
        year_bar[0] = snap.year
        pct    = snap.year / iv.n_years
        filled = int(28 * pct)
        bar    = (click.style("█" * filled,          fg="bright_red")
                + click.style("░" * (28 - filled),   fg="bright_black"))
        T  = _v(snap.surface_temperature)
        dF = _v(snap.delta_F)
        line = (f"  [{bar}] "
              + click.style(f"{pct*100:5.1f}%", fg="bright_yellow")
              + "  T=" + _styled_temp(T)
              + "  ΔF=" + click.style(f"{dF:.1f} W/m²", fg="cyan", bold=True))
        click.echo(f"\r{line}    ", nl=False)
        if snap.year == iv.n_years:
            click.echo()

    history = ic.run(n_years=iv.n_years, callback=_progress)

    click.echo()
    _print_intervention_summary(history)

    return [RunResult(name="intervention", history=history,
                      lat=cfg.planet.latitude, lon=cfg.planet.longitude)]


def _print_intervention_summary(history: list) -> None:
    from src.interventions.controller import InterventionSnapshot

    T0  = _v(history[0].surface_temperature)
    T_f = _v(history[-1].surface_temperature)
    P0  = _v(history[0].surface_pressure)
    P_f = _v(history[-1].surface_pressure)
    dF  = _v(history[-1].delta_F)
    ghf = _v(history[-1].greenhouse_factor)

    def lbl(s: str) -> str:
        return click.style(s, fg="bright_black")

    click.echo(
        f"  {lbl('years')}      {click.style(str(len(history)), fg='bright_white')}\n"
        f"  {lbl('T (K)')}      {_styled_temp(T0)}  →  {_styled_temp(T_f)}"
        + click.style(f"   ΔT={T_f-T0:+.1f}", fg="bright_green" if T_f > T0 else "bright_blue") + "\n"
        f"  {lbl('P (Pa)')}     {_styled_pressure(P0)}  →  {_styled_pressure(P_f)}"
        + click.style(f"   ΔP={P_f-P0:+.1f}", fg="bright_black") + "\n"
        f"  {lbl('ΔF final')}   {click.style(f'{dF:.2f} W/m²', fg='cyan', bold=True)}\n"
        f"  {lbl('GHF final')}  {click.style(f'{ghf:.4f}', fg='bright_white')}"
    )


# ── Output helpers ─────────────────────────────────────────────────────────────

def _print_run_header(lat: float, lon: float, sols: float,
                      engine, label: str | None = None) -> None:
    ns    = "N" if lat >= 0 else "S"
    loc   = click.style(f"{abs(lat):.1f}°{ns}, {lon:.1f}°E", fg="bright_white")
    sol_s = click.style(f"{sols:.0f} sol{'s' if sols != 1 else ''}", fg="bright_yellow")
    mode  = click.style(f"{engine.accuracy.value} mode  dt={engine.dt:.0f} s",
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

    def lbl(s: str) -> str:
        return click.style(s, fg="bright_black")

    click.echo(
        f"  {lbl('steps')}      {click.style(str(len(history)), fg='bright_white')}\n"
        f"  {lbl('T (K)')}      {_styled_temp(min(temps))}  →  {_styled_temp(max(temps))}"
        + click.style(f"   ΔT={max(temps)-min(temps):.1f}", fg="bright_black") + "\n"
        f"  {lbl('P (Pa)')}     {_styled_pressure(min(press))}  →  {_styled_pressure(max(press))}"
        + click.style(f"   ΔP={max(press)-min(press):.1f}", fg="bright_black") + "\n"
        f"  {lbl('flux')}       {_styled_flux(min(fluxes))}  →  {_styled_flux(max(fluxes))}\n"
        f"  {lbl('final ice')}  {_styled_ice(ice_f)}"
    )
