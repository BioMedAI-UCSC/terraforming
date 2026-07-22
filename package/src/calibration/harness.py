"""Run the 0-D model over a Mars year and calibrate/validate its seasonal cycle.

Pipeline:
  1. :func:`simulate_seasonal_cycle` — build a :class:`~src.celestials.planets.mars.Mars`
     at a site, run it to a repeating annual limit cycle, and bin the daily-mean
     surface temperature and pressure onto a solar-longitude (Ls) grid.
  2. :func:`evaluate` — compare that cycle to a
     :class:`~src.calibration.reference.ReferenceClimatology` (metrics in
     :mod:`src.calibration.metrics`).
  3. :func:`calibrate` — tune a named subset of the model's free physics
     parameters to minimise the seasonal-cycle RMSE against a reference.

Free parameters are injected by overriding the planet's cached ``self._*``
constants after construction — the same values that were originally hand-tuned to
REMS Gale-Crater observations. ``calibrate`` reports exactly which (and how many)
were tuned, so the calibration is auditable.
"""

from __future__ import annotations

import dataclasses
import math
from typing import Callable, Mapping, Sequence

import numpy as np
import torch

from src.calibration.metrics import CycleMetrics, compute_metrics, rmse
from src.calibration.reference import ReferenceClimatology
from src.celestials.planets.mars import MARS_LS_PERIHELION, Mars
from src.engine.time_controller import Accuracy, TimeController

# Named free parameters → the cached planet attribute they override. These are
# the constants originally calibrated to REMS; exposing them by name makes the
# tuned set explicit.
PARAMETERS: dict[str, str] = {
    "polar_cap_fraction": "_CAP_FRAC",
    "thermal_inertia": "_TI",
    "diurnal_swing_amp": "_DIURNAL_AMP",
    "thermal_tide_pa": "_TIDE_PA",
}


@dataclasses.dataclass(frozen=True)
class SeasonalCycle:
    """Model seasonal cycle binned onto ``ls_deg`` (diurnal cycle averaged out)."""

    ls_deg: np.ndarray
    temperature_k: np.ndarray
    pressure_pa: np.ndarray


def _apply_overrides(mars: Mars, overrides: Mapping[str, float] | None) -> None:
    """Override named free parameters on a constructed planet (cached ``_*``)."""
    for name, value in (overrides or {}).items():
        if name not in PARAMETERS:
            raise KeyError(f"unknown parameter {name!r}; known: {sorted(PARAMETERS)}")
        attr = PARAMETERS[name]
        current = getattr(mars, attr)
        setattr(mars, attr, torch.tensor(float(value), dtype=current.dtype, device=current.device))


def simulate_seasonal_cycle(
    reference: ReferenceClimatology,
    *,
    overrides: Mapping[str, float] | None = None,
    dt: float = 3600.0,
    n_years: int = 2,
    n_bins: int | None = None,
    initial_ls_deg: float = 0.0,
    **mars_kwargs,
) -> SeasonalCycle:
    """Run the model at ``reference``'s site and bin its annual cycle.

    Places the planet at the reference latitude/elevation, integrates ``n_years``
    Mars years (FAST path), and bins the **final** year onto ``n_bins`` Ls bins
    (defaults to the reference's own grid), so the returned cycle aligns with the
    reference index-for-index. Running >1 year discards spin-up so the result is
    a repeating annual limit cycle.
    """
    n_bins = len(reference.ls_deg) if n_bins is None else n_bins
    mars = Mars(
        latitude=reference.latitude_deg,
        elevation_m=reference.elevation_m,
        initial_ls_deg=initial_ls_deg,
        **mars_kwargs,
    )
    _apply_overrides(mars, overrides)

    year_s = float(mars.orbital_params.orbital_period)
    tc = TimeController(mars, dt=dt, accuracy=Accuracy.FAST)
    history = tc.run(duration=n_years * year_s)

    time_s = np.array([float(s.time) for s in history])
    oa = np.array([float(s.orbital_angle) for s in history])
    T = np.array([float(s.surface_temperature) for s in history])
    P = np.array([float(s.surface_pressure) for s in history])

    # Keep only the final year (limit cycle); guard tiny/one-year runs.
    last_year = time_s >= max(0.0, (n_years - 1) * year_s)
    if last_year.sum() < n_bins:
        last_year = np.ones_like(time_s, dtype=bool)

    ls = (np.degrees(oa + float(MARS_LS_PERIHELION))) % 360.0
    step = 360.0 / n_bins
    idx = np.floor(ls / step).astype(int) % n_bins

    ls_centers = np.arange(0.0, 360.0, step)
    T_cycle = np.full(n_bins, np.nan)
    P_cycle = np.full(n_bins, np.nan)
    for b in range(n_bins):
        m = last_year & (idx == b)
        if m.any():
            T_cycle[b] = T[m].mean()
            P_cycle[b] = P[m].mean()
    return SeasonalCycle(ls_deg=ls_centers, temperature_k=T_cycle, pressure_pa=P_cycle)


def evaluate(
    cycle: SeasonalCycle, reference: ReferenceClimatology, field: str = "pressure"
) -> CycleMetrics:
    """Metrics comparing ``cycle`` to ``reference`` for ``field`` (pressure|temperature)."""
    if field == "pressure":
        model, ref = cycle.pressure_pa, reference.pressure_pa
    elif field == "temperature":
        model, ref = cycle.temperature_k, reference.temperature_k
    else:
        raise ValueError(f"field must be 'pressure' or 'temperature', got {field!r}")
    if ref is None:
        raise ValueError(f"reference {reference.name!r} has no {field} climatology")
    return compute_metrics(cycle.ls_deg, model, ref)


@dataclasses.dataclass
class CalibrationResult:
    """Outcome of a calibration run."""

    tuned_parameters: dict[str, float]
    n_parameters: int
    field: str
    rmse_before: float
    rmse_after: float
    metrics_before: CycleMetrics
    metrics_after: CycleMetrics


def calibrate(
    reference: ReferenceClimatology,
    param_names: Sequence[str],
    x0: Sequence[float],
    *,
    field: str = "pressure",
    bounds: Sequence[tuple[float, float]] | None = None,
    maxiter: int = 40,
    simulate: Callable[..., SeasonalCycle] = simulate_seasonal_cycle,
    **sim_kwargs,
) -> CalibrationResult:
    """Tune ``param_names`` to minimise seasonal-cycle RMSE against ``reference``.

    Uses SciPy Nelder-Mead (derivative-free; the model is a black box here). The
    number of tuned parameters is ``len(param_names)`` — reported in the result so
    the calibration is fully auditable. ``x0`` are the starting values (the
    current calibrated defaults are a sensible choice).
    """
    from scipy.optimize import minimize

    unknown = set(param_names) - set(PARAMETERS)
    if unknown:
        raise KeyError(f"unknown parameters {sorted(unknown)}; known: {sorted(PARAMETERS)}")
    if len(x0) != len(param_names):
        raise ValueError("x0 length must match param_names")

    target = reference.pressure_pa if field == "pressure" else reference.temperature_k
    if target is None:
        raise ValueError(f"reference {reference.name!r} has no {field} climatology")

    def _cycle_field(cycle: SeasonalCycle) -> np.ndarray:
        return cycle.pressure_pa if field == "pressure" else cycle.temperature_k

    def loss(x: np.ndarray) -> float:
        overrides = {n: float(v) for n, v in zip(param_names, x)}
        cyc = simulate(reference, overrides=overrides, **sim_kwargs)
        return rmse(_cycle_field(cyc), target)

    cyc0 = simulate(reference, overrides=dict(zip(param_names, x0)), **sim_kwargs)
    metrics_before = compute_metrics(cyc0.ls_deg, _cycle_field(cyc0), target)

    res = minimize(
        loss, np.asarray(x0, dtype=float), method="Nelder-Mead",
        bounds=bounds, options={"maxiter": maxiter, "xatol": 1e-4, "fatol": 1e-2},
    )
    tuned = {n: float(v) for n, v in zip(param_names, res.x)}
    cyc1 = simulate(reference, overrides=tuned, **sim_kwargs)
    metrics_after = compute_metrics(cyc1.ls_deg, _cycle_field(cyc1), target)

    return CalibrationResult(
        tuned_parameters=tuned,
        n_parameters=len(param_names),
        field=field,
        rmse_before=float(metrics_before.rmse),
        rmse_after=float(metrics_after.rmse),
        metrics_before=metrics_before,
        metrics_after=metrics_after,
    )
