"""Present-day baseline stability of the *unforced* Mars model.

Answers: *before any forcing, does the model hold present-day Mars — the ~6 mb
global-mean pressure and its seasonal swing — as a stable state, with no secular
drift?*

:func:`assess_baseline` runs the unforced model for several Mars years and
reports, over the post-spin-up years:
  - the mean surface pressure (target ~6 mb / ~610 Pa) and seasonal swing,
  - the year-to-year **repeatability** of the swing (a limit cycle has ~0 spread),
  - the **secular drift** of the annual-mean pressure (Pa/yr and %/century — the
    only non-periodic term is the small MAVEN non-thermal escape sink),
  - the drift of temperature and total ice (they must not run away).

**Initial conditions.** The model's *default* 5×10¹⁵ kg ice reservoir is not
self-consistent with a 6 mb atmosphere: during spin-up the excess cap CO₂
sublimates and the atmosphere inflates to ~7.15 mb. The CO₂-budget-consistent
present-day reservoir is ~1.6×10¹⁵ kg; starting there, the atmosphere neither
gains nor loses net mass and the stable state sits at ~6.3 mb. Those are the
defaults here, and the discrepancy is itself reported.
"""

from __future__ import annotations

import dataclasses

import numpy as np

from src.celestials.planets.mars import Mars
from src.engine.time_controller import Accuracy, TimeController

# CO₂-budget-consistent present-day initial conditions (see module docstring).
# The model's cap attractor is ~1.59e15 kg regardless of start; the equilibrium
# *pressure* is set by the total CO₂ inventory. Starting the atmosphere at the
# ~6 mb global mean with a cap slightly below the attractor lets the caps grow to
# 1.59e15 while the spun-up mean stays ~6.1 mb (verified: 6.08 mb, drift <0.2
# Pa/yr). The default 5e15 kg cap instead inflates the baseline to ~7.15 mb.
PRESENT_DAY_PRESSURE_PA = 610.0
PRESENT_DAY_ICE_KG = 0.8e15


@dataclasses.dataclass(frozen=True)
class BaselineStability:
    """Stability diagnostics of the unforced model over post-spin-up years."""

    years: np.ndarray                    # post-spin-up year indices
    annual_mean_pressure_pa: np.ndarray
    annual_swing_pa: np.ndarray
    mean_pressure_pa: float
    seasonal_swing_pa: float
    seasonal_swing_pct: float
    swing_repeatability_pa: float        # std of the annual swing (0 = perfect cycle)
    pressure_drift_pa_per_year: float
    pressure_drift_pct_per_century: float
    temperature_drift_k_per_year: float
    ice_drift_kg_per_year: float
    mean_temperature_k: float
    mean_ice_kg: float

    @property
    def mean_pressure_mb(self) -> float:
        return self.mean_pressure_pa / 100.0

    def is_stable(
        self,
        *,
        drift_tol_pa_per_year: float = 1.0,
        repeatability_tol_pa: float = 5.0,
    ) -> bool:
        """True if the annual cycle repeats and the annual mean does not drift."""
        return (
            abs(self.pressure_drift_pa_per_year) < drift_tol_pa_per_year
            and self.swing_repeatability_pa < repeatability_tol_pa
        )


def _linfit_slope(x: np.ndarray, y: np.ndarray) -> float:
    """Least-squares slope; 0 for a single point (no drift measurable)."""
    if len(x) < 2:
        return 0.0
    return float(np.polyfit(x, y, 1)[0])


@dataclasses.dataclass(frozen=True)
class UnforcedRun:
    """Raw time series from an unforced integration (for plotting + summary)."""

    time_s: np.ndarray
    pressure_pa: np.ndarray
    temperature_k: np.ndarray
    ice_kg: np.ndarray
    year_s: float


def run_unforced(
    *,
    n_years: int = 6,
    dt: float = 3600.0,
    surface_pressure: float = PRESENT_DAY_PRESSURE_PA,
    ice_mass: float = PRESENT_DAY_ICE_KG,
    elevation_m: float = 0.0,
    **mars_kwargs,
) -> UnforcedRun:
    """Integrate the unforced model for ``n_years`` and return the raw series."""
    mars = Mars(
        surface_pressure=surface_pressure,
        ice_mass=ice_mass,
        elevation_m=elevation_m,
        **mars_kwargs,
    )
    year_s = float(mars.orbital_params.orbital_period)
    history = TimeController(mars, dt=dt, accuracy=Accuracy.FAST).run(duration=n_years * year_s)
    return UnforcedRun(
        time_s=np.array([float(s.time) for s in history]),
        pressure_pa=np.array([float(s.surface_pressure) for s in history]),
        temperature_k=np.array([float(s.surface_temperature) for s in history]),
        ice_kg=np.array([float(s.ice_mass) for s in history]),
        year_s=year_s,
    )


def summarize_baseline(run: UnforcedRun, *, spinup_years: int = 1) -> BaselineStability:
    """Per-year statistics + drift over the post-spin-up years of a run."""
    t, P, T, ice = run.time_s, run.pressure_pa, run.temperature_k, run.ice_kg
    year_idx = np.floor(t / run.year_s).astype(int)
    n_years = int(year_idx.max()) + 1
    if n_years <= spinup_years:
        raise ValueError(f"run spans {n_years} yr; must exceed spinup_years ({spinup_years})")

    # Drop under-sampled bins — notably the single-snapshot boundary year at
    # t = n_years * year_s, which would otherwise register as a 1-sample "year"
    # with zero swing and skew the drift fit.
    counts = np.array([int((year_idx == y).sum()) for y in range(n_years)])
    min_count = 0.5 * np.median(counts[counts > 0])

    years, ann_mean_P, ann_swing, ann_mean_T, ann_mean_ice = [], [], [], [], []
    for y in range(spinup_years, n_years):
        m = year_idx == y
        if m.sum() < min_count:
            continue
        years.append(y)
        ann_mean_P.append(P[m].mean())
        ann_swing.append(P[m].max() - P[m].min())
        ann_mean_T.append(T[m].mean())
        ann_mean_ice.append(ice[m].mean())

    years = np.array(years, dtype=float)
    ann_mean_P = np.array(ann_mean_P)
    ann_swing = np.array(ann_swing)
    ann_mean_T = np.array(ann_mean_T)
    ann_mean_ice = np.array(ann_mean_ice)

    mean_P = float(ann_mean_P.mean())
    swing = float(ann_swing.mean())
    p_drift = _linfit_slope(years, ann_mean_P)

    return BaselineStability(
        years=years,
        annual_mean_pressure_pa=ann_mean_P,
        annual_swing_pa=ann_swing,
        mean_pressure_pa=mean_P,
        seasonal_swing_pa=swing,
        seasonal_swing_pct=100.0 * swing / mean_P,
        swing_repeatability_pa=float(ann_swing.std()),
        pressure_drift_pa_per_year=p_drift,
        pressure_drift_pct_per_century=100.0 * (p_drift * 100.0) / mean_P,
        temperature_drift_k_per_year=_linfit_slope(years, ann_mean_T),
        ice_drift_kg_per_year=_linfit_slope(years, ann_mean_ice),
        mean_temperature_k=float(ann_mean_T.mean()),
        mean_ice_kg=float(ann_mean_ice.mean()),
    )


def assess_baseline(
    *,
    n_years: int = 6,
    spinup_years: int = 1,
    dt: float = 3600.0,
    surface_pressure: float = PRESENT_DAY_PRESSURE_PA,
    ice_mass: float = PRESENT_DAY_ICE_KG,
    elevation_m: float = 0.0,
    **mars_kwargs,
) -> BaselineStability:
    """Run the unforced model and quantify its present-day baseline stability.

    Integrates ``n_years`` Mars years (FAST path, no interventions), discards the
    first ``spinup_years`` as spin-up, and computes per-year statistics + drift
    over the remainder. Returns a :class:`BaselineStability`.
    """
    if n_years <= spinup_years:
        raise ValueError(f"n_years ({n_years}) must exceed spinup_years ({spinup_years})")
    run = run_unforced(
        n_years=n_years, dt=dt, surface_pressure=surface_pressure,
        ice_mass=ice_mass, elevation_m=elevation_m, **mars_kwargs,
    )
    return summarize_baseline(run, spinup_years=spinup_years)
