"""Conservation diagnostics — physics CI for the mass budget.

The prognostic surface pressure is pure mass bookkeeping (the thermal tide is
a diagnostic overlay, not a mass source), so the CO₂ budget must close exactly:

    M_atm(t₁) − M_atm(t₀)  +  M_ice(t₁) − M_ice(t₀)  +  R_escape·(t₁ − t₀)  =  ΔM_injected

where M_atm = P · 4πR² / g is the hydrostatic atmospheric mass.  A nonzero
residual (beyond integration tolerance) means a new term is leaking mass —
the failure mode that motivated this module (see the pressure audit and the
AmesGCM ``testconserv`` pattern this mirrors).

Usage::

    from src.engine.diagnostics import atmosphere_mass_kg, mass_budget_residual_kg

    before = (mars.atmosphere.surface_pressure.clone(), mars.water.ice_mass.clone())
    tc.run(duration=...)
    residual = mass_budget_residual_kg(
        mars, p_start=before[0], ice_start=before[1], elapsed_s=duration,
    )
    assert abs(float(residual)) < tolerance
"""

from __future__ import annotations

import math

import torch


def atmosphere_mass_kg(
    surface_pressure: torch.Tensor,
    gravity: torch.Tensor,
    radius: torch.Tensor,
) -> torch.Tensor:
    """Hydrostatic atmospheric mass:  M = P · 4πR² / g."""
    return surface_pressure * 4.0 * math.pi * radius ** 2 / gravity


def mass_budget_residual_kg(
    planet,
    p_start: torch.Tensor,
    ice_start: torch.Tensor,
    elapsed_s: float | torch.Tensor,
    injected_kg: float | torch.Tensor = 0.0,
) -> torch.Tensor:
    """Residual of the CO₂ mass budget over a completed run, in kg.

    Zero (to integrator tolerance) for a conservative run.  Uses the
    planet's *prognostic* pressure (``atmosphere.surface_pressure``), which
    excludes diagnostic overlays such as the thermal tide by design.

    Parameters
    ----------
    planet : Planet
        The planet in its end-of-run state.
    p_start, ice_start : torch.Tensor
        Prognostic surface pressure (Pa) and total ice mass (kg) captured
        before the run.
    elapsed_s : float | torch.Tensor
        Simulated duration in seconds (for the escape term).
    injected_kg : float | torch.Tensor
        Total mass added by interventions during the window (kg).
    """
    g = planet.intrinsic_params.gravity
    r = planet.intrinsic_params.radius
    d_atm = atmosphere_mass_kg(planet.atmosphere.surface_pressure, g, r) - \
        atmosphere_mass_kg(torch.as_tensor(p_start, dtype=g.dtype, device=g.device), g, r)
    d_ice = planet.water.ice_mass - torch.as_tensor(
        ice_start, dtype=g.dtype, device=g.device
    )
    escape = planet._ESCAPE_RATE * torch.as_tensor(
        elapsed_s, dtype=g.dtype, device=g.device
    )
    injected = torch.as_tensor(injected_kg, dtype=g.dtype, device=g.device)
    return d_atm + d_ice + escape - injected


def composition_consistency_residual_pa(planet) -> torch.Tensor:
    """|Σ partial pressures − total surface pressure| in Pa.

    Zero when composition is in sync with the prognostic total (the
    single-source-of-truth invariant maintained by Mars._sync_composition_co2).
    """
    total = sum(planet.atmosphere.composition.values())
    return (total - planet.atmosphere.surface_pressure).abs()
