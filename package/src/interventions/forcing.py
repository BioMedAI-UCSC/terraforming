"""Radiative forcing from super-greenhouse gases — Mars.

Converts atmospheric GHG masses to concentration (ppb), computes total
radiative forcing ΔF (W/m²), and updates the planet's greenhouse factor.

All tensor operations stay on the planet's device.  Python-float constants
are broadcast safely by PyTorch without allocating CPU intermediates.

References
----------
Marinova et al. (2005), "Radiative-convective model of warming Mars with
artificial greenhouse gases", J. Geophys. Res., 110, E03002.
"""

from __future__ import annotations

import torch

from src.constants import TF_DTYPE, STEFAN_BOLTZMANN
from src.interventions.compounds import get_compound
from src.interventions.state import GHGState

# ---------------------------------------------------------------------------
# Mars physical constants (Python floats — device-agnostic)
# ---------------------------------------------------------------------------
_MARS_SURFACE_AREA_M2 = 1.4437e14   # m²  (4π R²,  R = 3.3895×10⁶ m)
_MARS_ATM_MW_G_MOL    = 43.45       # g/mol  (95% CO₂ + 3% N₂ + 1.6% Ar)
_MARS_EMISSIVITY      = 0.95        # surface emissivity (same as mars.py)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_concentration_ppb(
    ghg_state: GHGState,
    total_atm_mass_kg: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Convert atmospheric masses to molar mixing ratios (ppb).

    Uses the molar-fraction definition:

        ppb_i = (M_i / MW_i) / (M_atm / MW_atm) × 1e9

    where M_i is the atmospheric mass of compound i (kg), MW_i is its
    molecular weight (g/mol), M_atm is the total atmospheric mass (kg),
    and MW_atm is the mean molecular weight of the Mars atmosphere (g/mol).

    Parameters
    ----------
    ghg_state : GHGState
        Current atmospheric GHG mass state.
    total_atm_mass_kg : torch.Tensor
        Total Mars atmospheric mass in kg (from mars.atmosphere.atmospheric_mass).

    Returns
    -------
    dict[str, torch.Tensor]
        Per-compound concentration in ppb, on the same device as ghg_state.
    """
    ppb: dict[str, torch.Tensor] = {}
    for name in ghg_state.compounds:
        mass_kg  = ghg_state.get_mass_kg(name)                    # kg, on device
        mw_ratio = get_compound(name).molecular_weight_g_mol / _MARS_ATM_MW_G_MOL
        ppb[name] = (mass_kg / total_atm_mass_kg) / mw_ratio * 1e9
    return ppb


def delta_F_total(
    ghg_state: GHGState,
    total_atm_mass_kg: torch.Tensor,
) -> torch.Tensor:
    """Compute total radiative forcing from all tracked GHGs (W/m²).

    ΔF = Σ_i  η_i [W m⁻² ppb⁻¹]  ×  C_i [ppb]

    This is the linear (optically thin) approximation appropriate for trace
    gases at concentrations < ~1000 ppb, which covers realistic injection
    rates over 50–100 year horizons.

    Parameters
    ----------
    ghg_state : GHGState
        Current atmospheric GHG mass state.
    total_atm_mass_kg : torch.Tensor
        Total Mars atmospheric mass in kg.

    Returns
    -------
    torch.Tensor
        Scalar total forcing (W/m²) on the GHGState's device.
    """
    device = ghg_state.device
    dF     = torch.zeros((), dtype=TF_DTYPE, device=device)
    ppb    = compute_concentration_ppb(ghg_state, total_atm_mass_kg)

    for name, conc in ppb.items():
        eta  = get_compound(name).rf_efficiency_W_m2_ppb   # Python float
        dF   = dF + eta * conc

    return dF


def delta_F_from_composition(
    composition: dict,
    total_pressure: torch.Tensor,
) -> torch.Tensor:
    """Compute total GHG radiative forcing from atmosphere composition (W/m²).

    Uses partial pressures (Pa) to derive mole-fraction concentrations (ppb),
    then applies compound-specific RF efficiencies:

        ppb_i  =  P_i / P_total × 10⁹
        ΔF     =  Σ_i  η_i × ppb_i

    Only species present in the COMPOUNDS registry contribute — background
    gases (CO₂, N₂, Ar) are silently skipped.

    Parameters
    ----------
    composition : dict[str, torch.Tensor]
        Species → partial pressure (Pa), as stored in ``mars.atmosphere.composition``.
    total_pressure : torch.Tensor
        Total atmospheric surface pressure (Pa).
    """
    from src.interventions.compounds import COMPOUNDS, get_compound
    device = total_pressure.device
    dF = torch.zeros((), dtype=TF_DTYPE, device=device)
    for name, P_i in composition.items():
        if name in COMPOUNDS:
            ppb = P_i.to(device) / total_pressure * 1e9
            eta = get_compound(name).rf_efficiency_W_m2_ppb
            dF = dF + eta * ppb
    return dF


def update_greenhouse_factor(
    mars,
    delta_F: torch.Tensor,
    baseline_ghf: torch.Tensor | None = None,
    baseline_olr: torch.Tensor | None = None,
) -> None:
    """Apply cumulative GHG radiative forcing via the energy-balance GHF formula.

    Derivation
    ----------
    At the original (CO₂-only) equilibrium::

        F_in  =  OLR_baseline  =  ε σ (T_eq / GHF_base)⁴          (1)

    Injected GHGs trap ΔF extra W/m².  At the **new** radiative equilibrium
    the atmosphere must emit more to compensate — T rises until::

        F_in + ΔF  =  ε σ (T_eq_new / GHF_base)⁴                  (2)

    Combining (1) and (2) gives the new effective greenhouse factor::

        GHF_new  =  GHF_base × (1 + ΔF / F_in_base)^(1/4)          (3)

    where F_in_base = ε σ (T₀ / GHF_base)⁴ is the baseline OLR (≈ mean
    absorbed solar flux for the initial conditions).

    This formula:
    * Is **independent of instantaneous surface temperature** — no singularity
      when Mars is in a cold polar winter (T ≈ 149 K).
    * Grows **monotonically** with cumulative ΔF — no year-to-year oscillation.
    * Is **stable** for any physically reachable ΔF over 50–200 year horizons.

    Parameters
    ----------
    mars : Mars
        The Mars instance whose greenhouse_factor will be updated.
    delta_F : torch.Tensor
        **Cumulative** total forcing from all GHGs (W/m²), on mars._device.
    baseline_ghf : torch.Tensor, optional
        Initial CO₂-only GHF.  Callers must pass the value saved at t=0 so
        that ΔF is applied relative to the fixed baseline (not compounded).
        Defaults to the current mars.thermal.greenhouse_factor (first call only).
    baseline_olr : torch.Tensor, optional
        Pre-computed F_in_base = ε σ (T₀/GHF_base)⁴ in W/m².
        Callers should cache this at initialisation to avoid recomputing
        it at every annual update.  Computed on-the-fly if not provided.
    """
    d        = mars._device
    sb       = STEFAN_BOLTZMANN.to(d)

    GHF_base = (baseline_ghf if baseline_ghf is not None
                else mars.thermal.greenhouse_factor)

    if baseline_olr is not None:
        F_in_base = baseline_olr
    else:
        T0        = mars.thermal.surface_temperature
        F_in_base = _MARS_EMISSIVITY * sb * (T0 / GHF_base) ** 4.0

    # Equation (3): monotonic, singularity-free.  relu keeps the
    # "no forcing when ΔF ≤ 0" semantics without a data-dependent branch,
    # so the autograd graph from delta_F to GHF is never cut.
    GHF_new = GHF_base * (1.0 + torch.relu(delta_F) / F_in_base) ** 0.25

    mars.thermal.greenhouse_factor = GHF_new.clamp(min=1.0)
