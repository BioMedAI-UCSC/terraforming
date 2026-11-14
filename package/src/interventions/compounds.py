"""GHG compound registry for Mars terraforming interventions.

Radiative efficiencies, lifetimes and GWP100 use IPCC AR6 WG1 Table 7.SM.7
(2021), pp. 20-21. These are Earth reference values, not Mars coefficients.
The linear efficiency is a trace-perturbation approximation. Its extension to
Mars or large concentrations is not validated. Marinova (2005) Table 2 gives
optimized mixtures and is not the source of these efficiencies.
"""

from __future__ import annotations

from dataclasses import dataclass

COMPOUND_REFERENCE = "https://www.ipcc.ch/report/ar6/wg1/downloads/report/IPCC_AR6_WGI_Chapter07_SM.pdf#page=20"
COMPOUND_REFERENCE_DOMAIN = "Earth atmosphere; trace perturbations; IPCC AR6 WG1 Table 7.SM.7 (2021)"

@dataclass(frozen=True)
class CompoundProperties:
    """Physical and radiative properties of a super-greenhouse gas.

    Attributes
    ----------
    molecular_weight_g_mol : float
        Molecular weight in g/mol.
    atmospheric_lifetime_yr : float
        First-order e-folding decay lifetime in Earth Julian years, used as
        an explicit Earth-reference assumption, not a measured Mars lifetime.
    rf_efficiency_W_m2_ppb : float
        Radiative forcing efficiency in W m⁻² per ppb (mixing ratio).
        Earth-reference sensitivity from IPCC AR6 WG1 Table 7.SM.7.
    gwp100 : float
        Global Warming Potential relative to CO₂ over 100 years (Earth-based
        reference, provided for familiarity).
    description : str
        Human-readable name.
    """

    molecular_weight_g_mol:  float
    atmospheric_lifetime_yr: float
    rf_efficiency_W_m2_ppb:  float
    gwp100:                  float
    description:             str


# ---------------------------------------------------------------------------
# Registry — keys are canonical IUPAC/common names used throughout the system
# ---------------------------------------------------------------------------
#
# IPCC AR6 WG1 Table 7.SM.7, Earth-reference radiative efficiencies:
#   CF4    0.0990 W/m²/ppb
#   C2F6   0.2610 W/m²/ppb
#   C3F8   0.2700 W/m²/ppb
#   SF6    0.5670 W/m²/ppb
#   NF3    0.2040 W/m²/ppb
#   C4F10  0.3690 W/m²/ppb
#   C6F14  0.4490 W/m²/ppb
#
# Earth-reference lifetimes are assumptions; reduced sunlight alone does not
# establish the sign or magnitude of Mars photochemical lifetime changes.

COMPOUNDS: dict[str, CompoundProperties] = {
    "CF4": CompoundProperties(
        molecular_weight_g_mol  = 88.0,
        atmospheric_lifetime_yr = 50_000.0,
        rf_efficiency_W_m2_ppb  = 0.0990,
        gwp100                  = 7_380,
        description             = "Carbon tetrafluoride (PFC-14)",
    ),
    "C2F6": CompoundProperties(
        molecular_weight_g_mol  = 138.0,
        atmospheric_lifetime_yr = 10_000.0,
        rf_efficiency_W_m2_ppb  = 0.2610,
        gwp100                  = 12_400,
        description             = "Hexafluoroethane (PFC-116)",
    ),
    "C3F8": CompoundProperties(
        molecular_weight_g_mol  = 188.0,
        atmospheric_lifetime_yr = 2_600.0,
        rf_efficiency_W_m2_ppb  = 0.2700,
        gwp100                  = 9_290,
        description             = "Octafluoropropane (PFC-218)",
    ),
    "SF6": CompoundProperties(
        molecular_weight_g_mol  = 146.1,
        atmospheric_lifetime_yr = 1_000.0,
        rf_efficiency_W_m2_ppb  = 0.5670,
        gwp100                  = 24_300,
        description             = "Sulfur hexafluoride",
    ),
    "NF3": CompoundProperties(
        molecular_weight_g_mol  = 71.0,
        atmospheric_lifetime_yr = 569.0,
        rf_efficiency_W_m2_ppb  = 0.2040,
        gwp100                  = 17_400,
        description             = "Nitrogen trifluoride",
    ),
    "C4F10": CompoundProperties(
        molecular_weight_g_mol  = 238.0,
        atmospheric_lifetime_yr = 2_600.0,
        rf_efficiency_W_m2_ppb  = 0.3690,
        gwp100                  = 10_000,
        description             = "Decafluorobutane (PFC-3-1-10)",
    ),
    "C6F14": CompoundProperties(
        molecular_weight_g_mol  = 338.0,
        atmospheric_lifetime_yr = 3_100.0,
        rf_efficiency_W_m2_ppb  = 0.4490,
        gwp100                  = 8_620,
        description             = "Tetradecafluorohexane (PFC-5-1-14)",
    ),
}


def get_compound(name: str) -> CompoundProperties:
    """Return properties for a compound by name.

    Raises
    ------
    KeyError
        If the compound is not in the registry.  Available names are
        listed by :func:`list_compounds`.
    """
    if name not in COMPOUNDS:
        available = ", ".join(sorted(COMPOUNDS))
        raise KeyError(
            f"Unknown compound '{name}'. Available: {available}"
        )
    return COMPOUNDS[name]


def list_compounds() -> list[str]:
    """Return sorted list of registered compound names."""
    return sorted(COMPOUNDS)
