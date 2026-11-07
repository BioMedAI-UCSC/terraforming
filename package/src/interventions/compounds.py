"""GHG compound registry for Mars terraforming interventions.

The retained linear radiative efficiencies are legacy sensitivity coefficients.
Their former attribution to all seven gases in Marinova et al. (2005), Table 2,
is unverified (#74). They must not be cited as validated Mars forcing data.

Atmospheric lifetimes are from IPCC AR5/AR6 and Ravishankara et al. (1993).
"""

from __future__ import annotations

from dataclasses import dataclass


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
        Legacy sensitivity coefficient; primary-source/domain audit pending.
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
# Legacy radiative forcing sensitivity coefficients (source audit pending #74):
#   CF4    0.0880 W/m²/ppb
#   C2F6   0.2600 W/m²/ppb
#   C3F8   0.2400 W/m²/ppb
#   SF6    0.5700 W/m²/ppb
#   NF3    0.2100 W/m²/ppb
#   C4F10  0.3600 W/m²/ppb
#   C6F14  0.4900 W/m²/ppb
#
# Earth-reference lifetimes are assumptions; reduced sunlight alone does not
# establish the sign or magnitude of Mars photochemical lifetime changes.

COMPOUNDS: dict[str, CompoundProperties] = {
    "CF4": CompoundProperties(
        molecular_weight_g_mol  = 88.0,
        atmospheric_lifetime_yr = 50_000.0,
        rf_efficiency_W_m2_ppb  = 0.0880,
        gwp100                  = 6_630,
        description             = "Carbon tetrafluoride (CFC-14)",
    ),
    "C2F6": CompoundProperties(
        molecular_weight_g_mol  = 138.0,
        atmospheric_lifetime_yr = 10_000.0,
        rf_efficiency_W_m2_ppb  = 0.2600,
        gwp100                  = 11_100,
        description             = "Hexafluoroethane (CFC-116)",
    ),
    "C3F8": CompoundProperties(
        molecular_weight_g_mol  = 188.0,
        atmospheric_lifetime_yr = 2_600.0,
        rf_efficiency_W_m2_ppb  = 0.2400,
        gwp100                  = 8_900,
        description             = "Octafluoropropane (CFC-218)",
    ),
    "SF6": CompoundProperties(
        molecular_weight_g_mol  = 146.1,
        atmospheric_lifetime_yr = 3_200.0,
        rf_efficiency_W_m2_ppb  = 0.5700,
        gwp100                  = 23_900,
        description             = "Sulfur hexafluoride",
    ),
    "NF3": CompoundProperties(
        molecular_weight_g_mol  = 71.0,
        atmospheric_lifetime_yr = 500.0,
        rf_efficiency_W_m2_ppb  = 0.2100,
        gwp100                  = 16_100,
        description             = "Nitrogen trifluoride",
    ),
    "C4F10": CompoundProperties(
        molecular_weight_g_mol  = 238.0,
        atmospheric_lifetime_yr = 2_600.0,
        rf_efficiency_W_m2_ppb  = 0.3600,
        gwp100                  = 8_860,
        description             = "Decafluorobutane (CFC-31-10)",
    ),
    "C6F14": CompoundProperties(
        molecular_weight_g_mol  = 338.0,
        atmospheric_lifetime_yr = 3_200.0,
        rf_efficiency_W_m2_ppb  = 0.4900,
        gwp100                  = 9_300,
        description             = "Tetradecafluorohexane (CFC-41-12)",
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
