"""Mars physical constants — pure Python, backend-neutral.

This module is the **single source of truth** for every Mars-specific physical
quantity in the project.  It deliberately imports nothing but ``math`` and the
standard library:

    >>> import src.celestials.planets.mars.constants as mc   # no torch, no jax

Why the isolation matters
-------------------------
The constants used to live as module-level ``torch.Tensor`` objects inside
``mars.py``, which made them unreadable to any non-PyTorch backend.  Keeping
them as plain ``float`` means the same numbers can be handed to:

  * the PyTorch 0-D box model — via ``Mars.constant_tensors()`` in
    :mod:`src.celestials.planets.mars.planet`, which converts on demand;
  * a JAX GCM (``jcm``) — via :meth:`MarsConstants.as_jcm_overrides`, which
    maps them onto ``jcm.constants.PhysicalConstants`` field names;
  * plain NumPy analysis, notebooks, and tests, with no framework import.

Nothing here allocates a tensor, touches a device, or depends on dtype.

Sources
-------
Bulk, orbital and rotational values: NASA Mars Fact Sheet
https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html

Values that are project calibrations rather than measurements (notably
:attr:`MarsConstants.polar_cap_fraction` and the thermal-tide pair) carry their
own justification in the field comment.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping

__all__ = [
    "MarsConstants",
    "MARS",
    "MARS_DEFAULT_COMPOSITION_PA",
    "SOLAR_CONSTANT_1AU_W_M2",
    "ASTRONOMICAL_UNIT_M",
    "UNIVERSAL_GAS_CONSTANT",
    "MOLAR_MASS_CO2",
]

# ── Universal reference values (not Mars-specific) ───────────────────────────
# Mirrors of the torch scalars in ``src.constants``; duplicated as floats so
# this module stays import-free.  Keep the two in sync.
SOLAR_CONSTANT_1AU_W_M2: float = 1361.0        # W m⁻²
ASTRONOMICAL_UNIT_M: float = 1.495978707e11    # m (IAU 2012 definition)
UNIVERSAL_GAS_CONSTANT: float = 8.314462618    # J mol⁻¹ K⁻¹ (CODATA 2018)
MOLAR_MASS_CO2: float = 0.0440095              # kg mol⁻¹

# ── Present-day atmospheric composition ──────────────────────────────────────
# Partial pressures in Pa summing to ~608 Pa.  ``Mars.__init__`` rescales these
# to whatever total surface pressure it is constructed with, so the ratios —
# not the absolute values — are what matter here.
MARS_DEFAULT_COMPOSITION_PA: Mapping[str, float] = MappingProxyType({
    "CO2": 580.0,
    "N2":   15.0,
    "Ar":   12.0,
    "O2":    0.8,
    "CO":    0.4,
})


@dataclass(frozen=True)
class MarsConstants:
    """An immutable, framework-neutral bundle of Mars's physical constants.

    Every field is a plain ``float`` in SI units; the unit is encoded in the
    field name so there is no ambiguity at the call site.  Derived quantities
    (rotation rate, gas constant, κ, insolation) are exposed as ``@property``
    so they can never drift out of sync with the base values they come from.

    The dataclass is frozen, so a variant is made with :func:`dataclasses.replace`
    rather than mutation::

        >>> from dataclasses import replace
        >>> thick = replace(MARS, polar_cap_fraction=0.08)
        >>> MARS.polar_cap_fraction          # the default is untouched
        0.04
    """

    # ── Bulk properties ──────────────────────────────────────────────────────
    mass_kg: float = 6.4171e23
    radius_m: float = 3.3895e6
    gravity_m_s2: float = 3.72076

    # ── Rotation ─────────────────────────────────────────────────────────────
    # Two distinct periods, and the distinction is load-bearing:
    #   solar_day_s    — one sol, noon-to-noon.  Drives the *diurnal cycle*
    #                    (insolation, thermal tide) in the 0-D box model.
    #   sidereal_day_s — one rotation w.r.t. the fixed stars.  This is the one
    #                    that yields Ω for a dynamical core; using the solar day
    #                    instead gives an Ω that is 0.15 % low.
    solar_day_s: float = 88_775.244        # 24h 39m 35.244s
    sidereal_day_s: float = 88_642.663     # 24h 37m 22.663s

    # ── Orbit ────────────────────────────────────────────────────────────────
    semi_major_axis_m: float = 2.27939200e11   # 1.5237 AU
    eccentricity: float = 0.0934
    orbital_period_s: float = 5.93568e7        # ~687 Earth days
    axial_tilt_deg: float = 25.19
    # Solar longitude at perihelion.  Ls ≈ 251° places perihelion in southern
    # summer, which is why the southern seasonal cap is the more dynamic one.
    ls_perihelion_deg: float = 251.0

    # ── Surface / thermal ────────────────────────────────────────────────────
    surface_emissivity: float = 0.95           # regolith, broadband IR
    # Bulk effective heat capacity per unit area of the thermally active layer.
    # Not the standard thermal inertia I (J m⁻² K⁻¹ s^-½) — it is the lumped
    # C of the 0-D slab model, hence the different units.
    thermal_inertia_j_k_m2: float = 6.0e4      # J K⁻¹ m⁻²
    diurnal_swing_amp_k: float = 50.0          # K, peak-to-mean surface swing
    # CO₂ atmospheric scale height, H = R T / g with T ≈ 210 K.  Used to
    # elevation-correct the surface pressure: P(z) = P_ref · exp(−z / H).
    scale_height_m: float = 11_100.0

    # ── CO₂ cycle ────────────────────────────────────────────────────────────
    co2_frost_point_k: float = 149.0           # K, at present-day ~610 Pa
    co2_latent_heat_j_kg: float = 5.7e5        # J kg⁻¹, sublimation
    # Effective fractional surface area of the seasonal CO₂ cap, per pole.
    # Sets the amplitude of the seasonal pressure swing.  0.01 gave only ~6 %
    # swing; the Viking Landers observe ~25–30 % (Hess et al. 1980; Tillman
    # et al. 1993), reproduced by MCD 6.1.  0.04 gives ~27 % at a stable
    # ~5.9 mb mean.  Effective area — it folds in partial coverage and
    # sublimation efficiency — and is still conservative w.r.t. the seasonal
    # cap's true mid-latitude reach.
    polar_cap_fraction: float = 0.04

    # ── Atmosphere ───────────────────────────────────────────────────────────
    thermal_tide_pa: float = 30.0              # Pa, diurnal tide amplitude
    thermal_tide_phase_rad: float = -0.7 * math.pi
    # Present-day atmospheric escape to space, MAVEN-derived.
    maven_escape_rate_kg_s: float = 0.2        # kg s⁻¹
    # Specific heat of CO₂ at constant pressure, near 200 K.  Together with
    # the derived gas constant this fixes κ = R/cp = 0.245 for a CO₂
    # atmosphere — materially different from Earth's diatomic 2/7 = 0.286.
    co2_cp_j_kg_k: float = 769.8

    composition_pa: Mapping[str, float] = field(
        default_factory=lambda: MARS_DEFAULT_COMPOSITION_PA
    )

    # ── Derived: angles ──────────────────────────────────────────────────────
    @property
    def axial_tilt_rad(self) -> float:
        """Obliquity in radians."""
        return math.radians(self.axial_tilt_deg)

    @property
    def ls_perihelion_rad(self) -> float:
        """Solar longitude at perihelion, in radians."""
        return math.radians(self.ls_perihelion_deg)

    # ── Derived: rotation ────────────────────────────────────────────────────
    @property
    def rotation_rate_rad_s(self) -> float:
        """Ω = 2π / T_sidereal (rad s⁻¹) — ≈ 7.0882 × 10⁻⁵.

        Deliberately built from the *sidereal* day.  A dynamical core's
        Coriolis term needs rotation w.r.t. inertial space, not w.r.t. the Sun.
        """
        return 2.0 * math.pi / self.sidereal_day_s

    # ── Derived: thermodynamics ──────────────────────────────────────────────
    @property
    def co2_gas_constant_j_kg_k(self) -> float:
        """Specific gas constant of CO₂, R = R_universal / M (≈ 188.92)."""
        return UNIVERSAL_GAS_CONSTANT / MOLAR_MASS_CO2

    @property
    def kappa(self) -> float:
        """κ = R/cp for a CO₂ atmosphere (≈ 0.2454).

        Earth's value is 2/7 ≈ 0.2857 (diatomic).  Any code that assumes the
        diatomic value on Mars misplaces the dry adiabat.
        """
        return self.co2_gas_constant_j_kg_k / self.co2_cp_j_kg_k

    # ── Derived: geometry and insolation ─────────────────────────────────────
    @property
    def surface_area_m2(self) -> float:
        """Total surface area, 4πR²."""
        return 4.0 * math.pi * self.radius_m ** 2

    @property
    def semi_major_axis_au(self) -> float:
        """Semi-major axis in astronomical units (≈ 1.5237)."""
        return self.semi_major_axis_m / ASTRONOMICAL_UNIT_M

    @property
    def solar_constant_w_m2(self) -> float:
        """Insolation at the semi-major axis, S₀ / a² (≈ 586 W m⁻²).

        This is the flux at r = a.  It is *not* the orbit-time-averaged value,
        which is larger by 1/√(1 − e²) — see :attr:`mean_solar_flux_w_m2`.
        Over one Mars year the instantaneous flux varies by roughly ±19 %
        about this figure because e = 0.0934.
        """
        return SOLAR_CONSTANT_1AU_W_M2 / self.semi_major_axis_au ** 2

    @property
    def mean_solar_flux_w_m2(self) -> float:
        """Orbit-time-averaged insolation, S₀ / (a²·√(1 − e²)) (≈ 589 W m⁻²).

        The time average of 1/r² over an orbit exceeds 1/a² because the planet
        lingers near aphelion but the flux is weighted toward perihelion.
        """
        return self.solar_constant_w_m2 / math.sqrt(1.0 - self.eccentricity ** 2)

    # ── Backend hand-off ─────────────────────────────────────────────────────
    def as_jcm_overrides(self) -> dict[str, float]:
        """Map these constants onto ``jcm.constants.PhysicalConstants`` fields.

        Returns a plain ``dict`` of keyword overrides, so this module still
        imports nothing.  Apply it at the call site::

            import jcm.constants as jc
            jc.set_constants(**MARS.as_jcm_overrides())

        Only the fields whose Earth defaults are *wrong for Mars* are
        overridden; universal constants (Stefan-Boltzmann, von Kármán,
        Boltzmann) are left alone.  Water-vapour constants (``rv``, ``eps``,
        ``alhc``) are also left at their Earth values: they are physically
        meaningless in a dry CO₂ atmosphere, and overriding them would imply a
        moist Mars the radiation code does not model.
        """
        return {
            "rearth": self.radius_m,
            "omega": self.rotation_rate_rad_s,
            "grav": self.gravity_m_s2,
            "cpd": self.co2_cp_j_kg_k,
            "akap": self.kappa,
            "solc": self.mean_solar_flux_w_m2,
            "alhs": self.co2_latent_heat_j_kg,
            # Reference pressures: on Mars the thermodynamic reference and the
            # mean surface pressure are the same ~610 Pa, unlike Earth where
            # p0 (1000 hPa) and p0s1_bg (1013.25 hPa) differ.
            "p0": self.total_composition_pa,
            "p0s1_bg": self.total_composition_pa,
        }

    @property
    def total_composition_pa(self) -> float:
        """Sum of the default partial pressures (≈ 608 Pa)."""
        return float(sum(self.composition_pa.values()))


#: The canonical present-day Mars.  Import this rather than constructing
#: ``MarsConstants()`` at each call site, so identity comparisons hold.
MARS: MarsConstants = MarsConstants()
