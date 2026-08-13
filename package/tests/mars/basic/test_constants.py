"""Tests for src/celestials/planets/mars/constants.py and its torch factories.

Covers:
  - MarsConstants: base values, immutability, dataclasses.replace variants
  - Derived properties: rotation rate, kappa, gas constant, insolation, area
  - as_jcm_overrides: field names and values a JAX GCM consumes
  - Framework isolation: constants.py must not import torch
  - constants_to_tensors / composition_to_tensors: dtype, device, values
  - Mars(constants=...): the model honours a non-default constant bundle
"""

from __future__ import annotations

import importlib
import math
from dataclasses import FrozenInstanceError, replace

import pytest
import torch

from src.celestials import Mars
from src.celestials.planets.mars.constants import (
    MARS,
    MARS_DEFAULT_COMPOSITION_PA,
    SOLAR_CONSTANT_1AU_W_M2,
    MarsConstants,
)
from src.celestials.planets.mars.planet import (
    composition_to_tensors,
    constants_to_tensors,
)

# ── Framework isolation ───────────────────────────────────────────────────────


class TestFrameworkIsolation:
    """The whole point of the split: constants.py must stay backend-neutral."""

    def test_constants_module_does_not_import_torch(self):
        """A JAX/NumPy backend must be able to read the constants torch-free."""
        source_path = importlib.util.find_spec(
            "src.celestials.planets.mars.constants"
        ).origin
        with open(source_path, encoding="utf-8") as fh:
            source = fh.read()

        for line in source.splitlines():
            stripped = line.strip()
            assert not stripped.startswith("import torch")
            assert not stripped.startswith("from torch")
            assert not stripped.startswith("import jax")

    def test_every_base_field_is_a_plain_float(self):
        """No tensors, no numpy scalars — plain Python floats only."""
        for name, value in vars(MARS).items():
            if name == "composition_pa":
                continue
            assert type(value) is float, f"{name} is {type(value)}, expected float"


# ── MarsConstants — base values ───────────────────────────────────────────────


class TestMarsConstantsBaseValues:

    def test_bulk_properties_match_nasa_fact_sheet(self):
        """Mass, radius, gravity against the NASA Mars Fact Sheet."""
        assert MARS.mass_kg == pytest.approx(6.4171e23, rel=1e-4)
        assert MARS.radius_m == pytest.approx(3.3895e6, rel=1e-4)
        assert MARS.gravity_m_s2 == pytest.approx(3.72076, rel=1e-5)

    def test_solar_day_is_longer_than_sidereal_day(self):
        """A sol exceeds one rotation because Mars advances along its orbit."""
        assert MARS.solar_day_s > MARS.sidereal_day_s
        # The gap is ~132.6 s; a prograde planet gains one rotation per orbit.
        assert MARS.solar_day_s - MARS.sidereal_day_s == pytest.approx(132.6, abs=1.0)

    def test_eccentricity_is_bounded(self):
        """Mars is on a closed elliptical orbit, so 0 <= e < 1."""
        assert 0.0 <= MARS.eccentricity < 1.0

    def test_emissivity_and_cap_fraction_are_fractions(self):
        assert 0.0 < MARS.surface_emissivity <= 1.0
        assert 0.0 < MARS.polar_cap_fraction < 1.0

    def test_default_composition_sums_to_present_day_pressure(self):
        """Partial pressures reproduce the ~610 Pa present-day atmosphere."""
        assert MARS.total_composition_pa == pytest.approx(608.2, abs=10.0)
        assert MARS.composition_pa["CO2"] / MARS.total_composition_pa > 0.94


# ── MarsConstants — immutability ──────────────────────────────────────────────


class TestImmutability:

    def test_assignment_raises_on_frozen_dataclass(self):
        """MARS is a shared singleton; mutating it would corrupt every caller."""
        with pytest.raises(FrozenInstanceError):
            MARS.gravity_m_s2 = 9.81  # type: ignore[misc]

    def test_replace_produces_a_variant_without_touching_the_default(self):
        thick = replace(MARS, polar_cap_fraction=0.08)
        assert thick.polar_cap_fraction == 0.08
        assert MARS.polar_cap_fraction == 0.04

    def test_default_composition_mapping_rejects_mutation(self):
        with pytest.raises(TypeError):
            MARS_DEFAULT_COMPOSITION_PA["CO2"] = 1.0  # type: ignore[index]


# ── MarsConstants — derived properties ────────────────────────────────────────


class TestDerivedProperties:

    def test_rotation_rate_uses_sidereal_not_solar_day(self):
        """Omega must come from the sidereal day.

        Regression: the solar day (88 775.244 s) gives 7.0776e-5, which is
        0.15 % low and misplaces the Coriolis term in any dynamical core.
        """
        assert MARS.rotation_rate_rad_s == pytest.approx(7.0882e-5, rel=1e-4)
        solar_omega = 2.0 * math.pi / MARS.solar_day_s
        assert MARS.rotation_rate_rad_s != pytest.approx(solar_omega, rel=1e-6)

    def test_co2_gas_constant(self):
        """R = R_universal / M_CO2 ≈ 188.92 J kg⁻¹ K⁻¹."""
        assert MARS.co2_gas_constant_j_kg_k == pytest.approx(188.92, rel=1e-4)

    def test_kappa_is_co2_not_diatomic(self):
        """κ = R/cp ≈ 0.245 for CO₂, distinctly below Earth's 2/7."""
        assert MARS.kappa == pytest.approx(0.2454, rel=1e-3)
        assert MARS.kappa < 2.0 / 7.0

    def test_angles_convert_degrees_to_radians(self):
        assert MARS.axial_tilt_rad == pytest.approx(math.radians(25.19))
        assert MARS.ls_perihelion_rad == pytest.approx(math.radians(251.0))

    def test_semi_major_axis_in_au(self):
        assert MARS.semi_major_axis_au == pytest.approx(1.5237, rel=1e-3)

    def test_solar_constant_follows_inverse_square_law(self):
        expected = SOLAR_CONSTANT_1AU_W_M2 / MARS.semi_major_axis_au ** 2
        assert MARS.solar_constant_w_m2 == pytest.approx(expected)
        assert MARS.solar_constant_w_m2 == pytest.approx(586.2, abs=2.0)

    def test_mean_flux_exceeds_flux_at_semi_major_axis(self):
        """Time-averaging 1/r² over an eccentric orbit raises the mean."""
        assert MARS.mean_solar_flux_w_m2 > MARS.solar_constant_w_m2
        assert MARS.mean_solar_flux_w_m2 == pytest.approx(588.8, abs=2.0)

    def test_surface_area(self):
        assert MARS.surface_area_m2 == pytest.approx(1.4441e14, rel=1e-3)

    def test_derived_values_track_a_replaced_base_value(self):
        """Properties recompute; they are never stale copies."""
        slow = replace(MARS, sidereal_day_s=MARS.sidereal_day_s * 2.0)
        assert slow.rotation_rate_rad_s == pytest.approx(
            MARS.rotation_rate_rad_s / 2.0
        )


# ── as_jcm_overrides ──────────────────────────────────────────────────────────


class TestJcmOverrides:

    def test_returns_plain_floats_under_jcm_field_names(self):
        overrides = MARS.as_jcm_overrides()
        for key in ("rearth", "omega", "grav", "cpd", "akap", "solc"):
            assert key in overrides
            assert type(overrides[key]) is float

    def test_values_match_the_source_constants(self):
        overrides = MARS.as_jcm_overrides()
        assert overrides["rearth"] == MARS.radius_m
        assert overrides["grav"] == MARS.gravity_m_s2
        assert overrides["omega"] == MARS.rotation_rate_rad_s
        assert overrides["akap"] == MARS.kappa

    def test_no_earth_values_leak_through(self):
        """Every overridden field must actually differ from Earth's default."""
        overrides = MARS.as_jcm_overrides()
        assert overrides["rearth"] != pytest.approx(6.371e6, rel=1e-3)
        assert overrides["grav"] != pytest.approx(9.81, rel=1e-3)
        assert overrides["akap"] != pytest.approx(2.0 / 7.0, rel=1e-3)

    def test_reference_pressures_are_martian(self):
        """~610 Pa, not Earth's 1e5 Pa."""
        overrides = MARS.as_jcm_overrides()
        assert overrides["p0"] == pytest.approx(608.2, abs=10.0)
        assert overrides["p0s1_bg"] == overrides["p0"]


# ── constants_to_tensors ──────────────────────────────────────────────────────


class TestConstantsToTensors:

    def test_returns_zero_dim_tensors_in_project_dtype(self):
        tensors = constants_to_tensors(MARS)
        for name, tensor in tensors.items():
            assert isinstance(tensor, torch.Tensor), name
            assert tensor.ndim == 0, name
            assert tensor.dtype is torch.float64, name

    def test_values_round_trip_from_the_float_source(self):
        tensors = constants_to_tensors(MARS)
        assert float(tensors["MARS_GRAVITY"]) == MARS.gravity_m_s2
        assert float(tensors["MARS_RADIUS"]) == MARS.radius_m
        assert float(tensors["MARS_ORBITAL_PERIOD"]) == MARS.orbital_period_s

    def test_angle_fields_are_converted_to_radians(self):
        """The legacy MARS_AXIAL_TILT tensor was in radians; preserve that."""
        tensors = constants_to_tensors(MARS)
        assert float(tensors["MARS_AXIAL_TILT"]) == pytest.approx(
            math.radians(25.19)
        )
        assert float(tensors["MARS_LS_PERIHELION"]) == pytest.approx(
            math.radians(251.0)
        )

    def test_rotation_period_exports_the_solar_day(self):
        """MARS_ROTATION_PERIOD has always meant one sol; keep it that way.

        Regression: swapping in the sidereal day here would silently shift
        every diurnal-cycle test by ~133 s per sol.
        """
        tensors = constants_to_tensors(MARS)
        assert float(tensors["MARS_ROTATION_PERIOD"]) == MARS.solar_day_s

    def test_honours_a_replaced_constant(self):
        variant = replace(MARS, gravity_m_s2=1.0)
        tensors = constants_to_tensors(variant)
        assert float(tensors["MARS_GRAVITY"]) == 1.0

    def test_respects_explicit_dtype(self):
        tensors = constants_to_tensors(MARS, dtype=torch.float32)
        assert tensors["MARS_GRAVITY"].dtype is torch.float32

    def test_allocates_on_the_requested_device(self):
        tensors = constants_to_tensors(MARS, device="cpu")
        assert tensors["MARS_GRAVITY"].device.type == "cpu"

    def test_legacy_names_are_all_present(self):
        """Existing import sites depend on these exact keys."""
        expected = {
            "MARS_MASS", "MARS_RADIUS", "MARS_GRAVITY", "MARS_ROTATION_PERIOD",
            "MARS_SEMI_MAJOR_AXIS", "MARS_ECCENTRICITY", "MARS_ORBITAL_PERIOD",
            "MARS_AXIAL_TILT", "MARS_LS_PERIHELION", "MARS_SURFACE_EMISSIVITY",
            "MARS_THERMAL_INERTIA", "MARS_MAVEN_ESCAPE_RATE",
            "MARS_CO2_FROST_POINT", "MARS_CO2_LATENT_HEAT",
            "MARS_POLAR_CAP_FRACTION", "MARS_DIURNAL_SWING_AMP",
            "MARS_THERMAL_TIDE_PA", "MARS_THERMAL_TIDE_PHASE",
        }
        assert set(constants_to_tensors(MARS)) == expected


class TestCompositionToTensors:

    def test_species_and_values_match_the_float_source(self):
        comp = composition_to_tensors(MARS)
        assert set(comp) == set(MARS.composition_pa)
        for species, tensor in comp.items():
            assert float(tensor) == MARS.composition_pa[species]

    def test_returns_a_fresh_dict_each_call(self):
        """Callers mutate composition during a run; they must not share state."""
        first, second = composition_to_tensors(MARS), composition_to_tensors(MARS)
        assert first is not second
        assert first["CO2"] is not second["CO2"]


# ── Mars honours the constant bundle ──────────────────────────────────────────


class TestMarsUsesConstants:

    def test_default_mars_exposes_the_canonical_bundle(self):
        assert Mars().constants is MARS

    def test_intrinsic_params_come_from_the_bundle(self):
        mars = Mars()
        assert float(mars.intrinsic_params.gravity) == MARS.gravity_m_s2
        assert float(mars.intrinsic_params.radius) == MARS.radius_m

    def test_a_replaced_constant_reaches_the_model(self):
        variant = replace(MARS, gravity_m_s2=1.0)
        mars = Mars(constants=variant)
        assert float(mars.intrinsic_params.gravity) == 1.0
        # The default is untouched for every other planet instance.
        assert float(Mars().intrinsic_params.gravity) == MARS.gravity_m_s2

    def test_scale_height_drives_the_elevation_correction(self):
        """P(z) = P_ref · exp(−z / H), with H taken from the bundle."""
        elevation = 5000.0
        mars = Mars(surface_pressure=610.0, elevation_m=elevation)
        expected = 610.0 * math.exp(-elevation / MARS.scale_height_m)
        assert float(mars.atmosphere.surface_pressure) == pytest.approx(
            expected, rel=1e-6
        )

    def test_ls_perihelion_anchors_the_initial_orbital_angle(self):
        """initial_ls_deg == Ls_perihelion must put the planet at perihelion."""
        mars = Mars(initial_ls_deg=MARS.ls_perihelion_deg)
        assert float(mars.orbital_angle) == pytest.approx(0.0, abs=1e-12)
