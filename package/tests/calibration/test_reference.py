"""Tests for src.calibration.reference.

Covers the reference-climatology fixtures: shapes, latitude match to the model's
default site, provenance fields, the registry, and validation errors.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.calibration.reference import (
    ReferenceClimatology,
    get_reference,
    viking_lander_1,
    viking_lander_2,
)


class TestVikingLander1:

    def test_latitude_matches_model_default_site(self):
        # VL1 at 22.3 N ~ the simulator's default 22 N site — the comparison is
        # latitude-consistent, which is why VL1 is the primary target.
        assert viking_lander_1().latitude_deg == pytest.approx(22.3, abs=0.5)

    def test_pressure_curve_shape_and_units(self):
        vl1 = viking_lander_1()
        assert vl1.pressure_pa.shape == vl1.ls_deg.shape == (12,)
        # Annual mean ~7.9 hPa; canonical ~26 % seasonal swing.
        assert np.mean(vl1.pressure_pa) == pytest.approx(780.0, abs=30.0)
        swing = (vl1.pressure_pa.max() - vl1.pressure_pa.min()) / np.mean(vl1.pressure_pa)
        assert 0.2 < swing < 0.35

    def test_minimum_near_ls_150_maximum_near_ls_270(self):
        vl1 = viking_lander_1()
        assert vl1.ls_deg[int(np.argmin(vl1.pressure_pa))] == pytest.approx(150.0)
        assert vl1.ls_deg[int(np.argmax(vl1.pressure_pa))] == pytest.approx(270.0)

    def test_provenance_is_recorded(self):
        vl1 = viking_lander_1()
        assert vl1.source == "viking-lander"
        assert "Hess" in vl1.citation
        assert vl1.temperature_k is None  # no fabricated T climatology


class TestRegistry:

    def test_get_reference_known(self):
        assert get_reference("vl1").name == "Viking Lander 1"
        assert get_reference("vl2").name == "Viking Lander 2"

    def test_get_reference_unknown_raises(self):
        with pytest.raises(KeyError):
            get_reference("nope")

    def test_vl2_higher_latitude_larger_swing(self):
        vl1, vl2 = viking_lander_1(), viking_lander_2()
        assert vl2.latitude_deg > vl1.latitude_deg
        amp1 = vl1.pressure_pa.max() - vl1.pressure_pa.min()
        amp2 = vl2.pressure_pa.max() - vl2.pressure_pa.min()
        assert amp2 > amp1  # poleward sites swing more


class TestValidation:

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            ReferenceClimatology(
                name="bad", latitude_deg=0.0, elevation_m=0.0,
                ls_deg=np.arange(0.0, 360.0, 30.0),
                pressure_pa=np.array([1.0, 2.0]),  # wrong length
                temperature_k=None, source="test", citation="",
            )

    def test_requires_at_least_one_field(self):
        with pytest.raises(ValueError):
            ReferenceClimatology(
                name="empty", latitude_deg=0.0, elevation_m=0.0,
                ls_deg=np.arange(0.0, 360.0, 30.0),
                pressure_pa=None, temperature_k=None, source="test", citation="",
            )
