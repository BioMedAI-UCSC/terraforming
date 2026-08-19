"""Regression tests for the staged Ames 12-band dry-CO2 coefficients."""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402

from src.framework.physics import ames_radiation  # noqa: E402
from src.framework.gcm._dinosaur import jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_staged_table_dimensions_weights_and_source_hashes():
    data = ames_radiation.load_ames_co2_tables()
    assert data["log10_k_ir"].shape == (7, 11, 5, 16)
    assert data["log10_k_sw"].shape == (7, 11, 7, 16)
    assert np.sum(data["gauss_weights"]) == pytest.approx(1.0)
    assert np.sum(data["solar_weights"]) == pytest.approx(1.0)
    assert np.allclose(np.sum(data["planck_fraction_ir"], axis=1), 1.0)
    assert str(data["source_ir_sha256"]) == (
        "5a75862bbc40a328dd55071edc07b9ec34c8834bb487196510ba1542dd530a46"
    )
    assert str(data["source_sw_sha256"]) == (
        "784f04c52e13b8ebc30a4dd5cf3d7392857b30204a36c33b60af34994b8af089"
    )


def test_decoded_coefficient_anchors_match_ames_records():
    data = ames_radiation.load_ames_co2_tables()
    assert data["log10_k_ir"][3, 6, 2, 7] == pytest.approx(-21.723412980814487)
    assert data["log10_k_sw"][3, 6, 4, 11] == pytest.approx(-27.634231050092964)
    # The shortest-wave Ames solar interval is 94% clear spectrum.
    assert data["clear_fraction_sw"][-1] == pytest.approx(0.94)


def test_correlated_k_optical_depth_is_finite_positive_and_differentiable():
    temperature = jnp.full((3, 2, 2), 200.0)
    pressure = jnp.array([0.5, 2.0, 5.0])[:, None, None] * 100.0
    delta_pressure = jnp.full((3, 2, 2), 200.0)
    sw, ir = ames_radiation.correlated_k_optical_depths(
        temperature, pressure, delta_pressure
    )
    assert sw.shape == (7, 17, 3, 2, 2)
    assert ir.shape == (5, 17, 3, 2, 2)
    assert np.isfinite(np.asarray(sw)).all() and np.isfinite(np.asarray(ir)).all()
    assert np.asarray(sw).min() >= 0.0 and np.asarray(ir).min() >= 0.0
    assert np.asarray(sw[:, :-1]).max() > 0.0
    assert np.asarray(ir[:, :-1]).max() > 0.0
    assert np.array_equal(np.asarray(sw[:, -1]), np.zeros_like(np.asarray(sw[:, -1])))

    def loss(offset):
        _, optical_depth = ames_radiation.correlated_k_optical_depths(
            temperature + offset, pressure, delta_pressure
        )
        return jnp.mean(optical_depth)

    gradient = float(jax.grad(loss)(0.0))
    assert math.isfinite(gradient)
    assert abs(gradient) > 0.0


def test_channel_weights_preserve_each_spectral_band():
    data = ames_radiation.load_ames_co2_tables()
    for key in ("clear_fraction_sw", "clear_fraction_ir"):
        weights = np.asarray(ames_radiation.channel_weights(data[key]))
        assert np.allclose(weights.sum(axis=1), 1.0)
