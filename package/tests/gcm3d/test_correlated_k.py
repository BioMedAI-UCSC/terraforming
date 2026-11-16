"""Regression tests for the staged Ames 12-band dry-CO2 coefficients."""

from __future__ import annotations

import math
import importlib.util
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("dinosaur")
import jax  # noqa: E402

from src.framework.physics import correlated_k  # noqa: E402
from src.framework.gcm._dinosaur import jnp  # noqa: E402

jax.config.update("jax_enable_x64", True)


def test_staged_table_dimensions_weights_and_source_hashes():
    data = correlated_k.load_ames_co2_tables()
    assert data["log10_k_ir"].shape == (7, 11, 5, 16)
    assert data["log10_k_sw"].shape == (7, 11, 7, 16)
    assert data["log10_k_ir_ames"].shape == (7,51,5,16)
    assert data["log10_k_sw_ames"].shape == (7,51,7,16)
    assert data["log10_k_ir_mixtures_ames"].shape == (7,51,10,5,16)
    assert data["log10_k_sw_mixtures_ames"].shape == (7,51,10,7,16)
    for kind in ("ir","sw"):
        np.testing.assert_allclose(data[f"log10_k_{kind}_ames"][:,::5],
            data[f"log10_k_{kind}"],rtol=2e-14,atol=2e-14)
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
    data = correlated_k.load_ames_co2_tables()
    assert data["log10_k_ir"][3, 6, 2, 7] == pytest.approx(-21.723412980814487)
    assert data["log10_k_sw"][3, 6, 4, 11] == pytest.approx(-27.634231050092964)
    # The shortest-wave Ames solar interval is 94% clear spectrum.
    assert data["clear_fraction_sw"][-1] == pytest.approx(0.94)


@pytest.mark.parametrize('linear',[False,True])
def test_correlated_k_optical_depth_is_finite_positive_and_differentiable(linear):
    temperature = jnp.full((3, 2, 2), 200.0)
    pressure = jnp.array([0.5, 2.0, 5.0])[:, None, None] * 100.0
    delta_pressure = jnp.full((3, 2, 2), 200.0)
    sw, ir = correlated_k.correlated_k_optical_depths(
        temperature, pressure, delta_pressure,linear_k_interpolation=linear
    )
    assert sw.shape == (7, 17, 3, 2, 2)
    assert ir.shape == (5, 17, 3, 2, 2)
    assert np.isfinite(np.asarray(sw)).all() and np.isfinite(np.asarray(ir)).all()
    assert np.asarray(sw).min() >= 0.0 and np.asarray(ir).min() >= 0.0
    assert np.asarray(sw[:, :-1]).max() > 0.0
    assert np.asarray(ir[:, :-1]).max() > 0.0
    assert np.array_equal(np.asarray(sw[:, -1]), np.zeros_like(np.asarray(sw[:, -1])))

    def loss(offset):
        _, optical_depth = correlated_k.correlated_k_optical_depths(
            temperature + offset, pressure, delta_pressure,linear_k_interpolation=linear
        )
        return jnp.mean(optical_depth)

    gradient = float(jax.grad(loss)(0.0))
    assert math.isfinite(gradient)
    assert abs(gradient) > 0.0


def test_linear_k_mixed_water_reverse_derivative_and_empty_zero_top():
    def objective(scale):
        sw,ir=correlated_k.correlated_k_optical_depths(jnp.array([170.03,203.03])*scale,
            jnp.array([0.,243.])*scale,jnp.array([0.,113.])*scale,
            water_mass_fraction=.003*scale,linear_k_interpolation=True)
        return jnp.sum(sw)+jnp.sum(ir)
    automatic=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(automatic) and automatic!=0.
    for epsilon in (1e-5,1e-6):
        np.testing.assert_allclose(automatic,(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon),rtol=2e-7)
    sw,ir=correlated_k.correlated_k_optical_depths(jnp.array([170.03,203.03]),
        jnp.array([0.,243.]),jnp.array([0.,113.]),linear_k_interpolation=True)
    for value in (sw,ir):
        assert np.isfinite(value).all()
        np.testing.assert_array_equal(value[:,:,0],0.)


def test_linear_k_last_water_bin_avoids_zero_weight_out_of_bounds_read():
    def lookup(water):
        return correlated_k.correlated_k_optical_depths(203.03,243.,113.,
            water_mass_fraction=water,linear_k_interpolation=True)
    for limit,outside in ((1e-7,0.),(.4,.5)):
        for actual,expected in zip(lookup(outside),lookup(limit),strict=True):
            assert np.isfinite(actual).all()
            np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('mode',['15band','extended'])
def test_linear_k_rejects_incompatible_bands(mode):
    from src.framework.physics.radiative_transfer import make_radiation_component
    with pytest.raises(ValueError,match='12-band'):
        correlated_k.correlated_k_optical_depths(200.,300.,100.,mode=mode,linear_k_interpolation=True)
    with pytest.raises(ValueError,match='12-band'):
        make_radiation_component(lambda *args:None,radiation_mode=mode,linear_k_interpolation=True)


def test_channel_weights_preserve_each_spectral_band():
    data = correlated_k.load_ames_co2_tables()
    for key in ("clear_fraction_sw", "clear_fraction_ir"):
        weights = np.asarray(correlated_k.channel_weights(data[key]))
        assert np.allclose(weights.sum(axis=1), 1.0)


def test_ames_pressure_staging_reproduces_cubic_and_preserves_source_points():
    path=Path(__file__).resolve().parents[3]/'scripts/stage_co2_radiation_tables.py'
    spec=importlib.util.spec_from_file_location('stage_co2_radiation_tables',path)
    stage=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage)
    axis=np.log10(stage.P_REF_MBAR)
    polynomial=lambda p:-20.+.1*p+.01*p*p+.001*p*p*p
    table=polynomial(axis)[None,:,None,None]
    refined=stage.ames_pressure_interpolation(table)
    np.testing.assert_allclose(refined[0,:,0,0],polynomial(stage.P_FINE_LOG_MBAR),rtol=2e-14,atol=2e-14)


def test_native_pressure_grid_changes_off_grid_opacity_and_preserves_gradients():
    data=correlated_k.load_ames_co2_tables()
    def lookup(temperature,pressure):
        return correlated_k._interp2_log_table(data['log10_k_ir_ames'],
            data['temperature_k'],data['pressure_mbar_ames'],temperature,pressure)
    actual=lookup(210.,3.)
    old=correlated_k._interp2_log_table(data['log10_k_ir'],
        data['temperature_k'],data['pressure_mbar'],210.,3.)
    assert float(jnp.max(jnp.abs(actual-old)))>1e-3
    def objective(scale):
        return jnp.sum(lookup(210.*scale,3.*scale))
    auto=jax.grad(objective)(1.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-7)


@pytest.mark.parametrize('mode',['12band','15band','extended'])
def test_absolute_planck_nodes_units_and_reverse_derivative(mode):
    data=correlated_k.load_radiation_tables(mode)
    for temperature in (50.,150.,250.,350.,900.):
        index=int(round((temperature-50.)*10.))
        actual=correlated_k.planck_band_intensities(temperature,mode=mode)
        np.testing.assert_allclose(actual,data['ames_planck_intensity_ir'][index],rtol=2e-14,atol=2e-15)
    def integrated(temperature):
        return jnp.sum(correlated_k.planck_band_intensities(temperature,mode=mode)*
            jnp.asarray(np.diff(data['ir_bounds_cm1'])))*jnp.pi
    assert 50.<float(integrated(250.))<500.
    auto=jax.grad(integrated)(250.05)
    for eps in (1e-4,1e-5,1e-6):
        finite=(integrated(250.05+eps)-integrated(250.05-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7)
