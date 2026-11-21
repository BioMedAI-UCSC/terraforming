"""Pressure-grid closure and differentiability, independent of native compilation."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.vertical_grid import (pressure_interfaces,sigma_boundaries,supported_layers,
    load_grids,initialization_reference_interfaces)

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('layers',supported_layers())
def test_hybrid_interfaces_close_column_and_sigma_adaptation_is_explicit(layers):
    pressure=np.asarray(pressure_interfaces(layers,705.,jnp.array([400.,705.,1000.])))
    assert np.all(np.diff(pressure,axis=0)>0.)
    np.testing.assert_array_equal(pressure[-1],[400.,705.,1000.])
    np.testing.assert_allclose(np.diff(pressure,axis=0).sum(axis=0),pressure[-1]-pressure[0],rtol=2e-15)
    sigma=sigma_boundaries(layers)
    np.testing.assert_allclose(sigma[1:]*705.,pressure[1:,1],rtol=2e-15)
    assert sigma[0]==0. and pressure[0,1]>0.
    assert abs(np.diff(sigma).sum()-1.)<3e-16
    if layers==56:
        assert .003<sigma[1]*705./2.<.004
        assert 1.-sigma[-2]<.0005


def test_hybrid_pressure_gradients_match_surface_and_reference_pressure_equations():
    layers=56
    surface=jax.jit(jax.jacrev(lambda p:pressure_interfaces(layers,705.,p)))(650.)
    reference=jax.jit(jax.jacrev(lambda p:pressure_interfaces(layers,p,650.)))(705.)
    data=load_grids()
    np.testing.assert_allclose(surface,data['b56'],rtol=2e-15,atol=0.)
    np.testing.assert_allclose(reference,data['a56']/725.,rtol=2e-15,atol=0.)
    epsilon=.01
    finite=(pressure_interfaces(layers,705.,650.+epsilon)-pressure_interfaces(layers,705.,650.-epsilon))/(2*epsilon)
    np.testing.assert_allclose(surface,finite,rtol=3e-11,atol=1e-12)


def test_unsupported_grids_and_invalid_pressure_are_not_silently_replaced():
    for layers in (4,0,56.5,True):
        with pytest.raises(ValueError,match='released vertical-grid'):
            sigma_boundaries(layers)
    assert np.isnan(pressure_interfaces(56,705.,0.)).all()
    assert np.isnan(pressure_interfaces(56,-1.,705.)).all()


@pytest.mark.parametrize('layers',supported_layers())
@pytest.mark.parametrize('reference',[610.,705.,50000.])
def test_initialization_selectors_use_native_reference_interfaces(layers,reference):
    actual=initialization_reference_interfaces(sigma_boundaries(layers),reference)
    np.testing.assert_allclose(actual,pressure_interfaces(layers,reference,reference),rtol=2e-15)
    assert actual[0]>0. and actual[-1]==reference


def test_custom_sigma_initialization_keeps_supplied_boundaries():
    sigma=np.array([0.,.1,.5,1.])
    np.testing.assert_array_equal(initialization_reference_interfaces(sigma,705.),sigma*705.)


@pytest.mark.parametrize('sigma,reference',[
    ([],705.),([0.,1.,.5],705.),([0.,.5],705.),([-1.,1.],705.),
    ([0.,np.nan,1.],705.),([0.,1.],0.),([0.,1.],np.inf)])
def test_initialization_selectors_reject_invalid_grid_or_reference(sigma,reference):
    with pytest.raises(ValueError): initialization_reference_interfaces(sigma,reference)


@pytest.mark.parametrize('layers',supported_layers())
def test_radiation_grid_retains_native_positive_top_and_order(layers):
    from src.framework.physics.vertical_grid import radiation_interfaces
    ps=jnp.array([100.,705.,1200.])
    half=radiation_interfaces(sigma_boundaries(layers),ps)
    assert np.all(half[0]>0.) and np.all(np.diff(half,axis=0)>0.)
    np.testing.assert_allclose(half[:,1],pressure_interfaces(layers,705.,705.),rtol=2e-15)
    np.testing.assert_array_equal(half[-1],ps)
