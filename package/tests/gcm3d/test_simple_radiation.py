import numpy as np
import pytest
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.simple_radiation import scatter,dust_depth,infrared,column_fluxes


def test_scattering_closes_flux_budget_and_has_clear_and_night_limits():
    tau=jnp.array([[.1,.2,.3],[.3,.4,.5],[.2,.1,.6]])
    mu=jnp.array([0.,.2,.8]);albedo=jnp.array([.1,.3,.9])
    result=scatter(tau,.92,.55,.55**2,mu,albedo)
    np.testing.assert_allclose(jnp.sum(result.heating_fraction,axis=0)+result.surface_fraction+
        result.outgoing_fraction,[0.,1.,1.],rtol=2e-6,atol=1e-7)
    np.testing.assert_array_equal(result.heating_fraction[:,0],0.)
    clear=scatter(jnp.zeros_like(tau),.92,.55,.55**2,jnp.ones(3),albedo)
    np.testing.assert_allclose(clear.surface_fraction,1.-albedo,rtol=2e-6)
    np.testing.assert_allclose(clear.outgoing_fraction,albedo,rtol=2e-6)
    thick=scatter(jnp.ones_like(tau)*1000.,.92,.55,.55**2,jnp.ones(3),albedo)
    assert all(np.isfinite(value).all() for value in thick)


def test_dust_normalization_and_infrared_energy_budget():
    ph=jnp.linspace(0.,705.,9);p=.5*(ph[1:]+ph[:-1])
    depth=dust_depth(ph,p,.7,.2,latitude_variation=True)
    np.testing.assert_allclose(jnp.sum(depth),.2*(.3+.7*np.cos(.7))*705./610.,rtol=2e-6)
    heating,up,down=infrared(jnp.linspace(140.,200.,8),190.,ph,p)
    np.testing.assert_allclose(jnp.sum(heating*jnp.diff(ph)*735.94/3.72),
        (up-down)[-1]-(up-down)[0],rtol=3e-6,atol=1e-5)
    with pytest.raises(ValueError,match='four levels'):
        infrared(jnp.ones(3)*180.,180.,ph[:4],p[:3],newton_damping=True)


def test_driver_ramp_emissivity_and_source_surface_flux_switch():
    ph=jnp.linspace(0.,705.,9);p=.5*(ph[1:]+ph[:-1]);t=jnp.full(8,180.)
    def flux(sols,emissivity=1.,**options):
        return column_fluxes(t,190.,ph,p,.5,610.,.3,emissivity,.7,sols,**options)
    start=flux(0.);middle=flux(2.5);full=flux(5.)
    np.testing.assert_array_equal(start.shortwave_down_w_m2,0.)
    np.testing.assert_allclose(middle.shortwave_down_w_m2,.5*full.shortwave_down_w_m2,rtol=2e-6)
    for actual,expected in zip(flux(5.,emissivity=0.),full,strict=True):
        np.testing.assert_array_equal(actual,expected)
    suppressed=flux(5.,no_downward_infrared=True)
    assert suppressed.longwave_down_w_m2[-1]==0.
    np.testing.assert_array_equal(suppressed.atmospheric_convergence_w_m2,full.atmospheric_convergence_w_m2)


def test_scattering_and_driver_keep_reverse_derivatives():
    ph=jnp.linspace(0.,705.,9);p=.5*(ph[1:]+ph[:-1]);t=jnp.linspace(140.,200.,8)
    def objective(opacity):
        flux=column_fluxes(t,190.,ph,p,.5,610.,.3,1.,.7,7.,optical_depth=opacity)
        return flux.shortwave_down_w_m2[-1]-flux.shortwave_up_w_m2[-1]
    automatic=jax.grad(objective)(.2)
    finite=(objective(.201)-objective(.199))/.002
    assert np.isfinite(automatic) and automatic!=0.
    np.testing.assert_allclose(automatic,finite,rtol=3e-3,atol=1e-3)
    def night_heating(surface_pressure):
        half=ph*surface_pressure/705.;full=.5*(half[1:]+half[:-1])
        flux=column_fluxes(t,190.,half,full,0.,610.,.3,1.,.7,7.,forget_shortwave_heating=True)
        return jnp.sum(flux.atmospheric_convergence_w_m2+flux.nonlte_heating_k_s)
    assert np.isfinite(jax.grad(night_heating)(705.))
