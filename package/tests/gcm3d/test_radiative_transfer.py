import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.radiative_transfer import column_fluxes,shortwave_ramp,make_radiation_component

jax.config.update('jax_enable_x64',True)


def fluxes(temperature=200.03,pressure=613.,mu=.6,scale_surface_flux=False,linear=False):
    half=jnp.array([0.,.13,.51,1.])*pressure
    return column_fluxes(jnp.full(3,temperature),temperature,half,
        (half[1:]+half[:-1])*.5,mu,jnp.array([12.7,24.2,54.6,145.9,354.9,657.5,106.3])/1.524**2,
        .25,.95,3.72,scale_surface_flux=scale_surface_flux,linear_k_interpolation=linear)


@pytest.mark.parametrize('linear',[False,True])
@pytest.mark.parametrize('scale_surface_flux',[False,True])
@pytest.mark.parametrize('mu',[.6,0.,-0.2])
def test_full_radiation_flux_budget_and_night(mu,scale_surface_flux,linear):
    result=jax.jit(lambda:fluxes(mu=mu,scale_surface_flux=scale_surface_flux,linear=linear))()
    if scale_surface_flux:
        emitted=result.longwave_up_w_m2[-1]-.05*result.longwave_down_w_m2[-1]
        # Native normalization uses 3.14159; the transfer solver uses full pi.
        np.testing.assert_allclose(emitted,5.67e-8*200.03**4*np.pi/3.14159,rtol=2e-13)
    assert all(np.isfinite(np.asarray(x)).all() for x in result)
    np.testing.assert_allclose(jnp.sum(result.atmospheric_convergence_w_m2)+result.surface_net_w_m2,
        result.toa_net_down_w_m2,rtol=2e-13,atol=1e-10)
    if mu<=0.:
        np.testing.assert_array_equal(result.shortwave_down_w_m2,0.)
        np.testing.assert_array_equal(result.shortwave_up_w_m2,0.)


@pytest.mark.parametrize('linear',[False,True])
@pytest.mark.parametrize('scale_surface_flux',[False,True])
def test_full_radiation_reverse_derivative_reaches_temperature_and_pressure(scale_surface_flux,linear):
    def objective(scale):
        result=fluxes(temperature=203.03*scale,pressure=613.*scale,scale_surface_flux=scale_surface_flux,linear=linear)
        return result.surface_net_w_m2+jnp.sum(result.atmospheric_convergence_w_m2*jnp.arange(1.,4.))
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for eps in (1e-6,1e-7):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-4,atol=1e-4)


def test_shortwave_ramp_reverse_derivative_reaches_time_and_duration():
    objective=lambda scale:shortwave_ramp(.73*scale,2.3/scale)
    automatic=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(automatic) and automatic>0.
    for epsilon in (1e-5,1e-6):
        np.testing.assert_allclose(automatic,(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon),rtol=2e-8)
    for duration in (-1.,0.):
        np.testing.assert_array_equal(shortwave_ramp(jnp.array([0.,1.,10.]),duration),1.)


@pytest.mark.parametrize('duration',[np.nan,np.inf,-np.inf])
def test_radiation_factory_rejects_nonfinite_ramp_duration(duration):
    with pytest.raises(ValueError,match='ramp duration must be finite'):
        make_radiation_component(lambda *args:None,shortwave_ramp_sols=duration)


@pytest.mark.parametrize('date',[1.5,np.nan,'100'])
def test_radiation_factory_requires_integer_fixed_date(date):
    with pytest.raises(ValueError,match='calendar date must be an integer'):
        make_radiation_component(lambda *args:None,fixed_calendar_date=date)


def test_reference_callback_uses_positive_native_top_and_current_top_temperature(monkeypatch):
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as body,reference_radiative_forcing
    from src.celestials.planets.mars.maps import initial_rest_state
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.physics.vertical_grid import sigma_boundaries,coefficients
    from src.framework.physics.dinosaur_physics import initial_state
    from src.framework.physics import radiative_transfer as module
    from src.framework.physics.gcm import RadiativeFluxDiagnostics
    coords=coordinate_system('T21',56,sigma_boundaries=sigma_boundaries(56));specs=physics_specs(body)
    dyn=initial_rest_state(coords,specs,body,np.zeros(coords.horizontal.nodal_shape),t_ref_k=195.,p0_pa=705.)
    import dataclasses
    dyn=dataclasses.replace(dyn,sim_time=0.)
    state=initial_state(dyn,coords,specs,195.,radiation_update_seconds=1848.)
    captured={}
    def flux(temperature,surface,half,full,*args,**kwargs):
        captured.update(half=np.asarray(half),full=np.asarray(full),strat=np.asarray(kwargs['stratosphere_temperature']))
        interfaces=jnp.zeros_like(half);ground=jnp.zeros_like(surface)
        return RadiativeFluxDiagnostics(interfaces,interfaces,interfaces,interfaces,jnp.zeros_like(temperature),ground,ground)
    monkeypatch.setattr(module,'column_fluxes',flux)
    make_radiation_component(lambda *args:None,solar_nonlte=False)(state,coords,specs,body,reference_radiative_forcing())
    top=float(coefficients(56,705.)[0][0])
    from src.framework.gcm._dinosaur import scales
    pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    resolved=np.exp(np.asarray(coords.horizontal.to_nodal(dyn.log_surface_pressure)))[0]*pressure_scale
    np.testing.assert_allclose(captured['half'][0]/resolved,top/705.,rtol=3e-14)
    np.testing.assert_allclose(captured['strat'],195.,rtol=3e-14)
    assert np.min(captured['half'][0])>0. and coords.vertical.boundaries[0]==0.
    np.testing.assert_allclose(captured['full'],.5*(captured['half'][:-1]+captured['half'][1:]),rtol=2e-15)
