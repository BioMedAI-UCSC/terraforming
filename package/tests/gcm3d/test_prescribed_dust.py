import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.prescribed_dust import (layer_opacities,fixed_optics,seasonal_fields,
    storm_opacity,make_fixed_dust_provider)

jax.config.update('jax_enable_x64',True)


def test_storm_retains_native_two_images_temporal_ramp_and_explicit_shape_inputs():
    center=np.deg2rad(10.);latitude=np.deg2rad(15.)
    value=storm_opacity(180.,center,latitude,slope_deg=0.,latitude_coefficient=5.,amplitude=.4)
    expected=.2*(1.+np.exp(-.5*(2.*np.pi/np.deg2rad(60.))**2))
    np.testing.assert_allclose(value,expected,rtol=2e-15)
    assert storm_opacity(170.,center,latitude,slope_deg=0.,latitude_coefficient=5.,amplitude=.4)<value
    assert storm_opacity(190.,center,latitude,slope_deg=0.,latitude_coefficient=5.,amplitude=.4)>value
    with pytest.raises(ValueError,match='undefined'):
        make_fixed_dust_provider(storm_options=dict(amplitude=.4))
    for options in (dict(longitude_width_deg=0.),dict(latitude_width_deg=-1.),dict(latitude_coefficient=np.inf)):
        configuration=dict(slope_deg=0.,latitude_coefficient=5.);configuration.update(options)
        assert np.isnan(storm_opacity(180.,center,latitude,**configuration))


def test_storm_reverse_derivative_includes_shape_amplitude_and_onset():
    lon=jnp.linspace(0.,2.*jnp.pi,9)[:,None];lat=jnp.linspace(-1.,1.,7)[None,:]
    def objective(scale):
        return jnp.sum(storm_opacity(180.+.2*scale,lon,lat,slope_deg=12.*scale,
            latitude_coefficient=5.*scale,amplitude=.4*scale))
    automatic=jax.jit(jax.grad(objective))(1.);epsilon=1e-5
    assert np.isfinite(automatic) and automatic!=0.
    np.testing.assert_allclose(automatic,(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon),rtol=2e-9)


@pytest.mark.parametrize('pulse',[0.,.2])
def test_prescribed_column_normalization_retains_surface_pressure_and_latitude_scaling(pulse):
    half=jnp.geomspace(.01,400.,13);full=jnp.sqrt(half[:-1]*half[1:])
    tau=layer_opacities(half,full,.3,latitude_rad=.7,latitude_variation=True,pulse_fraction=pulse)
    np.testing.assert_allclose(jnp.sum(tau),.3*(.3+.7*jnp.cos(.7))*400./610.,rtol=2e-14)
    assert np.min(tau)>=0.


def test_fixed_optics_preserves_visible_reference_column_and_keeps_ghosts_clear():
    half=jnp.array([0.,100.,300.,610.]);full=.5*(half[:-1]+half[1:])
    p=jnp.zeros(9).at[2::2].set(half).at[3::2].set(full)
    tau=jnp.array([.01,.02,.03])
    result=jax.jit(fixed_optics)(tau,p)
    np.testing.assert_allclose(jnp.sum(result.visible[0][:,5]),jnp.sum(tau),rtol=2e-14)
    for values in (*result.visible,*result.infrared):
        np.testing.assert_array_equal(values[:2],0.)
        assert np.isfinite(values).all()


def test_profile_and_size_optics_have_reverse_derivatives_through_pulses_and_pressure():
    def objective(scale):
        half=jnp.geomspace(.01,610.,13)*scale;full=jnp.sqrt(half[:-1]*half[1:])
        tau=layer_opacities(half,full,.3*scale,pulse_fraction=.2*scale)
        p=jnp.zeros(27).at[2::2].set(half).at[3::2].set(full)
        result=fixed_optics(tau,p,effective_radius_m=1.5e-6*scale)
        return jnp.sum(result.infrared[0]*jnp.arange(1.,27.)[:,None])
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-5
    eps=1e-5
    np.testing.assert_allclose(auto,(objective(1.+eps)-objective(1.-eps))/(2.*eps),rtol=2e-7,atol=1e-8)


def test_seasonal_interpolation_wraps_with_the_supplied_360_degree_endpoint():
    axis=jnp.array([0.,90.,180.,270.,360.]);values=jnp.array([1.,2.,3.,4.,1.])
    for ls,expected in ((0.,1.),(45.,1.5),(315.,2.5),(360.,1.),(-45.,2.5)):
        actual=seasonal_fields(ls,axis,values,values*10.)
        np.testing.assert_allclose(actual,[expected,expected*10.],rtol=1e-15)


def test_missing_tracer_height_inputs_are_rejected_and_empty_source_height_stays_invalid():
    half=jnp.geomspace(.01,610.,9);full=jnp.sqrt(half[:-1]*half[1:])
    with pytest.raises(ValueError,match='requires an aerosol'):
        layer_opacities(half,full,.3,cycle_scheme=1)
    invalid=layer_opacities(half,full,.3,cycle_scheme=1,aerosol_ratio=jnp.zeros(8))
    assert np.isnan(invalid).all()


@pytest.mark.parametrize('bands',[12,15])
def test_user_fixed_optics_keep_source_coefficients_and_reverse_derivatives(bands):
    nir=5 if bands==12 else 8
    half=jnp.array([0.,100.,300.,610.]);full=.5*(half[:-1]+half[1:])
    p=jnp.zeros(9).at[2::2].set(half).at[3::2].set(full)
    tau=jnp.array([.01,.02,.03])
    visible=jnp.stack([jnp.linspace(1.,3.,7),jnp.linspace(.2,3.,7),jnp.linspace(-.2,.8,7)])
    infrared=jnp.stack([jnp.linspace(.1,2.,nir),jnp.linspace(.01,2.,nir),jnp.linspace(-.1,.7,nir)])
    result=jax.jit(lambda x:fixed_optics(tau,p,bands=bands,optical_properties=x))((visible,infrared))
    np.testing.assert_allclose(jnp.sum(result.visible[0][:,5]),jnp.sum(tau),rtol=2e-15)
    np.testing.assert_array_equal(result.diagnostics['size_distribution'],0.)
    np.testing.assert_array_equal(result.visible[0][:,-1],result.visible[1][:,-1])
    np.testing.assert_allclose(result.visible[2],result.visible[1]*visible[2],rtol=2e-15)
    for values in (*result.visible,*result.infrared): np.testing.assert_array_equal(values[:2],0.)
    def objective(scale):
        properties=(visible.at[0,5].multiply(scale),infrared.at[1].multiply(scale**2))
        optics=fixed_optics(tau*scale,p,bands=bands,optical_properties=properties)
        return jnp.sum(optics.infrared[1])+jnp.sum(optics.visible[0])
    derivative=jax.jit(jax.grad(objective))(1.);epsilon=1e-5
    assert np.isfinite(derivative) and abs(derivative)>1e-4
    np.testing.assert_allclose(derivative,(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon),rtol=2e-9)
    for invalid in ((visible,), (visible[:2],infrared), (visible,infrared[:,:-1])):
        with pytest.raises(ValueError,match='fixed dust'):
            fixed_optics(tau,p,bands=bands,optical_properties=invalid)


def test_fixed_dust_provider_passes_user_coefficients_to_optics(monkeypatch):
    from types import SimpleNamespace
    from src.framework.physics import prescribed_dust
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    coords=coordinate_system('T21',4)
    p=jnp.array([0.,0.,0.,50.,100.,200.,300.,455.,610.])
    properties=(jnp.ones((3,7)),jnp.ones((3,5)))
    seen={}
    def optics(*args,**kwargs):
        seen.update(kwargs)
        return SimpleNamespace(visible=1,infrared=2)
    monkeypatch.setattr(prescribed_dust,'fixed_optics',optics)
    provider=make_fixed_dust_provider(optical_properties=properties)
    assert provider(None,coords,physics_specs(AMES_BODY_3D),AMES_BODY_3D,
        reference_radiative_forcing(),SimpleNamespace(pressure=jnp.broadcast_to(p[:,None,None],(9,*coords.horizontal.nodal_shape))))==(1,2)
    assert seen['optical_properties'] is properties


@pytest.mark.parametrize('fixed_date',[-1,100])
def test_fixed_dust_diagnostic_uses_selected_calendar_without_mutating_state(monkeypatch,fixed_date):
    from types import SimpleNamespace
    from src.framework.physics import prescribed_dust,gcm
    from src.framework.gcm._dinosaur import scales
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    coords=coordinate_system('T21',4);specs=physics_specs(AMES_BODY_3D);forcing=reference_radiative_forcing()
    scale=float(specs.dimensionalize(1.,scales.units.second).magnitude)
    seen=[]
    def orbit(time,forcing):
        seen.append(float(time))
        return time/forcing.rotation_period_s*.001
    monkeypatch.setattr(gcm,'_true_anomaly',orbit)
    monkeypatch.setattr(prescribed_dust,'seasonal_fields',lambda *args:(.2,35.))
    monkeypatch.setattr(prescribed_dust,'fixed_optics',lambda *args,**kwargs:SimpleNamespace(visible=1,infrared=2))
    scenario=dict(seasons=None,opacity=None,top_height_km=None)
    provider=make_fixed_dust_provider(scenario=scenario,fixed_calendar_date=fixed_date)
    p=jnp.array([0.,0.,0.,50.,100.,200.,300.,455.,610.])
    grid=SimpleNamespace(pressure=jnp.broadcast_to(p[:,None,None],(9,*coords.horizontal.nodal_shape)))
    for sol in (2.25,5.25):
        state=SimpleNamespace(dynamics=SimpleNamespace(sim_time=sol*forcing.rotation_period_s/scale))
        initial=state.dynamics.sim_time
        assert provider(state,coords,specs,AMES_BODY_3D,forcing,grid)==(1,2)
        np.testing.assert_allclose(seen[-1],(sol if fixed_date<0 else 100.25)*forcing.rotation_period_s,rtol=2e-15)
        assert state.dynamics.sim_time==initial
    with pytest.raises(ValueError,match='integer'): make_fixed_dust_provider(fixed_calendar_date=1.5)
