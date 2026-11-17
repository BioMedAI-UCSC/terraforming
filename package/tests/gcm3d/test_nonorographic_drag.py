"""Gravity-wave limits, energy accounting and reverse derivatives."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.nonorographic_drag import (spectrum,source_parameters,tendencies,wave_forcing,
    initial_drag_state,driver_step,schedule)

jax.config.update('jax_enable_x64',True)


def column(amplitude=.08,conserve=False):
    p=jnp.geomspace(.03,610.,12)
    z=9000.*jnp.log(610./p);t=jnp.linspace(150.,220.,12)
    u=jnp.linspace(-31.7,20.3,12);v=-.3*u+1.7
    result=tendencies(t,p,z,u,v,10,amplitude,300.,conserve_energy=conserve)
    return t,u,v,result


def test_phase_grid_keeps_exact_zero_and_patched_source_latitude_units():
    c,k=spectrum()
    assert len(c)==121 and c[60]==0.
    np.testing.assert_allclose(k,2.*np.pi/300e3,rtol=1e-15)
    level,amplitude=source_parameters([10.,50.,140.,180.,350.,610.],jnp.array([0.,np.pi/2.]))
    assert level==4
    np.testing.assert_allclose(amplitude,.08*(1.-jnp.tanh(jnp.array([0.,np.pi/2.])/45.)**2),rtol=1e-15)


def test_zero_source_and_calm_symmetric_spectrum_are_finite():
    _,_,_,zero=column(amplitude=0.)
    np.testing.assert_array_equal(zero.u_tendency,jnp.zeros(12))
    p=jnp.geomspace(.03,610.,12);z=9000.*jnp.log(610./p);t=jnp.full(12,190.)
    result=tendencies(t,p,z,jnp.zeros(12),jnp.zeros(12),10,.08,300.)
    assert all(np.all(np.isfinite(x)) for x in result[:3])
    np.testing.assert_allclose(result.u_tendency,0.,atol=1e-17)


def test_optional_reheating_closes_each_cells_kinetic_plus_sensible_energy():
    _,u,v,result=column(conserve=True);dt=300.
    assert np.max(np.abs(result.u_tendency))>0.
    residual=735.94*dt*result.heating_k_s+.5*((u+dt*result.u_tendency)**2-u**2+
        (v+dt*result.v_tendency)**2-v**2)
    np.testing.assert_allclose(residual,0.,atol=1e-12)


def test_reverse_derivative_matches_amplitude_finite_difference():
    def objective(amplitude):
        _,_,_,result=column(amplitude,True)
        return jnp.sum(result.u_tendency)+jnp.sum(result.heating_k_s)
    auto=jax.jit(jax.grad(objective))(.08);eps=1e-5
    forward=jax.jit(objective)
    fd=(forward(.08+eps)-forward(.08-eps))/(2.*eps)
    assert np.isfinite(auto) and abs(float(auto))>0.
    np.testing.assert_allclose(auto,fd,rtol=1e-8,atol=1e-13)


def test_equal_density_and_source_index_one_have_finite_extensions():
    rho=jnp.full(6,.02);wind=jnp.linspace(-17.3,25.7,6)
    bf=jnp.full(6,.01);z=jnp.linspace(50000.,0.,6)
    result=wave_forcing(rho,wind,bf,z,4,.08)
    assert np.all(np.isfinite(result.wind_tendency))
    p=jnp.geomspace(.03,610.,4);z=9000.*jnp.log(610./p)
    result=tendencies(jnp.full(4,190.),p,z,jnp.arange(4.)*10.,jnp.zeros(4),1,.08,300.)
    assert np.all(np.isfinite(result.u_tendency))


def test_invalid_spectrum_and_profile_shapes_are_rejected():
    with pytest.raises(ValueError,match='positive'):
        spectrum(speed_interval=0.)
    with pytest.raises(ValueError,match='increasing'):
        source_parameters([10.,5.,610.],0.)
    with pytest.raises(ValueError,match='below a reference'):
        source_parameters([10.,50.,610.],0.,source_pressure_pa=1000.)
    with pytest.raises(ValueError,match='matching'):
        tendencies(jnp.zeros(2),jnp.zeros(2),jnp.zeros(2),jnp.zeros(2),jnp.zeros(2),1,.08,300.)


def test_held_forcing_recomputes_heating_from_current_winds():
    p=jnp.geomspace(.03,610.,12);z=9000.*jnp.log(610./p);t=jnp.linspace(150.,220.,12)
    u=jnp.linspace(-31.7,20.3,12);v=-.3*u+1.7
    state=initial_drag_state(t,update_seconds=900,offset_seconds=300)
    first,state=driver_step(t,p,z,u,v,10,.08,300.,state,update_seconds=900,
        offset_seconds=300,conserve_energy=True)
    held,state=driver_step(t,p,z,u+10.,v-4.,10,.08,300.,state,update_seconds=900,
        offset_seconds=300,conserve_energy=True)
    assert first.diagnostics['refreshed'] and not held.diagnostics['refreshed']
    assert state.alarm_seconds==600.
    np.testing.assert_array_equal(held.u_tendency,first.u_tendency)
    np.testing.assert_array_equal(held.v_tendency,first.v_tendency)
    np.testing.assert_allclose(held.heating_k_s-first.heating_k_s,
        -(10.*first.u_tendency-4.*first.v_tendency)/735.94,atol=1e-18)


@pytest.mark.parametrize('frequency,offset',[(1.5,0),(-1,0),(0,np.nan),(0,-1)])
def test_invalid_timer_is_rejected(frequency,offset):
    with pytest.raises(ValueError,match='whole seconds'):
        schedule(frequency,offset)
def test_source_selection_excludes_bottom_interface_and_caps_extended_index():
    level,_=source_parameters([0.,20.,100.,200.],0.,source_pressure_pa=50.)
    assert level==2
    with pytest.raises(ValueError,match='source pressure'):
        source_parameters([0.,20.,100.,200.],0.,source_pressure_pa=100.)
