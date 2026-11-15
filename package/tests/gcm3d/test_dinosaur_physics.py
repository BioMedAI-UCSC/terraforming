import dataclasses
import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp

from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY
from src.framework.physics.dinosaur_physics import (
    remap_gas,initial_state,build_step,axial_angular_momentum,exchange_energy)

jax.config.update('jax_enable_x64',True)


def test_sigma_remap_conserves_energy_momentum_and_passive_material():
    weights=jnp.array([.1,.2,.3,.4])
    temperature=jnp.array([140.,160.,180.,200.])
    u=jnp.array([50.,30.,10.,-10.]);v=u*.2
    q=jnp.array([[.001],[.002],[.003],[.004]])
    loss=jnp.array([.01,.02,.1,.2])
    t,unew,vnew,qnew,ps=remap_gas(temperature,u,v,q,jnp.asarray(610.),loss,weights,BODY)
    gas=610.*weights/BODY.gravity_m_s2
    donor=gas-loss
    target=ps*weights/BODY.gravity_m_s2
    np.testing.assert_allclose(jnp.sum(target),jnp.sum(donor),rtol=2e-14)
    np.testing.assert_allclose(jnp.sum(target*(BODY.cp_j_kg_k*t+.5*(unew**2+vnew**2))),
        jnp.sum(donor*(BODY.cp_j_kg_k*temperature+.5*(u*u+v*v))),rtol=2e-14)
    for a,b in ((unew,u),(vnew,v)):
        np.testing.assert_allclose(jnp.sum(target*a),jnp.sum(donor*b),rtol=2e-14)
    np.testing.assert_allclose(jnp.sum(target[:,None]*qnew,axis=0),jnp.sum(gas[:,None]*q,axis=0),rtol=2e-14)
    assert np.min(np.asarray(qnew))>=0.


def test_impossible_donor_mass_is_nonfinite_instead_of_clipped():
    result=remap_gas(jnp.full(4,200.),jnp.zeros(4),jnp.zeros(4),jnp.empty((4,0)),
        jnp.asarray(610.),jnp.ones(4)*1000.,jnp.ones(4)*.25,BODY)
    assert not np.isfinite(np.asarray(result[0])).all()
    assert not np.isfinite(float(result[-1]))


def test_sigma_remap_reverse_derivative():
    temperature=jnp.array([140.,160.,180.,200.])
    loss=jnp.array([.01,.02,.1,.2])
    def objective(scale):
        result=remap_gas(temperature,jnp.array([50.,30.,10.,-10.]),jnp.zeros(4),
            jnp.full((4,1),.001),jnp.asarray(610.),loss*scale,jnp.ones(4)*.25,BODY)
        return jnp.sum(result[0])+jnp.sum(result[1])+result[-1]
    auto=jax.grad(objective)(1.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-7)


def backend_case(temperature_k=140.,surface_temperature_k=148.,return_diagnostics=False,
                 radiation_component=None,wind_speed=0.):
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm._dinosaur import primitive_equations,scales
    from src.framework.gcm.dynamics import reference_temperature
    from src.celestials.planets.mars.gcm import radiative_forcing
    from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
    from src.framework.physics.gcm import two_stream_radiative_fluxes
    coords=coordinate_system('T21',4);specs=physics_specs(BODY);grid=coords.horizontal
    zeros=jnp.zeros((4,)+grid.modal_shape)
    pressure=float(specs.nondimensionalize(610.*scales.units.pascal))
    ref=jnp.asarray(reference_temperature(coords,BODY))[:,None,None]
    temperature=jnp.full((4,)+grid.nodal_shape,temperature_k)
    dyn=primitive_equations.State(vorticity=zeros,divergence=zeros,
        temperature_variation=grid.to_modal(temperature-ref),
        log_surface_pressure=grid.to_modal(jnp.full((1,)+grid.nodal_shape,np.log(pressure))),sim_time=0.)
    if wind_speed:
        from src.framework.gcm._dinosaur import spherical_harmonic
        speed_scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
        wind=jnp.broadcast_to(wind_speed*jnp.asarray(grid.cos_lat),temperature.shape)/speed_scale
        vor,div=spherical_harmonic.uv_nodal_to_vor_div_modal(grid,wind,jnp.zeros_like(wind))
        dyn=dataclasses.replace(dyn,vorticity=vor,divergence=div)
    state=initial_state(dyn,coords,specs,temperature_k=surface_temperature_k)
    forcing=(radiative_forcing(co2_radiation_enabled=True) if radiation_component is None else
        reference_radiative_forcing())
    radiation=two_stream_radiative_fluxes if radiation_component is None else radiation_component
    step=build_step(coords,specs,BODY,forcing,5.,radiation_component=radiation,
        soil_diffusivity=(250./(1500.*627.9))**2,return_diagnostics=return_diagnostics)
    return state,step,grid


def test_dinosaur_backend_runs_cold_column_and_preserves_total_inventory():
    state,step,_=backend_case(return_diagnostics=True)
    final,d=jax.jit(step)(state)
    assert bool(d.valid_inventory)
    np.testing.assert_allclose(d.corrected_total_mass_kg,d.target_total_mass_kg,rtol=2e-13)
    assert float(jnp.max(final.co2_ice))>0.
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(final))
    assert np.isfinite(float(d.raw_axial_angular_momentum_kg_m2_s))
    assert np.isfinite(float(d.corrected_axial_angular_momentum_kg_m2_s))
    assert np.isfinite(float(d.raw_exchange_energy_j))
    assert np.isfinite(float(d.corrected_exchange_energy_j))


def test_surface_drag_transfers_axial_momentum_to_ground_and_reports_projection():
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    state,step,grid=backend_case(temperature_k=180.,surface_temperature_k=185.,
        return_diagnostics=True,wind_speed=20.)
    coords=coordinate_system('T21',4);specs=physics_specs(BODY)
    before=axial_angular_momentum(state,coords,specs,BODY)
    final,d=jax.jit(step)(state)
    ground=jnp.sum(final.surface_momentum[0]*grid.quadrature_weights)
    assert float(ground)>0.
    np.testing.assert_allclose(d.raw_axial_angular_momentum_kg_m2_s,before,rtol=2e-11)
    np.testing.assert_allclose(d.corrected_axial_angular_momentum_kg_m2_s,
        axial_angular_momentum(final,coords,specs,BODY),rtol=2e-14)


def test_exchange_energy_surface_derivative_uses_soil_node_capacity():
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.physics.soil_frost import soil_layer_thickness
    coords=coordinate_system('T21',4);specs=physics_specs(BODY)
    state,_,_=backend_case(temperature_k=180.,surface_temperature_k=185.)
    objective=lambda delta:exchange_energy(state._replace(
        surface_temperature=state.surface_temperature+delta),coords,specs,BODY)
    derivative=jax.grad(objective)(0.)
    expected=1500.*627.9*soil_layer_thickness(16)[0]*BODY.surface_area_m2
    np.testing.assert_allclose(derivative,expected,rtol=2e-14)


@pytest.mark.parametrize('native_radiation',[False,True])
def test_multistep_dinosaur_reverse_derivative_reaches_surface_state(native_radiation):
    from src.framework.physics.radiative_transfer import make_radiation_component
    radiation=(make_radiation_component(lambda *args:None) if native_radiation else None)
    state,step,grid=backend_case(temperature_k=180.,surface_temperature_k=185.,
        radiation_component=radiation)

    def objective(perturbation):
        initial=state._replace(surface_temperature=state.surface_temperature+perturbation)
        final,_=jax.lax.scan(lambda carry,_:(step(carry),None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)+jnp.mean(
            grid.to_nodal(final.dynamics.temperature_variation))

    value_and_grad=jax.jit(jax.value_and_grad(objective))
    value,auto=value_and_grad(0.)
    assert np.isfinite(value) and np.isfinite(auto) and abs(float(auto))>1e-6
    forward=jax.jit(objective)
    for epsilon in (1e-2,1e-3):
        finite=(forward(epsilon)-forward(-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,finite,rtol=2e-5,atol=1e-7)
