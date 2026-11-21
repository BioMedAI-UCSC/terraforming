"""Independent mass, locality, positivity and derivative checks for tracers."""
import dataclasses
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.physics.tracer_transport import tracer_to_nodal
from src.framework.gcm._dinosaur import jax,jnp,primitive_equations
from src.framework.gcm.coordinates import coordinate_system
from src.framework.physics.tracer_transport import project_tracers

jax.config.update('jax_enable_x64',True)


def states():
    coords=coordinate_system('T21',4);grid=coords.horizontal
    lon=jnp.asarray(grid.longitudes)[:,None];lat=jnp.asarray(grid.latitudes)[None,:]
    q=jnp.broadcast_to(1.+.7*jnp.cos(lon)*jnp.cos(lat),(4,*grid.nodal_shape))
    zero=jnp.zeros((4,*grid.modal_shape))
    state=primitive_equations.State(vorticity=zero,divergence=zero,temperature_variation=zero,
        log_surface_pressure=grid.to_modal((jnp.log(2.)+.1*jnp.cos(lon)*jnp.cos(lat))[None]),
        tracers={'dust':grid.to_modal(q),'empty':jnp.zeros_like(zero)})
    current=dataclasses.replace(state,tracers={**state.tracers,'dust':grid.to_modal(.8*q)},
        log_surface_pressure=state.log_surface_pressure+grid.to_modal(jnp.ones((1,*grid.nodal_shape))*.05))
    return coords,state,current


def inventory(state,coords,name):
    grid=coords.horizontal
    return jnp.sum(tracer_to_nodal(grid,state.tracers[name])*jnp.exp(grid.to_nodal(state.log_surface_pressure))*
        jnp.diff(jnp.asarray(coords.vertical.boundaries))[:,None,None]*grid.quadrature_weights)


def test_inventory_correction_preserves_grid_space_and_independent_species_mass():
    coords,old,current=states();final,diagnostics=jax.jit(lambda x:project_tracers(old,x,coords))(current)
    assert diagnostics['dust']['minimum_before']>0.
    for name in old.tracers:
        assert final.tracers[name].shape[-2:]==coords.horizontal.nodal_shape
        assert np.min(tracer_to_nodal(coords.horizontal,final.tracers[name]))>=0.
        np.testing.assert_allclose(inventory(final,coords,name),inventory(old,coords,name),rtol=2e-15)
        assert diagnostics[name]['valid']
    np.testing.assert_array_equal(final.vorticity,current.vorticity)
    np.testing.assert_array_equal(final.log_surface_pressure,current.log_surface_pressure)


def test_postphysics_projection_retains_physical_inventory_change():
    coords,old,current=states();final,_=project_tracers(old,current,coords,conserve_previous=False)
    np.testing.assert_allclose(inventory(final,coords,'dust'),inventory(current,coords,'dust'),rtol=2e-15)
    assert inventory(final,coords,'dust')!=inventory(old,coords,'dust')


def test_positive_unchanged_state_has_no_projection():
    coords,old,_=states();final,diagnostics=project_tracers(old,old,coords)
    np.testing.assert_array_equal(final.tracers['dust'],coords.horizontal.to_nodal(old.tracers['dust']))
    assert diagnostics['dust']['redistributed_inventory']==0.


def test_inventory_correction_reverse_derivative():
    coords,old,current=states();grid=coords.horizontal
    def objective(amplitude):
        supplied=dataclasses.replace(old,tracers={**old.tracers,'dust':old.tracers['dust']*amplitude})
        final,_=project_tracers(supplied,current,coords)
        return jnp.mean(tracer_to_nodal(grid,final.tracers['dust'])**2)
    auto=jax.jit(jax.grad(objective))(1.);eps=1e-5
    fd=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
    assert auto>0.
    np.testing.assert_allclose(auto,fd,rtol=1e-9,atol=1e-10)


def test_negative_total_and_changed_species_are_rejected():
    coords,old,_=states()
    invalid=dataclasses.replace(old,tracers={**old.tracers,'dust':-old.tracers['dust']})
    final,diagnostics=project_tracers(old,invalid,coords)
    assert not diagnostics['dust']['valid'] and np.isnan(final.tracers['dust']).all()
    with pytest.raises(ValueError,match='same tracer names'):
        project_tracers(old,dataclasses.replace(old,tracers={}),coords)


def test_a_material_negative_cell_is_rejected_without_global_blending():
    coords,old,_=states();grid=coords.horizontal
    value=grid.to_nodal(old.tracers['dust']).at[-1,0,0].set(-.01)
    current=dataclasses.replace(old,tracers={**old.tracers,'dust':value})
    final,info=project_tracers(old,current,coords)
    assert not info['dust']['valid']
    assert np.isnan(final.tracers['dust']).all()


def test_local_flux_transport_preserves_material_locality_and_derivatives():
    from src.framework.physics.tracer_transport import transport_tracers
    from src.framework.gcm._dinosaur import spherical_harmonic
    coords,base,_=states();grid=coords.horizontal
    plume=jnp.zeros((4,*grid.nodal_shape)).at[-1,10:14,-4:].set(1e-4)
    base=dataclasses.replace(base,log_surface_pressure=grid.to_modal(jnp.zeros((1,*grid.nodal_shape))),
        tracers={'plume':plume})
    def advance(speed):
        wind=jnp.broadcast_to(speed*jnp.asarray(grid.cos_lat),plume.shape)
        vor,div=spherical_harmonic.uv_nodal_to_vor_div_modal(grid,wind,jnp.zeros_like(wind))
        old=dataclasses.replace(base,vorticity=vor,divergence=div)
        return transport_tracers(old,dataclasses.replace(old,tracers={}),coords,.3)[0]
    final=jax.jit(advance)(.15)
    q=final.tracers['plume']
    assert jnp.min(q)>=0.
    assert jnp.max(jnp.abs(q[:-1]))<1e-20
    assert jnp.max(jnp.abs(q[:,:,:16]))<1e-20
    np.testing.assert_allclose(inventory(final,coords,'plume'),inventory(base,coords,'plume'),rtol=3e-15)
    objective=lambda speed:jnp.sum(advance(speed).tracers['plume']**2)
    automatic=jax.jit(jax.grad(objective))(.15)
    forward=jax.jvp(objective,(jnp.asarray(.15),),(jnp.asarray(1.),))[1]
    epsilon=1e-5
    finite=(objective(.15+epsilon)-objective(.15-epsilon))/(2*epsilon)
    np.testing.assert_allclose(automatic,finite,rtol=2e-7,atol=1e-14)
    np.testing.assert_allclose(forward,finite,rtol=2e-7,atol=1e-14)


def test_reference_step_keeps_a_localized_polar_plume_in_its_layers_and_hemisphere():
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm.dynamics import reference_temperature
    from src.framework.gcm._dinosaur import scales
    from src.framework.physics.dinosaur_physics import initial_state,build_step
    from src.framework.physics.vertical_grid import sigma_boundaries
    coords=coordinate_system('T21',56,sigma_boundaries=sigma_boundaries(56))
    grid=coords.horizontal;body=AMES_BODY_3D;specs=physics_specs(body)
    zero=jnp.zeros((56,*grid.modal_shape))
    pressure=float(specs.nondimensionalize(705.*scales.units.pascal))
    ref=jnp.asarray(reference_temperature(coords,body))[:,None,None]
    dyn=primitive_equations.State(vorticity=zero,divergence=zero,
        temperature_variation=grid.to_modal(jnp.broadcast_to(170.-ref,(56,*grid.nodal_shape))),
        log_surface_pressure=grid.to_modal(jnp.full((1,*grid.nodal_shape),np.log(pressure))),sim_time=0.)
    plume=np.zeros((56,*grid.nodal_shape))
    plume[-5:,:,np.asarray(grid.latitudes)>np.deg2rad(65.)]=1e-4
    state=initial_state(dyn,coords,specs,bulk_water=True,initial_water_vapor_ratio=plume)
    step=build_step(coords,specs,body,reference_radiative_forcing(),231.,bulk_water=True,
        no_forcing=True,return_diagnostics=True,radiation_component=lambda *args:None,soil_diffusivity=1e-7)
    final,diagnostics=jax.jit(step)(state)
    budget=diagnostics.tracer_transport['dynamics']['water_vapor']
    assert budget['redistributed_inventory_kg']/budget['target_inventory_kg']<1e-12
    value=final.dynamics.tracers['water_vapor']
    actual=np.asarray(value if value.shape[-2:]==grid.nodal_shape else grid.to_nodal(value))
    weights=np.diff(coords.vertical.boundaries)[:,None,None]*np.asarray(grid.quadrature_weights)
    leaked=np.sum(np.abs(actual[plume==0.])*np.broadcast_to(weights,plume.shape)[plume==0.])
    initial=np.sum(plume*weights)
    assert leaked/initial<1e-8, f'fraction of polar inventory moved outside plume: {leaked/initial}'
    np.testing.assert_allclose(np.sum(actual*weights),initial,rtol=3e-13)
