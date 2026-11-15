"""Independent mass, positivity and derivative checks for spectral tracers."""
import dataclasses
import numpy as np
import pytest
pytest.importorskip('dinosaur')
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
    negative=q-1.1
    negative=negative.at[1:].add(1.)
    current=dataclasses.replace(state,tracers={**state.tracers,'dust':grid.to_modal(negative)},
        log_surface_pressure=state.log_surface_pressure+grid.to_modal(jnp.ones((1,*grid.nodal_shape))*.05))
    return coords,state,current


def inventory(state,coords,name):
    grid=coords.horizontal
    return jnp.sum(grid.to_nodal(state.tracers[name])*jnp.exp(grid.to_nodal(state.log_surface_pressure))*
        jnp.diff(jnp.asarray(coords.vertical.boundaries))[:,None,None]*grid.quadrature_weights)


def test_projection_retains_modal_space_positivity_and_independent_species_mass():
    coords,old,current=states();final,diagnostics=jax.jit(lambda x:project_tracers(old,x,coords))(current)
    assert diagnostics['dust']['minimum_before']<0.
    for name in old.tracers:
        assert np.min(coords.horizontal.to_nodal(final.tracers[name]))>=0.
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
    np.testing.assert_array_equal(final.tracers['dust'],old.tracers['dust'])
    assert diagnostics['dust']['redistributed_inventory']==0.


def test_projection_reverse_derivative_with_active_undershoot():
    coords,old,current=states();grid=coords.horizontal
    def objective(amplitude):
        supplied=dataclasses.replace(old,tracers={**old.tracers,'dust':old.tracers['dust']*amplitude})
        final,_=project_tracers(supplied,current,coords)
        return jnp.mean(grid.to_nodal(final.tracers['dust'])**2)
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
