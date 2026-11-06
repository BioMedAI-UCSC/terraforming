import dataclasses
import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm._dinosaur import primitive_equations,scales
from src.framework.physics.gcm import initial_column_state
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.interventions.aerosols import aluminium_rod
from src.interventions.aerosols.state import initialize,inventory,correct_transport,MASS,NUMBER,put_burdens,burdens,gas_mass,quadrature_weights

jax.config.update('jax_enable_x64',True)


def setup_state():
    coords=coordinate_system('T21',4);specs=physics_specs(BODY);grid=coords.horizontal
    z=jnp.zeros((4,)+grid.modal_shape)
    p=float(specs.nondimensionalize(610*scales.units.pascal))
    dyn=primitive_equations.State(vorticity=z,divergence=z,temperature_variation=z,
        log_surface_pressure=grid.to_modal(jnp.full((1,)+grid.nodal_shape,np.log(p))),sim_time=0.)
    return coords,specs,initial_column_state(dyn,coords,210.,specs)


def test_initial_inventory_and_explicit_correction_gate():
    coords,specs,climate=setup_state()
    state=initialize(climate,coords,specs,BODY,aluminium_rod(),1e-6)
    mass,number=inventory(state,coords,specs,BODY)
    assert float(mass)==pytest.approx(1e-6*BODY.surface_area_m2,rel=1e-12)
    tracers={k:v*.99 if k in (MASS,NUMBER) else v for k,v in state.climate.dynamics.tracers.items()}
    changed=state._replace(climate=state.climate._replace(dynamics=dataclasses.replace(state.climate.dynamics,tracers=tracers)))
    repaired=jax.jit(lambda s:correct_transport(s,mass,number,coords,specs,BODY))(changed)
    assert bool(repaired.ledger.invalid)
    assert float(repaired.ledger.maximum_raw_relative_mass_error)==pytest.approx(.01)
    assert float(repaired.ledger.numerical_mass_correction_kg)==pytest.approx(.01*float(mass))
    np.testing.assert_allclose(inventory(repaired,coords,specs,BODY),[mass,number],rtol=1e-12)


def test_positive_storage_survives_spectral_roundtrip_and_preserves_layer_inventory():
    coords,specs,climate=setup_state();grid=coords.horizontal
    lon=jnp.asarray(grid.longitudes)[:,None]
    pressure=grid.to_nodal(climate.dynamics.log_surface_pressure)+.2*jnp.cos(lon)
    climate=climate._replace(dynamics=dataclasses.replace(climate.dynamics,
        log_surface_pressure=grid.to_modal(pressure)))
    gas=gas_mass(climate,coords,specs,BODY)
    mass=gas*jnp.where(jnp.cos(lon)>.5,1e-8,0.)[None]
    number=mass/aluminium_rod().mass_kg
    # A nonnegative nodal field acquires Gibbs undershoots when truncated.
    raw=put_burdens(climate,mass,number,coords,specs,BODY)
    assert np.min(burdens(raw,coords,specs,BODY)[1])<0
    store=jax.jit(lambda m,n:put_burdens(climate,m,n,coords,specs,BODY,positive=True))
    repaired=store(mass,number)
    weights=quadrature_weights(coords)
    for original,actual in zip((mass,number),burdens(repaired,coords,specs,BODY)):
        assert np.all(np.asarray(actual)>=0)
        np.testing.assert_allclose(jnp.sum(actual*weights,axis=(1,2)),
            jnp.sum(original*weights,axis=(1,2)),rtol=1e-12)
    empty=store(jnp.zeros_like(mass),jnp.zeros_like(number))
    for actual in burdens(empty,coords,specs,BODY):
        np.testing.assert_array_equal(actual,0.)
    negative=store(-mass,-number)
    for actual in burdens(negative,coords,specs,BODY):
        assert np.isnan(actual).all()
