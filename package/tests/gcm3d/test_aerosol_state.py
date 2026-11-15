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
from src.interventions.aerosols.state import initialize,inventory,correct_transport,MASS,NUMBER

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
