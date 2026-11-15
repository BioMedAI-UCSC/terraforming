"""Local-time scenario history, map boundaries and reverse derivatives."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.dust_control import (LocalTimeState,local_time_step,seasonal_sample,
    validate_local_time_scenario)

jax.config.update('jax_enable_x64',True)


def test_local_time_updates_history_after_current_opacity_and_wraps_longitude():
    seasons=jnp.array([0.,180.,360.]);maps=jnp.array([.1,.5,.1])[:,None,None]
    fraction=jnp.broadcast_to(jnp.array([.2,.3]),(3,1,1,2))
    history=LocalTimeState(jnp.array([[.1]]),jnp.array([[0]]))
    # At lon 0, sol .5 is noon and changes the half-day bin.
    opacity,current,next_state=local_time_step(maps,fraction,seasons,[0.,12.,24.],90.,0.,.5,history)
    np.testing.assert_allclose(opacity,.22,atol=1e-15)
    assert current==.3 and next_state.bin_index==1
    np.testing.assert_array_equal(next_state.opacity,opacity)
    second,_,held=local_time_step(maps,fraction,seasons,[0.,12.,24.],90.,2.*jnp.pi,.6,next_state)
    np.testing.assert_allclose(second,.34,atol=1e-15)
    np.testing.assert_array_equal(held.opacity,next_state.opacity)
    falling,_,_=local_time_step(maps,fraction,seasons,[0.,12.,24.],270.,0.,.6,next_state)
    np.testing.assert_array_equal(falling,maps[-1])


def test_initial_local_time_opacity_uses_next_season_and_exact_bin_boundaries():
    axis=jnp.array([0.,180.,360.]);maps=jnp.array([.1,.5,.1])[:,None,None]
    fractions=jnp.ones((3,1,1,2))*.2
    opacity,_,state=local_time_step(maps,fractions,axis,[0.,12.,24.],90.,0.,0.,None)
    np.testing.assert_array_equal(opacity,maps[1]);assert state.bin_index==0
    for ls,expected in ((0.,.1),(180.,.5),(360.,.1)):
        value,_,_,_=seasonal_sample(axis,maps,ls)
        np.testing.assert_allclose(value,expected,atol=1e-15)


def test_local_time_history_reverse_derivative():
    axis=jnp.array([0.,180.,360.]);maps=jnp.array([.1,.5,.1])[:,None,None]
    fractions=jnp.ones((3,1,1,2))*.2
    def objective(old):
        value,_,_=local_time_step(maps,fractions,axis,[0.,12.,24.],90.,0.,.5,
            LocalTimeState(jnp.array([[old]]),jnp.array([[0]])))
        return jnp.sum(value)
    np.testing.assert_allclose(jax.jit(jax.grad(objective))(.15),1.,atol=1e-15)


def test_bad_local_time_maps_are_rejected():
    scenario=dict(seasons=[0.,180.,360.],bins_hours=[0.,12.,24.],fractions=np.zeros((3,2,4,2)))
    validate_local_time_scenario(scenario,scenario['seasons'],(2,4))
    with pytest.raises(ValueError,match='seasonal axes'):
        validate_local_time_scenario(scenario,[0.,90.,360.],(2,4))
    with pytest.raises(ValueError,match='0 to 24'):
        validate_local_time_scenario({**scenario,'bins_hours':[0.,12.,23.]},scenario['seasons'],(2,4))
    with pytest.raises(ValueError,match='match seasons'):
        validate_local_time_scenario(scenario,scenario['seasons'],(4,2))
