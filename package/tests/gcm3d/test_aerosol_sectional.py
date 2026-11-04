"""CI checks for size-resolved material accounting and kinetic refinement."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
jax.config.update('jax_enable_x64',True)
from src.terraforming.aerosols.sectional import coagulator,collision_kernels
from src.terraforming.aerosols import aluminium_rod
from src.terraforming.aerosols.microphysics import brownian_kernel


def test_exact_self_channel_and_no_material_loss_at_large_step():
    step=coagulator([1.,2.])
    kernel=jnp.array([[1.,0.],[0.,0.]])
    for dt in (0.,1.,1e6):
        result=step(jnp.array([1.,0.]),kernel,dt)
        assert bool(result.valid)
        assert float(result.number_m3[0])==pytest.approx(1/(1+dt),abs=2e-15)
        assert np.all(np.asarray(result.number_m3)>=0)
        assert float(jnp.dot(result.number_m3,jnp.array([1.,2.])))==pytest.approx(1.,abs=2e-15)
        assert float(result.unresolved_mass_kg_m3)==0.
    integer_input=step(jnp.array([1,0]),jnp.array([[1,0],[0,0]]),1.)
    assert bool(integer_input.valid)
    assert float(integer_input.number_m3[0])==pytest.approx(.5)


def test_overflow_is_suspended_and_counted_once():
    result=coagulator([1.,2.])(jnp.array([0.,1.]),jnp.ones((2,2)),100.)
    assert bool(result.valid)
    assert float(jnp.dot(result.number_m3,jnp.array([1.,2.]))+result.unresolved_mass_kg_m3)==pytest.approx(2.,abs=1e-13)
    assert float(result.unresolved_number_m3)>0
    assert float(result.unresolved_mass_kg_m3)==pytest.approx(4*float(result.unresolved_number_m3))


def test_constant_kernel_number_and_monomer_refine_to_smoluchowski_solution():
    pivots=np.arange(1.,33.)
    step=coagulator(pivots)
    kernel=jnp.ones((32,32))
    errors=[]
    # n0=1, K=1, t=.5: tau=K*n0*t/2=.25.
    expected_total=1/1.25;expected_monomer=1/1.25**2
    for count in (8,32,128):
        def advance(carry,_):
            n,overflow=carry
            result=step(n,kernel,.5/count)
            return (result.number_m3,overflow+result.unresolved_mass_kg_m3),None
        initial=jnp.zeros(32).at[0].set(1.)
        (number,overflow),_=jax.lax.scan(advance,(initial,jnp.asarray(0.)),None,length=count)
        assert float(jnp.dot(number,jnp.asarray(pivots))+overflow)==pytest.approx(1.,abs=2e-12)
        assert float(overflow)<1e-8
        errors.append(abs(float(number.sum())-expected_total)+abs(float(number[0])-expected_monomer))
    assert errors[2]<errors[1]<errors[0]
    assert errors[2]<.01


def test_source_kernel_limits_and_unsupported_regimes():
    particle=aluminium_rod();pivots=particle.mass_kg*np.arange(1.,5.)
    neutral,settling=collision_kernels(particle,pivots,210.,610.,3.72)
    charged,_=collision_kernels(particle,pivots,210.,610.,3.72,charges_e=1.)
    assert float(neutral[0,0])==pytest.approx(float(brownian_kernel(particle,210.,610.)),rel=1e-12)
    assert np.all(np.asarray(charged)<=np.asarray(neutral))
    assert np.all(np.asarray(settling)==0)
    _,settling=collision_kernels(particle,pivots,210.,610.,3.72,settling_efficiency=1.)
    np.testing.assert_array_equal(np.diag(np.asarray(settling)),np.zeros(4))
    zero,_=collision_kernels(particle,pivots,210.,610.,3.72,sticking_efficiency=0.)
    assert np.all(np.asarray(zero)==0)
    with pytest.raises(ValueError,match='force law'):
        collision_kernels(particle,pivots,210.,610.,3.72,charges_e=1.,settling_efficiency=1.)
    with pytest.raises(ValueError,match='dimers'):
        coagulator([1.,3.])
    invalid=coagulator([1.,2.])(jnp.ones(2),jnp.array([[1.,-1.],[0.,1.]]),1.)
    assert not bool(invalid.valid)


def test_larger_size_grid_bounds_unresolved_material():
    unresolved=[]
    for size in (8,16,32):
        pivots=jnp.arange(1.,size+1.)
        step=coagulator(np.asarray(pivots))
        kernel=jnp.ones((size,size))
        def advance(carry,_):
            number,overflow=carry
            result=step(number,kernel,2/128)
            return (result.number_m3,overflow+result.unresolved_mass_kg_m3),None
        (number,overflow),_=jax.lax.scan(advance,
            (jnp.zeros(size).at[0].set(1.),jnp.asarray(0.)),None,length=128)
        assert float(jnp.dot(number,pivots)+overflow)==pytest.approx(1.,abs=2e-12)
        unresolved.append(float(overflow))
    assert unresolved[2]<unresolved[1]<unresolved[0]
    assert unresolved[2]<1e-6
