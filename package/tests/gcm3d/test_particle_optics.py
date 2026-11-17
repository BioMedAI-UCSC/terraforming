import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.particle_optics import (
    dust_optics,co2_cloud_optics,water_bulk_optics,water_moment_optics,load_tables)

jax.config.update('jax_enable_x64',True)


def pressure():
    return jnp.array([0.,0.,0.,25.,50.,100.,150.,250.,350.,480.,610.])


@pytest.mark.parametrize('kind',['dust','co2','water_bulk','water_moment'])
@pytest.mark.parametrize('clear',[False,True])
def test_particle_gradients_and_zero_ghost_mass(kind,clear):
    mass=jnp.zeros(4) if clear else jnp.array([1e-5,1e-4,1e-3,1e-2])
    def evaluate(scale):
        p=pressure()*scale;q=mass*scale
        if kind=='dust': return dust_optics(q[:,None],jnp.full((4,1),1e8),p,bands=15)
        if kind=='co2': return co2_cloud_optics(q,p,nuclei_per_kg=1e5*scale)
        if kind=='water_bulk': return water_bulk_optics(q,p,jnp.linspace(200.,300.,11),nuclei_per_kg=1e5*scale)
        return water_moment_optics(q,.03*q,jnp.full(4,1e8),p,bands=15)
    def objective(scale):
        result=evaluate(scale)
        return jnp.sum(result.visible[0])+jnp.sum(result.infrared[0])
    actual=jax.jit(evaluate)(1.)
    for spectrum in actual[:2]:
        for value in spectrum:
            assert np.isfinite(value).all()
            np.testing.assert_array_equal(value[:2],0.)
            if clear: np.testing.assert_array_equal(value,0.)
        assert np.min(spectrum[0])>=0.
        assert np.all(np.asarray(spectrum[1])<=.99999*np.asarray(spectrum[0])+1e-15)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto)
    for eps in (1e-4,1e-5):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-6,atol=1e-9)


def test_dust_area_weighting_is_independent_of_population_order_and_conserves_scaling():
    mass=jnp.array([1e-5,1e-4])
    q=jnp.broadcast_to(mass,(4,2));number=jnp.broadcast_to(jnp.array([1e8,2e8]),q.shape)
    original=dust_optics(q,number,pressure())
    flipped=dust_optics(q[:,::-1],number[:,::-1],pressure())
    doubled=dust_optics(q*2.,number*2.,pressure())
    for a,b,c in zip((*original.visible,*original.infrared),
            (*flipped.visible,*flipped.infrared),(*doubled.visible,*doubled.infrared),strict=True):
        np.testing.assert_allclose(a,b,rtol=2e-14,atol=1e-14)
        np.testing.assert_allclose(c,2.*a,rtol=2e-14,atol=1e-14)


def test_batched_particle_columns_and_repeated_tracing_are_independent():
    q=jnp.full((4,2,3),.001)
    p=jnp.broadcast_to(pressure()[:,None,None],(11,2,3))
    for _ in range(2):
        batched=jax.jit(co2_cloud_optics)(q,p)
        single=co2_cloud_optics(q[:,0,0],pressure())
        for a,b in zip(batched.infrared,single.infrared,strict=True):
            np.testing.assert_allclose(a[:,:,0,0],b,rtol=2e-14)


def test_co2_does_not_invent_a_twelve_band_table():
    with pytest.raises(ValueError,match='15-band'):
        co2_cloud_optics(jnp.zeros(4),pressure(),bands=12)


def test_staged_tables_preserve_source_single_precision_and_size_grid():
    data=load_tables()
    assert data['dust_12_sw'].dtype==np.float32
    np.testing.assert_array_equal(data['dust_12_sw'][0,0,0,0],np.float32(.47489e-2))
    assert data['moment_radii_m'][0]==1e-7
    np.testing.assert_allclose(data['moment_radii_m'][-1],50e-6,rtol=2e-14)
    np.testing.assert_allclose(data['co2_radii_m'][-1],260e-6,rtol=2e-14)
    np.testing.assert_allclose(data['bulk_radii_m'][19],50e-6,rtol=2e-14)
    assert data['moment_edges_m'][0]==1e-10 and data['moment_edges_m'][-1]==.01
