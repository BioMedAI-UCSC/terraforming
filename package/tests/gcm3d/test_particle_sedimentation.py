"""Independent material and derivative checks for moment settling."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.particle_sedimentation import sedimentation_step,dust_sedimentation_step

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('modes',[1,2,3])
@pytest.mark.parametrize('dt',[5.,30000.])
def test_dust_only_settling_closes_each_population_without_water_slots(modes,dt):
    half=jnp.array([0.,50.,200.,400.,610.]);full=.5*(half[1:]+half[:-1])
    pairs=jnp.array([1e-5,1e8]*modes)[None]*jnp.array([.5,1.,2.,5.])[:,None]
    result=dust_sedimentation_step(pairs,half,full,jnp.array([150.,180.,200.,220.]),230.,
        jnp.array([.001,1.,10.]),dt)
    gas=jnp.diff(half)/3.72
    change=jnp.sum(gas[:,None]*(result.tracers-pairs),axis=0)
    np.testing.assert_allclose(change[::2]+result.surface_deposit,0.,atol=2e-18)
    np.testing.assert_allclose(change[1::2]+result.diagnostics['per_species_deposit'][1::2],0.,atol=1e-5)
    assert result.tracers.shape==pairs.shape and result.surface_deposit.shape==(modes,)
    assert result.radius_m.shape==pairs.shape
    assert result.diagnostics['water_budget_residual']==0.


def test_dust_only_settling_lifting_mask_and_reverse_derivative():
    half=jnp.array([0.,50.,200.,400.,610.]);full=.5*(half[1:]+half[:-1]);q=jnp.full((4,2),1e-5).at[:,1].set(1e8)
    def settle(amplitude,mask=None):
        return dust_sedimentation_step(q*amplitude,half,full,jnp.array([150.,180.,200.,220.]),
            230.,jnp.array([.001,1.,10.]),300.,lifting_mask=mask)
    held=settle(1.,jnp.array([True,False]))
    assert held.surface_deposit[0]==0.
    np.testing.assert_array_equal(held.tracers[-1,0],q[-1,0])
    objective=lambda a:settle(a).surface_deposit[0]
    auto=jax.grad(objective)(1.);eps=1e-5
    assert np.isfinite(auto) and auto>0.
    np.testing.assert_allclose(auto,(objective(1.+eps)-objective(1.-eps))/(2.*eps),rtol=3e-7,atol=1e-15)


def column(dt,scavenging=True,lifting_mask=None,amplitude=1.):
    half=jnp.array([0.,50.,200.,400.,610.]);full=.5*(half[1:]+half[:-1])
    t=jnp.array([150.,180.,200.,220.])
    q=jnp.array([1e-5,1e8,1e-6,1e6,1e-8,1e-6])[None]*jnp.array([.5,1.,2.,5.])[:,None]*amplitude
    result=sedimentation_step(q,half,full,t,230.,jnp.array([.001,1.,10.]),dt,
        scavenging=scavenging,lifting_mask=lifting_mask)
    return q,result


@pytest.mark.parametrize('dt',[5.,300.,3e4])
@pytest.mark.parametrize('scavenging',[False,True])
def test_settling_closes_water_dust_and_reports_number_loss(dt,scavenging):
    q,result=column(dt,scavenging)
    assert np.min(result.tracers)>=0.
    for key in ('dust_budget_residual','water_budget_residual'):
        np.testing.assert_allclose(result.diagnostics[key],0.,atol=2e-18)
    np.testing.assert_array_equal(result.tracers[...,5],q[...,5])
    if not scavenging: np.testing.assert_array_equal(result.tracers[...,4],q[...,4])
    change=result.diagnostics['atmospheric_change']
    deposit=result.diagnostics['per_species_deposit']
    indices=jnp.array([1,3])
    np.testing.assert_allclose(change[indices]+deposit[indices],0.,atol=1e-5)


def test_settling_reverse_derivative_matches_finite_differences():
    def objective(amplitude):
        _,result=column(300.,amplitude=amplitude)
        return result.surface_deposit[0]+result.surface_deposit[5]
    auto=jax.jit(jax.grad(objective))(1.);eps=1e-5
    fd=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
    assert np.isfinite(auto) and auto>0.
    np.testing.assert_allclose(auto,fd,rtol=3e-7,atol=1e-15)


def test_source_lifting_override_retains_bottom_and_exposes_budget_error():
    mask=jnp.array([True,False,False,False,False,False])
    q,result=column(300.,lifting_mask=mask)
    np.testing.assert_array_equal(result.tracers[-1,0],q[-1,0])
    assert result.surface_deposit[0]==0.
    assert abs(float(result.diagnostics['dust_budget_residual']))>0.


@pytest.mark.parametrize('scavenging',[False,True])
def test_additional_physical_dust_modes_close_material_and_use_own_size(scavenging):
    q,_=column(300.)
    half=jnp.array([0.,50.,200.,400.,610.]);full=.5*(half[1:]+half[:-1])
    extra=jnp.stack([q[...,0]*2.,q[...,1]*.25,q[...,4]*.5],axis=-1)
    q=jnp.concatenate([q,extra],axis=-1)
    def settle(amplitude):
        return sedimentation_step(q.at[...,6:].multiply(amplitude),half,full,
            jnp.array([150.,180.,200.,220.]),230.,jnp.array([.001,1.,10.]),300.,scavenging=scavenging)
    result=settle(1.)
    change=result.diagnostics['atmospheric_change']
    np.testing.assert_allclose(change[6]+change[8]+result.surface_deposit[6],0.,atol=2e-18)
    np.testing.assert_allclose(change[7]+result.diagnostics['per_species_deposit'][6],0.,atol=1e-5)
    np.testing.assert_allclose(result.diagnostics['dust_budget_residual'],0.,atol=2e-18)
    assert np.all(result.radius_m[...,5]>result.radius_m[...,0])
    np.testing.assert_array_equal(result.radius_m[...,7],result.radius_m[...,2])
    if not scavenging: np.testing.assert_array_equal(result.tracers[...,8],q[...,8])
    objective=lambda a:settle(a).surface_deposit[6]
    auto=jax.jit(jax.grad(objective))(1.);eps=1e-5
    np.testing.assert_allclose(auto,(objective(1.+eps)-objective(1.-eps))/(2.*eps),rtol=3e-7,atol=1e-15)
