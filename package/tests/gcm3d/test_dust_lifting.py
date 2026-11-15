"""Independent material and reverse-gradient checks for interactive lifting."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.dust_lifting import stress_flux,devil_flux,interactive_step

jax.config.update('jax_enable_x64',True)


@pytest.mark.parametrize('scheme',[1,2,3,4,5])
def test_stress_masks_strict_boundaries_and_finite_calm_derivative(scheme):
    def flux(stress,lat=0.,snow=0.,frost=0.,tau=0.):
        return stress_flux(stress,.022,.01,lat,snow,frost,tau,scheme=scheme)
    assert flux(0.)==0. and np.isfinite(jax.grad(flux)(0.))
    assert flux(.05)>0.
    assert flux(.05,snow=2.,lat=0.)>0.
    assert flux(.05,snow=2.,lat=.5)==0.
    assert flux(.05,snow=2.,lat=-.5)==0.
    assert flux(.05,frost=1.,tau=100.)>0.
    assert flux(.05,frost=1.001)==0. and flux(.05,tau=100.001)==0.


@pytest.mark.parametrize('pbl',[False,True])
@pytest.mark.parametrize('limited',[False,True])
def test_lifting_transfers_surface_mass_into_selected_layers(pbl,limited):
    half=jnp.array([0.,50.,200.,400.,610.]);full=.5*(half[:-1]+half[1:])
    q=jnp.full(4,1e-5);number=jnp.full(4,1e8);t=jnp.full(4,230.)
    result=interactive_step(q,number,t,half,full,260.,.05,20.,1,200.,.5,0.,0.,.1,30.,100.,
        inject_pbl_stress=pbl,inject_pbl_devils=pbl,limited_reservoir=limited)
    change=jnp.sum(jnp.diff(half)/3.72*(result.mass_ratio-q))
    np.testing.assert_allclose(change+result.surface_dust_kg_m2,30.,atol=1e-14,rtol=1e-15)
    np.testing.assert_array_equal(result.mass_ratio[:1 if pbl else 3],q[:1 if pbl else 3])
    assert result.surface_dust_kg_m2<30.
    particle_mass=4./3.*jnp.pi*2500.*(2e-6*jnp.exp(-.5*.63676**2))**3
    np.testing.assert_allclose((result.number_ratio-number)*particle_mass,result.mass_ratio-q,
        rtol=1e-10,atol=1e-20)


@pytest.mark.parametrize('scheme',[1,2,3,4,5])
def test_stress_reverse_derivative_matches_centered_difference(scheme):
    objective=lambda x:stress_flux(x,.022,.01,.5,0.,0.,.1,scheme=scheme)
    auto=jax.jit(jax.grad(objective))(.05);eps=1e-7
    fd=(objective(.05+eps)-objective(.05-eps))/(2.*eps)
    np.testing.assert_allclose(auto,fd,rtol=1e-8,atol=1e-14)


def test_collapsed_pbl_devils_have_finite_limits_and_gradients():
    objective=lambda t:devil_flux(t,260.,500.,500.,3,3,20.,.5,0.,0.,dda=True,dth=True)
    assert np.isfinite(objective(230.))
    assert np.isfinite(jax.grad(objective)(230.))


@pytest.mark.parametrize('mode',[1,2,21])
def test_dual_opacity_gate_preserves_strict_boundary_and_both_sources(mode):
    half=jnp.array([0.,50.,200.,400.,610.]);full=.5*(half[:-1]+half[1:])
    def lift(opacity):
        return interactive_step(jnp.full(4,1e-5),jnp.full(4,1e8),jnp.full(4,230.),
            half,full,260.,.05,20.,1,200.,.5,0.,0.,opacity,30.,300.,
            dual_mode=mode,dual_scale=.5,scenario_opacity=.2)
    boundary=lift(.1);above=lift(.10001)
    assert boundary.diagnostics['devil_lift_kg_m2']>0.
    assert boundary.diagnostics['stress_lift_kg_m2']>0.
    if mode in (1,21):
        assert above.diagnostics['devil_lift_kg_m2']==0.
        assert above.diagnostics['stress_lift_kg_m2']==0.
        assert above.surface_dust_kg_m2==30.
    else:
        assert above.surface_dust_kg_m2<30.
