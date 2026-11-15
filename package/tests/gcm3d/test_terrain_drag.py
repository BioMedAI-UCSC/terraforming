import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from src.framework.physics.terrain_drag import tendencies
from scripts.validate_terrain_drag import column

jax.config.update('jax_enable_x64',True)


def test_terrain_drag_surface_cap_and_reported_force_budget():
    args=column(12,'sheared')
    result=jax.jit(tendencies)(*args,1500.)
    cap=tendencies(*args,1000.)
    np.testing.assert_allclose(result.u_tendency,cap.u_tendency,rtol=0.,atol=1e-15)
    np.testing.assert_array_equal(result.u_tendency[-4:],0.)
    u,v,t,p,ph,z,zh=args
    direction=np.array([u[-1],v[-1]])/np.hypot(u[-1],v[-1])
    force=np.array([jnp.sum(x*jnp.diff(ph)/3.72) for x in (result.u_tendency,result.v_tendency)])
    np.testing.assert_allclose(force+result.surface_stress_pa*direction,
        result.unresolved_stress_pa*direction,rtol=2e-13,atol=2e-15)


@pytest.mark.parametrize('calm',[False,True])
def test_zero_shear_and_calm_wind_have_finite_forward_and_reverse_derivatives(calm):
    args=list(column(12,'sheared'))
    args[0]=jnp.full(12,0. if calm else 30.);args[1]=jnp.zeros(12)
    def objective(offset):
        result=tendencies(args[0]+offset,args[1],*args[2:],300.)
        return jnp.sum(result.u_tendency+result.v_tendency)
    value,grad=jax.jit(jax.value_and_grad(objective))(0.)
    assert np.isfinite(value) and np.isfinite(grad)
    if calm: assert value==0.


def test_terrain_drag_reverse_gradient_matches_centered_differences():
    args=column(12,'sheared')
    def objective(sd):
        result=tendencies(*args,sd)
        return jnp.sum(result.u_tendency*jnp.arange(12))+jnp.sum(result.v_tendency)
    auto=jax.jit(jax.grad(objective))(100.)
    assert np.isfinite(auto) and abs(float(auto))>1e-10
    for eps in (1e-2,1e-3):
        finite=(objective(100.+eps)-objective(100.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-7,atol=1e-11)


def test_hydrostatic_heights_match_source_layer_mean_formula_without_changing_zero_top_mass():
    from src.framework.physics.terrain_drag import hydrostatic_heights
    ph=jnp.array([0.,1.,10.,100.,610.]);p=jnp.array([.1,5.,50.,350.])
    z,zh=hydrostatic_heights(jnp.full(4,200.),ph,p,top_pressure_pa=.001)
    effective=ph.at[0].set(.001)
    expected_bottom=189.02*200./3.72*jnp.log(610./ph[1:])
    np.testing.assert_allclose(z,expected_bottom+189.02*200./3.72*(1.-effective[:-1]/p),rtol=2e-15)
    np.testing.assert_allclose(zh,189.02*200./3.72*jnp.log(610./effective),atol=3e-11)
    assert ph[0]==0.
    invalid,_=hydrostatic_heights(jnp.full(4,200.),ph,p,top_pressure_pa=1.)
    assert np.isnan(invalid).all()
