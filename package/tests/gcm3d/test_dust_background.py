"""Material and derivative checks for scenario-controlled dust."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.dust_lifting import background_step

jax.config.update('jax_enable_x64',True)
HALF=jnp.array([0.,50.,200.,400.,610.])
MASS=jnp.array([3e-6,2e-6,1e-6,1e-7])
NUMBER=MASS*1e13


def update(current=.1,scenario=.2,**options):
    return background_step(MASS,NUMBER,HALF,1,230.,0.,current,scenario,1.,30.,300.,
        extinction_efficiency=2.45,**options)


@pytest.mark.parametrize('injection_type',range(1,8))
@pytest.mark.parametrize('pbl',[False,True])
def test_scenario_injection_preserves_surface_plus_air_mass(injection_type,pbl):
    result=update(injection_type=injection_type,inject_through_pbl=pbl)
    air_change=jnp.sum((result.mass_ratio-MASS)*jnp.diff(HALF)/3.72)
    np.testing.assert_allclose(air_change+result.surface_dust_kg_m2,30.,rtol=1e-15,atol=1e-14)
    assert air_change>0.
    np.testing.assert_array_equal(result.mass_ratio[:1 if pbl else 3],MASS[:1 if pbl else 3])


@pytest.mark.parametrize('bottom',[False,True])
def test_sink_removes_across_cells_without_negative_material(bottom):
    result=update(current=10.,scenario=0.,remove_excess=True,bottom_sink=bottom)
    air_change=jnp.sum((result.mass_ratio-MASS)*jnp.diff(HALF)/3.72)
    np.testing.assert_allclose(air_change+result.surface_dust_kg_m2,30.,rtol=1e-15,atol=1e-14)
    assert np.all(result.mass_ratio>=0.) and np.all(result.number_ratio>=0.)
    assert air_change<0.
    if bottom: assert result.mass_ratio[-1]==0. and result.mass_ratio[-2]<MASS[-2]


def test_all_source_injection_masks_and_local_time_fraction():
    for ts,snow,change,fraction in ((100.,0.,1.,1.),(230.,1.,1.,1.),
        (230.,0.,-1000.,1.),(230.,0.,1.,0.)):
        result=background_step(MASS,NUMBER,HALF,1,ts,snow,.1,.2,change,30.,300.,
            extinction_efficiency=2.45,local_time_fraction=fraction)
        np.testing.assert_array_equal(result.mass_ratio,MASS)


def test_injection_and_bottom_sink_keep_reverse_derivatives():
    for bottom in (False,True):
        objective=(lambda x:jnp.sum(update(current=x,scenario=0.,remove_excess=True,
            bottom_sink=True).mass_ratio)) if bottom else (
            lambda x:jnp.sum(update(scenario=x,injection_type=4).mass_ratio))
        point=10. if bottom else .2;eps=1e-5
        auto=jax.jit(jax.grad(objective))(point)
        fd=(objective(point+eps)-objective(point-eps))/(2.*eps)
        assert np.isfinite(auto) and abs(float(auto))>0.
        np.testing.assert_allclose(auto,fd,rtol=1e-7,atol=1e-14)


def test_source_negative_sink_allowance_is_repaired_and_reported():
    result=background_step(jnp.zeros(4),jnp.zeros(4),HALF,1,230.,0.,10.,0.,-1.,30.,300.,
        extinction_efficiency=2.45,remove_excess=True)
    assert np.all(result.mass_ratio>=0.) and np.all(result.number_ratio>=0.)
    assert result.diagnostics['negative_sink_repair_kg_m2']>0.
