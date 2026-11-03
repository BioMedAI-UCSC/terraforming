"""Independent boundary-value checks for particle thermal scattering."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp
from scipy.linalg import expm
from src.framework.physics.twostream import layer_coefficients, solve_fluxes, symmetric_embed
from src.terraforming.aerosols import aluminium_rod
from src.terraforming.aerosols.data import rod_table
from src.terraforming.aerosols.optics import make_band_table, integrate_band

jax.config.update('jax_enable_x64', True)


@pytest.mark.parametrize('tau,omega,g', [(0., .5, .2), (.7, 0., 0.), (2., .8, -.2), (4., 1., .6)])
def test_layer_against_independent_matrix_boundary_problem(tau, omega, g):
    a=(1-omega*(1+g)/2)/.5
    b=omega*(1-g)/(2*.5)
    # d(D,U)/dtau=(-a D+b U, -b D+a U), D(0)=1,U(tau)=0.
    matrix=expm(tau*np.array([[-a,b],[-b,a]]))
    expected_r=-matrix[1,0]/matrix[1,1]
    expected_t=matrix[0,0]+matrix[0,1]*expected_r
    r,t=layer_coefficients(tau,omega,g)
    np.testing.assert_allclose([r,t],[expected_r,expected_t],atol=2e-12)


def test_thermal_equilibrium_and_conservative_scattering():
    r,t=layer_coefficients(jnp.array([.2,1.,3.]),.8,.3)
    up,down=jax.jit(solve_fluxes)(r,t,jnp.full(3,170.),170.,.7*170.,.3)
    np.testing.assert_allclose(up,170.,atol=1e-11)
    np.testing.assert_allclose(down,170.,atol=1e-11)
    r,t=layer_coefficients(jnp.array([.2,1.,3.]),1.,.3)
    up,down=solve_fluxes(r,t,jnp.zeros(3),100.,0.,.2)
    np.testing.assert_allclose(down-up,(down-up)[0],atol=1e-11)


def test_zero_particle_embedding_and_gradient():
    r,t=symmetric_embed(.2,.6,0.,1.)
    np.testing.assert_allclose([r,t],[.2,.6],atol=1e-14)
    derivative=jax.grad(lambda x: sum(layer_coefficients(x,.8,.2)))(0.)
    assert np.isfinite(derivative)
    assert derivative<0


def test_source_band_quadrature_converges_and_mass_scales():
    particle=aluminium_rod(); table=rod_table(particle)
    a=integrate_band(table,8e-6,24e-6,215.,tail='source_constant')
    b=integrate_band(table,8e-6,24e-6,215.,tail='source_constant',points_per_interval=48)
    np.testing.assert_allclose(a,b,rtol=2e-10)
    bands=make_band_table(table,particle,temperature_step_k=50.)
    burden=jnp.full((2,1,1),1e-6); temperature=jnp.full((2,1,1),200.)
    one=bands.optical_depths(burden,temperature)
    two=jax.jit(bands.optical_depths)(2*burden,temperature)
    for name in one:
        np.testing.assert_allclose(two[name],2*one[name],rtol=1e-12)
    assert np.all(np.asarray(one['lw_extinction'])>=np.asarray(one['lw_scattering']))


@pytest.mark.parametrize('temperature',[100.,215.,350.])
def test_planck_band_against_independent_adaptive_quadrature(temperature):
    from scipy.integrate import quad
    from src.terraforming.aerosols.optics import planck_lambda
    table=rod_table(aluminium_rod())
    lo,hi=8e-6,24e-6
    knots=table.wavelength_m[(table.wavelength_m>lo)&(table.wavelength_m<hi)]
    # Scale the integrand to avoid absolute-tolerance dominance for SI spectra.
    norm=quad(lambda x:float(planck_lambda(x,temperature)),lo,hi,points=knots,epsabs=1e-12)[0]
    expected=quad(lambda x:float(planck_lambda(x,temperature))*
        np.interp(x,table.wavelength_m,table.absorption_m2)*1e12,
        lo,hi,points=knots,epsabs=1e-12)[0]/norm/1e12
    actual=integrate_band(table,lo,hi,temperature,tail='source_constant')[0]
    assert actual==pytest.approx(expected,rel=1e-8)


def test_cross_section_and_mass_opacity_are_consistent():
    particle=aluminium_rod();table=rod_table(particle)
    burden=2e-6;number=burden/particle.mass_kg
    cross=table.absorption_m2+table.scattering_m2
    np.testing.assert_allclose(cross/particle.mass_kg*burden,cross*number,rtol=2e-15)


def test_coupled_ames_zero_loading_parity_and_energy_budget():
    from src.framework.physics.gcm import column_radiative_fluxes, RadiativeForcing
    from src.celestials.planets.mars import MARS_BODY_3D
    particle=aluminium_rod()
    bands=make_band_table(rod_table(particle),particle,temperature_step_k=50.)
    air=jnp.array([150.,180.,210.])[:,None,None]
    import dataclasses
    from src.celestials.planets.mars import gcm as mars_gcm
    forcing=dataclasses.replace(mars_gcm.radiative_forcing(co2_radiation_enabled=True),emissivity=1.)
    def run(burden):
        return column_radiative_fluxes(air,jnp.array([[220.]]),610.,300.,1.,.2,.1,
            np.linspace(0,1,4),MARS_BODY_3D,forcing,
            aerosol_optics=None if burden is None else bands.optical_depths(jnp.full_like(air,burden),air))
    baseline=run(None); zero=run(0.)
    for before,after in zip(baseline,zero):
        np.testing.assert_allclose(after,before,atol=2e-11)
    active=run(1e-6)
    np.testing.assert_allclose(active.atmospheric_convergence_w_m2.sum(axis=0)+active.surface_net_w_m2,
        active.toa_net_down_w_m2,atol=2e-11)
    assert all(np.all(np.isfinite(x)) for x in active)
