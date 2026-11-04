"""CI-only angular checks independent of the production two-stream equations."""
import numpy as np
import pytest
pytest.importorskip('dinosaur')
import jax
jax.config.update('jax_enable_x64',True)
from scipy.special import expn
from src.terraforming.aerosols.angular_reference import homogeneous_slab,compare_two_stream,layered_column,compare_two_stream_column


def parameters(**changes):
    values=dict(optical_depth=.7,single_scattering_albedo=0.,asymmetry=0.,
        blackbody_w_m2=0.,down_top_w_m2=100.,surface_emission_w_m2=0.,angles=64)
    values.update(changes)
    return values


def test_absorbing_angular_flux_matches_exponential_integral():
    result=homogeneous_slab(**parameters())
    assert result['down_w_m2'][-1]==pytest.approx(200*expn(3,.7),rel=1e-7)
    np.testing.assert_allclose(result['up_w_m2'],0.,atol=1e-12)


def test_absorption_emission_and_reflective_surface_match_angular_solution():
    result=homogeneous_slab(**parameters(blackbody_w_m2=40.,surface_emission_w_m2=20.,surface_reflectivity=.3))
    transmission=2*expn(3,.7)
    down=40+(100-40)*transmission
    surface=20+.3*down
    assert result['down_w_m2'][-1]==pytest.approx(down,rel=1e-7)
    assert result['up_w_m2'][0]==pytest.approx(40+(surface-40)*transmission,rel=1e-7)


def test_isotropic_equilibrium_and_conservative_scattering():
    equilibrium=homogeneous_slab(**parameters(optical_depth=3.,single_scattering_albedo=.95,asymmetry=.2,
        blackbody_w_m2=170.,down_top_w_m2=170.,surface_emission_w_m2=119.,surface_reflectivity=.3,angles=16))
    np.testing.assert_allclose(equilibrium['up_w_m2'],170.,atol=1e-7)
    np.testing.assert_allclose(equilibrium['down_w_m2'],170.,atol=1e-7)
    scattering=homogeneous_slab(**parameters(optical_depth=2.,single_scattering_albedo=1.,angles=16,surface_reflectivity=.2))
    np.testing.assert_allclose(scattering['layer_convergence_w_m2'],0.,atol=1e-6)


def test_forward_scattering_limit_and_explicit_failed_accuracy_gate():
    forward=homogeneous_slab(**parameters(optical_depth=5.,single_scattering_albedo=1.,asymmetry=1.,surface_reflectivity=.2))
    np.testing.assert_allclose(forward['down_w_m2'],100.,atol=1e-12)
    np.testing.assert_allclose(forward['up_w_m2'],20.,atol=1e-12)
    # Hemispheric exp(-2*tau) and angular 2*E3(tau) differ at tau=1.
    # The reference must expose that error, rather than call both exact.
    comparison=compare_two_stream(**parameters(optical_depth=1.),
        max_flux_error_fraction=.01,max_layer_heating_error_fraction=.01)
    assert not comparison['passes_frozen_gate']
    assert comparison['maximum_flux_error_fraction']>.01
    assert comparison['frozen_limits']['flux_fraction']==.01


def test_layered_reference_preserves_continuity_and_homogeneous_solution():
    column=layered_column(optical_depths=[.2,.2,.2],single_scattering_albedos=[.7]*3,asymmetries=[.2]*3,
        blackbody_w_m2=[60.]*3,down_top_w_m2=0.,surface_emission_w_m2=150.,surface_reflectivity=.1,angles=16)
    slab=homogeneous_slab(**parameters(optical_depth=.6,single_scattering_albedo=.7,asymmetry=.2,
        blackbody_w_m2=60.,down_top_w_m2=0.,surface_emission_w_m2=150.,surface_reflectivity=.1,angles=16,samples=4))
    np.testing.assert_allclose(column['up_w_m2'],slab['up_w_m2'],rtol=1e-7,atol=1e-7)
    np.testing.assert_allclose(column['down_w_m2'],slab['down_w_m2'],rtol=1e-7,atol=1e-7)
    assert column['boundary_residual']<1e-7


def test_temperature_profile_comparison_records_errors_and_frozen_limits():
    case=dict(optical_depths=[.1,.4,.5],single_scattering_albedos=[.5,.95,.8],asymmetries=[0.,.2,-.1],
        blackbody_w_m2=[30.,70.,140.],down_top_w_m2=0.,surface_emission_w_m2=200.,angles=16)
    reference=layered_column(**case)
    assert reference['boundary_residual']<1e-7
    assert np.all(np.isfinite(reference['up_w_m2']))
    comparison=compare_two_stream_column(**case,max_flux_error_fraction=.01,max_layer_heating_error_fraction=.01)
    assert np.isfinite(comparison['maximum_flux_error_fraction'])
    assert np.isfinite(comparison['maximum_layer_heating_error_fraction'])
    assert comparison['interfaces']==4
    assert comparison['frozen_limits']['layer_heating_fraction']==.01
