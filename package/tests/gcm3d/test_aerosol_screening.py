"""CI tests of bounded lifetime sensitivities; no measured Mars-life assertions."""
import math
import pytest
pytest.importorskip('dinosaur')
import jax
jax.config.update('jax_enable_x64',True)
from src.terraforming.aerosols import aluminium_rod
from src.terraforming.aerosols.screening import lifetime_screen,CollectorScenario


def arguments(**overrides):
    result=dict(horizon_s=100.,dt_s=10.,column_mass_kg_m2=1e-6,mixing_depth_m=1000.,
        temperature_k=210.,pressure_pa=610.,gravity_m_s2=3.72,year_convention='earth_julian',
        max_monomers=4)
    result.update(overrides)
    return result


def test_prescribed_pulse_has_distinct_reservoir_and_refined_residence_integral():
    errors=[]
    for dt in (10.,2.,.5):
        config=arguments();config['dt_s']=dt
        report=lifetime_screen(aluminium_rod(),**config,removal_mode='prescribed',
            prescribed_lifetime_s=20.,prescribed_source='analytic pulse check, not a Mars lifetime',sticking_efficiency=0.)
        assert report['status']=='completed'
        assert report['suspended_mass_kg_m2']==pytest.approx(1e-6*math.exp(-5),rel=1e-12)
        assert report['deposited_mass_kg_m2']==0.
        assert report['prescribed_removed_mass_kg_m2']>0
        assert abs(report['material_budget_error_kg_m2'])<1e-18
        residence=report['finite_horizon_residence_lower_bounds_s']['mass']
        upper=report['finite_horizon_residence_upper_sums_s']['mass']
        assert residence<=20*(1-math.exp(-5))<=upper
        assert residence<=20.
        errors.append(abs(residence-20*(1-math.exp(-5))))
        assert report['radiatively_effective_residence_time_s'] is None
        assert report['roadmap_criterion_s']==365.25*86400/3
    assert errors[2]<errors[1]<errors[0]


def test_large_step_rectangle_bound_does_not_become_a_false_long_lifetime():
    report=lifetime_screen(aluminium_rod(),**arguments(dt_s=100.),removal_mode='prescribed',
        prescribed_lifetime_s=20.,prescribed_source='analytic pulse test',sticking_efficiency=0.)
    assert report['finite_horizon_residence_lower_bounds_s']['mass']<20.
    assert report['finite_horizon_residence_upper_sums_s']['mass']>=20*(1-math.exp(-5))


def test_joint_resolved_sinks_are_positive_conservative_and_refine():
    collector=CollectorScenario(2e-6,2500.,1e8,'explicit dilute stress scenario; not inferred from dust opacity')
    errors=[]
    for dt in (1000.,250.,62.5):
        config=arguments();config.update(horizon_s=10000.,dt_s=dt,mixing_depth_m=10.)
        report=lifetime_screen(aluminium_rod(),**config,sticking_efficiency=0.,collector=collector)
        assert report['status']=='completed'
        for field in ('suspended_mass_kg_m2','deposited_mass_kg_m2','collected_mass_kg_m2'):
            assert report[field]>=0
        assert abs(report['material_budget_error_kg_m2'])<1e-18
        dep=report['assumptions']['monomer_deposition_or_prescribed_rate_s']
        coll=report['assumptions']['monomer_collection_rate_s']
        expected=1e-6*dep/(dep+coll)*(1-math.exp(-(dep+coll)*10000))
        errors.append(abs(report['deposited_mass_kg_m2']-expected))
    assert errors[2]<errors[1]<errors[0]


def test_year_units_and_unsupported_combinations_fail_explicitly():
    config=arguments();config['year_convention']='year'
    with pytest.raises(ValueError,match='explicitly'):
        lifetime_screen(aluminium_rod(),**config)
    with pytest.raises(ValueError,match='separate'):
        lifetime_screen(aluminium_rod(),**arguments(),removal_mode='prescribed',prescribed_lifetime_s=10.,prescribed_source='scenario')
    with pytest.raises(ValueError,match='provenance'):
        CollectorScenario(1e-6,2500.,1.,'')
    with pytest.raises(ValueError,match='capture law'):
        lifetime_screen(aluminium_rod(),**arguments(),charge_e=1.,collector=CollectorScenario(1e-6,2500.,1.,'scenario'))
