"""Small planning/plotting regressions; no JAX/GCM simulation is invoked."""
from argparse import Namespace
import json

import numpy as np
import pytest

from scripts import aerosol_paper_experiments as experiments


def make_plan(tmp_path, lifetime_year="earth"):
    experiments.plan(Namespace(output=tmp_path, sols=1., spinup_sols=1., dt=60.,
        layers=12, truncation="T21", material="aluminium", lifetime_year=lifetime_year))
    return experiments.read_json(tmp_path / "suite.json")


def test_plan_keeps_loading_release_and_lifetime_assumptions_distinct(tmp_path):
    manifest = make_plan(tmp_path)
    cases = {case["name"]: case for case in manifest["cases"]}
    assert cases["loading_2mg"]["configuration"]["intervention"]["aerosol"]["initial_burden_kg_m2"] == 2e-6
    for name, rate in (("release_2l_s", 2.), ("release_60l_s", 60.)):
        aerosol = cases[name]["configuration"]["intervention"]["aerosol"]
        assert aerosol["initial_burden_kg_m2"] == 0.
        assert aerosol["sources"][0]["rate"] == rate
        assert aerosol["sources"][0]["unit"] == "L/s"
        assert aerosol["hyperdiffusion_tau_seconds"] > 0
    assert cases["release_60l_s"]["expectation_kind"] == "greater_than"
    assert cases["shutdown_2l_s"]["configuration"]["intervention"]["aerosol"]["sources"][0]["stop_s"] == manifest["duration_s"] / 2
    assert manifest["assumptions"]["effective_lifetime_s"] == experiments.EARTH_YEAR_S
    assert not manifest["assumptions"]["lifetime_is_simulated"]
    assert not (tmp_path / "runtime.json").exists()
    with pytest.raises(ValueError, match="already exists"):
        make_plan(tmp_path)


def test_one_box_reference_has_analytic_release_and_shutdown_limits():
    # Before shutdown M=rate*tau*(1-exp(-t/tau)); afterward it decays.
    values = experiments.reference_inventory([0., 10., 20.], 0., 3., 10., 100.)
    at_shutdown = 300 * (1-np.exp(-.1))
    np.testing.assert_allclose(values, [0., at_shutdown, at_shutdown*np.exp(-.1)])
    np.testing.assert_allclose(experiments.reference_inventory([0., 100.], 7., 0., 0., 100.),
                               [7., 7./np.e])


def test_plot_partial_history_keeps_actual_warming_and_reference_separate(tmp_path):
    manifest = make_plan(tmp_path, "mars")
    case = manifest["cases"][0]
    directory = tmp_path / case["name"]
    experiments.write_json(directory / "configuration.json", dict(cases=dict(active=dict(
        body=dict(radius_m=3389500.), particle=dict(density_kg_m3=2700.),
        initial_burden_kg_m2=2e-6, forcing=dict(orbital_period_s=experiments.MARS_YEAR_S)))))
    rows = []
    for elapsed, warming in ((60., .01), (experiments.MARS_YEAR_S, .02)):
        active = dict(surface_temperature_k=210.+warming, suspended_mass_kg=2e8,
                      injected_mass_kg=0., material_budget_error_kg=1e-8,
                      maximum_raw_relative_mass_error=1e-12)
        rows.append(dict(elapsed_seconds=elapsed, status="running", cases=dict(active=active,
            off=dict(surface_temperature_k=210.), passive=dict(surface_temperature_k=210.))))
    rows[-1]["warming"] = dict(completed_mars_years=1, consecutive_annual_warming_k=[.015],
                                paired_time_mean_warming_k=.015)
    # A plot of a live job ignores only the unfinished final JSON line.
    (directory / "history.ndjson").write_text("\n".join(json.dumps(r) for r in rows)+"\n{\"partial\":")
    experiments.plot(Namespace(output=tmp_path))
    comparison = experiments.read_json(tmp_path / "paper_comparison.json")["cases"][0]
    assert comparison["final_warming_k"] == pytest.approx(.02)
    assert comparison["paper_reference_warming_k"] == 4.
    assert comparison["reference_year"] == "mars"
    assert comparison["status"] == "running"
    assert comparison["simulated_mars_years"] == pytest.approx(1.)
    assert comparison["last_complete_annual_warming_k"] == [.015]
    assert (tmp_path / "paper_comparison.png").is_file()
    assert (tmp_path / "annual_warming.png").is_file()
    assert (tmp_path / "paper_comparison.csv").is_file()


def test_damping_configuration_rejects_nonpositive_timescale():
    from cli.models import AerosolConfig
    with pytest.raises(ValueError):
        AerosolConfig(hyperdiffusion_tau_seconds=0.)


@pytest.mark.parametrize('options',[
    ['--spinup-sols','1'],['--lifetime-year','earth']])
def test_planner_requires_explicit_spinup_and_year_convention(tmp_path,monkeypatch,options):
    monkeypatch.setattr('sys.argv',['aerosol','plan','--output',str(tmp_path),*options])
    with pytest.raises(SystemExit) as error:
        experiments.main()
    assert error.value.code==2
    assert not (tmp_path/'suite.json').exists()
