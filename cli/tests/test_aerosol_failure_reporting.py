"""Failure explanations and checkpoint probing, with no GCM integration."""
import io
import json
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import aerosol_paper_experiments as experiments
from src.interventions.aerosols.validity import explain_arrays,json_safe


def test_cold_upper_air_within_source_domain_is_not_rejected():
    air=np.full((2,3,2),200.)
    air[1,2,0]=99.5
    zero=np.zeros_like(air)
    details=explain_arrays(air,np.full_like(air,25.),zero,zero,np.ones_like(air),
        max_mass_loading=.001,lat_deg=[-30,30],lon_deg=[0,120,240],sigma=[.25,.75])
    assert details['reasons']==[]
    assert details['air_temperature_k']['min']==99.5
    assert details['air_temperature_k']['minimum_location']==dict(
        layer=1,sigma=.75,longitude_deg=240.,latitude_deg=-30.)
    assert details['limits']['air_temperature_k']==[50.,350.]
    assert details['limits']['layer_pressure_pa']==pytest.approx([.0001,1e6])


def test_failed_values_are_json_safe_without_losing_reason():
    array=np.ones((1,1,1))
    details=explain_arrays(array*360,array*2001,array*0,array*np.nan,array,
        max_mass_loading=.001,lat_deg=[0],lon_deg=[0],sigma=[.5])
    assert 'air_temperature_above_radiation_table' in details['reasons']
    assert 'layer_pressure_above_radiation_table' not in details['reasons']
    assert 'nonfinite_particle_number' in details['reasons']
    json.dumps(details,allow_nan=False)
    assert json_safe(dict(bad=float('inf'),nested=[float('nan')]))==dict(bad=None,nested=[None])


def test_dataset_bounds_and_optional_user_loading_criterion():
    array=np.ones((1,1,1))
    inputs=dict(lat_deg=[0],lon_deg=[0],sigma=[.5])
    supported=explain_arrays(array*80,array*.1,array*.002,array,array,
        max_mass_loading=None,**inputs)
    assert supported['reasons']==[]
    selected=explain_arrays(array*80,array*.1,array*.002,array,array,
        max_mass_loading=.001,**inputs)
    assert selected['reasons']==['particle_mass_loading_exceeded']
    invalid=explain_arrays(array*49,array*1e7,array*0,array,array,
        max_mass_loading=None,**inputs)
    assert invalid['reasons']==['air_temperature_below_radiation_table','layer_pressure_above_radiation_table']


@pytest.mark.parametrize('status,invalid,bad_value',[
    ('invalid_regime',True,0.),('numerical_failure',False,float('inf'))])
def test_stream_keeps_failed_candidate_details_and_last_valid_state(monkeypatch,status,invalid,bad_value):
    from src.interventions.aerosols import reporting
    valid=SimpleNamespace(elapsed_seconds=59280.)
    candidate=SimpleNamespace(elapsed_seconds=59340.,ledger=SimpleNamespace(invalid=invalid))
    def diagnostics(state,*args):
        return dict(elapsed_seconds=state.elapsed_seconds,
            surface_temperature_k=214.782,status=status if state is candidate else 'running',
            numerical_mass_correction_kg=bad_value if state is candidate else 0.)
    monkeypatch.setattr(reporting,'diagnostics',diagnostics)
    monkeypatch.setattr(reporting,'failure_details',lambda *args:dict(
        reasons=['air_temperature_below_100_k'] if invalid else ['nonfinite_diagnostics'],
        candidate_report=json_safe(diagnostics(candidate))))
    stream=io.StringIO()
    retained,report=list(reporting.stream_run(valid,lambda state:candidate,1,None,None,None,None,output=stream))[0]
    assert retained is valid
    assert report['status']==status
    assert report['elapsed_seconds']==59280.
    assert report['attempted_elapsed_seconds']==59340.
    assert report['failure_diagnostics']['candidate_report']['elapsed_seconds']==59340.
    assert json.loads(stream.getvalue())==report


def test_pair_preserves_failed_case_details(monkeypatch):
    from src.interventions.aerosols import paired,experiment
    valid=SimpleNamespace(elapsed_seconds=59280.)
    bad=SimpleNamespace(elapsed_seconds=59340.)
    pair=SimpleNamespace(off=valid,passive=valid,active=valid,warming=None)
    monkeypatch.setattr(paired,'diagnostics',lambda state,**kwargs:dict(
        status='invalid_regime' if state is bad else 'running'))
    monkeypatch.setattr(paired,'paired_report',lambda *args,**kwargs:dict(elapsed_seconds=59280.))
    monkeypatch.setattr(paired,'warming_report',lambda *args:{})
    monkeypatch.setattr(experiment,'failure_details',lambda *args,**kwargs:dict(reasons=['layer_pressure_below_1_pa']))
    stream=io.StringIO()
    retained,report=list(paired.stream_pair(pair,dict(off=lambda s:bad,passive=lambda s:valid,active=lambda s:valid),
        1,None,None,None,None,orbital_period_s=100.,output=stream))[0]
    assert retained is pair
    assert report['failed_cases']==['off']
    assert report['failure_diagnostics']['off']['reasons']==['layer_pressure_below_1_pa']
    assert json.loads(stream.getvalue())==report


def test_probe_does_one_step_without_rewriting_original_checkpoint(tmp_path,monkeypatch):
    from src.interventions.aerosols import pilot,checkpoint,experiment
    directory=tmp_path/'spinup'
    directory.mkdir()
    path=directory/'restart.npz'
    path.write_bytes(b'original saved checkpoint')
    (directory/'summary.json').write_text('{"status":"invalid_regime"}')
    archived=dict(options=dict(mode='off',dt_s=60.,max_relative_transport_correction=1e-4),
        particle=dict(name='aluminium_rod_9um'),forcing=dict(albedo=.25,diurnal=True,
        init_orbital_angle_rad=0.,ames_co2_longwave_opacity_scale=1.,
        ames_dust_longwave_opacity_scale=1.,surface_exchange_multiplier=1.),
        n_layers=12,truncation='T21',sources=[],initial_burden_kg_m2=0.,
        initial_surface_temperature_k=210.,initial_surface_pressure_pa=610.)
    original=SimpleNamespace(elapsed_seconds=59280.)
    candidate=SimpleNamespace(elapsed_seconds=59340.)
    steps=[]
    def step(state):
        assert state is original
        steps.append(state)
        return candidate
    monkeypatch.setattr(pilot,'prepare',lambda **kwargs:(original,step,dict(
        configuration=archived,coords=None,specs=None,body=None,particle=None)))
    monkeypatch.setattr(checkpoint,'read_configuration',lambda path:archived)
    monkeypatch.setattr(checkpoint,'load',lambda *args:original)
    monkeypatch.setattr(experiment,'diagnostics',lambda *args,**kwargs:dict(status='invalid_regime'))
    monkeypatch.setattr(experiment,'failure_details',lambda *args,**kwargs:dict(reasons=['air_temperature_below_100_k']))
    experiments.diagnose_spinup(SimpleNamespace(output=tmp_path,allow_cpu=True))
    assert len(steps)==1
    assert path.read_bytes()==b'original saved checkpoint'
    assert (directory/'summary.json').read_text()=='{"status":"invalid_regime"}'
    probe=experiments.read_json(directory/'failure-probe.json')
    assert probe['checkpoint_elapsed_s']==59280.
    assert probe['attempted_elapsed_s']==59340.
    assert probe['failure_diagnostics']['reasons']==['air_temperature_below_100_k']


def test_plot_reports_spinup_failure_without_fake_comparison(tmp_path,capsys):
    experiments.write_json(tmp_path/'suite.json',dict(cases=[]))
    experiments.write_json(tmp_path/'spinup'/'summary.json',dict(status='invalid_regime'))
    experiments.plot(SimpleNamespace(output=tmp_path))
    assert 'spinup stopped' in capsys.readouterr().out
    assert not (tmp_path/'paper_comparison.png').exists()
