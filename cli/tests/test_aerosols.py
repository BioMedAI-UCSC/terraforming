import json
import pytest
from pydantic import ValidationError
from cli.models import AerosolConfig,InterventionConfig,PlanetConfig


def test_shared_configuration_rejects_unsupported_inputs():
    from cli.server import RunRequest
    with pytest.raises(ValidationError): AerosolConfig(material='graphene')
    with pytest.raises(ValidationError): AerosolConfig(mode='active',coagulation=True)
    with pytest.raises(ValidationError): AerosolConfig(sources=[dict(start_s=10,stop_s=5,rate=1)])
    with pytest.raises(ValidationError): InterventionConfig(aerosol={},injection={'CF4':1})
    with pytest.raises(ValidationError): RunRequest(aerosol={})
    with pytest.raises(ValidationError): RunRequest(aerosol={},accuracy='gcm',inject={'CF4':1})


def test_real_cli_adapter_source_and_restart(tmp_path):
    pytest.importorskip('dinosaur')
    from cli.aerosols import run
    config=AerosolConfig(duration_s=5,dt_s=5,n_layers=4,
        sources=[dict(start_s=1,stop_s=8,rate=10)])
    reports=[]
    report=run(config,PlanetConfig(),tmp_path/'first',callback=lambda r,f:reports.append((r,f)))
    assert report['injected_mass_kg']==pytest.approx(40.)
    assert report['status']=='completed'
    archived=json.loads((tmp_path/'first'/'configuration.json').read_text())
    assert len(archived['physics_contract_sha256'])==64
    assert 'particle_feedback' in archived['model_assumptions']
    assert 'aerosol_burden' in reports[0][1]['maps']
    second=run(config.model_copy(update={'duration_s':10}),PlanetConfig(),tmp_path/'second',
        resume=tmp_path/'first'/'restart.npz')
    assert second['injected_mass_kg']==pytest.approx(70.)
    assert second['elapsed_seconds']==10


def test_server_uses_continuous_aerosol_runner(tmp_path,monkeypatch):
    pytest.importorskip('dinosaur')
    from cli import server
    monkeypatch.setattr(server,'_OUTPUTS_DIR',tmp_path)
    req=server.RunRequest(exp_type='intervention',accuracy='gcm',aerosol=dict(
        duration_s=5,dt_s=5,n_layers=4,sources=[dict(start_s=1,stop_s=4,rate=10)]))
    rid='a10b20c3'
    server._runs[rid]=dict(id=rid,status='running',progress=0.,data=[],label='aerosol',error=None)
    try:
        server._run_simulation(rid,req)
        result=server._runs[rid]
        assert result['status']=='done',result.get('error')
        assert result['data'][-1]['injected_mass_kg']==pytest.approx(30.)
        assert 'aerosol_burden' in result['fields']['maps']
        assert (tmp_path/'aerosols'/rid/'restart.npz').is_file()
        resumed=server.RunRequest(**{**req.model_dump(),
            'aerosol':{**req.aerosol.model_dump(),'duration_s':10},'aerosol_resume_run_id':rid})
        server._runs['resumed']=dict(id='resumed',status='running',progress=0.,data=[],label='resume',error=None)
        server._run_simulation('resumed',resumed)
        assert server._runs['resumed']['status']=='done',server._runs['resumed'].get('error')
        assert server._runs['resumed']['data'][-1]['injected_mass_kg']==pytest.approx(30.)
        assert server._runs['resumed']['data'][-1]['elapsed_seconds']==10.
    finally:
        server._runs.pop(rid,None)
        server._runs.pop('resumed',None)


def test_paired_cli_resume_preserves_all_cases_and_annual_integrals(tmp_path):
    pytest.importorskip('dinosaur')
    from cli.aerosols import run
    import dataclasses
    from src.interventions.aerosols.pilot import prepare
    from src.interventions.aerosols.starting import save_physical_start
    seed,_,context=prepare(dt_s=5,n_layers=4)
    climate=seed.climate._replace(surface_temperature=seed.climate.surface_temperature+3.,
        dynamics=dataclasses.replace(seed.climate.dynamics,tracers={}))
    seed_path=tmp_path/'physical.npz'
    save_physical_start(seed_path,climate,context['configuration']['start_contract'])
    config=AerosolConfig(paired=True,duration_s=5,dt_s=5,n_layers=4,
        initial_climate=str(seed_path),sources=[dict(start_s=1,stop_s=8,rate=10)])
    first=run(config,PlanetConfig(),tmp_path/'first')
    assert first['cases']['off']['injected_mass_kg']==0.
    assert first['cases']['active']['injected_mass_kg']==40.
    assert first['surface_temperature_k']>212.
    seed_path.unlink()
    resumed=run(config.model_copy(update={'duration_s':10}),PlanetConfig(),tmp_path/'second',
        resume=tmp_path/'first'/'restart.npz')
    assert resumed['cases']['active']['injected_mass_kg']==70.
    assert resumed['cases']['passive']['injected_mass_kg']==70.
    assert resumed['warming']['completed_mars_years']==0
    assert not any(resumed['warming']['sustained_thresholds_k'].values())


def test_screening_command_preserves_unsupported_case_records(tmp_path,monkeypatch):
    pytest.importorskip('dinosaur')
    from click.testing import CliRunner
    from cli.main import cli
    from src.interventions.aerosols import study
    def cases(definition):
        assert definition=={'base':{},'sweep':{}}
        yield dict(case=0,result=dict(status='completed',radiatively_effective_residence_time_s=None))
        yield dict(case=1,result=dict(status='unsupported_configuration',error='missing force law'))
    monkeypatch.setattr(study,'screening_cases',cases)
    source=tmp_path/'study.json';source.write_text(json.dumps(dict(base={},sweep={})))
    target=tmp_path/'cases.ndjson'
    result=CliRunner().invoke(cli,['mars','aerosol-screen','--study',str(source),'--output',str(target)])
    assert result.exit_code==0,result.output
    saved=[json.loads(line) for line in target.read_text().splitlines()]
    assert [x['result']['status'] for x in saved]==['completed','unsupported_configuration']
    assert 'radiatively effective lifetime remains unknown' in result.output


def test_radiation_command_keeps_failed_gate_and_exits_nonzero(tmp_path,monkeypatch):
    pytest.importorskip('dinosaur')
    from click.testing import CliRunner
    from cli.main import cli
    from src.interventions.aerosols import angular_reference
    def compare(**inputs):
        assert inputs['max_flux_error_fraction']==.01
        return dict(passes_frozen_gate=False,maximum_flux_error_fraction=.08,
            scope='illustrative HG control')
    monkeypatch.setattr(angular_reference,'compare_two_stream_column',compare)
    source=tmp_path/'case.json'
    source.write_text(json.dumps(dict(kind='column',case={},limits=dict(max_flux_error_fraction=.01))))
    output=tmp_path/'comparison.json'
    result=CliRunner().invoke(cli,['mars','aerosol-radiation-check','--case',str(source),'--output',str(output)])
    assert result.exit_code!=0
    saved=json.loads(output.read_text())
    assert saved['status']=='outside_frozen_limits'
    assert not saved['comparison']['passes_frozen_gate']
    assert 'exceeds the supplied accuracy limits' in result.output
