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
