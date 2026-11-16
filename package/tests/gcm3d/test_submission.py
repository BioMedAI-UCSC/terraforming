"""Scientific metadata and metric regressions for submission tooling."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest
import xarray as xr


def _script(name):
    path = Path(__file__).resolve().parents[3] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_gaussian_area_mean_integrates_latitude_polynomial():
    benchmark = _script("benchmark_mcd")
    nodes, _ = np.polynomial.legendre.leggauss(32)
    lat = np.degrees(np.arcsin(nodes))
    model = np.broadcast_to(nodes[:, None] ** 2, (32, 64))
    result = benchmark.weighted_metrics(model, np.zeros_like(model), lat)
    assert result["model_area_mean"] == pytest.approx(1 / 3, abs=1e-14)
    with pytest.raises(ValueError, match="No valid"):
        benchmark.weighted_metrics(model, np.full_like(model, np.nan), lat)


def test_checkpoint_history_rejects_reversed_time(tmp_path):
    driver = _script("run_gcm3d_ablation")
    path = tmp_path / "history.csv"
    driver._append_checkpoint_diagnostics(path, {"step": 10, "value": 1.0})
    driver._append_checkpoint_diagnostics(path, {"step": 10, "value": 1.0})
    assert len(path.read_text().splitlines()) == 2
    with pytest.raises(ValueError, match="ahead of restart"):
        driver._append_checkpoint_diagnostics(path, {"step": 9, "value": 1.0})
    with pytest.raises(ValueError, match="schema changed"):
        driver._append_checkpoint_diagnostics(
            path, {"step": 11, "value": 1.0, "new_value": 2.0}
        )


def test_reference_cold_start_sets_global_dry_mass_and_keeps_hydrostatic_terrain():
    pytest.importorskip('dinosaur')
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm._dinosaur import jax,scales
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as body
    jax.config.update('jax_enable_x64',True)
    driver=_script('run_gcm3d_ablation');coords=coordinate_system('T21',4);grid=coords.horizontal
    specs=physics_specs(body)
    elevation=np.broadcast_to(2000.*np.sin(np.asarray(grid.latitudes))[None,:],grid.nodal_shape)
    state=driver.reference_rest_state(coords,specs,body,elevation)
    scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    pressure=np.exp(np.asarray(grid.to_nodal(state.log_surface_pressure)))[0]*scale
    weights=np.asarray(grid.quadrature_weights)
    assert np.sum(pressure*weights)/np.sum(weights)==pytest.approx(705.,rel=2e-14)
    expected=np.exp(-body.gravity_m_s2*elevation/(body.gas_constant_j_kg_k*170.))
    from src.celestials.planets.mars.maps import hydrostatic_surface_pressure_pa
    hydrostatic=hydrostatic_surface_pressure_pa(elevation,body,t_ref_k=170.,p0_pa=705.)
    np.testing.assert_allclose(hydrostatic,705.*expected,rtol=2e-15)
    original=driver.initial_rest_state(coords,specs,body,elevation,t_ref_k=170.,p0_pa=705.)
    original_pressure=np.exp(np.asarray(grid.to_nodal(original.log_surface_pressure)))[0]*scale
    np.testing.assert_allclose(pressure/pressure.mean(),original_pressure/original_pressure.mean(),rtol=2e-14)
    from src.framework.gcm.dynamics import reference_temperature
    air=grid.to_nodal(state.temperature_variation)+np.asarray(reference_temperature(coords,body))[:,None,None]
    np.testing.assert_allclose(air,170.,atol=1e-12)


def test_reference_cold_surface_profile_retains_source_transition_and_polar_values():
    pytest.importorskip('dinosaur')
    driver=_script('run_gcm3d_ablation')
    lat=np.deg2rad(np.array([-90.,-15.0001,-15.,0.,90.]))
    values=np.asarray(driver.reference_surface_temperature_k(lat))
    expected=[165.+55.5*.5*(1.+np.cos((x+.261799)*180./75.)) if x<np.deg2rad(-15.)
        else 167.+52.*np.cos((x+.261799)*90./105.) for x in lat]
    np.testing.assert_allclose(values,expected,rtol=2e-15)
    assert values[0]<values[3] and values[-1]<values[3]


def test_alternate_reference_radiation_records_profile_and_timescale_for_restarts(tmp_path):
    from src.framework.gcm.coordinates import coordinate_system
    import hashlib
    driver=_script('run_gcm3d_ablation');coords=coordinate_system('T21',40)
    profile=tmp_path/'equilibrium.nc'
    values=np.broadcast_to(np.linspace(170.,210.,40)[:,None,None],(40,2,4))
    heights=np.broadcast_to(np.linspace(1e6,-1000.,40)[:,None,None],values.shape)
    xr.Dataset({'teq':(('level','lat','lon'),values),'zzf':(('level','lat','lon'),heights),
        'pfull':('level',np.geomspace(.000001,7.05,40)),
        'lonb':('lonb',np.linspace(0.,360.,5)),'latb':('latb',[-90.,0.,90.])},
        coords={'lat':[-45.,45.],'lon':[45.,135.,225.,315.]}).to_netcdf(profile)
    expected=hashlib.sha256(profile.read_bytes()).hexdigest()
    for mode in ('fixed','height'):
        component,options,metadata=driver.alternate_reference_radiation(coords,mode,profile,timescale=-2.)
        assert component is None and options['radiation_update_seconds'] is None
        assert options['thermal_relaxation_options']['timescale']==-2.
        assert metadata['equilibrium_profile_sha256']==expected
        assert metadata['radiation_mode']=='thermal-'+mode
        assert options['thermal_relaxation_options']['equilibrium'].shape==(40,*coords.horizontal.nodal_shape)
        if mode=='height': assert metadata['diagnostic_top_pressure_pa']>0.
    _,_,changed=driver.alternate_reference_radiation(coords,'fixed',profile,timescale=-3.)
    assert changed['thermal_timescale']==-3.
    component,options,metadata=driver.alternate_reference_radiation(coords,'simple')
    assert callable(component) and options['radiation_update_seconds'] is None
    assert metadata['simple_geometry_interval_seconds']==1800.
    with pytest.raises(ValueError,match='does not read'):
        driver.alternate_reference_radiation(coords,'simple',profile)
    with pytest.raises(ValueError,match='requires an equilibrium'):
        driver.alternate_reference_radiation(coords,'fixed')


def test_alternate_reference_cli_rejects_unsupported_combinations(monkeypatch):
    import sys
    driver=_script('run_gcm3d_ablation')
    for arguments in (['--reference-radiation','simple'],['--thermal-timescale','-2.'],['--no-co2-mass-feedback'],
                      ['--no-vertical-diffusion'],['--no-reference-pbl'],['--no-moment-water'],
                      ['--freeze-tracer-fields'],['--no-forcing'],['--gust-factor','2.'],['--gust-factor','nan'],
                      ['--recalculate-heat-flux'],['--lifting-suppresses-settling'],
                      ['--co2-exchange-ratio-pole','.8'],['--co2-exchange-ratio-trop','.9'],
                      ['--config','reference','--co2-exchange-ratio-trop','nan'],
                      ['--config','reference','--co2-exchange-ratio-pole','0'],
                      ['--config','reference','--no-mars-surface'],
                      ['--config','reference','--reference-radiation','height']):
        monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
        with pytest.raises(SystemExit) as stopped: driver.main()
        assert stopped.value.code==2


@pytest.mark.parametrize('flag',['--no-particle-scavenging','--no-moment-clouds'])
def test_disabled_reference_process_switch_rejects_nonreference_config(monkeypatch,capsys,flag):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',flag])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    assert 'reference process controls require --config reference' in capsys.readouterr().err


@pytest.mark.parametrize('water',[False,True])
def test_reference_configuration_keeps_dust_and_frost_independent_of_water(monkeypatch,tmp_path,water):
    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars import surface,boundary_inputs,dust_scenario
    from src.framework.physics import particle_optics,radiative_transfer
    grid=coordinate_system('T21',4).horizontal
    frost=np.arange(np.prod(grid.nodal_shape)).reshape(grid.nodal_shape)*.01
    monkeypatch.setattr(surface,'reference_surface_fields_on_grid',lambda *args:
        dict(albedo=.25,emissivity=.95,thermal_inertia=250.,grs_ice_latitude_deg=60.))
    monkeypatch.setattr(boundary_inputs,'polar_water_on_grid',lambda *args:frost)
    monkeypatch.setattr(dust_scenario,'load_scenario_on_grid',lambda *args,**kwargs:dict(fixture=True,**kwargs))
    monkeypatch.setattr(particle_optics,'make_particle_provider',lambda **kwargs:kwargs)
    monkeypatch.setattr(radiative_transfer,'make_radiation_component',lambda provider,**kwargs:provider)
    inputs=('mars_thermal_inertia8_2011.nc','mars_TES_albedo8_new.nc','mars_sfc_emissivity8.nc',
        'GRS_interp16.nc','npcflag8.nc','DustScenario_MY30.nc')
    for name in inputs: (tmp_path/name).write_bytes(name.encode())
    _,_,provider,options,hashes=_script('run_gcm3d_ablation').reference_configuration(
        grid,tmp_path,231.,moment_water=water)
    assert options['moment_water'] is water and options['moment_dust'] is not water
    assert (options['moment_cloud_options'] is not None)==water
    assert (provider['water_moment_tracers'] is not None)==water
    assert provider['dust_tracers']==(('dust_mass','dust_number'),)
    assert options['dust_background_options']['scenario']['map_scale']==3.67
    assert options['dust_opacity_channel']==1 and options['dust_optical_bands']==12
    np.testing.assert_array_equal(options['initial_water_surface_kg_m2'],frost)
    assert set(hashes)==set(inputs)


def test_staged_arco_time_is_martian_sol_not_cf_datetime(tmp_path):
    stage = _script("stage_arco_macda")
    time = xr.DataArray([0.5], dims="time")
    time.attrs.update(units="days since 0001-01-01", calendar="none")
    time.encoding.update(units="days since 0001-01-01", calendar="none")
    source = xr.Dataset(
        data_vars={
            "tsurf": (
                ("time", "lat", "lon"),
                np.ones((1, 3, 4), dtype=np.float64),
            ),
            "Ls": ("time", [1.0]),
            "MY_Ls": ("time", [24.0]),
        },
        coords={
            "time": time,
            "lat": [-90.0, 0.0, 90.0],
            "lon": [-180.0, -90.0, 0.0, 90.0],
        },
    )
    staged = stage._to_dinosaur_grid(source, "T21", 12)
    assert staged.time.attrs["units"] == "sol"
    assert "calendar" not in staged.time.attrs
    assert "units" not in staged.time.encoding
    assert "calendar" not in staged.time.encoding

    path = tmp_path / "arco.nc"
    staged.to_netcdf(path, engine="h5netcdf")
    with xr.open_dataset(path) as reopened:
        assert reopened.time.attrs["units"] == "sol"
        assert np.array_equal(reopened.time.values, [0.5])


def test_arco_ls_window_avoids_source_time_gap():
    stage = _script("stage_arco_macda")
    regular = np.arange(72, dtype=np.float64) / stage.SAMPLES_PER_SOL
    regular[36:] += 30.0
    source = xr.Dataset(
        {"Ls": ("time", np.linspace(0.0, 60.0, 72))},
        coords={"time": ("time", regular)},
    )
    start, stop = stage._nearest_contiguous_window(
        source, 0, 72, center_ls=30.0, sols=2
    )
    selected = np.asarray(source.time.isel(time=slice(start, stop)))
    assert stop - start == 2 * stage.SAMPLES_PER_SOL
    assert np.allclose(np.diff(selected), 1.0 / stage.SAMPLES_PER_SOL)
