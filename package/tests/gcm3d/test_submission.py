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


@pytest.mark.parametrize('cells',[24,48])
def test_native_grid_reconstruction_accepts_only_checked_source_controls(tmp_path,cells):
    module=_script('reconstruct_native_grid');path=tmp_path/'input.nml'
    base=f'&fv_core_nml\n npx={cells+1}, npy={cells+1},\n /\n'
    path.write_text(base)
    assert module.grid_cells(path)==cells
    for control in ('grid_type=1','shift_fac=0','do_schmidt=T','do_cube_transform=T',
                    'regional=T','nested=T',"grid_file='other.nc'",'do_schmidt=invalid'):
        path.write_text(base+control+'\n')
        with pytest.raises(ValueError,match='grid control'): module.grid_cells(path)
    for shape in ('npx=25,npy=49','npx=37,npy=37','npx=25','npx=25,npx=25,npy=25'):
        path.write_text(shape)
        with pytest.raises(ValueError,match='square C24 or C48'): module.grid_cells(path)


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


def test_reference_cold_start_keeps_native_log_weighted_pressure_correction():
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
    expected=np.exp(-body.gravity_m_s2*elevation/(body.gas_constant_j_kg_k*170.))
    from src.celestials.planets.mars.maps import hydrostatic_surface_pressure_pa
    hydrostatic=hydrostatic_surface_pressure_pa(elevation,body,t_ref_k=170.,p0_pa=705.)
    np.testing.assert_allclose(hydrostatic,705.*expected,rtol=2e-15)
    base=.8*hydrostatic
    expected=base+(705.-np.sum(base*weights)/np.sum(np.broadcast_to(weights,base.shape)))*np.log(base)/np.log(705.)
    expected=np.exp(np.asarray(grid.to_nodal(grid.clip_wavenumbers(grid.to_modal(np.log(expected[None]/scale))))))[0]*scale
    np.testing.assert_allclose(pressure,expected,rtol=2e-13)
    assert np.sum(pressure*weights)/np.sum(np.broadcast_to(weights,pressure.shape))<705.
    from src.framework.gcm.dynamics import reference_temperature
    air=grid.to_nodal(state.temperature_variation)+np.asarray(reference_temperature(coords,body))[:,None,None]
    np.testing.assert_allclose(air,170.,atol=1e-12)


@pytest.mark.parametrize('isothermal',[False,True])
def test_reference_atmosphere_controls_retain_global_initial_state_derivatives(isothermal):
    pytest.importorskip('dinosaur')
    import jax
    import jax.numpy as jnp
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as body
    jax.config.update('jax_enable_x64',True)
    driver=_script('run_gcm3d_ablation');coords=coordinate_system('T21',4);grid=coords.horizontal
    specs=physics_specs(body)
    elevation=jnp.broadcast_to(2000.*jnp.sin(grid.latitudes)[None,:],grid.nodal_shape)
    def objective(scale):
        state=driver.reference_rest_state(coords,specs,body,elevation*scale,
            temperature_k=170.+7.*(scale-1.),isothermal=isothermal)
        return jnp.mean(grid.to_nodal(state.temperature_variation))+jnp.mean(
            jnp.exp(grid.to_nodal(state.log_surface_pressure)))
    automatic=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(automatic) and abs(float(automatic))>1e-5
    for epsilon in (1e-4,1e-5):
        finite=(objective(1.+epsilon)-objective(1.-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(automatic,finite,rtol=2e-5,atol=1e-7)


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
    for arguments in (['--linear-k-interpolation'],
                      ['--config','reference','--reference-radiation','simple','--linear-k-interpolation'],
                      ['--scale-surface-ir-flux'],
                      ['--solar-constant','589'],
                      ['--config','reference','--solar-constant','nan'],
                      ['--config','reference','--solar-constant','0'],
                      ['--config','reference','--solar-constant','-1'],
                      ['--config','reference','--solar-constant','inf'],
                      ['--config','reference','--reference-radiation','height','--solar-constant','589'],
                      ['--config','reference','--reference-radiation','simple','--scale-surface-ir-flux'],
                      ['--reference-radiation','simple'],['--thermal-timescale','-2.'],['--no-co2-mass-feedback'],
                      ['--no-vertical-diffusion'],['--no-reference-pbl'],['--no-moment-water'],
                      ['--freeze-tracer-fields'],['--no-forcing'],['--gust-factor','2.'],['--gust-factor','nan'],
                      ['--recalculate-heat-flux'],['--lifting-suppresses-settling'],['--dust-opacity-from-aerosol'],
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


@pytest.mark.parametrize('arguments,message',[
    (['--shortwave-ramp-sols','5'],'shortwave ramp requires --config reference'),
    (['--config','reference','--reference-radiation','simple','--shortwave-ramp-sols','5'],
        'shortwave ramp requires --config reference and spectral radiation'),
    (['--config','reference','--shortwave-ramp-sols','nan'],'shortwave ramp duration must be finite'),
    (['--config','reference','--shortwave-ramp-sols','inf'],'shortwave ramp duration must be finite')])
def test_shortwave_ramp_cli_rejects_invalid_configuration(monkeypatch,capsys,arguments,message):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('arguments,message',[
    (['--fixed-radiation-date','100'],'fixed radiation date requires --config reference'),
    (['--config','reference','--reference-radiation','height','--fixed-radiation-date','100'],
        'fixed radiation date requires --config reference and spectral or simple radiation'),
    (['--config','reference','--fixed-radiation-date','1.5'],"invalid int value: '1.5'")])
def test_fixed_radiation_date_cli_rejects_invalid_configuration(monkeypatch,capsys,arguments,message):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    assert message in capsys.readouterr().err


@pytest.mark.parametrize('mode,explicit,expected',[
    ('simple',None,589.),('simple',600.,600.),
    ('spectral',None,None),('spectral',589.,589.)])
def test_reference_cli_passes_mode_specific_solar_constant(monkeypatch,tmp_path,mode,explicit,expected):
    import sys
    driver=_script('run_gcm3d_ablation')
    class ConfigurationReached(Exception): pass
    def capture(*args,**kwargs):
        assert kwargs['solar_constant_w_m2']==expected
        raise ConfigurationReached
    monkeypatch.setattr(driver,'reference_configuration',capture)
    mola=tmp_path/'mola';mola.touch()
    arguments=['run_gcm3d_ablation.py',str(tmp_path),'--config','reference',
        '--reference-radiation',mode,'--mola-path',str(mola),'--truncation','T21','--layers','24']
    if explicit is not None: arguments+=['--solar-constant',str(explicit)]
    monkeypatch.setattr(sys,'argv',arguments)
    with pytest.raises(ConfigurationReached): driver.main()


def test_simple_radiation_forwards_fixed_date_and_records_restart_identity(monkeypatch):
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.physics import simple_radiation
    seen={}
    def factory(**kwargs):
        seen.update(kwargs)
        return lambda *args:None
    monkeypatch.setattr(simple_radiation,'make_radiation_component',factory)
    driver=_script('run_gcm3d_ablation')
    coords=coordinate_system('T21',4)
    _,_,metadata=driver.alternate_reference_radiation(coords,'simple',fixed_calendar_date=100)
    assert seen['fixed_calendar_date']==100
    assert metadata['reference_radiation_controls']=={'fixed_calendar_date':100}
    with pytest.raises(ValueError,match='spectral or simple'):
        driver.alternate_reference_radiation(coords,'height',fixed_calendar_date=100)


@pytest.mark.parametrize('water',[False,True])
@pytest.mark.parametrize('from_aerosol',[False,True])
@pytest.mark.parametrize('scale_surface',[False,True])
@pytest.mark.parametrize('linear',[False,True])
@pytest.mark.parametrize('solar_constant',[None,589.])
@pytest.mark.parametrize('fraction_maps',[False,True])
def test_reference_configuration_keeps_dust_and_frost_independent_of_water(monkeypatch,tmp_path,water,from_aerosol,scale_surface,solar_constant,linear,fraction_maps):
    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars import surface,boundary_inputs,dust_scenario
    from src.framework.physics import particle_optics,radiative_transfer,prescribed_dust
    grid=coordinate_system('T21',4).horizontal
    frost=np.arange(np.prod(grid.nodal_shape)).reshape(grid.nodal_shape)*.01
    surface_options={}
    def surface_fields(*args,**kwargs):
        surface_options.update(kwargs)
        return dict(albedo=.25,emissivity=.95,thermal_inertia=250.,grs_ice_latitude_deg=60.)
    monkeypatch.setattr(surface,'reference_surface_fields_on_grid',surface_fields)
    monkeypatch.setattr(boundary_inputs,'polar_water_on_grid',lambda *args:frost)
    monkeypatch.setattr(dust_scenario,'load_scenario_on_grid',lambda *args,**kwargs:dict(fixture=True,**kwargs))
    fractions=dict(seasons=[0.,360.],fractions=np.ones((2,*grid.nodal_shape,1)))
    def fraction_loader(target,path,modes):
        assert target is grid and path==tmp_path/'radius_scenario.nc' and modes==1
        return fractions
    monkeypatch.setattr(dust_scenario,'load_injection_fractions_on_grid',fraction_loader)
    local_fractions=dict(seasons=[0.,360.],bins_hours=[0.,12.,24.],fractions=np.full((2,*grid.nodal_shape,2),.5))
    def local_loader(target,path):
        assert target is grid and path==tmp_path/'localtime_frac.nc'
        return local_fractions
    monkeypatch.setattr(dust_scenario,'load_local_time_fractions_on_grid',local_loader)


    monkeypatch.setattr(particle_optics,'make_particle_provider',lambda **kwargs:kwargs)
    fixed_options={}
    def fixed_factory(**kwargs):
        fixed_options.update(kwargs)
        return lambda *args:None
    monkeypatch.setattr(prescribed_dust,'make_fixed_dust_provider',fixed_factory)
    radiation_options={}
    def radiation_factory(provider,**kwargs):
        radiation_options.update(kwargs)
        return provider
    monkeypatch.setattr(radiative_transfer,'make_radiation_component',radiation_factory)
    inputs=('mars_thermal_inertia8_2011.nc','mars_TES_albedo8_new.nc','mars_sfc_emissivity8.nc',
        'GRS_interp16.nc','npcflag8.nc','DustScenario_MY30.nc')
    if fraction_maps: inputs+=('radius_scenario.nc','localtime_frac.nc')
    for name in inputs: (tmp_path/name).write_bytes(name.encode())
    _,forcing,provider,options,hashes=_script('run_gcm3d_ablation').reference_configuration(
        grid,tmp_path,231.,moment_water=water,dust_opacity_from_aerosol=from_aerosol,scale_surface_flux=scale_surface,
        solar_constant_w_m2=solar_constant,linear_k_interpolation=linear,shortwave_ramp_sols=5.,fixed_calendar_date=100,
        surface_map_options=dict(north_inertia=900.,regional_scale=1.2))
    assert surface_options==dict(north_inertia=900.,regional_scale=1.2)
    assert radiation_options['linear_k_interpolation'] is linear
    assert radiation_options['shortwave_ramp_sols']==5.
    assert radiation_options['fixed_calendar_date']==100
    assert radiation_options['scale_surface_flux'] is scale_surface
    np.testing.assert_allclose(forcing.tsi_1au_w_m2,1356.1 if solar_constant is None else solar_constant*1.524**2)
    assert options['moment_water'] is water and options['moment_dust'] is not water
    if fraction_maps:
        assert options['dust_background_options']['injection_fraction_scenario'] is fractions
        assert options['dust_background_options']['local_time_scenario'] is local_fractions
    else:
        assert 'injection_fraction_scenario' not in options['dust_background_options']
        assert 'local_time_scenario' not in options['dust_background_options']
    assert (options['moment_cloud_options'] is not None)==water
    assert (provider['water_moment_tracers'] is not None)==water
    assert provider['dust_tracers']==(('dust_mass','dust_number'),)
    if from_aerosol:
        assert fixed_options['fixed_calendar_date']==100
        assert callable(options['dust_background_options']['fixed_dust_provider'])
        assert 'scenario' not in options['dust_background_options']
        assert forcing.dust_visible_optical_depth==0. and options['dust_opacity_from_aerosol']
    else:
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


@pytest.mark.parametrize('arguments',[
    ['--north-inertia','500'],['--north-inertia-latitude','60'],
    ['--north-inertia-maximum','700'],['--regional-inertia-scale','1.5'],
    ['--config','reference','--north-inertia','nan'],
    ['--config','reference','--north-inertia-latitude','91'],
    ['--config','reference','--north-inertia-maximum','0'],
    ['--config','reference','--regional-inertia-scale','inf']])
def test_surface_inertia_cli_rejects_invalid_controls_before_input_loading(monkeypatch,capsys,arguments):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    assert 'surface inertia controls require' in capsys.readouterr().err


def test_surface_inertia_cli_forwards_controls_to_reference_configuration(monkeypatch,tmp_path):
    import sys
    driver=_script('run_gcm3d_ablation')
    class ConfigurationReached(Exception): pass
    def capture(*args,**kwargs):
        assert kwargs['surface_map_options']==dict(north_inertia=900.,north_latitude_deg=60.,
            north_maximum=700.,regional_scale=1.5)
        raise ConfigurationReached
    monkeypatch.setattr(driver,'reference_configuration',capture)
    mola=tmp_path/'mola';mola.touch()
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py',str(tmp_path),'--config','reference',
        '--mola-path',str(mola),'--truncation','T21','--layers','24',
        '--north-inertia','900','--north-inertia-latitude','60',
        '--north-inertia-maximum','700','--regional-inertia-scale','1.5'])
    with pytest.raises(ConfigurationReached): driver.main()


def test_reference_uniform_surface_preserves_temperature_and_reverse_derivative():
    from src.framework.gcm._dinosaur import jax,jnp
    jax.config.update('jax_enable_x64',True)
    driver=_script('run_gcm3d_ablation');latitude=jnp.deg2rad(jnp.array([-90.,-15.,0.,70.,90.]))
    actual=driver.reference_surface_temperature_k(latitude,equilibrated=False,soil_temperature_k=180.)
    np.testing.assert_array_equal(actual,180.)
    derivative=jax.jit(jax.grad(lambda temperature:jnp.sum(driver.reference_surface_temperature_k(
        latitude,equilibrated=False,soil_temperature_k=temperature))))(180.)
    assert derivative==5.


@pytest.mark.parametrize('arguments',[
    ['--no-equilibrated-surface'],['--soil-temperature','180'],
    ['--config','reference','--soil-temperature','nan'],
    ['--config','reference','--soil-temperature','0'],
    ['--config','reference','--no-mars-surface','--no-equilibrated-surface']])
def test_uniform_soil_cli_rejects_invalid_controls_before_input_loading(monkeypatch,capsys,arguments):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    assert 'soil' in capsys.readouterr().err


@pytest.mark.parametrize('arguments',[
    ['--no-surface-condensation'],['--soil-alpha','.75'],
    ['--config','reference','--soil-alpha','nan'],['--config','reference','--soil-alpha','inf']])
def test_soil_solver_cli_rejects_invalid_configuration_before_inputs(monkeypatch,capsys,arguments):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    assert 'soil' in capsys.readouterr().err


@pytest.mark.parametrize('arguments',[
    ['--no-isothermal-atmosphere'],['--initial-air-temperature','180'],
    ['--config','reference','--initial-air-temperature','nan'],
    ['--config','reference','--initial-air-temperature','0']])
def test_atmosphere_cli_rejects_invalid_cold_start_before_inputs(monkeypatch,capsys,arguments):
    import sys
    driver=_script('run_gcm3d_ablation')
    monkeypatch.setattr(sys,'argv',['run_gcm3d_ablation.py','unused-surface',*arguments])
    with pytest.raises(SystemExit) as stopped: driver.main()
    assert stopped.value.code==2
    error=capsys.readouterr().err
    assert 'initial atmosphere' in error or 'initial air temperature' in error


def test_release_namelist_inventory_ignores_comments_inside_continuations(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[3]/'scripts'))
    audit=_script('validate_release_interfaces')
    text='namelist /dust/ a, b, & ! continue\n !removed,\n &c !end\nreal :: other'
    assert audit.namelist_controls(text,'dust')==['a','b','c']
    assert audit.namelist_controls('namelist /dust/ a,b\nreal :: other','dust')==['a','b']
    with pytest.raises(AssertionError): audit.namelist_controls(text,'absent')
