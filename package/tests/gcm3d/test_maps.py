"""Tests for src.celestials.planets.mars.maps — dry-dynamics Mars maps over MOLA terrain
(requires the optional 'gcm3d' extra and the staged MOLA raster).

Covers:
  - hydrostatic_surface_pressure_pa: Mars barometry (sea-level = p0, sign).
  - run_maps: field shapes, finiteness, physical ranges, and the key
    invariant — surface pressure anti-correlates with terrain elevation.
  - save_netcdf / plot_maps: output files are written.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
from src.framework.physics.tracer_transport import tracer_to_nodal
import jax  # noqa: E402

from src.celestials.planets.mars import MARS_BODY_3D  # noqa: E402
from src.celestials.planets.mars import maps  # noqa: E402
from src.celestials.planets.mars import topography as topo  # noqa: E402
from src.framework.gcm.dynamics import horizontal_hyperdiffusion_filter  # noqa: E402

jax.config.update("jax_enable_x64", True)

needs_mola = pytest.mark.skipif(
    not topo._DEFAULT_MOLA.exists(), reason="MOLA MEGDR raster not staged"
)


class TestHydrostaticPressure:

    def test_sea_level_equals_reference(self):
        p = maps.hydrostatic_surface_pressure_pa(0.0, MARS_BODY_3D)
        assert float(p) == pytest.approx(MARS_BODY_3D.reference_surface_pressure_pa)

    def test_higher_terrain_lower_pressure(self):
        low = maps.hydrostatic_surface_pressure_pa(-4000.0, MARS_BODY_3D)
        high = maps.hydrostatic_surface_pressure_pa(20000.0, MARS_BODY_3D)
        assert float(high) < MARS_BODY_3D.reference_surface_pressure_pa < float(low)
        # ~10 km scale height ⇒ Olympus (20 km) near ~85 Pa.
        assert 60.0 < float(high) < 120.0


@pytest.mark.parametrize('boundaries',[None,[0.,.001,.05,.5,1.]])
@pytest.mark.parametrize('temperature',[None,170.])
def test_cold_start_balances_the_same_terrain_modes_as_the_dycore(boundaries,temperature):
    import jax.numpy as jnp
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm.dynamics import primitive_equations,stepper
    from src.framework.gcm._dinosaur import spherical_harmonic,scales
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as body
    coords=coordinate_system('T21',4,sigma_boundaries=boundaries);grid=coords.horizontal;specs=physics_specs(body)
    terrain_modes=jnp.zeros(grid.modal_shape).at[0,2].set(1500.).at[0,-1].set(1500.)
    elevation=np.asarray(grid.to_nodal(terrain_modes))
    orography=topo.mola_modal_orography(coords,specs,elevation_nodal_m=elevation)
    initial=maps.initial_rest_state(coords,specs,body,elevation,t_ref_k=temperature)
    equation=primitive_equations(coords,body,specs,orography=orography)
    final=jax.jit(stepper(equation,5.,specs))(initial)
    u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,final.vorticity,final.divergence)
    scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    assert float(jnp.max(jnp.hypot(u,v)))*scale<1e-8


@needs_mola
class TestRunMaps:

    @pytest.fixture(scope="class")
    def fields(self):
        # T21 / few layers / short run keeps this fast but exercises the whole path.
        return maps.run_maps(
            truncation="T21", n_layers=8, dt_seconds=600.0, n_steps=40
        )

    def test_field_shapes(self, fields):
        n_lon = fields.lon_deg.size
        n_lat = fields.lat_deg.size
        for arr in (
            fields.elevation_m,
            fields.surface_pressure_pa,
            fields.temperature_k,
            fields.u_ms,
            fields.v_ms,
        ):
            assert arr.shape == (n_lat, n_lon)

    def test_fields_finite_and_physical(self, fields):
        assert np.all(np.isfinite(fields.surface_pressure_pa))
        assert np.all(fields.surface_pressure_pa > 0.0)
        assert np.all(fields.temperature_k > 100.0) and np.all(fields.temperature_k < 350.0)
        assert np.all(np.isfinite(fields.wind_speed_ms))

    def test_surface_pressure_tracks_topography(self, fields):
        # The defining validation signature: low p over high terrain
        # (Tharsis/Olympus), high p in Hellas. Expect strong anti-correlation.
        r = np.corrcoef(
            fields.elevation_m.ravel(), fields.surface_pressure_pa.ravel()
        )[0, 1]
        assert r < -0.9

    def test_run_maps_rejects_nonpositive_steps(self):
        with pytest.raises(ValueError):
            maps.run_maps(truncation="T21", n_layers=8, n_steps=0)

    def test_chunked_run_emits_progress_and_diagnostic_states(self):
        progress, diagnostics = [], []
        maps.run_maps(
            truncation="T21", n_layers=4, dt_seconds=300.0, n_steps=4,
            progress_chunk_steps=2,
            progress_callback=lambda done, total: progress.append((done, total)),
            diagnostic_callback=lambda done, total, state, coords, specs:
                diagnostics.append((done, total, state is not None, coords.vertical.layers)),
        )
        assert progress == [(2, 4), (4, 4)]
        assert diagnostics == [(2, 4, True, 4), (4, 4, True, 4)]


class TestScalePresets:

    def test_known_presets_resolve(self):
        for name in ("fast", "balanced", "high", "ultra"):
            cfg = maps.resolve_scale(name)
            assert set(cfg) == {
                "truncation", "n_layers", "dt_seconds", "n_steps",
                "hyperdiffusion_tau_seconds",
            }
            assert cfg["hyperdiffusion_tau_seconds"] == "auto"
            assert cfg["n_layers"] > 0 and cfg["n_steps"] > 0

    def test_default_and_ordering(self):
        # Higher scales are at least as fine as lower ones.
        fast = maps.resolve_scale("fast")
        ultra = maps.resolve_scale("ultra")
        assert ultra["n_layers"] >= fast["n_layers"]
        assert ultra["n_steps"] >= fast["n_steps"]
        assert maps.resolve_scale(None) == maps.resolve_scale(maps.DEFAULT_SCALE)

    def test_unknown_scale_raises(self):
        with pytest.raises(ValueError, match="unknown scale"):
            maps.resolve_scale("gigantic")


@pytest.mark.parametrize("tau", ["auto", None, 1000.0])
def test_map_runner_applies_and_reports_damping_without_changing_inventories(monkeypatch, tmp_path, tau):
    import jax.numpy as jnp
    import xarray as xr
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs

    coords = coordinate_system("T21", 4)
    specs = physics_specs(MARS_BODY_3D)
    grid = coords.horizontal
    terrain = np.zeros(grid.nodal_shape)
    monkeypatch.setattr(maps, "regrid_to_nodal", lambda *args, **kwargs: terrain)
    monkeypatch.setattr(maps, "mola_modal_orography", lambda *args, **kwargs: jnp.zeros(grid.modal_shape))
    initial = maps.initial_rest_state(coords, specs, MARS_BODY_3D, terrain)
    initial = dataclasses.replace(
        initial,
        temperature_variation=initial.temperature_variation.at[:, 0, -1].set(1.0),
    )
    # Exercise the real runner/filter, but isolate the damping from PE evolution.
    monkeypatch.setattr(
        maps, "stepper",
        lambda equation, dt_seconds, specs, **kwargs:
            (lambda state: state) if kwargs["hyperdiffusion_tau_seconds"] is None else
            (lambda state: horizontal_hyperdiffusion_filter(
                coords, dt_seconds, specs, kwargs["hyperdiffusion_tau_seconds"]
            )(state, state)),
    )
    fields, final = maps.run_maps(
        truncation="T21", n_layers=4, dt_seconds=100.0, n_steps=3,
        initial_state=initial, hyperdiffusion_tau_seconds=tau, return_final_state=True,
    )
    expected_tau = 0.2 * MARS_BODY_3D.solar_day_seconds if tau == "auto" else tau
    expected_decay = 1.0 if expected_tau is None else np.exp(-300.0 / expected_tau)
    assert float(final.temperature_variation[0, 0, -1]) == pytest.approx(expected_decay)
    np.testing.assert_array_equal(final.log_surface_pressure, initial.log_surface_pressure)
    assert fields.hyperdiffusion_tau_seconds == expected_tau
    maps.save_netcdf(fields, tmp_path / "maps.nc")
    with xr.open_dataset(tmp_path / "maps.nc") as dataset:
        if expected_tau is None:
            assert np.isnan(dataset.attrs["hyperdiffusion_tau_seconds"])
        else:
            assert dataset.attrs["hyperdiffusion_tau_seconds"] == expected_tau


def test_explicit_surface_fields_match_grid_and_physical_bounds(tmp_path):
    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars.surface import surface_fields_on_grid
    import xarray as xr

    grid = coordinate_system("T21", 8).horizontal
    path = tmp_path / "surface.nc"
    lat = np.linspace(-90.0, 90.0, 7)
    lon = np.linspace(0.0, 330.0, 12)
    shape = (lat.size, lon.size)
    xr.Dataset(
        {
            "albedo": (("lat", "lon"), np.full(shape, 0.25)),
            "thermal_inertia": (("lat", "lon"), np.full(shape, 250.0)),
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)
    albedo, inertia = surface_fields_on_grid(grid, path)
    assert albedo.shape == grid.nodal_shape
    assert inertia.shape == grid.nodal_shape
    assert np.all((albedo >= 0.0) & (albedo <= 1.0))
    assert np.all(inertia > 0.0)


def test_surface_fields_are_threaded_into_radiation_and_regolith(tmp_path):
    import xarray as xr

    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars.gcm import radiative_forcing

    grid = coordinate_system("T21", 8).horizontal
    path = tmp_path / "surface.nc"
    lat = np.linspace(-90.0, 90.0, 7)
    lon = np.linspace(0.0, 330.0, 12)
    lon_pattern = np.linspace(0.1, 0.4, lon.size)[None, :]
    albedo = np.broadcast_to(lon_pattern, (lat.size, lon.size))
    inertia = np.broadcast_to(100.0 + 900.0 * lon_pattern, albedo.shape)
    xr.Dataset(
        {
            "albedo": (("lat", "lon"), albedo),
            "thermal_inertia": (("lat", "lon"), inertia),
        },
        coords={"lat": lat, "lon": lon},
    ).to_netcdf(path)

    forcing = maps.forcing_with_surface_properties(
        radiative_forcing(diurnal=False), grid, path
    )
    assert np.asarray(forcing.albedo).shape == grid.nodal_shape
    assert np.ptp(np.asarray(forcing.albedo)) > 0.2
    assert np.ptp(np.asarray(forcing.surface_thermal_inertia_tiu)) > 200.0
    assert forcing.regolith_enabled
    assert forcing.stability_exchange_enabled
    assert forcing.pbl_diffusion_enabled
    assert forcing.convective_adjustment_enabled


def test_ames_dust_climatology_is_regridded_for_seasonal_forcing(tmp_path):
    import xarray as xr

    from src.framework.gcm.coordinates import coordinate_system
    from src.celestials.planets.mars.gcm import radiative_forcing

    grid = coordinate_system("T21", 4).horizontal
    path = tmp_path / "ames_dust.nc"
    shape = (3, 5, 8)
    xr.Dataset(
        {
            "dust_visible_optical_depth": (
                ("ls", "lat", "lon"), np.full(shape, 0.3)
            ),
            "dust_longwave_optical_depth": (
                ("ls", "lat", "lon"), np.full(shape, 0.1)
            ),
        },
        coords={
            "ls": [0.0, 120.0, 240.0],
            "lat": np.linspace(-90.0, 90.0, shape[1]),
            "lon": np.linspace(0.0, 315.0, shape[2]),
        },
    ).to_netcdf(path)
    forcing = maps.forcing_with_ames_dust_climatology(
        radiative_forcing(), grid, path
    )
    assert np.asarray(forcing.dust_climatology_ls_deg).shape == (3,)
    assert np.asarray(forcing.dust_visible_climatology).shape == (
        3, *grid.nodal_shape
    )
    assert np.allclose(np.asarray(forcing.dust_longwave_climatology), 0.1)


@needs_mola
class TestOutputs:

    def test_full_state_comparison_schema(self):
        from src.celestials.planets.mars.gcm import radiative_forcing
        from src.framework.gcm.coordinates import coordinate_system
        from src.framework.gcm.specs import physics_specs

        forcing = radiative_forcing(diurnal=True)
        _, state = maps.run_maps(
            truncation="T21", n_layers=4, dt_seconds=300.0, n_steps=2,
            forcing=forcing, return_final_state=True,
        )
        coords = coordinate_system("T21", 4)
        dataset = maps.state_to_comparison_dataset(
            state, coords, physics_specs(MARS_BODY_3D), MARS_BODY_3D, forcing
        )
        assert dataset.sizes == {
            "time": 1, "lat": 32, "lon": 64, "sigma": 4,
            "soil_layer": 12,
        }
        assert set((
            "surface_pressure", "surface_temperature", "air_temperature",
            "eastward_wind", "northward_wind", "wind_speed", "air_pressure",
            "co2_frost", "soil_temperature", "local_solar_time",
            "solar_longitude",
        )).issubset(dataset.data_vars)
        assert np.isfinite(dataset.to_array()).all()
        assert np.all((dataset.local_solar_time >= 0.0) & (dataset.local_solar_time < 24.0))

    def test_save_netcdf_and_plots(self, tmp_path):
        f = maps.run_maps(truncation="T21", n_layers=8, dt_seconds=600.0, n_steps=20)
        nc = maps.save_netcdf(f, tmp_path / "mars_maps.nc")
        assert nc.exists()
        import xarray as xr

        ds = xr.open_dataset(nc)
        assert set(["surface_pressure", "temperature", "u", "v", "elevation"]).issubset(ds.data_vars)
        assert 0.0 < ds.attrs["wind_level_sigma"] < 1.0
        assert ds.attrs["approximate_wind_height_m"] > 0.0
        ds.close()

        pngs = maps.plot_maps(f, tmp_path, prefix="mars")
        assert len(pngs) == 4
        for p in pngs:
            assert p.exists() and p.stat().st_size > 0


def test_comparison_export_preserves_geography_units_season_and_cloud_inventory(tmp_path):
    import jax.numpy as jnp
    import xarray as xr
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.physics.gcm import initial_column_state, mean_anomaly_for_ls
    from src.framework.physics.co2_clouds import CO2_CLOUD_TRACER
    from src.celestials.planets.mars.gcm import radiative_forcing

    coords = coordinate_system("T21", 4)
    grid, body = coords.horizontal, MARS_BODY_3D
    specs = physics_specs(body)
    forcing = radiative_forcing(diurnal=True)
    forcing = dataclasses.replace(forcing, init_orbital_angle_rad=mean_anomaly_for_ls(0., forcing))
    dynamics = maps.initial_rest_state(coords, specs, body, np.zeros(grid.nodal_shape))
    cloud_ratio = .01
    dynamics = dataclasses.replace(dynamics, sim_time=0.,
        tracers={**dynamics.tracers, CO2_CLOUD_TRACER:
                 grid.to_modal(jnp.full((4,) + grid.nodal_shape, cloud_ratio)),
                 'dust_number':grid.to_modal(jnp.full((4,)+grid.nodal_shape,1e8)),
                 'co2_mmr_gas':grid.to_modal(jnp.full((4,)+grid.nodal_shape,.95))})
    state = initial_column_state(dynamics, coords, 200., specs, forcing=forcing)
    # This asymmetric landmark detects north/south flips and lon/lat transposes.
    landmark = 200. + np.arange(64)[:, None] * .1 + np.arange(32)[None, :] * .01
    dust=jnp.stack([landmark,2.*landmark],axis=-1)[None]
    state = state._replace(surface_temperature=jnp.asarray(landmark[None]),
        auxiliary_fields=dict(water_surface_kg_m2=jnp.asarray(landmark[None]),dust_surface_kg_m2=dust))
    path = maps.save_comparison_netcdf(state, coords, specs, body, forcing, tmp_path / "state.nc")
    with xr.open_dataset(path, decode_times=False) as ds:
        np.testing.assert_allclose(ds.surface_temperature.isel(time=0), landmark.T)
        np.testing.assert_allclose(ds.lon, np.degrees(grid.longitudes))
        np.testing.assert_allclose(ds.lat, np.degrees(grid.latitudes))
        assert ds.surface_temperature.attrs["units"] == "K"
        assert ds.surface_pressure.attrs["units"] == "Pa"
        assert ds.co2_cloud_column.attrs["units"] == "Pa-equivalent"
        np.testing.assert_allclose(ds.co2_cloud_column, ds.surface_pressure * cloud_ratio, atol=1e-12)
        np.testing.assert_allclose(ds.co2_frost, 0.)
        assert float(ds.solar_longitude[0]) % 360. == pytest.approx(0., abs=1e-10)
        np.testing.assert_allclose(ds.local_solar_time[0], (12. + ds.lon / 15.) % 24.)
        np.testing.assert_allclose(ds.air_pressure,
                                   (ds.surface_pressure * ds.sigma).transpose(*ds.air_pressure.dims))
        np.testing.assert_allclose(ds['tracer_'+CO2_CLOUD_TRACER],cloud_ratio,atol=1e-14)
        assert ds['tracer_'+CO2_CLOUD_TRACER].attrs['units']=='kg kg-1'
        assert ds.tracer_dust_number.attrs['units']=='kg-1'
        assert ds.tracer_co2_mmr_gas.attrs['units']=='kg kg-1'
        np.testing.assert_allclose(ds.tracer_co2_mmr_gas,.95,atol=1e-14)
        expected_number=np.asarray(tracer_to_nodal(grid,dynamics.tracers['dust_number'])).transpose(0,2,1)
        np.testing.assert_array_equal(ds.tracer_dust_number[0],expected_number)
        np.testing.assert_allclose(ds.water_surface_kg_m2[0],landmark.T)
        assert ds.dust_surface_kg_m2.dims==('time','dust_mode','lat','lon')
        np.testing.assert_allclose(ds.dust_surface_kg_m2[0,1],2.*landmark.T)


@pytest.mark.parametrize('dust_mode_count,boundaries,thermal',[(1,None,False),(2,None,False),
    (2,[0.,.01,.1,.5,1.],False),(1,None,True),(1,None,'simple'),(1,None,'height'),
    (1,None,'feedback-off'),(1,None,'no-surface'),(1,None,'no-diffusion'),(1,None,'no-pbl'),
    (1,None,'dust-only'),(2,None,'dust-only'),(1,None,'aerosol-control')])
def test_reference_maps_backend_runs_water_and_preserves_restart_state(monkeypatch,dust_mode_count,boundaries,thermal):
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.physics.dinosaur_physics import water_inventory
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    monkeypatch.setattr(maps,'regrid_to_nodal',lambda coords,mola_path=None:np.zeros(coords.horizontal.nodal_shape))
    radiation=make_radiation_component(lambda *args:None)
    options=dict(soil_diffusivity=(250./(1500.*627.9))**2,bulk_water=True,
        initial_water_surface_kg_m2=100.,initial_water_vapor_ratio=1e-6,form_bulk_clouds=False)
    if thermal=='no-surface':
        import jax.numpy as jnp
        from src.framework.physics.simple_radiation import make_radiation_component as simple_radiation
        simple=simple_radiation()
        def radiation(state,coords,specs,body,forcing):
            flux=simple(state,coords,specs,body,forcing)
            valid=jnp.all(state.surface_temperature==170.)&jnp.all(forcing.albedo==.25)&jnp.all(forcing.emissivity==1.)
            return flux._replace(atmospheric_convergence_w_m2=jnp.where(valid,flux.atmospheric_convergence_w_m2,jnp.nan))
        options.update(bulk_water=False,do_mars_surface=False,do_vert_diff=False)
    elif thermal in ('no-diffusion','no-pbl'):
        options['do_vert_diff' if thermal=='no-diffusion' else 'do_ames_pbl']=False
        options['do_vert_diff']=False
    elif thermal=='feedback-off':
        options.update(do_co2_condensation=False,do_atmos_co2_loss=False,do_co2_condensation_cycle=False)
    elif thermal=='simple':
        from src.framework.physics.simple_radiation import make_radiation_component as simple_radiation
        radiation=simple_radiation()
    elif thermal=='dust-only':
        options.update(bulk_water=False,moment_dust=True,
            dust_mode_count=dust_mode_count)
    elif thermal=='aerosol-control':
        from src.framework.physics.prescribed_dust import make_fixed_dust_provider
        grid=coordinate_system('T21',4).horizontal
        scenario=dict(seasons=[0.,360.],opacity=np.full((2,*grid.nodal_shape),.4),
            top_height_km=np.full((2,*grid.nodal_shape),35.))
        options.update(bulk_water=False,moment_water=True,dust_opacity_from_aerosol=True,
            radiation_update_seconds=24.,initial_dust_mass_ratio=1e-5,initial_dust_number_per_kg=1e8,
            dust_background_options=dict(fixed_dust_provider=make_fixed_dust_provider(scenario=scenario)))
    elif thermal:
        radiation=None
        options['thermal_relaxation_options']=dict(equilibrium=185.,timescale=100.)
        if thermal=='height':
            source=np.broadcast_to(np.array([1e6,1e5,1e4,-1000.])[:,None,None],(4,*coordinate_system('T21',4).horizontal.nodal_shape))
            options['thermal_relaxation_options'].update(equilibrium=185.+source*1e-5,
                source_height=source,top_pressure_pa=1e-6)
    if dust_mode_count>1:
        options.update(dust_mode_count=dust_mode_count,
            do_qmass=True,initial_co2_gas_ratio=.95,
            initial_dust_mass_ratio=[1e-5,2e-5],initial_dust_number_per_kg=[1e8,3e7],
            initial_dust_surface_kg_m2=[30.,20.])
        if thermal!='dust-only':
            options.update(bulk_water=False,moment_water=True)
    kwargs=dict(body=AMES_BODY_3D,truncation='T21',n_layers=4,dt_seconds=6.,
        sigma_boundaries=boundaries,
        t_ref_k=180.,forcing=reference_radiative_forcing(),radiation_component=radiation,
        physics_backend='reference',reference_options=options,return_final_state=True)
    fields,first=maps.run_maps(**kwargs,n_steps=1)
    _,second=maps.run_maps(**kwargs,n_steps=1,initial_state=first)
    _,whole=maps.run_maps(**kwargs,n_steps=2)
    if thermal is False and dust_mode_count==1 and boundaries is None:
        import dataclasses
        horizontal=coordinate_system('T21',4).horizontal
        legacy=first._replace(dynamics=dataclasses.replace(first.dynamics,tracers={
            name:horizontal.to_modal(value) for name,value in first.dynamics.tracers.items()}))
        _,converted=maps.run_maps(**kwargs,n_steps=1,initial_state=legacy)
        for name,value in second.dynamics.tracers.items():
            np.testing.assert_allclose(converted.dynamics.tracers[name],value,rtol=3e-12,atol=1e-18)
    assert fields.co2_ice_pa is not None
    assert 'reference column physics' in fields.physics
    expected_height=-AMES_BODY_3D.gas_constant_j_kg_k*180./AMES_BODY_3D.gravity_m_s2*np.log(fields.wind_level_sigma)
    assert fields.approximate_wind_height_m==pytest.approx(expected_height)
    assert fields.wind_level_sigma==pytest.approx(.75 if boundaries is not None else .875)
    if boundaries is not None:
        with pytest.raises(ValueError,match='restart sigma boundaries'):
            maps.run_maps(**{**kwargs,'sigma_boundaries':[0.,.02,.1,.5,1.]},n_steps=1,initial_state=first)
    if dust_mode_count>1:
        assert first.auxiliary_fields['dust_surface_kg_m2'].shape[-1]==dust_mode_count
        assert 'dust_mass_2' in first.dynamics.tracers
        assert 'co2_mmr_gas' in first.dynamics.tracers and 'ar_mmr_gas' in first.dynamics.tracers
    if thermal=='dust-only':
        assert 'water_ice_number' not in first.dynamics.tracers
        assert 'water_core_mass' not in first.dynamics.tracers
        frost='water_surface_kg_m2'
        np.testing.assert_array_equal(first.auxiliary_fields[frost],second.auxiliary_fields[frost])
    if thermal=='aerosol-control':
        assert first.auxiliary_fields['dust_fixed_reference_opacity'].shape[0]==2
        assert np.all(np.asarray(first.auxiliary_fields['dust_fixed_reference_opacity'])>0.)
    for actual,expected in zip(jax.tree_util.tree_leaves(second),jax.tree_util.tree_leaves(whole),strict=True):
        if boundaries is not None and actual.shape[-2:]==first.dynamics.temperature_variation.shape[-2:]:
            # Different scan lengths can round tiny spectral coefficients differently.
            from src.framework.gcm.coordinates import grid as spectral_grid
            actual=spectral_grid('T21').to_nodal(actual)
            expected=spectral_grid('T21').to_nodal(expected)
        np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=3e-13)
    coords=coordinate_system('T21',4,sigma_boundaries=boundaries);specs=physics_specs(AMES_BODY_3D)
    np.testing.assert_allclose(water_inventory(second,coords,specs,AMES_BODY_3D),
        water_inventory(first,coords,specs,AMES_BODY_3D),rtol=3e-12)


@pytest.mark.parametrize('kwargs,match',[
    ({'physics_backend':'other'},'physics_backend'),
    ({'reference_options':{}},'reference_options'),
    ({'physics_backend':'reference'},'requires explicit'),
])
def test_reference_maps_backend_rejects_incomplete_configuration(kwargs,match):
    with pytest.raises(ValueError,match=match): maps.run_maps(**kwargs)


def test_reference_maps_rejects_bulk_water_with_moment_dust(monkeypatch):
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D,reference_radiative_forcing
    from src.framework.physics.radiative_transfer import make_radiation_component
    monkeypatch.setattr(maps,'regrid_to_nodal',lambda coords,mola_path=None:np.zeros(coords.horizontal.nodal_shape))
    with pytest.raises(ValueError,match='cannot combine bulk water with moment dust or water'):
        maps.run_maps(body=AMES_BODY_3D,truncation='T21',n_layers=4,dt_seconds=5.,n_steps=1,
            forcing=reference_radiative_forcing(),physics_backend='reference',
            radiation_component=make_radiation_component(lambda *args:None),
            reference_options=dict(soil_diffusivity=1e-8,bulk_water=True,moment_dust=True))


def test_reference_checkpoints_finish_averaging_window_after_partial_restart(monkeypatch):
    import dataclasses
    from src.framework.physics import dinosaur_physics
    from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as body,reference_radiative_forcing
    from src.framework.gcm._dinosaur import scales
    monkeypatch.setattr(maps,'regrid_to_nodal',lambda coords,mola_path=None:np.zeros(coords.horizontal.nodal_shape))
    def factory(coords,specs,*args,**kwargs):
        tick=float(specs.nondimensionalize(231.*scales.units.second))
        return lambda state:state._replace(surface_temperature=state.surface_temperature+1.,
            dynamics=dataclasses.replace(state.dynamics,sim_time=state.dynamics.sim_time+tick))
    monkeypatch.setattr(dinosaur_physics,'build_step',factory)
    options=dict(body=body,truncation='T21',n_layers=4,dt_seconds=231.,
        forcing=reference_radiative_forcing(),radiation_component=lambda *args:None,
        physics_backend='reference',reference_options=dict(soil_diffusivity=1e-7,comparison_window_steps=4),
        return_final_state=True)
    _,partial=maps.run_maps(**options,n_steps=3)
    seen=[]
    _,final=maps.run_maps(**options,n_steps=5,initial_state=partial,progress_chunk_steps=4,
        diagnostic_callback=lambda done,total,state,*args:seen.append((done,int(state.auxiliary_fields['comparison_count']))))
    assert seen==[(1,4),(5,4)]
    np.testing.assert_allclose(final.auxiliary_fields['comparison_sum_surface_temperature']/4.,206.5,rtol=2e-14)
