# API Reference — gcm3d

Reusable imports live under `src.framework.gcm` and `src.framework.physics`.
Mars forcing, maps, datasets, and seasonal components live under
`src.celestials.planets.mars`. There is no top-level `src.gcm3d` package.
`BodyConstants`/`EARTH` are always available; dynamics and physics kernels require
the `gcm3d` extra. Dataset adapters may require xarray or the `arco` extra.
The entries below summarize interfaces; full signatures are available in source.

## Body abstraction — `body.py`

### `BodyConstants`
**Purpose**: frozen SI constants a planet/moon supplies to the dry dynamical core.
**Location**: `package/src/framework/gcm/body.py`

| Field | Type | Description |
|-------|------|-------------|
| `name` | str | Body name |
| `radius_m` | float | Mean radius (m) |
| `gravity_m_s2` | float | Surface gravity (m s⁻²) |
| `rotation_period_s` | float | Sidereal rotation period (s) → Ω |
| `gas_constant_j_kg_k` | float | Specific gas constant R = R_univ/M |
| `cp_j_kg_k` | float | Isobaric heat capacity (sets κ = R/cp) |
| `reference_temperature_k` | float | Semi-implicit linearisation anchor (default 250) |
| `reference_surface_pressure_pa` | float | Reference p_s |

Properties: `angular_velocity_s` (2π/period), `kappa` (R/cp), `surface_area_m2`
(4πR²). Validates all quantities positive in `__post_init__`. `EARTH` is a
standard dry-air instance used to prove the core is planet-agnostic in tests.

## Discretisation — `coordinates.py`, `specs.py`

### `grid(truncation="T42") -> spherical_harmonic.Grid`
Named triangular truncations: `T21`(64×32), `T31`, `T42`(default), `T85`, `T106`, `T170`.

### `coordinate_system(truncation="T42", n_layers=25) -> CoordinateSystem`
Spectral grid × `n_layers` equidistant terrain-following sigma (p/pₛ) levels.

### `physics_specs(body) -> units.SimUnits`
Body-anchored dinosaur `SimUnits`; **Raises** if the body's radius ≠ 1 under the scale.
### `nondimensionalization_scale(body) -> scales.Scale`
Length = radius, time = 1/(2Ω), mass = 1 kg, temperature = 1 K.

## Dry dynamics — `dynamics.py`

- `primitive_equations(coords, body, specs=None, orography=None) -> PrimitiveEquationsSigma` — dry `ImplicitExplicitODE`; flat orography by default.
- `reference_temperature(coords, body) -> np.ndarray[n_layers]` — constant linearisation profile (nondimensional).
- `stepper(equation, dt_seconds, specs) -> step_fn` — semi-implicit IMEX-RK-SIL3; nondimensionalises `dt`.
- `integrate(step_fn, state, n_steps) -> state` — `jax.lax.scan` rollout; differentiable. **Raises** `ValueError` if `n_steps < 1`.

## Column physics — `framework/physics/gcm.py`

### `RadiativeForcing` (dataclass) / Mars `radiative_forcing(...) -> RadiativeForcing`
Radiative + orbital + surface-scheme constants for the per-column energy balance.
`src.celestials.planets.mars.gcm.radiative_forcing` supplies Mars's
obliquity/precession/orbit/emissivity/thermal-inertia. Key toggles: `diurnal`, `co2_radiation_enabled`,
`ames_correlated_k_enabled` (default True, used only with resolved radiation), `regolith_enabled`,
`stability_exchange_enabled`, `pbl_diffusion_enabled`,
`convective_adjustment_enabled`, dust fields (`dust_visible_optical_depth`,
`dust_longwave_optical_depth`, `dust_top_height_km`, `dust_conrath_parameter`).

The advanced surface flags and `co2_radiation_enabled` default to False in the
generic forcing. Use `dataclasses.replace` for fields outside the Mars factory's
arguments. Calibration fields are `ames_co2_longwave_opacity_scale`,
`ames_dust_longwave_opacity_scale` and `surface_exchange_multiplier` (all 1.0).
`solar_slant_path_enabled=True` applies solar air-mass correction. Seasonal dust
uses `dust_climatology_ls_deg`, `dust_visible_climatology` and
`dust_longwave_climatology`. See [configuration examples](quickstart.md).

### `CO2Forcing` (dataclass) / Mars `co2_forcing(...) -> CO2Forcing`
Constants for the 3-D CO₂ condensation/sublimation cycle. `energy_limited=True`
drives phase change off the surface-energy residual (no tunable relaxation rate);
`use_pressure_frost` uses the Clausius–Clapeyron frost point; `escape_rate_kg_s`
is a uniform non-thermal mass sink.

The Mars factory defaults to `energy_limited=True`; the generic dataclass defaults
to False. These defaults are distinct.

### State & tendency containers
- `ColumnPhysicsState(dynamics, surface_temperature, co2_ice, ground_temperature)` — JAX pytree.
- `ColumnPhysicsTendencies(vorticity, divergence, temperature_variation, log_surface_pressure, surface_temperature, co2_ice, ground_temperature, tracers)`.
- `SurfaceEnergyDiagnostics`, `RadiativeFluxDiagnostics` — NamedTuples of diagnostic fields.

### Equation builders
- `forced_primitive_equations(coords, body, forcing, specs=None, orography=None) -> ImplicitExplicitODE` — dry dynamics + radiative energy balance.
- `forced_co2_primitive_equations(coords, body, forcing, co2_forcing, specs=None, orography=None) -> ImplicitExplicitODE` — the above + CO₂ cycle.
- `column_primitive_equations(base, parameterization) -> ImplicitExplicitODE` — the generic dynamics⊕physics seam (conventional or learned).
- `compose_column_parameterizations(*parameterizations)` — sum matching tendency trees.
- Both forced equation builders accept `radiation_component=None` and
  `neural_tendency=None`; these bind replacement radiation and additive heating.
- `initial_column_state(dyn_state, coords, surface_temperature_k, specs, ice_pa=0.0, forcing=None) -> ColumnPhysicsState` — pass the run's `RadiativeForcing` when using a custom regolith layer layout.

### Physics functions
- `two_stream_radiative_fluxes(state, coords, specs, body, f, *, air_temperature_k=None, surface_pressure_pa=None) -> RadiativeFluxDiagnostics`.
- `surface_energy_tendencies(state, coords, specs, body, f, …) -> (atm_modal, surface_nd, diagnostics)`.
- `radiative_heating_tendency(state, coords, specs, body, f)` — atmospheric-only accessor; requires a `ColumnPhysicsState`.
- `project_co2_reservoirs(state, coords, specs)` — repair frost while balancing
  the local pressure deficit.
- `positivity_preserving_co2_step(step_fn, coords, specs, *, body=None,
  co2_forcing=None, dt_seconds=None)` — also restore global CO₂ inventory;
  `body` and `dt_seconds` are required for nonzero escape.

### Orbit / geometry helpers
- `cos_zenith_nodal(t_s, lat_rad, lon_rad, f)` — cos(zenith) on the `(n_lon, n_lat)` grid; diurnal terminator or daily-mean.
- `orbital_distance(t_s, f)`, `mean_anomaly_for_ls(ls_rad, f)`, `co2_frost_point_k(pressure_pa)`.

## Correlated-k radiation — `framework/physics/ames_radiation.py`
- `load_ames_co2_tables()` / `load_ames_co2_tables_jax()` — the bundled 12-band correlated-k asset.
- `correlated_k_optical_depths(temperature_k, pressure_mid_pa, delta_pressure_pa) -> (sw_tau, lw_tau)`.
- `channel_weights(clear_fraction)`, `planck_band_fractions(temperature_k)`.
- Fixed-dust optics constants: `DUST_SW_EXTINCTION`, `DUST_IR_EXTINCTION`, etc. (Reff=1.5 µm).

## 3-D maps — `maps.py`
- `run_maps(body=None, truncation="T42", n_layers=25, dt_seconds=600.0,
  n_steps=200, ..., forcing=None, co2_forcing=None, initial_state=None,
  return_final_state=False, hyperdiffusion_tau_seconds=None,
  radiation_component=None, neural_tendency=None)` returns `MarsMapFields` or
  `(fields, final_state)`. These options are not keyword-only in the current API.
- `MarsMapFields` — lat/lon fields (`surface_pressure_pa`, `temperature_k`, `u_ms`, `v_ms`, `co2_ice_pa`, `elevation_m`) + provenance; properties `wind_speed_ms`, `duration_sols`, `is_transient`, `approximate_wind_height_m`.
- `save_netcdf(fields, path)`, `plot_maps(fields, outdir, prefix="mars")`, `save_maps(fields, outdir, prefix="mars")`.
- Scale presets: `MAP_SCALES` (`fast`/`balanced`/`high`/`ultra`), `DEFAULT_SCALE`, `resolve_scale(name)`.
- `forcing_with_surface_properties`, `forcing_with_ames_dust` and
  `forcing_with_ames_dust_climatology` attach boundary data. The climatology helper
  retains separate visible/IR fields throughout evolving seasons.
- `state_to_comparison_dataset` and `save_comparison_netcdf` export full-state
  comparison fields, coordinates and forcing provenance.

Map metadata includes `temperature_kind`, `insolation_sampling` and solar
longitude; full-state comparison exports also retain forcing controls.
Forced temperature maps are surface temperature;
dry maps are lowest-layer air temperature. Lowest-layer winds are not fixed-height
10 m diagnostics. `is_transient` is a duration heuristic, not equilibrium acceptance.
**Raises**: `ValueError` for a diurnal timestep coarser than the terminator CFL; `FloatingPointError` if the state becomes non-finite.

## 0-D seasonal ODE — `celestials/planets/mars/seasonal.py`
- `SeasonalForcing` (dataclass), `initial_seasonal_state(T, P, ice_north, ice_south, t0_s=0.0)`.
- `seasonal_tendency(y, f)`, `seasonal_ode(f)`, `solar_flux(t, f)`, `solar_longitude(t, f)`.
- `run_seasonal(f, y0, dt_seconds, n_steps, sample_every=1) -> SeasonalTrajectory` — **Raises** if `dt_seconds > rotation_period/8` (diurnal aliasing) or bad sampling.
- `SeasonalTrajectory` — Ls-indexed arrays; `.write_csv(path)` matches the torch Mars exports.

## Restart / averaging — `restart.py`
- `RESTART_FORMAT_VERSION`, `save_restart(state, path)`, `load_restart(path)` — NPZ+JSON, version-checked, no pickle.
- `integrate_with_averaging(step_fn, initial_state, spinup_steps, average_steps, sample_every=1, diagnostic_fn=…) -> (final, time_mean)`.

Keep the associated grid, forcing, units, diffusion and precision configuration
with the restart; serialization stores state rather than experiment settings.

## Parameterized integration and neural interfaces

- `make_parameterized_step(equation_fn, dt_seconds, specs)`
  from `framework.gcm.learning` creates `step(params, state)`.
- `rollout(step_fn, params, initial_state, n_steps, save_every=None, *, remat=False)`
  returns final state, sampled trajectory and step indices.
- `framework.neural` provides `ColumnMLP`, `NeuralRadiation`, `NeuralTendency`,
  train-only normalization and NPZ inference checkpoint helpers.

See [neural experiments](neural-experiments.md) for shape, feature, flux and
checkpoint contracts, and [temperature-only](temperature-only.md) for the separate
cached postprocessing workflow.

## Topography — `topography.py`
- `load_mola_meg(path=None) -> (elevation_m, lats_deg, lons_deg)` — SHA-256-verified MEGDR raster.
- `regrid_to_nodal(coords, elevation_m=None, mola_path=None)`, `mola_modal_orography(coords, specs, …)`.

## Benchmarks — `benchmarks.py`
Dycore acceptance + conservation diagnostics: `run_resting_atmosphere`,
`run_solid_body_tracer`, `run_balanced_jet`, `run_held_suarez`,
`dry_conserved_quantities`, `conservation_drift`, `dry_drift_diagnostics`,
`resting_convergence_matrix`, and their `*Diagnostics` dataclasses.

## MCD validation client — `mcd.py`
`query_url`, `fetch_ascii`, `parse_ascii`, `interpolate_periodic`,
`weighted_metrics` — fetch Mars Climate Database v6.1 reference maps and compute
area-weighted bias/RMSE/correlation against a `gcm3d` run.
