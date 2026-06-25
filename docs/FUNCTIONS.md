# Functions

This file is reserved for documenting project functions and runtime interfaces.

## Mathematical model (current baseline)

### Solar flux with zenith angle
Used by `PlanetModel.evolve()` in `src/planet.py`.

Inputs:
- `S_1AU` (solar constant at 1 AU, `W/m2`)
- `r_AU` (Sun-Mars distance, `AU`)
- `theta_z` (solar zenith angle, `deg`)
- `tau_atm` (effective atmospheric transmittance, unitless)
- `alpha` (surface albedo, unitless)

Equations:
1. Top-of-atmosphere incident solar flux:
   - `F_TOA = (S_1AU / r_AU^2) * max(0, cos(theta_z))`
2. Surface incident solar flux:
   - `F_sfc_inc = F_TOA * tau_atm`
3. Surface reflected solar flux:
   - `F_sfc_refl = alpha * F_sfc_inc`

Zenith behavior:
- `theta_z = 0 deg` -> maximum direct flux
- `theta_z = 60 deg` -> 50% of normal-incidence direct component
- `theta_z = 90 deg` -> direct shortwave approximately zero

### Orbital distance model (used for `r_AU`)
Inputs:
- `a_AU` (semi-major axis, `AU`)
- `e` (orbital eccentricity)
- `Ls` (solar longitude, `deg`)

Equation:
- `r_AU = a_AU * (1 - e^2) / (1 + e * cos(Ls - 251))`

### Thermal IR model
Inputs:
- `epsilon_up` (surface emissivity)
- `epsilon_down` (effective atmospheric IR emissivity)
- `sigma` (Stefan-Boltzmann constant)
- `T` (surface/near-surface temperature, `K`)

Equations:
1. Upwelling IR:
   - `F_IR_up = epsilon_up * sigma * T^4`
2. Downwelling IR:
   - `F_IR_down = epsilon_down * sigma * T^4`

### Temperature tendency (reduced-order)
Inputs:
- `C_eff` (effective heat capacity, `J m^-2 K^-1`)
- `F_extra` (optional external forcing, `W/m2`)

Equations:
1. Net surface energy flux:
   - `F_net = F_sfc_inc - F_sfc_refl + F_IR_down - F_IR_up + F_extra`
2. Temperature update over timestep `dt`:
   - `T_next = T + (F_net / C_eff) * dt`

### TOA flux by zenith angle (reference table)
Equation:
- `F_TOA(theta_z) = F_normal * max(0, cos(theta_z))`

Baseline reference (`Ls ~ 0 deg`, `F_normal = 713.246626897437 W/m2`):
- `0 deg` -> `713.2466 W/m2`
- `30 deg` -> `617.6897 W/m2`
- `45 deg` -> `504.3415 W/m2`
- `60 deg` -> `356.6233 W/m2`
- `90 deg` -> `~0 W/m2` (floating-point epsilon in runtime)

## External inputs for `evolve()`

`PlanetModel.evolve()` supports structured external forcings through:
- `forcings["solar_radiation"]`
- `forcings["solar_wind"]`
- `forcings["cosmic_radiation"]`
- `forcings["giant_planets_gravity"]`
- `forcings["mars_moons_gravity"]`

### `forcings["solar_radiation"]` fields
- `solar_zenith_angle_deg`
- `sun_mars_distance_AU` (optional override)
- `solar_constant_1au_W_m2` (optional override)
- `toa_incident_solar_flux_W_m2` (optional direct TOA override)
- `atmospheric_transmittance`
- `surface_energy_forcing_W_m2`

### `forcings["solar_wind"]` fields
- `electron_number_density_cm3`
- `magnetic_field_nT`
- `wind_speed_km_s`
- `proton_density_cm3`
- `dynamic_pressure_nPa` (optional; auto-estimated if omitted and speed+density are provided)

### `forcings["cosmic_radiation"]` fields
- `gcr_flux_particles_cm2_s` (galactic cosmic ray flux)
- `sep_flux_particles_cm2_s` (solar energetic particle flux)
- `dose_rate_toa_mSv_day` (top-of-atmosphere dose rate)
- `atmospheric_shielding_factor` (0 to 1 fraction attenuated by atmosphere)

### `forcings["giant_planets_gravity"]` fields
- `jupiter_accel_m_s2` (effective external acceleration contribution)
- `saturn_accel_m_s2` (effective external acceleration contribution)

### `forcings["mars_moons_gravity"]` fields
Optional direct inputs:
- `phobos_accel_m_s2`
- `deimos_accel_m_s2`
- `phobos_tidal_accel_m_s2`
- `deimos_tidal_accel_m_s2`

Optional derived-input mode (if direct values are not provided):
- `phobos_mass_kg` (default `1.0659e16`)
- `phobos_distance_m` (default `9_376_000`)
- `deimos_mass_kg` (default `1.4762e15`)
- `deimos_distance_m` (default `23_463_000`)

Derived formulas:
- direct acceleration: `a = G * M / d^2`
- tidal acceleration at Mars surface: `a_tidal = 2 * G * M * R_mars / d^3`

Dynamic pressure auto-estimation uses:
- `P_dyn = n * m_p * v^2`
- with unit conversion to `nPa`

Magnetic-field evolution (`dynamic["plasma_magnetic"]["magnetic_field_nT"]`) can now be modified by:
- external forcing target: `forcings["solar_wind"]["magnetic_field_nT"]`
- manual control tendency: `controls["magnetic_field_tendency_nT_s"]`
- wind-pressure coupling: `params["magnetic_pressure_coupling_nT_per_nPa_s"]`
- relaxation to forcing target: `params["magnetic_forcing_relaxation_timescale_s"]`
- recovery to background field:
  - `static["magnetic_field"]["background_field_nT"]`
  - `params["magnetic_background_recovery_timescale_s"]`

Traceability output:
- `dynamic["plasma_magnetic"]["magnetic_field_tendency_nT_s"]`

Giant-planets gravity traceability outputs:
- `dynamic["time"]["jupiter_accel_m_s2"]`
- `dynamic["time"]["saturn_accel_m_s2"]`
- `dynamic["time"]["net_giant_planets_accel_m_s2"]`
- `dynamic["atmosphere"]["giant_gravity_pressure_tendency_Pa_s"]`

Mars-moons gravity traceability outputs:
- `dynamic["time"]["phobos_accel_m_s2"]`
- `dynamic["time"]["deimos_accel_m_s2"]`
- `dynamic["time"]["phobos_tidal_accel_m_s2"]`
- `dynamic["time"]["deimos_tidal_accel_m_s2"]`
- `dynamic["time"]["net_moons_accel_m_s2"]`
- `dynamic["time"]["net_moons_tidal_accel_m_s2"]`
- `dynamic["atmosphere"]["moons_gravity_pressure_tendency_Pa_s"]`

Optional coupling params:
- `params["giant_gravity_orbit_coupling_au_per_s_per_m_s2"]`
  - perturbs Sun-Mars distance update from net giant-planet acceleration
- `params["giant_gravity_pressure_pa_s_per_m_s2"]`
  - adds pressure tendency from net giant-planet acceleration
- `params["moons_gravity_pressure_pa_s_per_m_s2"]`
  - adds pressure tendency from net Phobos+Deimos direct acceleration
- `params["moons_tidal_pressure_pa_s_per_m_s2"]`
  - adds pressure tendency from net Phobos+Deimos tidal acceleration

Cosmic radiation diagnostics:
- `dose_surface = dose_toa * (1 - atmospheric_shielding_factor)`
- stored under `dynamic["radiation_energy"]` as:
  - `cosmic_dose_rate_toa_mSv_day`
  - `cosmic_dose_rate_surface_mSv_day`
  - `cosmic_gcr_flux_particles_cm2_s`
  - `cosmic_sep_flux_particles_cm2_s`

Optional temperature coupling parameter:
- `params["cosmic_forcing_W_m2_per_mSv_day"]` (default `0.0`)
- added to energy tendency as:
  - `F_cosmic = dose_surface * cosmic_forcing_W_m2_per_mSv_day`

Optional pressure/composition coupling parameters:
- `params["cosmic_escape_pressure_loss_pa_s_per_mSv_day"]`
  - applies pressure loss:
  - `dP_cosmic/dt = -dose_surface * cosmic_escape_pressure_loss_pa_s_per_mSv_day`
- `params["cosmic_ozone_loss_kg_m2_s_per_mSv_day"]`
  - applies additional ozone column loss tendency on `O3`
- `params["cosmic_h_escape_kg_m2_s_per_mSv_day"]`
  - applies additional escape tendency on `H`
- `params["cosmic_h2_escape_kg_m2_s_per_mSv_day"]`
  - applies additional escape tendency on `H2`

Traceability outputs:
- `dynamic["atmosphere"]["cosmic_escape_pressure_tendency_Pa_s"]`
- `dynamic["composition"]["cosmic_escape_tendency_kg_m2_s"]` for `O3`, `H`, `H2`

### Example
```python
forcings = {
    "solar_radiation": {
        "solar_zenith_angle_deg": 45.0,
        "atmospheric_transmittance": 0.55,
    },
    "solar_wind": {
        "magnetic_field_nT": 35.0,
        "electron_number_density_cm3": 1500.0,
        "wind_speed_km_s": 400.0,
        "proton_density_cm3": 3.0,
    },
    "cosmic_radiation": {
        "gcr_flux_particles_cm2_s": 4.0,
        "sep_flux_particles_cm2_s": 0.2,
        "dose_rate_toa_mSv_day": 0.9,
        "atmospheric_shielding_factor": 0.25,
    },
}
```

## Atmospheric composition controls (add more gases)

Use `controls` in `PlanetModel.evolve()` to increase/decrease atmospheric species.

### `controls["species_column_tendency_kg_m2_s"]`
Per-species mass-column tendency (`kg m^-2 s^-1`), integrated each timestep:
- positive value -> add species
- negative value -> remove species

Default supported species in runtime:
- `O2`, `N2`, `H2`, `H`, `CO2`, `CO`, `O3`, `Ar`, `He`
- `super_ghg` (aggregate super greenhouse gases bucket)
- optional inert rare gases: `Ne`, `Kr`, `Xe`

Any additional species key is also accepted.

### `controls["species_vmr_target"]` (optional override)
Directly set target volume-mixing-ratio values (scenario forcing).  
The model renormalizes VMRs to sum to 1.

### Ozone layer note
- Ozone is represented by species key `O3`.
- Increase `O3` via `species_column_tendency_kg_m2_s["O3"]` or `species_vmr_target["O3"]`.

### Example (increase key species)
```python
controls = {
    "species_column_tendency_kg_m2_s": {
        "O2": 1.0e-7,
        "N2": 2.0e-7,
        "H2": 2.0e-8,
        "CO2": 1.5e-7,
        "CO": 1.0e-8,
        "O3": 1.0e-9,
        "Ar": 1.0e-8,
        "super_ghg": 5.0e-10,
    }
}
```

## Water controls (solid and liquid)

Use `controls` in `PlanetModel.evolve()` to add/remove water reservoirs:

- `water_ice_tendency_kg_m2_s`
  - Adds/removes bulk solid water (`water_ice_column_kg_m2`)
- `water_liquid_tendency_kg_m2_s`
  - Adds/removes bulk liquid water (`water_liquid_column_kg_m2`)
- `water_phase_change_tendency_kg_m2_s`
  - Positive: melt transfer from ice to liquid
  - Negative: freeze transfer from liquid to ice

Polar layers remain separate:
- `polar_ice_h2o_tendency_kg_m2_s`
- `polar_ice_co2_tendency_kg_m2_s`

### Example (add more water)
```python
controls = {
    "water_ice_tendency_kg_m2_s": 2.0e-6,      # add solid water
    "water_liquid_tendency_kg_m2_s": 5.0e-7,   # add liquid water
    "water_phase_change_tendency_kg_m2_s": 1.0e-7,  # melt some ice into liquid
}
```

## Soil composition preset (Earth-like)

`PlanetModel` provides:
- `apply_earthlike_soil(planet, blend=1.0, overrides=None)`

What it does:
- Applies an Earth-like loam preset to `static["soil_regolith"]`
- Supports blending with existing soil values (`blend` in `0..1`)
- Accepts `overrides` for custom tuning

Preset includes:
- `chemistry_mass_fraction` (normalized mass fractions)
- `porosity`
- `permeability_m2`
- `bulk_density_kg_m3`
- `ph`
- `depth_m`
- `water_holding_capacity`

Example:
```python
model = PlanetModel()
state = model.initialize(static_data, dynamic_initial)
state = model.apply_earthlike_soil(
    state,
    blend=1.0,
    overrides={"ph": 7.0, "depth_m": 1.5},
)
```

## Regolith compound changes (runtime)

Use `controls` in `PlanetModel.evolve()` to modify regolith chemistry over time.

### Controls
- `soil_compound_tendency_mass_fraction_per_s`
  - dictionary of `{compound: d_fraction_dt}`
  - integrated each step as `fraction += d_fraction_dt * dt_s`
- `soil_compound_delta_mass_fraction`
  - dictionary of `{compound: delta_fraction}`
  - applied directly per timestep

The model clamps negatives to zero and renormalizes
`static["soil_regolith"]["chemistry_mass_fraction"]` to sum to 1.

### Example
```python
controls = {
    "soil_compound_tendency_mass_fraction_per_s": {
        "organic_matter": 2.0e-9,
        "Fe_oxides": -1.0e-9,
    },
    "soil_compound_delta_mass_fraction": {
        "SiO2": -0.001,
        "clay_minerals": 0.001,
    },
}
```
