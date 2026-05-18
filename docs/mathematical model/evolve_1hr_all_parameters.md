# Mathematical Model: `evolve()` for 1 Hour (All Parameters)

This document captures the full 1-hour (`dt_s = 3600`) `evolve()` setup currently used in `src/planet.py`, including the full parameter set and resulting diagnostics.

## Time step
- `dt_s = 3600` seconds (1 hour)

## Full input parameter set

### Static data
- `identity_orbit.gravity_m_s2 = 3.71`
- `identity_orbit.mean_radius_m = 3_389_500.0`
- `identity_orbit.semi_major_axis_AU = 1.523679`
- `identity_orbit.orbital_eccentricity = 0.0934`
- `soil_regolith.gcm_surface_bare_ground_albedo = 0.25`

### Dynamic initial state
- `time.solar_longitude_deg = 251.0` (perihelion in the runtime Mars convention)
- `time.solar_zenith_angle_deg = 60.0`
- `atmosphere.temperature_K = 210.0`
- `atmosphere.pressure_Pa = 610.0`
- `atmosphere.density_kg_m3 = 0.02`
- `hydro_cryosphere.polar_ice_h2o_kg_m2 = 0.0`
- `hydro_cryosphere.polar_ice_co2_kg_m2 = 0.0`

### Forcings
- `solar_zenith_angle_deg = 60.0`
- `surface_energy_forcing_W_m2 = 0.0`
- `electron_number_density_cm3 = 1500.0`
- `magnetic_field_nT = 35.0`

### Controls
- `pressure_tendency_Pa_s = 0.0`
- `polar_ice_h2o_tendency_kg_m2_s = 0.0`
- `polar_ice_co2_tendency_kg_m2_s = 0.0`

### Model parameters
- `solar_constant_1au_W_m2 = 1361.0`
- `atmospheric_transmittance = 0.55`
- `surface_emissivity = 0.95`
- `down_ir_effective_emissivity = 0.2`
- `effective_heat_capacity_J_m2_K = 2.0e6`

## Equations used in this 1-hour step

1. Sun-Mars distance (AU):
   - `nu = Ls - Ls_perihelion`, with `Ls_perihelion = 251 deg`
   - `r_AU = a_AU * (1 - e^2) / (1 + e * cos(nu))`

2. TOA incident shortwave:
   - `F_TOA = (S_1AU / r_AU^2) * max(0, cos(theta_z))`

3. Surface incident shortwave:
   - `F_sfc_inc = F_TOA * tau_atm`

4. Reflected shortwave:
   - `F_sfc_refl = alpha * F_sfc_inc`

5. Thermal IR:
   - `F_IR_up = epsilon_up * sigma * T^4`
   - `F_IR_down = epsilon_down * sigma * T^4`

6. Net energy and temperature update:
   - `F_net = F_sfc_inc - F_sfc_refl + F_IR_down - F_IR_up + F_extra`
   - `T_next = T + (F_net / C_eff) * dt_s`

7. Pressure and polar-ice control tendencies:
   - `P_next = P + (dP_dt) * dt_s`
   - `IceH2O_next = IceH2O + (dIceH2O_dt) * dt_s`
   - `IceCO2_next = IceCO2 + (dIceCO2_dt) * dt_s`

## Example output diagnostics (current baseline run, initialized near perihelion)
- `sol_elapsed = 0.04055184573753466`
- `solar_longitude_deg = 251.02183467613747`
- `toa_incident_solar_flux_W_m2 = 356.62331344871853`
- `surface_incident_solar_flux_W_m2 = 196.1428223967952`
- `surface_thermal_ir_flux_W_m2 = 22.05560174763078`
- `pressure_Pa = 610.0`
- `polar_ice_h2o_kg_m2 = 0.0`
- `polar_ice_co2_kg_m2 = 0.0`
- `atmosphere.temperature_K = 210.11591749843916`
- `plasma_magnetic.electron_number_density_cm3 = 1500.0`
- `plasma_magnetic.magnetic_field_nT = 35.0`
