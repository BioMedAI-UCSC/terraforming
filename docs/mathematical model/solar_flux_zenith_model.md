# Mathematical Model: Solar Flux With Zenith Angle

This model documents the shortwave radiation equations used in the baseline `evolve()` step.

## Inputs
- `S_1AU`: solar constant at 1 AU (`W/m2`)
- `r_AU`: Sun-Mars distance (`AU`)
- `theta_z`: solar zenith angle (`deg`)
- `tau_atm`: effective atmospheric transmittance (unitless, `0..1`)
- `alpha`: surface albedo (unitless, `0..1`)

## Core equations
1. **Top-of-atmosphere (TOA) incident solar flux**

   `F_TOA = (S_1AU / r_AU^2) * max(0, cos(theta_z))`

2. **Surface incident solar flux**

   `F_sfc_inc = F_TOA * tau_atm`

3. **Surface reflected shortwave flux (horizontal surface)**

   `F_sfc_refl = alpha * F_sfc_inc`

## Zenith-angle behavior
- `theta_z = 0 deg` -> maximum direct incidence (`cos = 1`)
- `theta_z = 60 deg` -> half of normal-incidence component (`cos = 0.5`)
- `theta_z = 90 deg` -> direct shortwave ~ `0` (`cos = 0`)

## Baseline example values
Using the current baseline settings from `src/planet.py`:
- `S_1AU = 1361 W/m2`
- `tau_atm = 0.55`
- near `Ls ~ 0 deg`, `r_AU ~ 1.38`

Example one-step outputs (`dt_s = 3600`):
- `theta_z = 0 deg`: `F_TOA ~ 713.25 W/m2`, `F_sfc_inc ~ 392.29 W/m2`
- `theta_z = 60 deg`: `F_TOA ~ 356.62 W/m2`, `F_sfc_inc ~ 196.14 W/m2`
- `theta_z = 90 deg`: `F_TOA ~ 0 W/m2`, `F_sfc_inc ~ 0 W/m2`
