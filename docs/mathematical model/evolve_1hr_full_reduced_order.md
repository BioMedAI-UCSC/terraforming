# Mathematical Model: `evolve()` 1-Hour Full Reduced-Order Scaffold

This document keeps the current full-scope `evolve()` scenario under `docs/mathematical model`.
It describes the reduced-order model used for a 1-hour Mars step with coupled atmosphere, radiation,
plasma/magnetic, cryosphere, composition, regolith chemistry, and gravity forcings.

## Scope
- Time step: `dt_s = 3600` (1 hour)
- Model type: reduced-order, box-style tendencies + diagnostic couplings
- Implementation: `src/planet.py` (`PlanetModel.evolve`)

## State groups included
- Orbital/time: `Ls`, Sun-Mars distance, axis tilt metadata, rotation/orbital rates
- Atmosphere: `temperature_K`, `pressure_Pa`, density
- Radiation: TOA/surface shortwave, thermal IR, cosmic dose diagnostics
- Plasma/magnetic: electron density, solar-wind pressure, magnetic field tendency/update
- Hydro/cryosphere: H2O/CO2 polar ice, bulk solid/liquid water columns
- Composition: species columns + VMR normalization + cosmic escape tendencies
- Soil/regolith: chemistry mass fractions with runtime compound changes
- External gravity: giant planets + Mars moons (Phobos/Deimos) accelerations/tidal diagnostics
- Interior metadata placeholders: mantle/core/crust

## Core equations (reduced-order)

1. Orbital distance:
- `r_AU = a_AU * (1 - e^2) / (1 + e * cos(Ls-251))`

2. TOA and surface shortwave:
- `F_TOA = (S_1AU / r_AU^2) * max(0, cos(theta_z))`
- `F_sfc_inc = F_TOA * tau_atm`
- `F_sfc_refl = alpha * F_sfc_inc`

3. Thermal IR:
- `F_IR_up = epsilon_up * sigma * T^4`
- `F_IR_down = epsilon_down * sigma * T^4`

4. Temperature tendency:
- `F_net = F_sfc_inc - F_sfc_refl + F_IR_down - F_IR_up + F_extra + F_cosmic`
- `T_next = T + (F_net / C_eff) * dt_s`

5. Pressure tendency:
- `dP/dt = dP_control + dP_cosmic + dP_giant + dP_moons`
- `P_next = P + (dP/dt) * dt_s`

6. Cosmic coupling:
- `dose_surface = dose_toa * (1 - shielding)`
- `dP_cosmic/dt = -dose_surface * k_cosmic_pressure_loss`
- `dO3/dt += -dose_surface * k_ozone_loss`
- `dH/dt  += -dose_surface * k_h_escape`
- `dH2/dt += -dose_surface * k_h2_escape`

7. Solar-wind dynamic pressure (if not directly provided):
- `P_dyn = n * m_p * v^2`

8. Magnetic field update:
- `dB/dt = dB_manual + dB_wind + dB_forcing_relax + dB_background_recovery`
- `B_next = B + (dB/dt) * dt_s`

9. Moons gravity diagnostics:
- direct acceleration: `a = G * M / d^2`
- tidal acceleration: `a_tidal = 2 * G * M * R_mars / d^3`

10. Composition update:
- `column_i,next = max(0, column_i + tendency_i * dt_s)`
- `vmr_i = column_i / sum_j(column_j)` (with normalization safeguards)

11. Regolith chemistry update:
- `f_i,next = max(0, f_i + df_i_dt * dt_s + delta_i)`
- renormalize `sum_i(f_i) = 1`

## Why this is a mathematical model
- It is a coded numerical model based on explicit equations and state evolution.
- It is intentionally low-order/scaffold-level and not a full Mars GCM.
