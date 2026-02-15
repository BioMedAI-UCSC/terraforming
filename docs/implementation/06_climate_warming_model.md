# 6) Climate warming model

## Input
- Atmospheric composition trajectories, GHG injection schedules, albedo/dust scenarios.

## Output
- Global mean temperature trajectory `T(t)` and climate forcing budget.

## Modeling approach (equations + parameters)
- Zero-D energy balance model:
  - `C_T * dT/dt = F_solar * (1 - alpha) / 4 + F_GHG(M_i) + F_dust - OLR(T, M_i)`
- Typical parameterization:
  - `OLR ~= A_olr + B_olr * T`
- Parameters: heat capacity `C_T`, albedo `alpha`, forcing coefficients for super-GHG and dust.

## Data needed
- Spectral forcing coefficients, dust optical depth statistics, albedo maps, thermal inertia estimates.

## Assumptions
- First-order global mean model before spatial GCM refinement.
- Forcing parameterization remains valid over scenario ranges.

## Pre-requisites
- Phases 1, 5.


