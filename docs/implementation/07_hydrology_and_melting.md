# 7) Hydrology and melting

## Input
- Temperature/pressure trajectories, cryosphere inventory, terrain and permeability.

## Output
- Liquid water inventory, melt rates, runoff/infiltration fluxes, stable water zones.

## Modeling approach (equations + parameters)
- Water-phase mass balances:
  - `dW_ice/dt = -Melt(T,P) + Freeze(T,P) + Deposition - Sublimation`
  - `dW_liq/dt = Melt - Freeze - Evap(T,P) - Infiltration + Runon - Runoff`
- Simple melt law example:
  - `Melt = k_m * max(0, T - T_m(P))`
- Parameters: melt coefficient `k_m`, permeability `K`, evaporation coefficients.

## Data needed
- Ice distribution/thickness, topography, regolith hydraulic properties, phase diagram constraints.

## Assumptions
- First implementation uses regional boxes/catchments, not full 3D hydro.
- Subsurface hydrology represented by effective parameters.

## Pre-requisites
- Phases 1, 5, 6.


