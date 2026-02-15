# 1) Baseline Mars system

## Input
- Raw present-day Mars datasets (topography, atmosphere, radiation, ice, dust, regolith).

## Output
- Calibrated initial state `x(0)` + uncertainty bounds `Sigma_0`.

## Modeling approach (equations + parameters)
- Data assimilation / state estimation:
  - `x_hat = argmin_x (y - Hx)^T R^{-1} (y - Hx) + (x - x_b)^T B^{-1} (x - x_b)`
- Parameters: `H` (observation operator), `R` (obs covariance), `B` (background covariance), `x_b` (background state).

## Data needed
- MOLA DEM, MCD (Mars Climate Database), MAVEN loss/radiation products, ice maps, dust climatology.
- MAVEN EUV modeled irradiance (minute, 0-190 nm):
  - Required files: daily `.cdf` data file plus matching `.xml` label (metadata).
  - Initial state minimum: one reference day plus +/- 7 days (15 files) to compute daily mean and variability.
  - Uncertainty bounds: 1 Mars year of daily files for seasonal range; full mission only if solar-cycle bounds are needed.

## Baseline data checklist (Phase 1)
| data component | status | source | notes |
|---|---|---|---|
| MOLA DEM | have | MOLA | `mola32.nc` present |
| MCD climatology + dust | have | MCD v6.1 | `clim_*.nc`, `dust_*` present |
| MAVEN loss + radiation products | missing | MAVEN SDC / PDS | escape rates, EUV/particle flux, solar wind |
| Ice maps (caps + subsurface) | missing | MRO/SHARAD, MARSIS, Odyssey GRS | cryosphere inventories |
| Regolith properties | missing | TES, THEMIS, CRISM | thermal inertia, albedo, mineralogy |
| Uncertainty metadata | missing | all sources | obs errors + covariances for `Sigma_0` |

## Assumptions
- Datasets are temporally harmonized to one reference epoch.
- Bias corrections are stable over simulation horizon.

## Pre-requisites
- None (foundation phase).


