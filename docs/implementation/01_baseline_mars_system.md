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
| MAVEN loss + radiation products | partial | MAVEN SDC / PDS | EUV modeled samples + KP samples + NGIMS L2 2024 + SWIA fine-arc |
| Ice maps (caps + subsurface) | partial | MARSIS optimized radargrams | 12 orbits: 5 south polar, 5 north polar, 2 equatorial |
| Regolith properties | partial | TES, THEMIS, CRISM | TES thermal inertia map + THEMIS projected albedo subset + CRISM MRDR starter tiles |
| Uncertainty metadata | missing | all sources | obs errors + covariances for `Sigma_0` |

## Data we currently have (local)
- MCD v6.1 core + climatology/dust grids (`data/mcd/MCD_6.1/data/`)
- MOLA DEM + terrain derivatives (`data/mcd/MCD_6.1/data/mola32.nc`, `slope_map.nc`, `mountain.nc`)
- MOLA global mosaics (`data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`, `data/mars_mgs_mola_dem_mosaic_global_1024.jpg`)
- MAVEN EUV modeled irradiance samples (`data/maven/euv/_maven.euv.modelled/`, `data/maven/euv/24.0/`, `data/maven/euv/test/`)
- MAVEN insitu KP daily samples (`data/maven/insitu/kp/`)
- MAVEN NGIMS L2 partial 2024 months (`data/maven/ngims/l2/2024/`)
- MAVEN SWIA fine-arc 3D partial (`data/maven/swia/fine_arc_3d/`)
- MARSIS optimized radargrams (`data/marsis/radargram_data/`):
  - South Polar Cap (SPLD): orbits 01867, 01883, 01900, 01901, 01902 (folders 018xx, 019xx)
  - North Polar Cap (NPLD): orbits 03002, 03003, 03004, 03006, 03023 (folder 030xx)
  - Equatorial: orbits 08000, 08004 (folder 080xx)
  - Format: `*_optim_wind_f.img` + `*_optim_wind_f.xml` (~64 MB per orbit)
  - Total: 12 orbits × ~64 MB = ~768 MB
  - Source: PDS Geosciences Node (https://pds-geosciences.wustl.edu/mex/urn-nasa-pds-mex_marsis_optim/)
- TES thermal inertia global map (`data/themis-thermal-inertia/`):
  - `ti16` VICAR numeric raster (`2880 x 1440`, 32-bit int, values ~25-602)
  - PNG browse layers: `TES_Thermal_Inertia.png`, `TES_Thermal_Inertia_mola.png`, `tes_ti_label.png`
- THEMIS projected albedo starter pack (`data/themis_odtgeo_albedo/`):
  - 10 visual projected albedo products (`*.IMG` + `*.xml`) across `v008xxalb`-`v012xxalb`
  - Includes `CMIDX_ODTVA.TAB` index + metadata bundle for scaling to larger pulls
- CRISM MRDR mineralogy starter pack (`data/crism/mrdr_3201_baseline/`):
  - Full metadata/index from `MROCR_3201`
  - Downloaded tiles: `MC02`, `MC15`, `MC24`, `MC30`
  - Product families: `MRRSU` (summary mineral indices), `MRRAL` (multispectral reflectance), `MRRDE` (geometry/context)
  - Current status: starter set complete for all four tiles (`MC02`, `MC15`, `MC24`, `MC30`)

## Still needed for baseline initial twin
- MAVEN full-year (or multi-year) coverage for EUV + insitu KP, or full L2 bundles for SWEA/SEP/MAG/STATIC/LPW to derive escape forcing.
- MAVEN escape-rate or equivalent derived products (species-wise loss rates).
- Ice maps: additional MARSIS orbits for denser polar coverage, plus MRO/SHARAD and Odyssey GRS for cross-validation.
- Regolith properties: CRISM starter set is complete; expand beyond 4 tiles to full regional coverage.
- Uncertainty metadata for all inputs (obs errors + covariance priors for `Sigma_0`).

## Download targets (baseline minimum)
- MAVEN insitu KP: at least one full Mars year of daily KP files (EUV+LPW+MAG+SEP+STATIC+SWEA+SWIA in one product).
- MAVEN EUV modeled irradiance: 1 Mars year of daily `.cdf` + `.xml` pairs (minute model).
- MAVEN NGIMS L2: at least one representative year (daily/periapsis passes) for species densities.
- MAVEN SWEA/SEP/MAG/STATIC/LPW L2 (if not relying on KP-only): 1 Mars year to drive solar wind/particle forcing.
- MAVEN SWIA L2: full-year coverage (if using SWIA outside KP).

## Missing / pending downloads
- MAVEN KP daily coverage beyond current samples (need a full Mars year).
- MAVEN EUV modeled irradiance beyond current samples (need a full Mars year).
- MAVEN NGIMS L2 coverage beyond partial 2024.
- MAVEN SWEA/SEP/MAG/STATIC/LPW L2 (not present yet).
- MAVEN escape-rate derived products (not present yet).
- Ice maps: denser MARSIS coverage + MRO/SHARAD + Odyssey GRS for cross-validation (have 12 baseline orbits).
- Regolith properties:
  - CRISM expansion beyond 4 starter tiles (or move to broader MRDR/MTRDR coverage).
  - Regolith porosity/permeability constraints not yet assembled from a dedicated source.

## Assumptions
- Datasets are temporally harmonized to one reference epoch.
- Bias corrections are stable over simulation horizon.

## Pre-requisites
- None (foundation phase).


