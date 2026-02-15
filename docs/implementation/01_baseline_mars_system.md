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
| data component | status | date range (local) | source | notes |
|---|---|---|---|---|
| MOLA DEM | have | static global | MOLA | `mola32.nc` present |
| MCD climatology + dust | have | climatology (multi-year bins, no single mission window) | MCD v6.1 | `clim_*.nc`, `dust_*` present |
| MAVEN KP insitu | have | `2014-10-01..2016-08-17` | MAVEN SDC | now covers full one-Martian-year baseline window; common overlap with EUV starts `2014-10-19` |
| MAVEN KP IUVS | have | `2014-10-01..2025-12-31` | MAVEN SDC | coverage exceeds baseline window; 2026 partial not complete |
| MAVEN EUV L3 minute | have | `2014-10-19..2016-08-17` | MAVEN SDC / PDS | baseline 1-Martian-year window covered |
| MAVEN NGIMS L2/L3 | have | `2014-10-01..2016-08-17` | MAVEN SDC | baseline 1-Martian-year window covered |
| MAVEN SWIA L2 fine-arc 3D | partial | `2014-07-07..2016-04-25` | MAVEN SDC / PDS | high-resolution ion velocity distributions; useful for solar-wind coupling and escape forcing, but not yet full-year overlap with EUV/NGIMS window |
| Ice maps (caps + subsurface) | partial | targeted orbit subset (not continuous time series) | MARSIS optimized radargrams | 12 orbits: 5 south polar, 5 north polar, 2 equatorial |
| Regolith properties | partial | mostly static global maps + selected tiles | TES, THEMIS, CRISM | TES thermal inertia map + THEMIS projected albedo subset + CRISM MRDR starter tiles |
| Uncertainty metadata | missing | n/a | all sources | obs errors + covariances for `Sigma_0` |

## Data we currently have (local)
- MCD v6.1 core + climatology/dust grids (`data/mcd/MCD_6.1/data/`)
- MOLA DEM + terrain derivatives (`data/mcd/MCD_6.1/data/mola32.nc`, `slope_map.nc`, `mountain.nc`)
- MOLA global mosaics (`data/Mars_MGS_MOLA_DEM_mosaic_global_463m.tif`, `data/mars_mgs_mola_dem_mosaic_global_1024.jpg`)
- MAVEN EUV modeled irradiance samples (`data/maven/euv/_maven.euv.modelled/`, `data/maven/euv/24.0/`, `data/maven/euv/test/`)
- MAVEN insitu KP daily samples (`data/maven/insitu/kp/`)
- MAVEN KP API pull (`data/maven/kp_api_full/`):
  - KP insitu: monthly chunks `2014-10-01..2016-08-17` (tail chunks `2016-05..2016-08-17` added)
  - KP IUVS: yearly chunks `2014-10-01..2025-12-31` (2026 partial not complete)
- MAVEN NGIMS L2 partial 2024 months (`data/maven/ngims/l2/2024/`, local filenames span `2024-01-01..2024-09-18`)
- MAVEN SWIA fine-arc 3D partial (`data/maven/swia/fine_arc_3d/`, local filenames span `2014-07-07..2016-04-25`)
- MAVEN SDC API pull (`data/maven/sdc_api_full/`):
  - EUV L3 minute: `2014-10-01..2016-08-17` requested, local files from `2014-10-19..2016-08-17`
  - NGIMS L2: `2014-10-01..2016-08-17` (1 Martian year, quarter-chunked)
  - NGIMS L3: `2014-10-01..2016-08-17` (1 Martian year, quarter-chunked)
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
- MAVEN escape-rate or equivalent derived products (species-wise loss rates), and/or full SWEA/SEP/MAG/STATIC/LPW L2 if we move beyond KP-driven forcing.
- Ice maps: additional MARSIS orbits for denser polar coverage, plus MRO/SHARAD and Odyssey GRS for cross-validation.
- Regolith properties: CRISM starter set is complete; expand beyond 4 tiles to full regional coverage.
- Uncertainty metadata for all inputs (obs errors + covariance priors for `Sigma_0`).

## Download targets (baseline minimum)
- MAVEN KP insitu: baseline one-Martian-year overlap target is complete (`2014-10-01..2016-08-17`).
- MAVEN EUV modeled irradiance: current one-Martian-year pull is sufficient for baseline minimum (`2014-10-19..2016-08-17` local coverage).
- MAVEN NGIMS L2/L3: current one-Martian-year pull is sufficient for baseline minimum (`2014-10-01..2016-08-17`).
- MAVEN SWEA/SEP/MAG/STATIC/LPW L2 (optional if not relying on KP-only): 1 Mars year to drive solar wind/particle forcing.
- MAVEN SWIA L2: expand to 1 full Mars year if using SWIA outside KP.

## Missing / pending downloads
- MAVEN SWEA/SEP/MAG/STATIC/LPW L2 (not present yet; optional if KP-only path is insufficient).
- MAVEN escape-rate derived products (not present yet).
- Ice maps: denser MARSIS coverage + MRO/SHARAD + Odyssey GRS for cross-validation (have 12 baseline orbits).
- Regolith properties:
  - CRISM expansion beyond 4 starter tiles (or move to broader MRDR/MTRDR coverage).
  - Regolith porosity/permeability constraints not yet assembled from a dedicated source.

## Date-range overlap remarks
- Common MAVEN overlap currently available:
  - EUV minute + NGIMS L2/L3 overlap: `2014-10-19..2016-08-17`
  - KP insitu + EUV minute + NGIMS L2/L3 overlap: `2014-10-19..2016-08-17`
- Non-overlap currently present:
  - KP IUVS extends to `2025-12-31` while EUV/NGIMS baseline pull stops in 2016.
  - NGIMS 2024 partial set in `data/maven/ngims/l2/2024/` is outside the 2014-2016 baseline window.

## Assumptions
- Datasets are temporally harmonized to one reference epoch.
- Bias corrections are stable over simulation horizon.

## Pre-requisites
- None (foundation phase).


