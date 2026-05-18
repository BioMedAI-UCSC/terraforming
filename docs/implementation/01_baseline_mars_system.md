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

## Planet object (baseline schema)
The baseline state should be represented by a grouped `Planet` object with explicit static and dynamic fields.

### Static data (slowly varying or constant for baseline runs)
- **Identity and orbital constants**
  - `name`
  - `gravity_m_s2`
  - `mean_radius_m` (radial distance from Mars center to reference surface)
  - `semi_major_axis_m` (distance from Sun, long-term mean)
  - `axis_tilt_deg` (obliquity; Mars is tilted)
  - `sidereal_day_s` (Martian day / rotation period)
  - `sidereal_year_s` (Mars orbital period around Sun)
  - `rotation_rate_rad_s`
- **Interior and solid planet**
  - `crust`: thickness, density, heat production priors
  - `mantle`: composition fractions, temperature priors, viscosity priors
  - `core`: radius, density, state (liquid/solid fraction), composition priors
- **Geomorphology**
  - `altitude_m` (MOLA-referenced elevation grid)
  - `slope_inclination_deg` (terrain slope grid)
  - `surface_type` (bare ground, dust, ice classes)
- **Soil / regolith**
  - `chemistry` (major/minor species maps)
  - `depth_m`
  - `percolation_capacity_m_s`
  - `porosity`
  - `permeability_m2`
  - `gcm_surface_thermal_inertia_tiu`
  - `gcm_surface_bare_ground_albedo`

### Dynamic data (time-dependent, weather/seasons/diurnal cycles)
- **Orbital and solar geometry**
  - `sun_mars_distance_m(t)` (distance evolution through Martian year)
  - `solar_longitude_deg(t)` (`Ls`)
  - `local_true_solar_time_h(t)` / sol clock
  - `solar_zenith_angle_deg(t, lat, lon)`
- **Atmosphere and weather**
  - `temperature_K(t, lat, lon, z)`
  - `pressure_Pa(t, lat, lon, z)`
  - `density_kg_m3(t, lat, lon, z)`
  - `winds_m_s(t, lat, lon, z)` (`u`, `v`, optional `w`)
  - `column_height_m(t, lat, lon)` (effective atmospheric column height)
  - `air_viscosity_Pa_s(t, lat, lon, z)` (estimated from state variables)
  - `convective_pbl_height_m(t, lat, lon)`
  - `seasons` and synoptic weather diagnostics
- **Atmospheric composition and columns**
  - `vmr(t, lat, lon, z)` for:
    - `CO2`, `N2`, `Ar`, `CO`, `O`, `O2`, `O3`, `H`, `H2`, `He`
  - `column_kg_m2(t, lat, lon)` for:
    - `CO2`, `N2`, `Ar`, `CO`, `O`, `O2`, `O3`, `H`, `H2`, `He`
  - `electron_number_density_cm3(t, lat, lon, z)`
- **Plasma and magnetic environment**
  - `plasma`:
    - ion/electron temperature, density, velocity
    - species-resolved ion densities (e.g., `H+`, `O+`, `O2+`, `CO2+`)
    - solar wind dynamic pressure and energy spectra
  - `magnetic_field_nT(t, lat, lon, z)`:
    - crustal field components
    - induced/external components
    - optional equivalent dipole diagnostics
- **Radiation and energy fluxes**
  - `toa_incident_solar_flux_W_m2(t, lat, lon)` (top of atmosphere)
  - `surface_incident_solar_flux_W_m2(t, lat, lon)`
  - `surface_reflected_solar_flux_W_m2(t, lat, lon)` (horizontal surface)
  - `thermal_ir_up_W_m2(t, lat, lon)` and `thermal_ir_down_W_m2(t, lat, lon)`
  - `surface_thermal_ir_flux_W_m2(t, lat, lon)` (explicit surface thermal IR)
- **Hydro/cryosphere**
  - `polar_ice_h2o_kg_m2(t, lat, lon)` (single layer covering both poles)
  - `polar_ice_co2_kg_m2(t, lat, lon)` (single layer covering both poles)
  - `water_vapor_column_kg_m2(t, lat, lon)` (if source provides `km/m2`, store raw field plus converted SI)
  - `water_vapor_vmr_mol_mol(t, lat, lon, z)`
  - `water_ice_column_kg_m2(t, lat, lon)`
  - `water_ice_vmr_mol_mol(t, lat, lon, z)`

### Suggested grouped object layout
```python
Planet = {
    "meta": {...},
    "static": {
        "identity_orbit": {...},
        "interior": {"crust": {...}, "mantle": {...}, "core": {...}},
        "geomorphology": {...},
        "soil_regolith": {...},
    },
    "dynamic": {
        "time": {...},                # sol, season, orbital geometry
        "atmosphere": {...},          # T, P, rho, winds, PBL, viscosity
        "composition": {...},         # VMR + atmospheric columns
        "plasma_magnetic": {...},     # plasma species + magnetic field
        "radiation_energy": {...},    # solar + reflected + thermal IR fluxes
        "hydro_cryosphere": {...},    # water vapor/ice + merged polar ice layers (H2O and CO2)
        "weather": {...},             # storm/season diagnostics
    },
}
```

### Evolution interface (`evolve()`)
Use a single step function to advance dynamic state while keeping static fields fixed.

```python
def evolve(planet, dt_s, forcings, controls, params):
    """
    Advance Mars state by one timestep.

    Updates dynamic groups only:
    - time/orbital geometry (Ls, Sun-Mars distance, local solar time, zenith)
    - atmosphere (T, P, density, winds, PBL, viscosity)
    - composition (VMR and atmospheric columns)
    - plasma/magnetic environment
    - radiation/energy fluxes (TOA, surface, reflected, thermal IR)
    - hydro/cryosphere (polar H2O/CO2 ice, vapor/ice columns and mixing ratios)
    - weather diagnostics
    """
    # 1) Update orbital/solar geometry
    # 2) Compute top-of-atmosphere forcing
    # 3) Integrate atmosphere + composition tendencies
    # 4) Update plasma/magnetic coupling and escape forcing terms
    # 5) Integrate hydro/cryosphere tendencies
    # 6) Recompute diagnostic fluxes and constraint checks
    return planet
```

Recommended companion methods:
- `initialize(static_data, dynamic_initial)`
- `validate_state(planet)` (units/range/physical checks)
- `diagnostics(planet)` (derived fields for logging and ranking)

### Runtime details and defaults
- `evolve()` is currently a reduced-order baseline stepper in `src/planet.py`.
- `solar_zenith_angle_deg` is user-provided (via `forcings`) or defaults to `60 deg` if omitted.
- TOA incident solar flux is computed as:
  - `F_toa = S_1AU / r_AU^2 * max(0, cos(zenith))`
  - where `S_1AU` default is `1361 W/m2`.
- Surface incident solar flux is currently:
  - `F_surface = F_toa * transmittance`
  - default transmittance is `0.55`.
- Flux naming conventions in runtime:
  - `toa_incident_solar_flux_W_m2`: incoming shortwave at top of atmosphere
  - `surface_incident_solar_flux_W_m2`: shortwave reaching surface
  - `surface_reflected_solar_flux_W_m2`: reflected shortwave from horizontal surface
  - `surface_thermal_ir_flux_W_m2`: downwelling thermal IR at surface

### Example one-hour step (`dt_s = 3600`) near perihelion (`Ls ~ 251 deg`)
Using baseline defaults in `src/planet.py`:

| solar zenith angle | TOA incident solar flux (W/m2) | Surface incident solar flux (W/m2) |
|---|---:|---:|
| `0 deg` | `~713.25` | `~392.29` |
| `60 deg` | `~356.62` | `~196.14` |
| `90 deg` | `~0.0` (floating-point epsilon) | `~0.0` (floating-point epsilon) |

Interpretation:
- `0 deg` is near local noon direct incidence (maximum in this simplified setup).
- `90 deg` is horizon condition, so direct shortwave is effectively zero.
- The modeled TOA value depends on Mars-Sun distance and zenith, so it will vary with `Ls`.
- Runtime Mars orbital phase uses `nu = Ls - 251 deg`; perihelion is `Ls ~= 251 deg` and aphelion is `Ls ~= 71 deg`.
- Runtime `Mars(elevation_m=...)` applies an initialization-only hydrostatic pressure correction, but MOLA terrain lookup and temperature lapse-rate effects are not yet coupled into the evolution step. See `docs/wiki/elevation-effects.md`.

### Notes
- Preserve native source units in raw ingestion, and maintain SI-normalized fields in harmonized outputs.
- Treat static fields as versioned constants (or slowly updated maps), and dynamic fields as time-indexed state variables.
- This schema is intended for Mars baseline but should remain portable to other planetary bodies with minimal renaming.

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

