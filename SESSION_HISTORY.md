# Terraforming Project — Session History & Changes

This document summarizes the development sessions, key changes, and the prompts/goals that drove them.

---

## Session 1 — Project Initialization
**Date:** 2026-02-14
**Commits:** `7a98ced`, `8b48e26`, `550a09b`, `470137d`, `3d1ffc6`, `08555c4`, `b42140b`

### Goal
Bootstrap the project with documentation, objectives, and real Mars data downloads.

### Changes
- Added initial documentation (`README.md`, `docs/OBJECTIVES.md`, `docs/MEASUREMENT_REGISTER.md`, `docs/PLAN.md`)
- Added download scripts for MAVEN EUV samples, MARSIS polar orbits, and MAVEN key parameter (KP) API data
- Downloaded NGI and EUV datasets; organized raw data into `data/`
- Added `Makefile` for automation of data download targets

### Prompts / Goals
- Define what terraforming targets to track (temperature, pressure, ice, magnetic field, radiation)
- Establish the measurement register mapping physical observables to data sources (MCD, MAVEN, MARSIS)
- Set up reproducible data acquisition pipeline

---

## Session 2 — System Architecture & Planet Object
**Date:** 2026-02-16
**Commits:** `02b2651`, `ab29ae7`, `eefcac1`, `4d953c3`

### Goal
Design and implement the high-level and low-level architecture for the simulation framework.

### Changes
- Implemented HLD and LLD for the base system (`docs/architecture/`, `docs/implementation/`)
- Added initial `Planet` base object with core physical attributes
- Added `docs/planet.md` describing the planet object design

### Prompts / Goals
- Design the `Planet` abstraction: what properties does it hold vs. what does the engine own?
- Define the separation of concerns: planet knows the physics equations, engine knows how to integrate them

---

## Session 3 — UV Project Setup & Planet Integration
**Date:** 2026-02-18 to 2026-02-20
**Commits:** `ae64cd2`, `3ab5300`

### Goal
Set up the `uv` Python package manager and wire the `Mars` planet object into a runnable experiment.

### Changes
- Initialized `uv` project (`pyproject.toml`, `uv.lock`) with dependencies: `torch`, `matplotlib`, `numpy`
- Created `src/` package layout:
  - `src/constants/__init__.py` — physical constants (AU, solar constant, PI, etc.)
  - `src/engine/__init__.py` and `src/engine/time_controller.py` — `TimeController` with `Accuracy.FAST` and `Accuracy.ACCURATE` modes
  - `src/celestials/__init__.py` — `Mars` class export
  - `src/celestials/planets/mars.py` — Mars-specific physics (albedo, greenhouse, CO2 ice sublimation)
  - `src/celestials/framework/planet.py` — abstract `Planet` base class
- Added `experiments/run_mars.py` — first runnable experiment: evolve Mars for 1 sol and 1 year, save CSVs and plots
- Added `tests/mars/` suite: planet interface tests, daily/yearly/time-controller tests

### Prompts / Goals
- Integrate the Mars planet object with the engine so a simulation can actually run
- Run Mars for one sol and one Martian year, produce output plots and CSVs
- Support both a fast reduced-order mode and an accurate RK4 ODE integration mode

---

## Session 4 — PyTorch Backend Conversion
**Date:** 2026-02-20
**Commit:** `d870dfc`

### Goal
Replace NumPy/plain-Python numerics with a PyTorch tensor backend throughout the framework.

### Changes
- Converted all state variables and constants to `torch.Tensor` scalars (`dtype=torch.float64`)
- `src/constants/__init__.py`: all constants are now `torch.Tensor`; added `_c()` helper to cast scalars
- `src/celestials/framework/planet.py`: state vector packed/unpacked as `torch.Tensor [3]`; ODE RHS returns a tensor
- `src/celestials/planets/mars.py`: all physics equations now use PyTorch ops
- `src/engine/time_controller.py`: RK4 integrator rewritten with tensor arithmetic
- Updated tests to handle `torch.Tensor` comparisons
- Reduced `uv.lock` size significantly (removed heavy non-torch dependencies)

### Prompts / Goals
- Make the framework GPU-ready and differentiable for future optimization or learned-physics use cases
- Ensure consistent float64 precision across all computations

---

## Session 5 — Framework Refactoring (Part 1)
**Date:** 2026-02-21
**Commit:** `04c8863`

### Goal
Refactor the module layout: move `planet.py` from `src/celestials/framework/` to `src/framework/`.

### Changes
- Moved `src/celestials/framework/planet.py` → `src/framework/planet.py`
- Updated imports in `src/celestials/__init__.py` and `src/celestials/planets/mars.py`
- Minor fixes to `src/engine/time_controller.py` imports
- Mars physics updated to align with the new module paths

### Prompts / Goals
- Separate the framework (generic abstractions) from the celestials (planet-specific implementations)
- Cleaner package layout: `src/framework/` for base classes, `src/celestials/` for concrete planets

---

## Session 6 — Seasonal Plots & Output Reorganization
**Date:** 2026-02-21
**Commit:** `41b1f52`

### Goal
Add seasonal temperature analysis and reorganize output file layout.

### Changes
- `experiments/run_mars.py`:
  - Added `plot_seasonal_temps()`: plots daily min/max/avg temperature vs. Solar Longitude (Ls)
  - Added `plot_history()`: now saves temperature, pressure, and ice mass as separate PNG files
  - Added `save_history_to_csv()`: writes full history to CSV
  - Reorganized outputs into `outputs/sol/` and `outputs/668_sols/` subdirectories
  - Fixed year simulation to run an integer number of complete sols (668 sols) to avoid biased partial-sol averages
- `src/celestials/planets/mars.py`: adjustments to thermal physics for better seasonal behavior
- Removed stale root-level output files (`outputs/mars_sol_evolution.*`, `outputs/mars_year_evolution.*`)
- New outputs: `outputs/sol/mars_sol_evolution_{temp,pressure,ice}.png`, `outputs/668_sols/mars_year_evolution_*.png`, `outputs/668_sols/mars_seasonal_temps.png`

### Prompts / Goals
- Produce a seasonal temperature plot (Ls 0°–360°) showing the ~90°C swing from Martian summer to winter
- Split combined plots into separate figures per variable (temperature, pressure, ice mass)
- Fix spike artifact near Ls ~256° caused by a partial last sol being included in daily averages

---

## Session 7 — Framework Systems Refactoring (Part 2)
**Date:** 2026-02-22
**Commit:** `c16acd8`

### Goal
Decompose the monolithic `Planet` base class into separate domain-specific property dataclasses.

### Changes
- Created new `src/framework/` modules, each holding a focused dataclass:
  - `atmosphere.py` — `Atmosphere` (surface pressure, CO2 partial pressure)
  - `thermal.py` — `Thermal` (surface temperature, heat capacity)
  - `water.py` — `Water` (ice mass, liquid water, water vapor)
  - `radiation.py` — `Radiation` (solar flux, albedo, greenhouse factor)
  - `magnetic.py` — `Magnetic` (field strength, dipole moment)
  - `intrinsic.py` — `IntrinsicParameters` (mass, radius, gravity)
  - `orbital.py` — `OrbitalParameters` (semi-major axis, eccentricity, orbital/rotation periods, Kepler distance formula)
  - `__init__.py` — re-exports all framework classes
- `src/framework/planet.py`: slimmed down; `Planet` now composes these dataclasses rather than defining all fields inline
- `src/celestials/planets/mars.py`: updated to use new property dataclasses; physics methods unchanged
- `src/celestials/__init__.py`: updated imports
- `experiments/run_mars.py`: minor import fix
- All tests updated to use new attribute paths (e.g., `state.thermal.surface_temperature`)

### Prompts / Goals
- Improve extensibility: each domain (atmosphere, water, etc.) can be extended or replaced independently
- Make it easy to add new planets by configuring only the relevant property dataclasses
- Enable future work where individual subsystems (e.g., magnetic field) can evolve independently

---

## Session 8 — Multi-Coordinate Experiment & Plot Updates
**Date:** 2026-02-23
**Commits:** `5f36796` (update plots), `95db291` (save work)

### Goal
Run simulations at three distinct Mars coordinates (North, Equator, Southern Hemisphere) and produce a combined seasonal temperature comparison plot.

### Changes
- `experiments/run_mars.py`:
  - Added `run_three_coordinates()`: runs separate simulations for 45°N/137°E, 0°/137°E, and -40°/137°E with a 10-minute timestep and 10-sol spin-up period
  - Added `plot_three_temperatures()`: overlays daily-average temperature vs. Ls for all three coordinates on a single figure
  - `run_one_year()` updated to use integer sol count and avoid partial-day bias
  - `plot_seasonal_temps()` updated with spin-up exclusion parameter (`spin_up_sols`)
  - Wrap-around handling for Ls near 0°/360° boundary in daily averaging
- `src/celestials/planets/mars.py`: `Mars` constructor now accepts `latitude` and `longitude` parameters to set location-dependent insolation geometry
- New outputs in `outputs/668_sols/`:
  - `mars_north_evolution_{temp,pressure,ice}.png` + `mars_north_seasonal.png`
  - `mars_equator_evolution_{temp,pressure,ice}.png` + `mars_equator_seasonal.png`
  - `mars_southern_hemisphere_evolution_{temp,pressure,ice}.png` + `mars_southern_hemisphere_seasonal.png`
  - `mars_three_coords_temps.png` — combined Ls vs. temperature comparison
  - Corresponding CSVs for each coordinate

### Prompts / Goals
- Compare seasonal temperature patterns at different latitudes to validate the thermal model
- Add spin-up period to let the model reach thermal equilibrium before recording seasonal statistics
- Show the latitude-dependent insolation asymmetry (southern hemisphere has warmer summers due to Mars perihelion)

---

## Current State (Branch: `v0-implementation`)

### Architecture Summary

```
src/
  constants/          # Physical constants as torch.float64 tensors
  framework/          # Abstract base classes and property dataclasses
    planet.py         # Abstract Planet (pack/unpack state, advance_orbit)
    atmosphere.py     # Atmosphere dataclass
    thermal.py        # Thermal dataclass
    water.py          # Water dataclass
    radiation.py      # Radiation dataclass
    magnetic.py       # Magnetic dataclass
    intrinsic.py      # IntrinsicParameters dataclass
    orbital.py        # OrbitalParameters dataclass (Kepler orbit)
  celestials/
    planets/
      mars.py         # Concrete Mars planet (physics equations, lat/lon support)
  engine/
    time_controller.py  # TimeController: FAST (relaxation) and ACCURATE (RK4) modes

experiments/
  run_mars.py         # Main experiment: 1-sol, 1-year, 3-coordinate simulations

outputs/
  sol/                # 1-sol simulation outputs
  668_sols/           # 1-year (668-sol) simulation outputs per coordinate
```

### Key Design Decisions
- **PyTorch backend**: all state is `torch.float64`; framework is GPU-ready and differentiable
- **Two integration modes**: `Accuracy.FAST` (analytic relaxation, cheap) vs. `Accuracy.ACCURATE` (RK4, exact)
- **Planet/Engine separation**: `Planet` owns physics equations; `Engine` owns integration strategy
- **Modular framework**: each physical domain is its own dataclass, composable into any planet
- **Spin-up handling**: multi-year runs discard initial transient sols before computing seasonal statistics
