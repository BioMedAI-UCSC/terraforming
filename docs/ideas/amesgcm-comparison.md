# AmesGCM physics comparison

## Purpose

This document compares the NASA Ames Mars GCM physics in `AmesGCM/` with the
current differentiable Dinosaur/JAX `gcm3d` implementation. The target is not a
line-by-line Fortran port. The target is a NeuralGCM-style system with a sound
deterministic Mars core, followed by learned residual physics where useful.

The main present-day Mars accuracy gap is not MOLA topography itself. It is the
chain connecting dust, radiation, the surface, and atmospheric mixing:

```text
seasonal dust
  -> solar and infrared heating
  -> surface and atmospheric temperature
  -> pressure gradients and boundary-layer depth
  -> winds, thermal tides, pressure, and polar frost
```

## JAX execution status

The deterministic physics remains expressed with static-shape, differentiable
JAX operations, but the expensive vertical kernels no longer build unnecessarily
large compiled programs:

- Boundary-layer diffusion uses a batched Thomas tridiagonal solve. Runtime and
  compiled graph size scale linearly with vertical layer count, rather than using
  a dense `L x L` solve.
- Local convection uses an exact weighted PAVA stack driven by a fixed `2L-1`
  `lax.scan`; its pushes and merges are O(L), and stable disconnected layers are
  retained.
- Shortwave and thermal two-stream vertical recurrences use `lax.scan`, avoiding
  Python-unrolled radiation graphs.
- Each forced-equation evaluation computes nodal wind, temperature, and surface
  pressure once and shares them among surface energy, drag, PBL, convection,
  radiation, and CO2 exchange.
- Numeric Ames lookup tables are converted to JAX arrays once through a cached
  loader. Small immutable optical constants are native JAX arrays, so they do not
  trigger repeated host-to-device conversion.

These changes improve compilation and execution scaling; they do not change the
calibration or add physical fidelity by themselves. Physics acceptance remains
based on conservation tests and MCD/profile benchmarks.

## Important qualification

AmesGCM contains more physics than its standard configuration enables. Its
standard present-Mars setup uses multiband radiation, an Ames boundary-layer
scheme, surface and atmospheric CO2 condensation, and prescribed radiatively
active background dust. Interactive dust, the bin water cycle, CO2 clouds,
gravity-wave drag, and several advanced modules are disabled by default.

Primary local references:

- `AmesGCM/README.md`: Ames uses an external GFDL FV3 finite-volume core.
- `AmesGCM/build_run/fms_mars_default_v3.2`: standard release configuration.
- `AmesGCM/atmos_param_mars/mars_physics.F90`: physics switches and coupling.
- `AmesGCM/atmos_param_mars/mars_surface.F90`: surface and soil physics.
- `AmesGCM/atmos_param_mars/Rad/radiation_driver.F90`: radiation and aerosols.
- `AmesGCM/atmos_param_mars/BLcode/pblmod_mgcm.F90`: boundary-layer mixing.
- `AmesGCM/atmos_param_mars/update_mars_atmos.F90`: convection and CO2 cycle.
- `AmesGCM/atmos_param_mars/testconserv.F90`: tracer/CO2 conservation checks.

## Simplified property comparison

| Property | What it means | AmesGCM | Current `gcm3d` | Main visible effect |
|---|---|---|---|---|
| Dynamical core | Moves air, heat, and momentum | FV3 finite-volume cubed sphere | Dinosaur spectral primitive equations | Pressure, waves, and global winds |
| Gas radiation | Where CO2 absorbs sunlight and emits infrared | Multiband lookup-table radiation | Ames-derived correlated-k bands with two-stream fluxes | Temperature profile, tides, and winds |
| Dust radiation | Dust heats air and shades the ground | Spectral and particle-size-aware | Ames/Wolff band optics with absorption and scattering | Daytime temperature and circulation |
| Dust distribution | Dust location by height, season, and geography | Seasonal scenarios and Conrath profiles; interactive mode optional | Prescribed seasonal scenarios and Conrath vertical profiles | Seasonal and latitudinal temperature structure |
| Surface energy | Heat stored during day and released later | Deep multilayer soil with restart state | Twelve prognostic ground layers with restart state | Diurnal range and seasonal lag |
| Surface properties | Albedo, emissivity, thermal inertia, and roughness | Spatial fields for each | Spatial albedo, TI, emissivity, and roughness | Regional temperature and surface wind |
| Boundary layer | Turbulent exchange between ground and atmosphere | Mellor-Yamada-style closure with implicit solve | Richardson/mixing-length closure with implicit tridiagonal solve | Near-surface wind and night inversions |
| Convection | Mixes vertically unstable air | Whole-column and surface-connected algorithms | Local conservative PAVA adjustment | Vertical temperature and wind smoothness |
| Surface CO2 frost | Polar atmosphere freezes onto and returns from surface | Coupled seasonal mass and energy exchange | Energy-limited exchange with mass projection | Ice caps and seasonal pressure |
| Atmospheric CO2 condensation | CO2 freezes in cold atmospheric layers | Enabled in standard setup | Missing | Polar atmospheric temperatures |
| CO2 clouds | Suspended atmospheric CO2 ice | Available, disabled by default | Missing | Mainly early/thick Mars climates |
| Water cycle | Vapor, frost, ice clouds, and transport | Available, disabled in standard setup | Missing | Polar hood, clouds, and water ice |
| Aerosol settling | Dust and ice fall under gravity | Sedimentation and surface deposition | Missing | Vertical dust/cloud profiles |
| Coagulation | Small particles collide into larger particles | Optional microphysics | Missing | Particle lifetime and opacity |
| Topographic drag | Unresolved hills exert drag on winds | Available, disabled by default | Missing | Jets and winds near rugged terrain |
| Conservation diagnostics | Detects artificial creation or loss of material | Explicit dust, water, and CO2 checks | CO2 and energy-budget tests | Model reliability |
| Photochemistry | Chemical conversion of atmospheric gases | Interfaces exist; release/default functionality is off or limited | Missing | Low priority for initial maps |
| Nested grids | Higher resolution over selected locations | Example Gale/Jezero configurations | Global grid only | Local crater circulations |

## Explanation and implications

### Dynamical core

The core is the air-motion engine. FV3 transfers material conservatively between
grid cells on six cube faces. Dinosaur represents global fields using spherical
waves. Neither choice alone guarantees a better Mars climate. Dinosaur remains a
good choice because it is JAX-native, differentiable, and suitable for learned
parameterizations.

Pending work is long-duration validation of mass, energy, and angular momentum,
plus conservative tracer tests once dust and water become prognostic. Replacing
the core is lower priority than improving the physical tendencies supplied to it.

### Multiband CO2 radiation

Radiation determines how every layer gains and loses energy. CO2 absorption
varies strongly with wavelength, pressure, and temperature. The current compact
two-band solver conserves column energy but cannot accurately reproduce the
vertical heating structure of the CO2 15-micron and near-infrared regions.

An incorrect vertical heating profile can produce a plausible global-mean
temperature while creating incorrect winds. Winds respond to horizontal and
vertical temperature gradients, not only the mean surface temperature.

Use the Ames tables and algorithms as a validation reference for a differentiable
pressure/temperature interpolation and multiband two-stream solver. Ames code and
data remain subject to the repository's NASA Open Source Agreement.

### Dust

Dust is a primary component of Mars climate. It heats the atmosphere by absorbing
sunlight, shades the surface, absorbs infrared radiation, changes thermal tides,
and reorganizes circulation.

The first target should match the simpler Ames default: prescribed seasonal dust
with a Conrath-like vertical distribution and band-dependent optical properties.
Interactive lifting, transport, settling, and coagulation should follow only after
that configuration is validated.

### Soil and surface properties

- **Albedo** is the fraction of sunlight reflected.
- **Emissivity** controls how efficiently the surface emits thermal infrared.
- **Thermal inertia** measures resistance to rapid temperature change.
- **Roughness** controls aerodynamic exchange of heat and momentum.

Ames defaults to 16 soil layers, supports deep-soil restart state, uses spatial
surface fields, and includes subsurface-ice effects. Our four-layer soil model is
a useful start, but short runs from uniform soil temperature are not comparable to
an equilibrated MCD climatology.

Every visualizer and benchmark path must load the same versioned TES surface data.
The soil should extend beyond the annual thermal skin depth and be spun through
repeated Mars years until its deep-temperature drift is small.

### Boundary layer

The boundary layer is the atmosphere directly stirred by the surface. Hot daytime
ground drives deep mixing; cold nighttime ground produces a stable inversion and
suppresses mixing. This strongly affects near-surface temperature, winds, and dust
lifting.

Ames calculates height-dependent heat and momentum diffusivities from shear,
stability, and a turbulence mixing length, then solves diffusion implicitly. Our
bulk-Richardson scheme captures the sign of the stability effect but is much less
complete and has shown excessive downward momentum transfer.

Implement and validate a level-resolved closure in isolated daytime and nighttime
columns before globally tuning drag coefficients.

### Convective adjustment

Convection removes unstable arrangements in which buoyant warm air lies beneath
colder air. The current scheme mixes the entire column if any adjacent pair is
unstable, which can erase unrelated stable layers. Ames also contains a
surface-connected convective-zone method.

The replacement should identify contiguous unstable blocks, mix potential
temperature conservatively only inside each block, and later mix momentum and
tracers consistently.

### CO2 seasonal cycle

During polar winter, atmospheric CO2 freezes onto the surface, removing
atmospheric mass. It sublimes during spring and returns that mass. The current
implementation already includes a pressure-dependent frost point, latent energy,
energy-limited exchange, non-negative frost projection, and mass checks.

Remaining work includes atmospheric CO2 condensation, frost-dependent albedo and
emissivity, full-year equilibration, and validation of cap edge and seasonal
surface-pressure amplitude. CO2 cloud microphysics is not required for the first
present-day Mars target.

### Water, aerosols, and gravity-wave drag

Water vapor and ice clouds matter for a complete modern-Mars climate, especially
near the poles, but they are not the first explanation for the current large
temperature and wind errors. Add a passive vapor tracer, surface exchange,
saturation adjustment, ice sedimentation, and finally cloud radiative effects.

Sedimentation makes particles fall; coagulation combines small particles into
larger ones. These become necessary for interactive dust and cloud simulations.

Subgrid topographic drag represents the effect of terrain too small for the model
grid. It should be added after radiation and boundary-layer winds are credible so
that it does not become a tuning term masking more fundamental errors.

## Implementation checklist

### P0 — reasonably accurate present-Mars maps

- [ ] Implement pressure- and temperature-dependent multiband CO2 radiation.
- [ ] Add a seasonal and spatial prescribed dust climatology.
- [ ] Add a pressure-dependent Conrath-like vertical dust distribution.
- [ ] Use band-dependent visible and infrared dust optical properties.
- [ ] Make every run path load the same versioned TES surface fields.
- [ ] Add spatial surface emissivity and physically bounded roughness.
- [ ] Expand the soil to roughly 12-16 geometrically spaced layers.
- [ ] Persist soil restart state and spin it past the annual thermal skin depth.
- [ ] Implement level-resolved stable and unstable PBL mixing.
- [ ] Replace whole-column convection with conservative unstable-block mixing.
- [ ] Add atmospheric CO2 condensation.
- [ ] Couple CO2 frost to surface albedo and emissivity.
- [ ] Spin up at least one complete Mars year, then repeat until drift is small.
- [ ] Validate by season, local time, latitude, altitude/pressure, and dust scenario.

### P1 — stronger research-quality Mars GCM

- [ ] Add prognostic dust tracer transport.
- [ ] Add dust lifting, sedimentation, and surface deposition.
- [ ] Add water-vapor and water-ice tracers.
- [ ] Add water surface exchange, saturation adjustment, and cloud sedimentation.
- [ ] Add cloud radiative effects after cloud mass is validated.
- [ ] Add subgrid orographic gravity-wave drag.
- [ ] Add angular-momentum and per-tracer conservation diagnostics.
- [ ] Benchmark thermal tides and zonal-mean circulation, not only map snapshots.

### P2 — advanced or scenario-specific physics

- [ ] Add dust coagulation and particle-size moments.
- [ ] Add CO2 cloud microphysics for early or thick-atmosphere Mars.
- [ ] Add photochemistry and variable atmospheric composition when required.
- [ ] Add nested regional grids for crater-scale applications.
- [ ] Add non-orographic gravity-wave parameterization if the model top requires it.
- [ ] Train NeuralGCM-style residual physics only after deterministic validation.

## Recommended implementation order

```text
multiband radiation
  -> prescribed seasonal dust
  -> deep soil and consistent surface maps
  -> boundary-layer correction
  -> local convective adjustment
  -> coupled CO2 seasonal validation
  -> interactive dust and water
  -> learned residual physics
```

## How AmesGCM should help

Use AmesGCM as a physics reference and teacher model rather than directly replacing
the differentiable core:

1. Reproduce its standard prescribed-background-dust configuration.
2. Export column radiative fluxes and physical tendencies for representative Mars
   columns.
3. Implement differentiable JAX equivalents of radiation, soil, and PBL physics.
4. Require isolated-column tests to agree before running global comparisons.
5. Run Ames, `gcm3d`, and MCD at matching season, local time, dust, and surface
   conditions.
6. Train learned residuals only on persistent, physically diagnosed differences.

This keeps the main advantage of the current architecture—JAX differentiation and
NeuralGCM compatibility—while adopting the mature Mars physics structure exposed
by AmesGCM.

## Minimum AmesGCM physics for the NeuralGCM goal

### Selection rule

Only port physics that satisfies at least one of these conditions:

1. It controls the large temperature, wind, pressure, or seasonal-cap errors that
   a learned model would otherwise have to compensate for.
2. It supplies a conservation law or hard physical constraint that should not be
   learned from data.
3. It produces stable, differentiable tendencies that can be used alongside a
   NeuralGCM residual parameterization.

"Import" should normally mean reimplementing the governing equations and table
interpolation in JAX, with Ames used as an offline reference. It should not mean
calling the Fortran model inside a JAX time step. Any direct reuse of Ames source
or data must comply with `AmesGCM/NOSA.pdf`.

### Priority 1 — minimum deterministic physics

These components are the smallest useful set for a credible NeuralGCM baseline.
They should be completed before training learned residual physics.

#### 1. Multiband CO2 radiative transfer

**Status: Ames coefficients integrated; full Ames flux regression pending.** The
default path now uses the bundled Ames 12-band tables: seven solar bands, five
thermal bands, 16 absorbing correlated-k quadrature points, and one clear channel
per band. `scripts/stage_ames_co2_radiation.py` reproducibly decodes the
little-endian sequential Fortran records, selects the nearly pure-CO2 mixture,
records source SHA-256 hashes, constructs temperature-dependent Planck fractions,
and writes the compact packaged JAX asset `ames_co2_12band.npz`.

The JAX implementation interpolates `log10(k)` bilinearly in temperature and
log-pressure and follows the Ames gas optical-depth relation
`tau = 3.51e22 * delta_p_mbar * k`. It uses Ames Gaussian weights, clear-spectrum
fractions, solar-band fractions, and direct-beam air-mass scaling. The earlier
two-solar/three-thermal compact coefficients remain only as an explicit ablation
fallback (`ames_correlated_k_enabled=False`).

Aggregate interface diagnostics and exact discrete column energy closure are
preserved. Tests pin source hashes and decoded coefficient values and cover table
dimensions, quadrature normalization, pressure response, temperature gradients,
configuration validation, dust-limit behavior, and column closure. The remaining
scientific gap is comparison against flux/heating outputs from a running Ames
column. The solar solver now carries upward and downward flux and uses
energy-conserving hemispheric layer reflection/transmission with the Ames fixed-
dust extinction, scattering, and asymmetry parameters. Thermal dust scattering
is represented through its absorption fraction; reproducing the complete Ames IR
scattering solver remains part of direct column validation.

**Import from Ames:** the division into solar and thermal spectral bands, the
pressure/temperature dependence of gaseous optical depth, two-stream flux
structure, and representative column regression cases from
`atmos_param_mars/Rad/`.

**Implement in JAX:** differentiable interpolation of optical-property tables and
layer-by-layer upward/downward fluxes. Enforce column energy closure explicitly.

**Why it is minimal:** radiation establishes the atmospheric temperature profile
and thermal tides. A learned closure cannot reliably repair winds if the resolved
radiative heating is deposited at the wrong heights.

**Acceptance criteria:**

- [ ] Top-of-atmosphere and surface fluxes match selected Ames columns within a
      documented tolerance.
- [x] Layer heating sums to net column flux convergence.
- [x] Gradients through temperature and pressure-dependent optical depth are finite.
- [ ] Clear-sky seasonal temperature structure improves against MCD without a
      learned correction.

#### 2. Prescribed seasonal dust with a Conrath vertical profile

**Status: implemented; climate validation pending.** `gcm3d.dust` reads the Ames
seasonal scenario, interpolates periodic solar longitude and longitude plus
latitude, and supplies nodal column opacity and dust-top height. Radiation uses
the Ames new-Conrath pressure profile and normalizes layer opacity to the requested
column total. Seven-band Ames fixed-dust solar optics drive two-stream scattering;
five-band IR extinction and absorption drive thermal heating.

**Import from Ames:** the prescribed-dust workflow, seasonal scenario convention,
Conrath-type pressure profile, and wavelength-dependent dust optical properties.

**Implement in JAX:** a prescribed column opacity field indexed by solar longitude
and position, converted into layer opacity with a differentiable vertical profile.
Feed dust absorption and scattering into the same radiation solver as CO2.

**Why it is minimal:** dust is part of the basic radiative state of present Mars,
not an optional aerosol refinement. It strongly changes atmospheric heating,
surface temperature, thermal tides, and winds.

**Acceptance criteria:**

- [x] Integrated layer opacity recovers the requested column opacity.
- [ ] Dust-free, background-dust, and dusty-column tests match Ames flux changes.
- [x] Dust varies with season and location rather than using one global constant.
- [ ] MCD temperature and wind errors are reported separately by dust scenario.

#### 3. Multilayer soil and consistent surface fields

**Status: implemented; spin-up validation pending.** The prognostic soil now has
12 geometrically deepening layers extending beyond the annual thermal skin depth.
The existing restart format already persists every ground layer. Surface loading
now optionally threads nodal emissivity and roughness in addition to TES albedo
and thermal inertia; scalar values remain explicit fallbacks when a dataset lacks
those variables.

**Import from Ames:** geometrically deepening soil layers, deep-temperature restart,
semi-implicit conductive coupling, and spatial albedo, emissivity, thermal inertia,
and roughness inputs from `mars_surface.F90`.

**Implement in JAX:** a compact 12–16-layer conductive column extending past the
annual skin depth. Surface properties must be explicit fields carried through
every CLI, benchmark, and visualizer execution path.

**Why it is minimal:** surface temperature is the lower boundary condition for the
atmosphere. Incorrect heat storage creates false temperature gradients that a
learned model may memorize instead of learning unresolved atmospheric physics.

**Acceptance criteria:**

- [x] Conductive energy is conserved between the surface and soil layers.
- [ ] Diurnal and annual skin-depth tests match the analytic diffusion solution.
- [x] Restarting a run does not reset deep-soil memory.
- [ ] TES fields and their checksums appear in output metadata.
- [ ] Deep-soil drift is small before an MCD climate comparison is accepted.

#### 4. Level-resolved boundary-layer mixing

**Status: implemented; global wind validation pending.** Each interface diagnoses
gradient Richardson number from resolved shear and potential temperature, applies
stable/unstable mixing-length factors, uses separate heat and momentum rates via
a turbulent Prandtl number, and solves both implicitly. Momentum dissipation is
returned as heat and tracers use the conservative implicit heat-mixing operator.

**Import from Ames:** the stability-, shear-, and mixing-length dependence of the
Ames PBL, separate heat and momentum diffusivities, and its implicit tridiagonal
vertical solve from `BLcode/pblmod_mgcm.F90`.

**Implement in JAX:** a differentiable column closure with positive bounded
diffusivities and an implicit solve. Cache wind transformations so the closure does
not repeatedly convert spectral and nodal winds.

**Why it is minimal:** near-surface winds and nighttime temperatures cannot be
recovered from radiation alone. The PBL determines how surface forcing is mixed
upward and how momentum is transferred downward.

**Acceptance criteria:**

- [ ] A convective daytime column mixes deeply while a stable nighttime column
      remains weakly mixed.
- [x] Momentum diffusion cannot create column momentum without surface stress.
- [x] Dissipated kinetic energy is returned as heat or explicitly diagnosed.
- [ ] Near-surface wind error improves without degrading free-atmosphere winds.

#### 5. Local dry convective adjustment

**Status: implemented.** The whole-column switch has been replaced by exact
weighted decreasing isotonic regression, the block/PAVA solution. It mixes only
connected unstable blocks, uses sigma/Exner weights that conserve column enthalpy,
and is static-shape JAX composed of piecewise-differentiable min/max operations.

**Import from Ames:** the surface-connected/contiguous convective-zone logic and
mass-weighted mixing principles in `update_mars_atmos.F90`.

**Implement in JAX:** conservative adjustment of only the unstable contiguous
block, initially for potential temperature and later for momentum and tracers.

**Why it is minimal:** the current whole-column response is overly diffusive and
can erase the vertical structure that the radiation and PBL schemes create.

**Acceptance criteria:**

- [x] Stable layers outside the adjusted block remain unchanged.
- [x] Column enthalpy is conserved to numerical tolerance.
- [x] The final potential-temperature profile is statically stable.
- [x] The operation has finite JAX gradients or a documented differentiable
      approximation at switching boundaries.

#### 6. Complete CO2 condensation coupling

**Import from Ames:** atmospheric condensation checks, transfer of condensed mass
to the surface reservoir when CO2 clouds are disabled, and frost-dependent surface
albedo/emissivity behavior.

**Keep from our model:** pressure-dependent frost point, energy-limited exchange,
latent-energy accounting, non-negative reservoir projection, and explicit total
CO2 conservation tests.

**Why it is minimal:** the seasonal CO2 cycle changes global atmospheric mass and
therefore surface pressure, polar temperature, and circulation. This is a hard
physical constraint rather than a suitable learned correction.

**Acceptance criteria:**

- [ ] Atmosphere plus surface frost conserves CO2 when escape is disabled.
- [ ] Atmospheric layers do not remain substantially below the CO2 frost point.
- [ ] Condensation latent heat and mass tendencies are mutually consistent.
- [ ] Seasonal pressure amplitude and polar-cap timing are compared with MCD.

### Priority 2 — add only after the deterministic baseline works

#### 7. Passive tracers and conservative aerosol sedimentation

Port the Ames tracer-conservation pattern and a minimal sedimentation operator,
but initially use it only for tests or a passive dust tracer. This creates the
infrastructure needed for later interactive dust and water without immediately
adding uncertain source parameterizations.

- [ ] Positive, mass-conservative tracer advection.
- [ ] Conservative settling and surface deposition.
- [ ] Column and global budgets reported for every tracer.

#### 8. Subgrid orographic drag

Port a compact form of the Ames Palmer topographic-drag option only if resolved
winds retain a systematic terrain-correlated bias after radiation, dust, surface,
and PBL fixes. Keep the scheme switchable and diagnose its momentum budget.

- [ ] Demonstrate the bias before enabling the scheme.
- [ ] Verify that drag removes, rather than creates, kinetic energy.
- [ ] Do not tune it to compensate for incorrect radiative or PBL forcing.

### Defer from the minimal NeuralGCM target

The following Ames capabilities should not be imported initially:

| Deferred capability | Reason to defer |
|---|---|
| FV3 dynamical core | Dinosaur is the intended differentiable NeuralGCM core; replacing it does not address the main physics gap. |
| Interactive dust lifting | Highly parameterized and difficult to validate before prescribed dust works. |
| Dust coagulation and size moments | Needed for aerosol research, not the first temperature/wind benchmark. |
| Full water cycle and cloud microphysics | Important later, but not the main cause of current broad temperature and wind errors. |
| CO2 cloud microphysics | Disabled in the Ames standard present-Mars setup and mainly useful for colder/thicker atmospheres. |
| Photochemistry | Low impact on the first surface-map and circulation target. |
| Nested grids | A resolution feature rather than missing global physics. |
| Non-orographic gravity-wave drag | Add only after diagnosing a model-top or upper-atmosphere wind bias. |
| Ames diagnostic/output framework | Keep the native JAX/xarray pipeline and reproduce only essential conservation diagnostics. |

### NeuralGCM handoff point

Begin learned-physics work when priorities 1–6 pass isolated-column tests and a
spun-up deterministic run produces stable seasonal climatology. The learned model
should predict residual tendencies for processes that remain unresolved, not
replace known conservation laws or compensate for incorrect boundary conditions.

Recommended learned outputs:

- bounded temperature residual tendency;
- horizontal momentum residual tendencies;
- optional subgrid vertical fluxes of heat and momentum;
- later, dust or water subgrid source tendencies with explicit budget projection.

Recommended inputs:

- resolved temperature, pressure, winds, and vertical gradients;
- solar longitude, local solar time, latitude, and topographic descriptors;
- surface temperature and spatial surface properties;
- prescribed dust state and boundary-layer stability diagnostics.

The learned tendency should be projected so that dry atmospheric mass is unchanged,
global axial angular momentum is controlled, energy changes are diagnosed, and CO2
or tracer reservoirs remain non-negative. Training targets can come from AmesGCM
high-resolution or detailed-physics runs, with MCD used as an independent climate
benchmark rather than as an exact instantaneous state.

### Minimal execution order

```text
1. multiband CO2 radiation
2. prescribed seasonal dust and vertical profile
3. deep soil plus consistent surface properties
4. level-resolved PBL
5. local convective adjustment
6. complete CO2 condensation coupling
7. full-year spin-up and Ames/MCD benchmarks
8. train constrained NeuralGCM residual tendencies
```
