# Ames physics on Dinosaur

Work branch: `feat/ames-physics-parity`, from `experimenal` at `9013f42`.
Reference: NASA Ames GCM v3.2.1, commit
`ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26`.

Keep the Dinosaur dynamical core and JAX differentiation. Match the Ames
physical equations, process switches and boundary conditions. Adapt array
layout and integration interfaces as needed for Dinosaur. Kernel agreement,
global climate agreement, gradient accuracy and speed are separate checks.
No complete-model equivalence or speed claim has passed yet.

## Verified ports

| Process | Implementation | Native comparison | Coupled status |
| --- | --- | --- | --- |
| Dry convection | `dry_convection.pass2` | Compiled original `pass2`; four 40-layer cases with 2,048 columns, temperature errors below 2e-13 K | `adjust_state` and `build_step` support Dinosaur; explicit reference backend selection is available in the map runner |
| PBL surface exchange | `boundary_layer.surface_exchange` | Compiled original `bndcond`, stable/unstable/zero-wind cases | Used by the new column driver |
| PBL turbulent mixing | `boundary_layer.eddy_coefficients` | Compiled original `eddycoef`, 4/12/40 layers | Used by the new column driver |
| PBL gas-column driver | `boundary_layer.column_step` | Compiled original `pbl_driver`, `newpbl`, matrix, scaling and solve routines; temperature errors below 1e-13 K | Connected to the soil/frost column and experimental Dinosaur adapter; tracer categories remain pending |
| Radiation pressure-table staging | `stage_co2_radiation_tables.ames_pressure_interpolation` | Original `laginterp` and `lagrange`; all ten gas mixtures match exactly in the recorded run | Enabled in the existing correlated-k kernel |
| Radiation dry-table lookup | `correlated_k._interp2_log_table` on the refined pressure grid | Original `tpindex15` and `boxinterp`; 100 temperature/pressure points, including values outside the tables | Enabled in the existing correlated-k kernel |
| Two-stream flux kernels | `scattering_transfer.visible_fluxes` and `infrared_fluxes` | Original `gfluxv`, `gfluxi`, `dsolver`, `dtridgl`; 1/4/12/40-layer thin/moderate/thick columns | Visible keeps the original solve. IR solves the same hemispheric equations with stable layer adding; the composed check uses unchanged Fortran routines at binary128 without relaxing flux tolerances |
| Radiation pressure/temperature grid | `radiation_grid.radiation_grid` | Original `fillpt`; all output arrays agree exactly in the recorded 1/4/12/40-layer cases, with positive and zero top pressure | Supports Dinosaur's zero top without adding ghost mass or a pressure floor |
| Spectral flux integration | `spectral_flux.visible_spectrum` and `infrared_spectrum` | Original `sfluxv`, `sfluxi`, `getdetau` and flux solvers; 27 cases with weak/mixed/strong gas opacity and both surface-emission scaling settings | Connected to optical aggregation; the special bottom IR sample uses its full, possibly floored, depth |
| Optical-depth aggregation | `optical_depth.gas_subgrid` and `aggregate_optics` | Original `optcv15`, `optci15`, `tpindex15`, `boxinterp`; 16 clear/dust cases at 1/4/12/40 layers, positive and zero top pressure | Gas absorption, supplied particle scattering/moments and Rayleigh scattering are connected |
| Composed radiation | `radiative_transfer.column_fluxes` and `make_radiation_component` | Composed original grid, optics and spectral solvers; 32 day/night and clear/dust cases at 4/12 layers, with 170/200 K stratospheres; unchanged interface-flux gates | Connected to Dinosaur; a three-step reverse derivative passes with full radiation |
| Absolute Planck means | `correlated_k.planck_band_intensities` | Original `setspi`; 50--900 K at 0.1 K nodes | Continuous interpolation preserves temperature derivatives; original integer indexing does not |
| Soil/surface frost column | `soil_frost.surface_step` and `subsurface_diffusivity` | Original `progts`, `trdslv`, `albedo_calc`; uniform soil plus case-7 subsurface ice at five northern/southern/equatorial latitudes | Connected to PBL; native TES inertia and GRS ice-boundary maps can supply the interfaces |
| Surface maps | `surface.reference_surface_fields_on_grid` | Four input files match pinned Git blobs; T21/T42 conservative area integrals pass; orientation, longitude seam and northern inertia-cap tests pass | Uses Gaussian cell areas and retains the native cap north of 65 degrees |
| Orbital setup | `reference_constants.reference_radiative_forcing` | Compiled original `mars_calender`, 37 cases over three model years | 668 model sols/year, 88704 s/sol, reference epoch/equinox and unscaled 1356.1 W/m2 solar bands |
| Solar geometry | `solar_geometry.daylight_geometry` and `averaged_zenith` | Compiled original `sol88`/`zenith`, 27 cases including poles, sunrise/sunset and integration windows | Connected to radiation; the averaging interval is explicit |
| Immediate CO2 deposition | `co2_condensation.simple_condensation` | Original `co2_condense`, followed by original `pass2` in the coupled column check | Enabled in the experimental backend; suspended-cloud mode remains separate |
| Rayleigh drag | `rayleigh_drag.rayleigh_tendencies` | Original `rayleigh_damping`; 32 cases with surface, top and tropical switches, masks and timescale units | Connected before PBL; optional dissipative reheating remains off by default, as in the source |
| Suspended CO2 clouds | `co2_microphysics.cloud_step` | Original `co2condsub`, `sedimco2`, `fallvel` and `dtridgl`; 288 cases at 2/4/12/40 layers, at 610 Pa and 500 mbar | Optional column and Dinosaur mode; formation, size, settling, deposition and 15-band cloud optical feedback are connected. A three-step rollout derivative passes finite differences |
| Particle optical properties | `particle_optics` | Original `opt_dst`, `calc_opts`, `opt_co2cld`, `opt_cld_blk`, `opt_cld`; 96 cases, both band configurations, including prescribed fixed-radius dust and 2/12/40 layers | Fourteen size-dependent tables and the brightness correction retain pinned input blobs and source REAL*4 precision; dust populations, lognormal distributions, coated ice and bulk ice/liquid switching are implemented |
| Extended gas radiation | `correlated_k`, `spectral_flux` | Original `tpindex15`, `boxinterp`, `sfluxv`, `sfluxi` and transfer solvers; 40 lookup/spectral cases | Both released 15-band options are connected: 32 Gaussian points to 350 K, or 16 points to 800 K. All ten water mixtures are available; source mixture selection remains discrete |
| Collision-induced absorption | `collision_absorption` | Original `optci15` and `kinter`; 81 cases at 610 Pa, 500 mbar and 10 bar | CO2/CO2 and CO2/H2 infrared continuum is connected, including the source temperature clamp before density/path calculation and zero ghost-layer absorption |
| Prescribed dust | `prescribed_dust`, `dust_scenario` | Original `dust_distrib_fixed2`, `dust_distrib_pulse2`, `season_table` and `ini_optdst`; 144 profile cases, fixed-size optics, 10 FMS map comparisons | Both Conrath formulas, tracer-height/profile modes, fractional/elevated pulses, reference-pressure scaling and size optics are connected to the radiation callback. All six pinned scenario files load; map sampling follows FMS bilinear weights on T21/T42/T85/T170, including polar rows and the seam |
| Bulk water cloud kernels | `water_bulk` | Original `blkh2o_driver`, `findq`, `sedimh2o` and `fallvel`; 24 roots and 36 columns | Phase change, precipitation threshold, settling and rain/snow accounting are implemented. The original root iteration is checked separately; three Newton corrections remove its stopping noise. Surface exchange, lagged PBL exchange history, cloud optical feedback and global water inventory checks are connected |
| Moment water nucleation/growth | `water_moments` | Original `nuclea`, `growthrate2`, `fshape2`, `scrit_tdep`, `nucleacond`; 175 kinetic, coupled and native-substep cases | One or two physical dust modes are checked in the kernel. The one-mode Dinosaur path connects surface exchange, convection, subcycling, latent heat and coated-cloud optics. Material, restarts and a three-step reverse derivative are checked |
| Dust/cloud moment settling | `particle_sedimentation` | Original `init_sedim`, `sedim`, `fallvel`, `dtridgl`; 432 columns with 2/4/12/40 levels, 610 Pa or 500 mbar, explicit/implicit transport and core scavenging on/off | Reuses the checked continuity solver. Dust and core enter the surface dust reservoir; ice enters surface water. Particle-number loss and the source lifting-override budget imbalance are exposed |
| Interactive dust lifting | `dust_lifting` | Unchanged `flux_stress` and contiguous interactive source block; 250 cases | All five stress formulas, three dust-devil options, PBL/fixed-layer injection and reservoir-dependent thresholds are checked. One-mode global coupling uses the current PBL, source drag coefficient and held radiation-time reference opacity |
| Dust coagulation | `dust_coagulation` | Original `coagul_main`, collision kernels, table setup and interpolation; 152 cases with four columns each, table or `FULLCOAG`, one or two physical modes | Brownian, gravitational and diffusion-enhanced collisions are implemented. Equal sizes have exact zero differential settling. Source interpolation quirks, number reconstruction, small mass regularization and inactive-column mass loss remain visible |
| Radiation holding/stratosphere | `dinosaur_physics`, `radiative_transfer` | Source first-call/modulo-day schedule, cold/warm native stratosphere fluxes and two focused schedule/gradient tests | Optional `radiation_update_seconds` requires matching initial cache fields. K/s rates stay fixed when pressure changes. Stratosphere temperature and held radiation survive restarts |
| Surface water exchange | `water_surface` | Original `update_water`, `update_water_fv3`, `update_water_blk`; 216 cases | Bulk/bin/moment surface kernels, liquid saturation and latent diagnostics are checked. Bulk mode is connected before PBL using saved exchange coefficients; cloud formation is a separate switch |
| Surface water optics | `water_surface.optical_properties` | Original `albedo_calc`; 120 threshold/temperature/snow cases | Radiation uses old water; soil uses water after surface exchange. Bare map values are restored when frost falls below the threshold |
| Legacy soil materials | `soil_frost.subsurface_properties` | Original `progts`; 20 modern/legacy and uniform/case-7 cases | Returns diffusivity and heat capacity. Distinct legacy northern/southern ice inertia and the fixed deep node are retained |
| Terrain gravity-wave drag | `terrain_drag` | Original `palmer_drag` and `IR_damping`; 90 columns and 40 damping pressures | Connected before convection. Calm wind and zero shear use finite limits. Original early exits and force imbalance are retained and reported. Zero-top grids require an explicit finite diagnostic top pressure for drag heights |
| Non-LTE heating | `nonlte_cooling` | Original `simple_nltecool` and `interpnlte`; 54 columns and interpolation-history cases | Pinned MGITM atmosphere is staged. Solar attenuation is connected; optional longwave cooling requires explicit CO2/atomic-oxygen tracers. The longwave option is excluded by the original RELEASE flag. Heating corrections are separate from interface fluxes |
| Water state/restarts | `dinosaur_physics`, restart format 4 | Independent global water inventories, full cloud-feedback restart and three-step reverse derivatives | Surface water and lagged PBL state persist across restarts. Versions 2 and 3 remain readable; trace-water thermodynamics follows the source approximation |
| Coupled gas column | `column_physics.column_physics` | Original Rayleigh, PBL, surface, CO2 and convection routines in reference order; 24 clear/winter/thaw/convective cases at 4/12/40 layers, with immediate deposition or suspended clouds; temperature errors below 2e-13 K | Radiation heating and downward surface flux are prescribed inputs |
| Dinosaur adaptation | `dinosaur_physics.build_step` and `remap_gas` | Cold global smoke, gas inventory, remap energy/momentum/passive material checks, and three-step reverse derivatives checked against finite differences | Ground impulse is recorded; raw/corrected angular momentum and exchange-energy ledgers expose numerical repair changes; whole-model validation remains pending |

The optional `moment_water=True` path uses six physical moments: dust mass and
number, cloud ice mass and number, embedded dust mass, and water vapor mass.
The four particle tracers bypass gas-only PBL mixing. Convection mixes all six.
Cloud formation follows the source `subeffmode=0` order. Other physics adds its
positive tracer changes across the microsteps; negative changes act at once.
The default `microtimestep=-1` gives one step, as in the source. Set
`moment_cloud_options={"microtimestep": 60., "latent_heat": True}` to enable
subcycling and latent heating. Moment settling follows cloud formation;
`moment_sedimentation=False` disables it. Dust surface mass starts at the source
30 kg/m2 default and persists in restarts.

The moment kernel retains the source's second weighting of multiple dust modes,
its small-core particle-count history, and its distinct 2.8343 and 2.8 MJ/kg
latent constants. Empty bins get zero weight and dry vapor uses the direct
saturation pressure. These finite limits replace source divisions by zero.
The embryo geometry uses equivalent factored terms to avoid cancellation.
Its native check uses unchanged binary128 routines and also records original
FP64 errors. Contact case 1 can activate below saturation and produce NaNs;
those cases are recorded as source undefined, with matching classifications.
The shared saturation root removes the native 1e-8 stopping noise and uses an
implicit derivative. Coupled native checks record a separate bound for that
change. No matched-climate claim follows from these short tests.

Set `dust_lifting_options={}` to enable one-mode interactive lifting. Use
`dust_optical_bands=15` with either 15-band radiation mode. The default is the
original 12-band configuration. `dust_opacity_channel=1` selects visible opacity;
2 selects infrared absorption. These diagnostics refresh with radiation and
persist across restarts. The source scales them by 610 Pa divided by the current
bottom full-level pressure before it applies the lifting cutoff.

Set `coagulation_options={}` to enable coagulation after lifting and before
cloud formation. `direct=True` within this mapping selects the source
`FULLCOAG` kernels. The default uses its temperature/path or pressure tables.
The source swaps the middle interpolation corners and uses pressure index 15
when pressure exceeds its 25-point table. Both rules are retained. The source
also removes an unresolved distribution below its particle-number threshold.
Its floor repair can select a different branch after a one-ulp arithmetic
change. Native checks report that exact floor bound separately from active
collision checks. The cumulative `dust_coagulation_mass_change_kg_m2` field
records the resulting mass change. It is not hidden as surface deposition.
The source turbulent-inertia kernel has an uninitialized dissipation rate;
there is no defined rate to port for that switch.

Reverse derivatives are checked within the source gas-mixture bins. Crossing a
bin can change the forward radiation result abruptly. A dust perturbation in
the initial global fixture crossed three such cells on step three. The updated
derivative fixture stays within a bin and checks two finite-difference sizes.

The convection sweep mixes heat, winds and tracers. It makes one bottom-to-top
pass, as the reference does. It does not replace this pass with a global neutral
column projection. Its JAX scan carries only the adjacent changing layer.

The PBL port retains the reference's fixed exponent 0.25684, 150 m asymptotic
mixing length, diffusivity minimum 0.001 m2/s, stability functions, implicit solve,
ice roughness and hard-coded disabled internal water sublimation. The outer Ames
interface resets the shear history each call. The low-level coefficient routine
can accept history, but the matched column driver resets it. Atmospheric ice
must not be silently treated as one of the gas tracers mixed by this driver.

The radiation path formerly interpolated directly from eleven pressure points.
Ames first uses four-point cubic interpolation of log(k) to prepare 51 pressure
points, then performs its runtime lookup. The corrected asset retains the raw
source slices and adds all ten mixtures on the refined grid. Staging all mixtures
does not implement a water cycle.

The new backend keeps Dinosaur dynamics, spectral transforms and JAX arrays.
It applies the coupled column after the dynamical step. It then remaps the
changed layer masses onto Dinosaur's sigma grid. The remap conserves donor gas
enthalpy plus kinetic energy, horizontal momentum and passive tracer material.
Suspended mode also carries cloud momentum and kinetic energy through the remap.
The shared pressure repair now preserves all tracer material when it changes
the gas carrier mass. This also fixes existing callers that carry passive gases.

The ground momentum reservoir receives the opposite of the gas-column impulse
from PBL and mass transfer. A rotating-wind test checks gas/ground axial momentum;
the diagnostic path separately records raw and pressure-corrected angular momentum.
These checks do not establish a conservation theorem for arbitrary global climates.
The exchange-energy ledger counts gas enthalpy/kinetic energy, evolving soil-node
sensible energy and frost enthalpy cp*T-L. It records raw and corrected values.
The fixed deep node is an external reservoir. The reference defaults omit some
drag/convection dissipative reheating; those losses must not be hidden as repairs.
The backend requires an explicit radiation callback and soil diffusivity. The
initial cold smoke uses the existing radiation callback; the multi-step gradient
test also exercises the new full-radiation callback. Nonzero suspended CO2 clouds are
rejected in immediate-deposition mode. Set `suspended_co2_clouds=True` in
`build_step` to use the source bulk cloud scheme. Its cloud tracer must be present
in the initial state; `initial_state` supplies it. Ice bypasses gas PBL mixing
and joins all tracers in dry convection. Cloud mass survives gas pressure changes.
The default radiation remains explicitly clear unless its particle provider
supplies cloud optics; enabling microphysics alone does not enable cloud opacity.
Use `make_particle_provider(bands=15, co2_cloud_tracer=CO2_CLOUD_TRACER)` with
`make_radiation_component(..., radiation_mode='extended')` for the released
high-pressure gas tables and CO2 cloud feedback. `radiation_mode='15band'`
selects the separate 32-point, 350 K tables. The default remains `12band`.

The bulk cloud source uses pre-phase density and interface temperatures for
settling, a 590200 J/kg latent heat, mass-derived lognormal radii and a
concentration-dependent explicit/implicit transport switch. The port preserves
these details, including the unusual below-cell fall-speed stencil. Its ground
momentum accounting excludes gas-to-cloud phase transfer and includes deposited
particle momentum. The local unresolved momentum source is reported separately
in `co2_cloud.unresolved_momentum_source_kg_m_s`; it is not assigned to the ground.
The source omits cloud thermal inertia and two-way momentum
feedback. The energy ledger includes cloud latent/kinetic energy, and angular
momentum includes suspended mass; neither ledger claims that the source
approximations conserve total energy or momentum. Native comparisons and gradient
checks are separate from long global validation.

Native surface input files are sampled with conservative area overlaps. Latitude
cells integrate Dinosaur's Gaussian weights; longitude wraps at the seam. The
source TES emissivity map includes some values above one, which the source model
retains. The loader preserves and reports that input behavior. Physical bounds
must be considered explicitly when defining a new calibrated surface dataset.

Radiation is evaluated at each Dinosaur step. The source's averaging interval is
explicit, including its midnight phase and daylight fraction. Matched performance
runs must also match the radiation evaluation schedule; Ames normally holds
radiative tendencies between its configured radiation updates.

The IR layer solver uses stable exponential remainders and boundary adding.
It retains the source hemispheric equations, linear Planck profile and exponent
cap. This avoids subtracting large terms proportional to `1/tau` in thin layers.
The original FP64 composed solver loses about 5e-7 W/m2 in one recorded case.
The binary128 native comparison retains the same strict interface-flux gates;
FP64 cancellation is recorded separately. Thin-layer formal-integral tests and
forward/reverse derivative checks pass without enlarging their tolerances.

The released 800 K IR input contains 2,941 positive infinite coefficients in
some water-mixture cells. The loader retains them. Native comparison checks
matching nonfinite results separately from finite coefficient errors. These
input regions cannot be called validated finite climate states. Gas-only and
tested cloudy rollouts use finite cells; water-cycle coupling must enforce and
audit its actual thermodynamic states. No replacement coefficients are invented.

Bulk water uses the original bracketed Newton/bisection iteration, including
its unusual derivative and absolute `1e-8` step tolerance. Three corrections
then solve the same saturation equation to roundoff. The smooth objective and
implicit derivative are checked against an independent Brent solve and actual
forward finite differences. The original iteration still matches Fortran;
whole-driver differences due to its stopping tolerance are recorded explicitly.

The source CO2 mass guard is retained and reported. It can break the latent heat
ledger in extreme states because the original routine limits removed mass while
retaining the full temperature adjustment. The backend rejects such states;
the recorded matched column cases do not activate this guard.

Gradients are piecewise derivatives of the retained physical switches. They are
not smooth at frost onset, branch boundaries, remap donor changes or table knots.
Finite-difference checks use points away from these boundaries. Continuous Planck
interpolation is the explicit change that replaces the source's integer indexing.

The infrared solver uses one exponent cap for each whole layer, including
samples inside very thick layers. Capping each sample segment on its own changes
the reference solution. Nine analytic thick-layer cases and the expanded native
cold-stratosphere cases check this limit without changing flux tolerances.

The MY24/MY30/MY31/MGS dust files end at Ls 359.74185 instead of 360 degrees.
The native lookup reads outside its array after that last sample. Staging keeps
all original samples and appends the first sample at 360 degrees to close the
periodic boundary. The manifest records this extension; native agreement is
claimed only where the original lookup is defined.

## Required remaining work

| Area | Remaining implementation and acceptance check |
| --- | --- |
| Global budgets | Momentum exchange and repair-stage ledgers are connected. Extend the energy audit across dissipation, fixed-deep-soil exchange, phase approximations and long global runs; select gas and particle tracers explicitly. |
| Radiation schedule | Optional integer-second source schedule now holds heating in K/s and surface fluxes, carries the 170 K initial stratosphere and its source update, and persists caches in restarts. Match the configured interval and callback averaging window in whole-model comparisons. |
| Dust and cloud opacity | Size-dependent dust and all three cloud optical kernels are implemented. CO2 optical feedback is connected; prescribed dust, moment clouds and interactive lifting pass three-step global gradient checks. Lifting and coagulation kernels are checked. Background/dual opacity control, global multimode coupling and matched climatology remain. |
| Soil and frost | Mapped TES/GRS inputs and default case-7 profiles are implemented and checked. Compare diurnal/seasonal response in global runs; modern and legacy material profiles now pass native checks. |
| CO2 modes | Immediate deposition and optional bulk suspended-cloud phase/settling match native routines and coupled columns. Cloud optical feedback passes a three-step gradient check. Full two-way exchange auditing and long global validation remain pending. The older conservative finite-settling scheme remains a separate model. |
| Interactive dust | Lifting, advection, sedimentation, surface reservoirs, coagulation and material budgets. |
| Water | Bulk and moment phase/growth/settling paths, surface exchange, state/restarts and optical feedback are connected. The moment kernel checks two physical dust modes; global coupling currently uses one. Previous-effective-state mode, bin cloud microphysics and long climate validation remain pending. |
| Drag | Rayleigh and Palmer terrain drag are implemented and checked. Polar water and terrain maps now pass pinned-blob and T21/T42 area checks. Complete non-orographic drag, and run long momentum/energy audits. They are off in the default Ames present-Mars setup. |
| Extended atmosphere | Both 15-band options and CIA are implemented and checked. Non-LTE cooling is implemented with the pinned reference atmosphere. Gas-mixture mass/thermodynamics and source infinite-table regions still need auditing or implementation. |
| Other release interfaces | Audit chemistry, bin microphysics and tagging against the release's real and null implementations. Do not label a stub as working physics. |
| Whole-model validation | Matched inputs, switches, grids, resolution studies, spin-up, conservation, held-out observations and native-output comparisons. Dinosaur and FV3 require climate/statistical comparison as well as kernel tests. |
| Performance | Matched physical workloads and precision; record hardware, threads, compilation, transfers, steady-state timings and derivatives separately. |

## Reproduce the native checks

Install a Fortran compiler and the normal locked JAX environment. Supply the
unchanged Ames source and its radiation input data. The scripts check the Git
blob identity before compiling the original routines. Small module declarations
and C adapters replace unused FMS infrastructure; the routine bodies stay intact.
The compiler uses double precision, as the Ames GNU build template does.

```sh
env PYTHONPATH=package:. python scripts/validate_dry_convection.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/convection.json
env PYTHONPATH=package:. python scripts/validate_boundary_layer.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/pbl.json
env PYTHONPATH=package:. python scripts/validate_correlated_k_tables.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/radiation-tables.json
env PYTHONPATH=package:. python scripts/validate_scattering_transfer.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/two-stream.json
env PYTHONPATH=package:. python scripts/validate_spectral_flux.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/spectral.json
env PYTHONPATH=package:. python scripts/validate_optical_depth.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/optics.json
env PYTHONPATH=package:. python scripts/validate_radiative_transfer.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/radiative-driver.json
env PYTHONPATH=package:. python scripts/validate_solar_geometry.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/solar-geometry.json
env PYTHONPATH=package:. python scripts/validate_surface_maps.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/surface-maps.json
env PYTHONPATH=package:. python scripts/validate_soil_frost.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/surface.json
env PYTHONPATH=package:. python scripts/validate_rayleigh_drag.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/rayleigh-drag.json
env PYTHONPATH=package:. python scripts/validate_co2_microphysics.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/co2-clouds.json
env PYTHONPATH=package:. python scripts/validate_column_physics.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/column.json
env PYTHONPATH=package:. python scripts/validate_particle_optics.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/particle-optics.json
env PYTHONPATH=package:. python scripts/validate_extended_radiation.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/extended-radiation.json
env PYTHONPATH=package:. python scripts/validate_collision_absorption.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/collision-absorption.json
env PYTHONPATH=package:. python scripts/validate_bulk_water.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/bulk-water.json
env PYTHONPATH=package:. python scripts/validate_water_moments.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/water-moments.json
env PYTHONPATH=package:. python scripts/validate_particle_sedimentation.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/particle-sedimentation.json
env PYTHONPATH=package:. python scripts/validate_dust_lifting.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/dust-lifting.json
env PYTHONPATH=package:. python scripts/validate_dust_coagulation.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/dust-coagulation.json
env PYTHONPATH=package:. python scripts/validate_water_surface.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/water-surface.json
env PYTHONPATH=package:. python scripts/validate_terrain_drag.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/terrain-drag.json
env PYTHONPATH=package:. python scripts/validate_nonlte_cooling.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/nonlte-cooling.json
env PYTHONPATH=package:. python scripts/validate_prescribed_dust.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/prescribed-dust.json
env PYTHONPATH=package:. python scripts/validate_dust_scenario.py \
  --ames-source /path/to/AmesGCM --fms-bilinear-source /path/to/horiz_interp_bilinear.inc \
  --output outputs/physics-parity/dust-scenario.json
```

Small recorded reports are in `validation/physics-parity/`. Local CPU timing is
kernel evidence only. It does not establish GPU or whole-model performance.

The broad Python regression run passed 758 tests with 4 skips, including the
slow group. Later water, terrain, soil and non-LTE work has additional focused
checks. The refreshed broad run passed 787 tests with 4 skips and 11 slow
tests deselected. Its one callback-test failure was fixed in the test inputs
and passed a focused rerun. Final Dinosaur, map-runner and non-LTE checks
passed all 44 tests. Terrain drag passed five further focused checks. The later radiation schedule,
stratosphere, restart and backend run passed 50 focused tests, and 17 infrared
solver tests passed. Package-directory test collection also passes. Recorded reference reports carry source and
asset hashes. Kernel checks do not establish held-out climate accuracy or a
whole-model speedup. Those experiments remain required.

NASA-derived algorithms remain subject to the [NASA Open Source Agreement
1.3](../iclr-results/06-co2-physics/ames-reference/NOSA.pdf). The source notices
and pinned reference must accompany these ports. The solar geometry derived from
the FMS patch uses that patch's Apache License 2.0, as its file header states.
