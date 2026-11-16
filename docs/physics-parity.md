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
| Vertical pressure grids | `vertical_grid` | Unchanged `set_eta` and released header; 168 cases across all 14 grids | Reference-column sigma adaptation, saved geometry, map continuation and reverse gradients; fixed-pressure terrain interfaces and full-level discretization differ from FV3 |
| Explicit gas-mass feedback | `dinosaur_physics.remap_gas` and `column_physics` | Unchanged gas-update loops; 108 condensation/sublimation cases | Optional CO2/argon state, initial-CO2-weighted surface exchange, independent species budgets, restart and reverse derivative |
| Dry convection | `dry_convection.pass2` | Compiled original `pass2`; four 40-layer cases with 2,048 columns, temperature errors below 2e-13 K | `adjust_state` and `build_step` support Dinosaur; explicit reference backend selection is available in the map runner |
| Legacy dry convection | `legacy_convection.adjust` | Unchanged `legacy_convect` and `potemp1`; 40 strict full-field cases and 8 neutral temperature/wind cases | Optional `convection_scheme="legacy"`; default `"fv3"` retains pass2 and `"none"` disables adjustment. Column budgets, global restart continuation and a three-step gradient pass. Exact neutral nonuniform tracer mixing has a floating-point branch limit |
| Saturation pressure for moist adjustment | `saturation_pressure` | Unchanged FMS table generators, quadratic lookups and `compute_qs`; 36 cases with 1,815 points each | Both Smithsonian and simple tables, three water/ice phase choices, native humidity caps and explicit temperature derivatives. NASA extended bounds and water constants are retained; nine unit/gradient checks pass |
| Moist convective adjustment | `moist_convection.adjust` | Unchanged GFDL `moist_conv`, `CONV_DETR` and FMS saturation kernels; 80 column cases | Connected after dry convection and before lifting/cloud microphysics. Five kernel checks, independent latent/material budgets, bulk/moment global water inventories, exact restarts and a three-step reverse derivative pass |
| PBL surface exchange | `boundary_layer.surface_exchange` | Compiled original `bndcond`, stable/unstable/zero-wind cases | Used by the new column driver |
| PBL turbulent mixing | `boundary_layer.eddy_coefficients` | Compiled original `eddycoef`, 4/12/40 layers | Used by the new column driver |
| PBL gas-column driver | `boundary_layer.column_step` | Compiled original `pbl_driver`, `newpbl`, matrix, scaling and solve routines; temperature errors below 1e-13 K | Connected to soil/frost and Dinosaur; gas species mix separately from cloud and moment particle tracers |
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
| Moment water nucleation/growth | `water_moments` | Original `nuclea`, `growthrate2`, `fshape2`, `scrit_tdep`, `nucleacond`; 187 kinetic, coupled and native-substep cases | One or two physical dust modes are checked in the kernel. The one-mode Dinosaur path connects surface exchange, convection, subcycling, latent heat, saved effective input and coated-cloud optics. Material, restarts and a three-step reverse derivative are checked |
| Dust/cloud moment settling | `particle_sedimentation` | Original `init_sedim`, `sedim`, `fallvel`, `dtridgl`; 432 columns with 2/4/12/40 levels, 610 Pa or 500 mbar, explicit/implicit transport and core scavenging on/off | Reuses the checked continuity solver. Dust and core enter the surface dust reservoir; ice enters surface water. Particle-number loss and the source lifting-override budget imbalance are exposed |
| Independent moment dust | `dust_sedimentation_step`, `column_physics`, `dinosaur_physics` | 432 unchanged native settling cases for each of one, two and three physical dust modes, with inactive water/core slots zero | Dust transport, lifting, seasonal control, coagulation, settling and optics work without moment-water physics. One/two-mode global material, restart and three-step gradient checks pass; bulk-water coupling keeps a separate fixed dust-gating frost boundary |
| Interactive dust lifting | `dust_lifting` | Unchanged `flux_stress` and contiguous interactive source block; 250 cases | All five stress formulas, three dust-devil options, PBL/fixed-layer injection and reservoir-dependent thresholds are checked. One-mode global coupling uses the current PBL, source drag coefficient and held radiation-time reference opacity |
| Seasonal dust control | `dust_lifting.background_step` | Unchanged background source block on defined branches, plus original size optics; 132 cases | All seven injection scalings, PBL/fixed-layer injection, seasonal cuts, cap/temperature masks, uniform and bottom removal are implemented. Native comparisons cover defined injection/uniform-sink cases. Independent material and gradient checks cover explicit sink repairs; global cloud feedback and restarts pass |
| Combined/local-time dust control | `dust_lifting`, `dust_control` | Unchanged interactive/background blocks and local-time history block; 72 combined cases and 210 local-time cases | Source opacity gates and process order are implemented. Mapped local-time fractions and saved opacity/bin history are connected to Dinosaur and restarts |
| Dust coagulation | `dust_coagulation` | Original `coagul_main`, collision kernels, table setup and interpolation; 152 cases with four columns each, table or `FULLCOAG`, one or two physical modes | Brownian, gravitational and diffusion-enhanced collisions are implemented. Equal sizes have exact zero differential settling. Source interpolation quirks, number reconstruction, small mass regularization and inactive-column mass loss remain visible |
| Radiation holding/stratosphere | `dinosaur_physics`, `radiative_transfer` | Source first-call/modulo-day schedule, cold/warm native stratosphere fluxes and two focused schedule/gradient tests | Optional `radiation_update_seconds` requires matching initial cache fields. K/s rates stay fixed when pressure changes. Stratosphere temperature and held radiation survive restarts |
| Surface water exchange | `water_surface` | Original `update_water`, `update_water_fv3`, `update_water_blk`; 216 cases | Bulk/bin/moment surface kernels, liquid saturation and latent diagnostics are checked. Bulk mode is connected before PBL using saved exchange coefficients; cloud formation is a separate switch |
| Surface water optics | `water_surface.optical_properties` | Original `albedo_calc`; 120 threshold/temperature/snow cases | Radiation uses old water; soil uses water after surface exchange. Bare map values are restored when frost falls below the threshold |
| Legacy soil materials | `soil_frost.subsurface_properties` | Original `progts`; 20 modern/legacy and uniform/case-7 cases | Returns diffusivity and heat capacity. Distinct legacy northern/southern ice inertia and the fixed deep node are retained |
| Terrain gravity-wave drag | `terrain_drag` | Original `palmer_drag` and `IR_damping`; 90 columns and 40 damping pressures | Connected before convection. Calm wind and zero shear use finite limits. Original early exits and force imbalance are retained and reported. Zero-top grids require an explicit finite diagnostic top pressure for drag heights |
| Non-orographic gravity-wave drag | `nonorographic_drag` | Unchanged GFDL `gwfc`, `cg_drag_calc`, timer routines and spectrum initialization at pinned revision `46817a1`; 160 cases | NASA-patched source pressure, spectrum and latitude amplitude. Connected after terrain drag and before convection. Held tendencies, current-wind reheating, exact restart continuation and three-step reverse derivatives are checked |
| Non-LTE heating | `nonlte_cooling` | Original `simple_nltecool` and `interpnlte`; 54 columns and interpolation-history cases | Pinned MGITM atmosphere is staged. Solar attenuation is connected; optional longwave cooling requires explicit CO2/atomic-oxygen tracers. The longwave option is excluded by the original RELEASE flag. Heating corrections are separate from interface fluxes |
| Water state/restarts | `dinosaur_physics`, restart format 4 | Independent global water inventories, full cloud-feedback restart and three-step reverse derivatives | Surface water and lagged PBL state persist across restarts. Versions 2 and 3 remain readable; trace-water thermodynamics follows the source approximation |
| Coupled gas column | `column_physics.column_physics` | Original Rayleigh, PBL, surface, CO2 and convection routines in reference order; 24 clear/winter/thaw/convective cases at 4/12/40 layers, with immediate deposition or suspended clouds; temperature errors below 2e-13 K | Radiation heating and downward surface flux are prescribed inputs |
| Dinosaur adaptation | `dinosaur_physics.build_step` and `remap_gas` | Cold global smoke, gas inventory, remap energy/momentum/passive material checks, and three-step reverse derivatives checked against finite differences | Ground impulse is recorded; raw/corrected angular momentum and exchange-energy ledgers expose numerical repair changes; whole-model validation remains pending |
| Positive spectral tracers | `tracer_transport.project_tracers` | Independent species inventories, grid positivity, physical-source retention, invalid-inventory rejection and reverse derivatives | Dinosaur numerical adaptation, separate from reference FV3 transport. Dynamics retains each species inventory. The post-physics projection retains physical sources and sinks. Per-species correction and redistribution are reported |

The optional `nonorographic_drag_options={"top_pressure_pa": .03}` enables
non-orographic gravity-wave drag. Supply the same options to `initial_state`
and `build_step`. The finite top pressure is used for diagnostic heights only.
The default calculation frequency is zero, as in the released Mars setup.
`update_seconds` and `offset_seconds` use the complete GFDL timer sequence:
decrement the integer alarm, compute or hold forcing, then add the frequency
once when the alarm expires. The released Mars caller omits the timer helper
calls, so its nonzero-frequency behavior is not a parity claim.
Saved tendencies and the integer-valued alarm survive restarts. Keep the offset
unchanged across a restart; changing only the offset requires reinitializing
the timer. Optional `conserve_energy=True` computes heating from current winds,
including on steps that hold the forcing.

The source index uses Dinosaur full levels plus the original top ghost level.
It selects the first reference pressure above 150 Pa and stays above the
bottom model level. This replaces the reference initialization's ambiguous
`size(pref)-1` interface when Mars supplies full-level pressures. Native checks
use explicit source indices. Index 1 has a finite extension that copies input
winds before ghost extrapolation; the original reads an uninitialized wind.
Native comparisons use indices at least 2. Separate IEEE64 spectrum operations
keep the zero phase speed exact; native compilation disables FP contraction.
Top flux dumping retains the source's last-relative-speed diffusion rule.
The port's LGPLv3 and NASA patch Apache-2.0 notices ship with the package.

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

Set `moment_cloud_options={"effective_input_mode": 1}` on initialization and
the step builder to use saved effective inputs. The first call uses the fields
before physics. Later calls use saved temperature after clouds and saved
tracers before clouds, as the source assignments specify. Positive changes
enter over the substeps; negative changes enter at once. History is saved
before settling and sigma remapping and survives restarts. The released source
allocates but never fills `qpi_dmprev` and `qpi_dnprev`. The port fills those
dust histories from the pre-cloud state. Seeded native substeps check the
defined calculations; no agreement is claimed with uninitialized memory.

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

Set `dust_background_options={"scenario": mapped_scenario}` for seasonal dust
control. Supply maps on the Dinosaur grid, in the selected visible or infrared
channel. `load_scenario_on_grid(..., map_scale=3.67)` gives the source visible
scaling; use `map_scale=1.` for infrared. The driver interpolates opacity and
retains the difference between the bounding seasonal maps. `segment=True`
selects the upper map. Fixed-radius injection coefficients come from the pinned
optical tables. `remove_excess=True` enables the optional sink;
`bottom_sink=True` removes material from the ground upward.

The original bottom sink clears a cell before adding its contents to the removed
amount. It can then read below the column. Its surface-deposit calculation also
uses loop indices after those loops have ended. The port counts material before
clearing a cell and uses the local column. It caps removal at available mass and
number, replacing the source's extra 1e-32 allowance for negative tracers.
`negative_sink_repair_kg_m2` records that mass correction. These branches have
independent mass checks; no native agreement is claimed where the source reads
out of bounds.

Combine interactive lifting and background control with
`dust_lifting_options={"dual_mode": 1}` and the background options above.
Modes 1 and 21 suppress both interactive sources only when current opacity
exceeds `dual_scale * scenario_opacity`. Mode 2 allows both sources. All use
the same old diagnostic opacity for lifting and background injection, as in
the reference. Background injection follows interactive lifting.

Optional `local_time_scenario` in the background options contains `seasons`,
`bins_hours` and `fractions`. Fractions have shape `(season, longitude,
latitude, bin)` on the Dinosaur grid. The seasonal axis must match the opacity
axis; this makes the source's shared-index initialization explicit. Hour-bin
edges must run from 0 to 24. Pass these options to `initial_state` as well.
The driver keeps the source longitude/sol clock, strict hour bins, opacity
clamps and update-after-evaluation history. Interactive gates use the target
before local-time changes. The saved history survives restarts so continuation
matches an uninterrupted run. No local-time fraction dataset is included in
the released reference data; supply measured or configured fractions explicitly.

Reverse derivatives are checked within the source gas-mixture bins. Crossing a
bin can change the forward radiation result abruptly. A dust perturbation in
the initial global fixture crossed three such cells on step three. The updated
derivative fixture stays within a bin and checks two finite-difference sizes.
The bulk-water derivative fixture also starts within a bin. At the exact
`1e-6` boundary, the conservative transport correction moved hundreds of cells
between source gas-mixture bins on the second step. Direct radiation-provider
instrumentation confirmed those switches. The replacement `1.1e-6` probe has
no bin changes at any of its checked perturbation sizes. Table selection and
the derivative tolerances are unchanged.

The convection sweep mixes heat, winds and tracers. It makes one bottom-to-top
pass, as the reference does. It does not replace this pass with a global neutral
column projection. Its JAX scan carries only the adjacent changing layer.

The optional legacy sweep runs from top to bottom and retains the source's
inner-loop continuation above stable pairs. Its returned winds are unchanged,
despite the source momentum-mixing flag. Unreturned top-ghost/interfacial
diagnostics are omitted; they do not affect the returned full-level fields.
At exact neutral potential temperature, native `pow` and XLA power roundoff
can select different tracer-mixing branches. Eight neutral cases record tracer
differences explicitly; they have strict temperature/wind gates only. Forty
other cases retain strict gates for every field, including both sides at
1e-8 K potential-temperature offsets. Column diagnostics report enthalpy,
kinetic energy, momentum and species-integral changes for all three switches.

Set `moist_convection_options={}` to enable the released AM4 adjustment.
Initialize bulk or moment water first. The default relative humidity is 1,
the initial tolerance is 0.02 K, and the source uses 30 iterations at each
of 0.02, 0.04 and 0.08 K. `unconverged` reports the source maximum-iteration
warning; output is retained as in the source. Global validation checks that
the default warm test columns finish without this warning. `simple_adjustment`
and `saturation_model` are separate source switches. The latter selects the
Smithsonian tables or the simple exponential tables. Saturation tables use
nearest-node quadratic interpolation, with their supplied temperature
derivatives. Invalid indices return NaNs and a validity flag; FMS raises FATAL.

The port retains the original initial lapse calculation without humidity
scaling, later lapse calculations with scaling, whole-column exit on a failed
block search, and skipped flag refresh at the last iteration of each tolerance.
Cold columns use 2.834 MJ/kg and snow when the pre-adjustment bottom air is
below 273.15 K. Warm columns use 2.5 MJ/kg and rain. The simple adjustment
always uses rain. Optional stratiform cloud fields are absent in the Mars
caller; the original routine returns zero changes for all other tracers.

Moist-convection rain and snow enter the bulk surface reservoir in the source,
even when moment clouds are active. The moment adapter therefore keeps a
separate `moist_bulk_surface_water_kg_m2` bank, includes it in water inventory,
and saves it in restarts. It does not feed active moment frost or surface
exchange. Set `initial_moist_bulk_surface_water_kg_m2` explicitly to match the
reference initial bank; the default is zero. `polar_water_on_grid` supplies
the pinned 500 kg/m2 polar-cap map for both source banks when matching a cold
start. With bulk clouds, precipitation enters the active bulk water bank.

The warm global derivative fixture starts at 250.037 K, inside a Planck-table
interval. At exactly 250.0 K, automatic differentiation uses the right-hand
slope while a centered difference averages the two slopes at the table knot.
Forward- and reverse-mode derivatives agree. Moving the probe inside the
interval removes the finite-difference discrepancy without changing physics,
interpolation or tolerances. Direct Planck derivative tests cover all three
radiation configurations.

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

## Multiple physical dust populations

Set `dust_mode_count` in both initialization and the step builder, or in the map
runner's `reference_options`. The default is one population. With more than one,
initial dust mass, number and surface reservoir inputs can be scalar or have a
last axis with one entry per population. Surface reservoirs keep that last axis
in checkpoints. Extra tracers are `dust_mass_2`, `dust_number_2`,
`dust_core_mass_2`, and so on. They are particles in PBL mixing and retain the
source's mass/number/core order. Radiation providers must select every active
mass/number pair explicitly.

Lifting uses the first population's surface reservoir to set all populations'
thresholds, as in the source. Background `effective_radius_m`,
`extinction_efficiency` and `opacity_fraction` can each have one entry per
population. Fractions that do not sum to one reset to `[1, 0, ...]`, as in source
initialization. Coagulation and cloud formation use all physical populations.
The source transfers nucleated core mass into the primary cloud core; the extra
core-tag update is excluded by its preprocessor. Existing extra cores still mix
and settle with the cloud size. Each population has its own dust size and dust
surface deposit. Total dust includes all dust mass and all core mass.

The unchanged native lifting block passes 250 two-population cases. Native
settling passes 432 cases each for two and three populations. Independent
checks cover separate material reservoirs, size, number loss and derivatives.
The two-population global case combines local-time control, lifting,
coagulation, cloud feedback and saved effective inputs. Its material budget,
exact restart continuation and three-step reverse derivative pass. The map
runner also checks two-population continuation. These short tests do not
establish climate accuracy or long-run transport convergence.

The preceding moist-convection checkpoint passed the full package, CLI and
calibration suite: 953 passed, 4 skipped, including slow tests. Later multimode
changes have focused checks; their wider backend regression is recorded
separately.

The three-step warm moist global reverse pass originally required
27,506,267,400 temporary bytes and exceeded CI memory. Recomputing radiation
and each moist iteration during reverse mode reduces the compiled requirement
to 5,633,589,656 bytes on the local CPU. The same three-step derivative test now
checks an 8 GiB temporary-memory ceiling. It still checks the original equations
against centered differences. Recomputing saves memory and adds reverse-pass
work; this is not a measured forward speedup.

The combined two-population reverse pass initially required 14,049,478,256
temporary bytes. Recomputing each coagulation bin update reduces this to
5,652,882,496 bytes on the local CPU. That test also enforces the 8 GiB ceiling
and retains its two finite-difference probes. Native coagulation equations and
the number of bins are unchanged.

## Run the released present-Mars cloud configuration

The production runner now has an explicit `reference` configuration. It uses
the released `build_run/fms_mars_clouds_v3.2` cloud switches: transported moment
dust, MY30 background dust, moment water clouds with contact parameter 0.95,
30-second cloud substeps without latent reheating, 12-band radiation, and the
top Rayleigh sponge. Radiation refreshes every eight physics steps. The surface
and soil use the four released TES/GRS maps; polar water uses the pinned cap map.
The cold start sets global mean gas pressure to 705 Pa. Terrain pressure
differences stay hydrostatic. This matches the source's dry-mass target rather
than assigning 705 Pa at zero elevation.
Air starts at the released 170 K isothermal default. The surface and all soil
nodes start from the released latitude profile. An unchanged native initializer
matches 184 latitude samples, including the southern branch transition.
The shared map initializer now uses an explicit temperature for both the air
and hydrostatic pressure; it previously changed pressure alone.

Use the released `mars_topo_mola16.nc` for matched comparisons. Its units and
global cell-center axes are checked, then its cell-area integral is retained
on the Gaussian grid. Orientation, the longitude seam and invalid input checks
have independent tests. The original 4-pixels-per-degree MEGDR image remains
available for existing map runs.

The reference runner defaults to the released 56-level grid. All 14 released
grid sizes (24 through 87 levels) are available. The unchanged native `set_eta`
matches 168 pressure-grid cases. Dinosaur freezes each grid's reference-column
pressure fractions, sets the top boundary to zero, and uses arithmetic centers.
Its sigma coordinates cannot retain FV3's fixed-pressure hybrid interfaces
over changing terrain. Saved restarts include the selected boundaries and reject
different vertical geometry.

The cold start filters pressure with the same terrain modes as the dynamical
core. An actual MOLA dry check reduced false wind after five seconds from
0.225396 m/s to 1.30e-12 m/s. Independent high-mode terrain tests cover uniform
and stretched grids. This removes an initialization imbalance.

```sh
env PYTHONPATH=package:. python scripts/run_gcm3d_ablation.py /path/to/AmesGCM/data \
  --config reference --mola-path /path/to/AmesGCM/data/mars_topo_mola16.nc \
  --truncation T42 --layers 56 --dt 231 --sols 668 --chunk-sols 5 \
  --cooldown-seconds 0 --output-dir outputs/reference-climate
```

Use `--resume` with the same source and inputs. The runner rejects changed
physics or input hashes. Each checkpoint stores the full restart, a NetCDF with
all atmospheric levels and physical coordinates, and CO2/water/dust inventories.
The source archive captures the imported package, including installed wheels,
its data assets and license notices. Separate runtime records identify the
devices, JAX/NumPy/Python versions and image package list. Wind-height metadata
uses the selected body's constants and reference temperature.

The two-step production smoke check is finite. Relative CO2, water and dust
inventory changes are below 5e-14. A separate 56-level run completed ten
231-second steps, including radiation cache refresh, with finite outputs;
relative total CO2 drift was below 5e-15. This checks the runner and its bookkeeping;
it does not measure a climate gain. Dinosaur still uses its spectral dynamics
and sigma grid. Whole-model comparison needs a matched native run, resolution
checks and climate statistics.

The corrected initial state and source MOLA16 file completed another ten steps
with finite outputs. The composed native radiation check now includes the
released 56-level grids, with positive and zero top pressure: 48 cases pass.
The top boundary matters. In a clear 170 K night column, the first-cell cooling
rate is about -0.000802 K/s with the native 0.001319 Pa top, and -0.0400 K/s
with the Dinosaur zero top. Lower cells agree closely in that check. The
excluded native top pressure is about 1.87e-6 of 705 Pa. This grid adaptation
must remain explicit in upper-atmosphere and climate comparisons.

The reusable Fortran/MPI runtime and parallel 100-sol preparation/compute
protocol are described in [the comparison experiment](../experiments/reference-comparison/README.md).

## Explicit CO2 and argon mass

Explicit gas-mass feedback is available with `do_qmass=True` and
`initial_co2_gas_ratio` in map `reference_options`. Direct initialization takes
the initial ratio, and `build_step` takes the switch. Initialization assigns the
remaining fraction to the source's argon tracer. Surface phase exchange uses
each layer's initial CO2 mass; atmospheric condensation removes CO2 material
while retaining argon and passive material. The sigma remap then conserves those
updated inventories. Missing required gas tracers raise an error; removing more
CO2 than a layer contains returns invalid output. The source uses fixed CO2 gas
and heat constants and total-pressure condensation even in this mode; the port
retains those approximations.

The source chooses one surface exchange depth for its entire MPI domain using
`ANY(abs(latitude)>45 degrees)`. Dinosaur's global domain therefore uses the
polar depth in this mode. This preserves that domain rule, but cannot reproduce
the spatial patchwork produced by a different FV3 MPI partition. The default
released cloud protocol leaves gas-mass feedback off.

The unchanged native gas-update loops pass 108 cases, including condensation,
sublimation, three pressures and up to 56 levels. Multilevel comparisons check
species inventories after the separate sigma remap; one-level comparisons also
check the normalized ratios. Independent checks cover CO2/argon closure,
initial-layer surface weighting, invalid depletion, global species budgets,
exact restarts and a three-step reverse derivative.

```sh
env PYTHONPATH=package:. python scripts/validate_gas_mass.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/gas-mass.json
env PYTHONPATH=package:. python scripts/validate_vertical_grid.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/vertical-grid.json
```

## Required remaining work

The reference backend projects transported species after dynamics and after
column physics. The dynamics projection restores each previous global species
inventory. The physics projection keeps each current inventory, so injection,
condensation and settling remain physical sources and sinks. Negative spectral
values are removed by a convex blend with the mass-weighted global mean. The
blend stays in the existing spectral space; clipping and transforming nodal
values can otherwise create new negative lobes. Negative total inventory is
invalid and returns NaNs. `diagnostics.tracer_transport` reports raw and target
mass, corrected mass, redistributed mass, and minimum values for both stages.
This projection can smooth sharp tracer fronts. Long runs must measure its
size and check resolution convergence. It does not reproduce FV3's local flux
limiter or establish matched transport accuracy.

| Area | Remaining implementation and acceptance check |
| --- | --- |
| Thermal relaxation | Fixed-profile, evolving-height and analytic Held--Suarez kernels match 36 unchanged native cases, including the original FMS interpolation. Fixed and height forcing connect to the Dinosaur step and command-line runner, replace radiation on every step, and retain profile hashes/timescales in restart identity. NetCDF readers preserve the source equal-level-count bypass and reuse the native-checked bilinear map. Analytic forcing remains an explicit kernel because the released dispatch does not call it. Native reader/composed-profile and long climate comparisons remain. |
| Simple radiation | `simple_radiation` implements the released fixed-dust profile, `ecoeffs`/`scatter_calc`, `swheat`, `lwheat` and five-sol SW ramp. Twenty-four native columns and 24 independent scattering cases pass. The callback, map/restart path and three-step reverse derivative are connected. CLI selection is `--config reference --reference-radiation simple`. The source's unused emissivity argument and surface-only IR suppression are preserved; SPEC_CORE-only modulation is excluded by the release flags. |
| Global budgets | Momentum exchange and repair-stage ledgers are connected. Extend the energy audit across dissipation, fixed-deep-soil exchange, phase approximations and long global runs; select gas and particle tracers explicitly. |
| Radiation schedule | Optional integer-second source schedule now holds heating in K/s and surface fluxes, carries the 170 K initial stratosphere and its source update, and persists caches in restarts. Match the configured interval and callback averaging window in whole-model comparisons. |
| Dust and cloud opacity | Size-dependent dust and all three cloud optical kernels are implemented. CO2 optical feedback is connected; prescribed dust, moment clouds, interactive lifting and seasonal control pass three-step global gradient checks. Multiple physical populations now connect lifting, coagulation, clouds, settling, opacity diagnostics, surface reservoirs and saved effective inputs. Matched climatology remains. |
| Soil and frost | Mapped TES/GRS inputs and default case-7 profiles are implemented and checked. Compare diurnal/seasonal response in global runs; modern and legacy material profiles now pass native checks. |
| CO2 modes | Immediate deposition and optional bulk suspended-cloud phase/settling match native routines and coupled columns. Cloud optical feedback passes a three-step gradient check. Full two-way exchange auditing and long global validation remain pending. The older conservative finite-settling scheme remains a separate model. |
| CO2 switches | Atmospheric condensation, atmospheric loss and gas-mass feedback are independent source controls. Ninety-six unchanged native control-block cases check all combinations and initial-CO2-weighted surface exchange. Column tests retain condensation heating when precipitation loss is disabled, and retain surface snow when mass feedback is disabled. Held suspended clouds still mix as tracers when their phase driver is off. Global tests check disabled-feedback behavior, exact restart continuation and three-step reverse derivatives. |
| Surface/diffusion controls | `do_mars_surface`, `do_vert_diff` and `do_ames_pbl` now select the released active branches. Without Mars surface physics, radiation sees 170 K, albedo 0.25 and emissivity 1; soil/CO2 surface exchange is inactive while drag and convection remain available. Twelve unchanged native `diffuse` cases prove the release fallback leaves tendencies unchanged because `fv3_turb` is excluded. Inactive PBL exchange fields are defined as zero instead of copying uninitialized source values. Surface-dependent water/dust modes are rejected without the source's surface reservoir initialization. |
| Remaining process controls | Audit and implement `freeze_tracer_fields`, `no_forcing` and `gust_fact`. The released tracer-source/sink helper has no caller; alternate turbulence calls are excluded by build flags. These exclusions do not cover the active tracer-freeze and dust-gust branches. |
| Interactive dust | Lifting, advection, sedimentation, surface reservoirs, coagulation and material budgets are connected independently of moment water. Matched climate validation remains. |
| Water | Bulk and moment phase/growth/settling paths, surface exchange, saved effective inputs, state/restarts and optical feedback are connected. Two physical dust modes pass combined global material, restart and three-step derivative checks. Long climate validation remains. Bin cloud microphysics is unavailable in the public release. |
| Drag | Rayleigh, Palmer terrain drag and non-orographic gravity-wave drag are implemented and checked. Polar water and terrain maps pass pinned-blob and T21/T42 area checks. Long momentum/energy audits remain. Gravity-wave schemes are off in the default Ames present-Mars setup. |
| Extended atmosphere | Both 15-band options and CIA are implemented and checked. Non-LTE cooling and explicit CO2/argon gas-mass feedback are implemented. Source mixture thermodynamic approximations and infinite-table regions still need auditing. |
| Other release interfaces | Pinned build/preprocessor audit verifies that bin clouds, photochemistry and legacy binned dust sources use fatal placeholders. Tagging imports are excluded by RELEASE. These equations are unavailable in v3.2.1; they are not implemented physics. |
| Whole-model validation | Matched inputs, switches, grids, resolution studies, spin-up, conservation, held-out observations and native-output comparisons. Dinosaur and FV3 require climate/statistical comparison as well as kernel tests. |
| Performance | Matched physical workloads and precision; record hardware, threads, compilation, transfers, steady-state timings and derivatives separately. |

The release interface audit checks pinned `compile.bash`, `mars_physics.F90`
and `null_physics_mod.F90`, then runs the actual release preprocessor. It extracts
all three build definitions from `compile.bash`. A second preprocessing pass
with `fv3_turb` proves that seven alternate surface-flux and turbulence driver
calls are excluded from the pinned build. Those alternate schemes are not
implemented or validated here. Each of
the five audited placeholder routines calls only `error_mesg(..., FATAL)`.
This result does not supply bin-cloud or chemistry equations. The public
release's real moment/bulk modules are checked separately. Alternate diffusion/radiation
branches also need explicit coverage before claiming every source switch.

Use `--config reference --no-moment-water` for independent moment dust in the
production runner. It keeps the released polar frost map as a fixed boundary
for dust lifting and surface optics. It does not create water vapor, cloud
moments or water exchange history. The persistent dust tracers contain only
mass/number pairs. The settling adapter supplies zero inactive cloud/core slots
locally because the unchanged native routine uses fixed tracer indices.
When bulk water and independent dust are both active, `dust_water_frost_kg_m2`
stores the fixed moment-frost boundary used by dust gates. It is not another
water inventory. NetCDF exports retain this field's map orientation and units.
The real-input 56-level T21 dust-only smoke completed one step, then resumed to
two steps. This checks startup and continuation, not climate stability.

## Reproduce the native checks

Install a Fortran compiler and the normal locked JAX environment. Supply the
unchanged Ames source and its radiation input data. The scripts check the Git
blob identity before compiling the original routines. Small module declarations
and C adapters replace unused FMS infrastructure; the routine bodies stay intact.
The compiler uses double precision, as the Ames GNU build template does.

Thermal relaxation also needs the pinned FMS checkout used by the native build.
Its height interpolation rejects out-of-range columns. The JAX kernel returns
nonfinite values for those columns. It does not extend endpoint temperatures.
Height forcing requires an explicit diagnostic top pressure for the Dinosaur
zero-pressure upper boundary, as in the existing terrain-height adaptation.

```sh
env PYTHONPATH=package:. python scripts/validate_thermal_relaxation.py \
  --ames-source /path/to/AmesGCM --fms-source /path/to/FMS \
  --output outputs/physics-parity/thermal-relaxation.json
env PYTHONPATH=package:. python scripts/validate_simple_radiation.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/simple-radiation.json
```

The production runner accepts `--reference-radiation fixed` or `height` with
`--equilibrium-profile /path/to/profile.nc`. `--thermal-timescale -4` means four
86400-second days, as in the source. A positive value is in seconds; zero
disables relaxation. These choices and the profile content hash are checked
when a run resumes. The spectral default keeps its existing restart metadata.

Simple scattering divides the layer equations by the growing exponential.
This preserves the finite source equations and keeps thick-layer evaluation
finite after the original algebra overflows. The source singularities at
exact conservative scattering or `lambda*zenith=1` are not hidden by clipping.
The optional near-infrared CO2 term is exactly zero at night; its inactive
square root uses a finite reverse-mode branch. The surface-only infrared
suppression does not recompute atmospheric heating, matching the source's
non-conservative diagnostic behavior for that optional switch.

The reference runner also accepts `--no-co2-condensation`,
`--no-atmos-co2-loss` and `--no-co2-mass-feedback`. The defaults keep all three
source switches enabled. Disabled atmospheric loss keeps condensation heating
but removes its precipitation deposit and gas loss. Disabled mass feedback
keeps phase heating and surface snow but resets the whole gas-mass tendency.
Those settings can deliberately break the source's gas-plus-ice inventory.
For these non-conservative modes the Dinosaur wrapper disables its closed-CO2
inventory repair, including pressure rescaling. Otherwise the repair would
silently restore the pressure response that the switch disables. Diagnostics
retain the closed inventory as a reference so the intended source imbalance
remains visible. Numerical drift and positivity without that repair still need
long-run checks. These are source experiment controls, not conservation fixes.
Non-default controls are saved in restart identity.

Use `--no-vertical-diffusion` or `--no-reference-pbl` to disable the released
PBL exchange. The alternative FV3 turbulence drivers are not compiled into
the pinned release. The source leaves several exchange/history values unset
on these inactive paths. The port uses zero heat/stress coefficients and zero
eddy diffusivity, and diagnoses the bottom cell as the inactive PBL top. This
is an explicit deterministic extension of undefined source fields. The soil
solver remains active when Mars surface physics is enabled.

`--no-mars-surface` fixes the radiative boundary at the source's 170 K,
albedo 0.25 and emissivity 1. The command-line preset requires simple or thermal
forcing and disables transported surface dust/water, whose reservoirs the
native no-surface initialization does not allocate. The map API can also use
an explicitly supplied spectral component with valid atmospheric inputs.
Unused soil storage is frozen; CO2 surface/atmospheric phase exchange is skipped.
Rayleigh/gravity-wave drag and convection retain their configured behavior.
The initial fixed boundary and non-default switches are recorded for restarts.

The CO2-control native check compiles the unchanged released routing block.
It seeds phase-driver outputs whose equations have separate native validators.
This isolates switch behavior from already checked phase and PBL kernels; it
does not claim a new whole-native-model comparison.

```sh
env PYTHONPATH=package:. python scripts/validate_co2_controls.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/co2-controls.json
env PYTHONPATH=package:. python scripts/validate_surface_controls.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/surface-controls.json
```

```sh
env PYTHONPATH=package:. python scripts/validate_dry_convection.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/convection.json
env PYTHONPATH=package:. python scripts/validate_legacy_convection.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/legacy-convection.json
env PYTHONPATH=package:. python scripts/validate_saturation_pressure.py \
  --fms-source /path/to/FMS --ames-source /path/to/AmesGCM \
  --output outputs/physics-parity/saturation-pressure.json
env PYTHONPATH=package:. python scripts/validate_moist_convection.py \
  --gfdl-source /path/to/atmos_phys --fms-source /path/to/FMS \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/moist-convection.json
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
env PYTHONPATH=package:. python scripts/validate_particle_sedimentation.py \
  --ames-source /path/to/AmesGCM --dust-modes 2 \
  --output outputs/physics-parity/multimode-sedimentation.json
env PYTHONPATH=package:. python scripts/validate_particle_sedimentation.py \
  --ames-source /path/to/AmesGCM --dust-modes 3 \
  --output outputs/physics-parity/three-mode-sedimentation.json
env PYTHONPATH=package:. python scripts/validate_particle_sedimentation.py \
  --ames-source /path/to/AmesGCM --dust-only \
  --output outputs/physics-parity/independent-dust-sedimentation.json
env PYTHONPATH=package:. python scripts/validate_particle_sedimentation.py \
  --ames-source /path/to/AmesGCM --dust-only --dust-modes 2 \
  --output outputs/physics-parity/independent-dust-2-mode-sedimentation.json
env PYTHONPATH=package:. python scripts/validate_particle_sedimentation.py \
  --ames-source /path/to/AmesGCM --dust-only --dust-modes 3 \
  --output outputs/physics-parity/independent-dust-3-mode-sedimentation.json
env PYTHONPATH=package:. python scripts/validate_dust_lifting.py \
  --ames-source /path/to/AmesGCM --dust-modes 2 \
  --output outputs/physics-parity/multimode-dust-lifting.json
env PYTHONPATH=package:. python scripts/validate_dust_control.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/dust-control.json
env PYTHONPATH=package:. python scripts/validate_nonorographic_drag.py \
  --gfdl-source /path/to/atmos_phys --ames-source /path/to/AmesGCM \
  --output outputs/physics-parity/nonorographic-drag.json
env PYTHONPATH=package:. python scripts/validate_release_interfaces.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/release-interfaces.json
env PYTHONPATH=package:. python scripts/validate_dust_coagulation.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/dust-coagulation.json
env PYTHONPATH=package:. python scripts/validate_dust_background.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/dust-background.json
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
