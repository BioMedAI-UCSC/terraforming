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
| Dry convection | `dry_convection.pass2` | Compiled original `pass2`; four 40-layer cases with 2,048 columns, temperature errors below 2e-13 K | `adjust_state` and `build_step` support Dinosaur; production runner selection remains pending |
| PBL surface exchange | `boundary_layer.surface_exchange` | Compiled original `bndcond`, stable/unstable/zero-wind cases | Used by the new column driver |
| PBL turbulent mixing | `boundary_layer.eddy_coefficients` | Compiled original `eddycoef`, 4/12/40 layers | Used by the new column driver |
| PBL gas-column driver | `boundary_layer.column_step` | Compiled original `pbl_driver`, `newpbl`, matrix, scaling and solve routines; temperature errors below 1e-13 K | Connected to the soil/frost column and experimental Dinosaur adapter; tracer categories remain pending |
| Radiation pressure-table staging | `stage_co2_radiation_tables.ames_pressure_interpolation` | Original `laginterp` and `lagrange`; all ten gas mixtures match exactly in the recorded run | Enabled in the existing correlated-k kernel |
| Radiation dry-table lookup | `correlated_k._interp2_log_table` on the refined pressure grid | Original `tpindex15` and `boxinterp`; 100 temperature/pressure points, including values outside the tables | Enabled in the existing correlated-k kernel |
| Two-stream flux kernels | `scattering_transfer.visible_fluxes` and `infrared_fluxes` | Original `gfluxv`, `gfluxi`, `dsolver`, `dtridgl`; 1/4/12/40-layer thin/moderate/thick columns | Connected to the full gas-radiation driver; retain the source reciprocal-pivot arithmetic in the rotated solve |
| Radiation pressure/temperature grid | `radiation_grid.radiation_grid` | Original `fillpt`; all output arrays agree exactly in the recorded 1/4/12/40-layer cases, with positive and zero top pressure | Supports Dinosaur's zero top without adding ghost mass or a pressure floor |
| Spectral flux integration | `spectral_flux.visible_spectrum` and `infrared_spectrum` | Original `sfluxv`, `sfluxi`, `getdetau` and flux solvers; 27 cases with weak/mixed/strong gas opacity and both surface-emission scaling settings | Connected to optical aggregation; the special bottom IR sample uses its full, possibly floored, depth |
| Optical-depth aggregation | `optical_depth.gas_subgrid` and `aggregate_optics` | Original `optcv15`, `optci15`, `tpindex15`, `boxinterp`; 16 clear/dust cases at 1/4/12/40 layers, positive and zero top pressure | Gas absorption, supplied particle scattering/moments and Rayleigh scattering are connected |
| Composed radiation | `radiative_transfer.column_fluxes` and `make_radiation_component` | Composed original grid, optics and spectral solvers; 16 day/night and clear/dust cases at 4/12 layers; flux errors below 1e-8 W/m2 | Connected to Dinosaur; a three-step reverse derivative passes with full radiation |
| Absolute Planck means | `correlated_k.planck_band_intensities` | Original `setspi`; 50--900 K at 0.1 K nodes | Continuous interpolation preserves temperature derivatives; original integer indexing does not |
| Soil/surface frost column | `soil_frost.surface_step` and `subsurface_diffusivity` | Original `progts`, `trdslv`, `albedo_calc`; uniform soil plus case-7 subsurface ice at five northern/southern/equatorial latitudes | Connected to PBL; native TES inertia and GRS ice-boundary maps can supply the interfaces |
| Surface maps | `surface.reference_surface_fields_on_grid` | Four input files match pinned Git blobs; T21/T42 conservative area integrals pass; orientation, longitude seam and northern inertia-cap tests pass | Uses Gaussian cell areas and retains the native cap north of 65 degrees |
| Orbital setup | `reference_constants.reference_radiative_forcing` | Compiled original `mars_calender`, 37 cases over three model years | 668 model sols/year, 88704 s/sol, reference epoch/equinox and unscaled 1356.1 W/m2 solar bands |
| Solar geometry | `solar_geometry.daylight_geometry` and `averaged_zenith` | Compiled original `sol88`/`zenith`, 27 cases including poles, sunrise/sunset and integration windows | Connected to radiation; the averaging interval is explicit |
| Immediate CO2 deposition | `co2_condensation.simple_condensation` | Original `co2_condense`, followed by original `pass2` in the coupled column check | Enabled in the experimental backend; suspended-cloud mode remains separate |
| Coupled gas column | `column_physics.column_physics` | Original PBL, surface, CO2 and convection routines in reference order; clear/winter/thaw/convective cases at 4/12/40 layers; temperature errors below 2e-13 K | Radiation heating and downward surface flux are prescribed inputs |
| Dinosaur adaptation | `dinosaur_physics.build_step` and `remap_gas` | Cold global smoke, gas inventory, remap energy/momentum/passive material checks, and three-step reverse derivatives checked against finite differences | Ground impulse is recorded; raw/corrected angular momentum and exchange-energy ledgers expose numerical repair changes; whole-model validation remains pending |

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
rejected with a nonfinite result in this immediate-deposition mode.

Native surface input files are sampled with conservative area overlaps. Latitude
cells integrate Dinosaur's Gaussian weights; longitude wraps at the seam. The
source TES emissivity map includes some values above one, which the source model
retains. The loader preserves and reports that input behavior. Physical bounds
must be considered explicitly when defining a new calibrated surface dataset.

Radiation is evaluated at each Dinosaur step. The source's averaging interval is
explicit, including its midnight phase and daylight fraction. Matched performance
runs must also match the radiation evaluation schedule; Ames normally holds
radiative tendencies between its configured radiation updates.

The source CO2 mass guard is retained and reported. It can break the latent heat
ledger in extreme states because the original routine limits removed mass while
retaining the full temperature adjustment. The backend rejects such states;
the recorded matched column cases do not activate this guard.

Gradients are piecewise derivatives of the retained physical switches. They are
not smooth at frost onset, branch boundaries, remap donor changes or table knots.
Finite-difference checks use points away from these boundaries. Continuous Planck
interpolation is the explicit change that replaces the source's integer indexing.

## Required remaining work

| Area | Remaining implementation and acceptance check |
| --- | --- |
| Global budgets | Momentum exchange and repair-stage ledgers are connected. Extend the energy audit across dissipation, fixed-deep-soil exchange, phase approximations and long global runs; select gas and particle tracers explicitly. |
| Radiation schedule | Gas-radiation assembly and Dinosaur coupling pass native and derivative checks. Match radiation evaluation/holding schedules in whole-model comparisons. |
| Dust and cloud opacity | Match prescribed dust, Rayleigh scattering, CO2/water cloud optics, radii, particle distributions and radiation feedback. |
| Soil and frost | Mapped TES/GRS inputs and default case-7 profiles are implemented and checked. Compare diurnal/seasonal response in global runs; optional legacy material settings remain separate. |
| CO2 modes | The simple immediate-deposition column matches the reference. Port and compare optional suspended-cloud microphysics separately. Existing conservative finite-settling code is not an exact port of either. |
| Interactive dust | Lifting, advection, sedimentation, surface reservoirs, coagulation and material budgets. |
| Water | Vapor/ice/frost transport, bulk and moment cloud options, latent heat, cloud nuclei and sedimentation, each with explicit switches. |
| Drag | Orographic and non-orographic gravity-wave options, their inputs, momentum/energy accounting and source-based tests. They are off in the default Ames present-Mars setup. |
| Extended atmosphere | Optional 15-band/extended gas tables, collision-induced absorption, gas-mixture mass/thermodynamics and non-LTE cooling. |
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
env PYTHONPATH=package:. python scripts/validate_column_physics.py \
  --ames-source /path/to/AmesGCM --output outputs/physics-parity/column.json
```

Small recorded reports are in `validation/physics-parity/`. Local CPU timing is
kernel evidence only. It does not establish GPU or whole-model performance.

Local validation for this checkpoint: the regular Python suite passed 681 tests
with 4 skips and 11 slow tests deselected. The earlier full suite passed 658 tests
with 4 skips before the later physics additions and renames. Focused checks also
run from `package/` without a repository-root PYTHONPATH, as CI does.
The reports are checked against source and asset hashes. These are implementation
checks, not held-out climate-accuracy results.

NASA-derived algorithms remain subject to the [NASA Open Source Agreement
1.3](../iclr-results/06-co2-physics/ames-reference/NOSA.pdf). The source notices
and pinned reference must accompany these ports. The solar geometry derived from
the FMS patch uses that patch's Apache License 2.0, as its file header states.
