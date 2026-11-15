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
| Dry convection | `ames_convection.pass2` | Compiled original `pass2`; four 40-layer cases with 2,048 columns, temperature errors below 2e-13 K | `adjust_state` and `build_step` support Dinosaur; production runner selection remains pending |
| PBL surface exchange | `ames_pbl.surface_exchange` | Compiled original `bndcond`, stable/unstable/zero-wind cases | Used by the new column driver |
| PBL turbulent mixing | `ames_pbl.eddy_coefficients` | Compiled original `eddycoef`, 4/12/40 layers | Used by the new column driver |
| PBL gas-column driver | `ames_pbl.column_step` | Compiled original `pbl_driver`, `newpbl`, matrix, scaling and solve routines; temperature errors below 1e-13 K | Connected to the soil/frost column and experimental Dinosaur adapter; tracer categories remain pending |
| Radiation pressure-table staging | `stage_ames_co2_radiation.ames_pressure_interpolation` | Original `laginterp` and `lagrange`; all ten gas mixtures match exactly in the recorded run | Enabled in the existing correlated-k kernel |
| Radiation dry-table lookup | `ames_radiation._interp2_log_table` on the refined pressure grid | Original `tpindex15` and `boxinterp`; 100 temperature/pressure points, including values outside the tables | Enabled in the existing correlated-k kernel |
| Two-stream flux kernels | `ames_twostream.visible_fluxes` and `infrared_fluxes` | Original `gfluxv`, `gfluxi`, `dsolver`, `dtridgl`; 1/4/12/40-layer thin/moderate/thick columns | Full optical-depth/subgrid driver integration remains pending |
| Radiation pressure/temperature grid | `ames_radiation_grid.radiation_grid` | Original `fillpt`; all output arrays agree exactly in the recorded 1/4/12/40-layer cases, with positive and zero top pressure | Supports Dinosaur's zero top without adding ghost mass or a pressure floor |
| Spectral flux integration | `ames_spectral.visible_spectrum` and `infrared_spectrum` | Original `sfluxv`, `sfluxi`, `getdetau` and flux solvers; 27 cases with weak/mixed/strong gas opacity, including both surface-emission scaling settings; flux errors below 5e-12 W/m2 | Requires supplied optical depths, scattering and gas-column opacities; optical aggregation remains pending |
| Absolute Planck means | `ames_radiation.planck_band_intensities` | Original `setspi`; 50--900 K at 0.1 K nodes | Continuous interpolation preserves temperature derivatives; original integer indexing does not |
| Soil/surface frost column | `ames_surface.surface_step` | Original `progts`, `trdslv`, `albedo_calc`; seven warming/cooling/phase cases at 4 and 16 soil nodes | Connected to PBL in the experimental backend; mapped soil properties remain pending |
| Immediate CO2 deposition | `ames_co2.simple_condensation` | Original `co2_condense`, followed by original `pass2` in the coupled column check | Enabled in the experimental backend; suspended-cloud mode remains separate |
| Coupled gas column | `ames_column.column_physics` | Original PBL, surface, CO2 and convection routines in reference order; clear/winter/thaw/convective cases at 4/12/40 layers; temperature errors below 2e-13 K | Radiation heating and downward surface flux are prescribed inputs |
| Dinosaur adaptation | `ames_backend.build_step` and `remap_gas` | Cold global smoke, gas inventory, remap energy/momentum/passive material checks, and three-step reverse derivatives checked against finite differences | Experimental; full radiation and global climate validation remain pending |

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

These column and remap checks do not establish a closed global energy or angular
momentum budget. The ground momentum reservoir still needs the matching PBL and
mass-transfer exchanges. Spectral projection also needs a separate budget check.
The backend requires an explicit radiation callback and soil diffusivity. The
global smoke test supplies the existing radiation callback only to test coupling;
it does not verify native Ames radiation. Nonzero suspended CO2 clouds are
rejected with a nonfinite result in this immediate-deposition mode.

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
| Constants and calendar | `AMES_BODY_3D` explicitly provides R=189.02, cp=735.94, g=3.72, radius=3386000 m, omega=7.0882e-5/s and model sol=88704 s. Match orbital/calendar settings; keep the old framework defaults for existing runs. |
| PBL/surface coupling | Column coupling and a three-step Dinosaur gradient check pass. Validate global surface/air energy and ground momentum exchanges; select gas and particle tracers explicitly. |
| Radiation transfer | The subgrid, flux solvers and spectral integration are verified. Complete native optical-depth aggregation and connect the resulting radiation driver to Dinosaur. |
| Thermal emission | Absolute Planck means are connected to spectral integration. Complete the full driver; retain explicit documentation of continuous interpolation replacing the integer temperature lookup. |
| Dust and cloud opacity | Match prescribed dust, Rayleigh scattering, CO2/water cloud optics, radii, particle distributions and radiation feedback. |
| Soil and frost | Connect the verified Ames grid and ground/frost solve to mapped surface/subsurface-ice properties, then compare diurnal and seasonal response. |
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
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_convection.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/convection.json
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_pbl.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/pbl.json
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_radiation_tables.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/radiation-tables.json
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_twostream.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/two-stream.json
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_spectral.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/spectral.json
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_surface.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/surface.json
rtk proxy env PYTHONPATH=package:. python scripts/validate_ames_column.py \
  --ames-source /path/to/AmesGCM --output outputs/ames-parity/column.json
```

Small recorded reports are in `validation/ames-parity/`. Local CPU timing is
kernel evidence only. It does not establish GPU or whole-model performance.

Local validation for this checkpoint: the full Python suite passed 658 tests
with 4 skips, including its slow tests. After the later spectral integration and
zero-top additions, the affected radiation tests passed 18 tests from `package/`
without a repository-root PYTHONPATH. This also checks the import path used by CI.
The seven native reports have been checked against the current source and asset
hashes. These are implementation checks, not held-out climate-accuracy results.

NASA-derived algorithms remain subject to the [NASA Open Source Agreement
1.3](../iclr-results/06-co2-physics/ames-reference/NOSA.pdf). The source notices
and pinned reference must accompany these ports.
