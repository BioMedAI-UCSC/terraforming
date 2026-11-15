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
| PBL gas-column driver | `ames_pbl.column_step` | Compiled original `pbl_driver`, `newpbl`, matrix, scaling and solve routines; temperature errors below 1e-13 K | Global surface coupling and tracer selection remain pending |
| Radiation pressure-table staging | `stage_ames_co2_radiation.ames_pressure_interpolation` | Original `laginterp` and `lagrange`; all ten gas mixtures match exactly in the recorded run | Enabled in the existing correlated-k kernel |
| Radiation dry-table lookup | `ames_radiation._interp2_log_table` on the refined pressure grid | Original `tpindex15` and `boxinterp`; 100 temperature/pressure points, including values outside the tables | Enabled in the existing correlated-k kernel |

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

## Required remaining work

| Area | Remaining implementation and acceptance check |
| --- | --- |
| Constants and calendar | Explicit matching setup: Ames uses R=189.02, cp=735.94, g=3.72, radius=3386000 m and omega=7.0882e-5/s; existing framework defaults differ. Match orbital/calendar settings and document physical versus model-sol clocks. |
| PBL/surface coupling | Connect the verified gas-column solver to the surface heat solve without duplicating current sensible heat or drag. Validate the coupled surface/air budgets and differentiate through a multi-step rollout. |
| Radiation transfer | Port and compare Ames visible delta-Eddington/quadrature and infrared hemispheric solvers, including reflected surface fluxes, special lower layer and optical-depth aggregation. |
| Thermal emission | Match Ames absolute band Planck integrals and temperature range. Preserve useful temperature derivatives; disclose any continuous interpolation replacing quantized lookup. |
| Dust and cloud opacity | Match prescribed dust, Rayleigh scattering, CO2/water cloud optics, radii, particle distributions and radiation feedback. |
| Soil and frost | Match the Ames thermal grid, ground solve, surface properties and frost-dependent behavior; compare diurnal and seasonal response. |
| CO2 modes | Compare the simple immediate-deposition scheme and optional suspended-cloud microphysics separately. Existing conservative finite-settling code is not an exact port of either. |
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
```

Small recorded reports are in `validation/ames-parity/`. Local CPU timing is
kernel evidence only. It does not establish GPU or whole-model performance.

NASA-derived algorithms remain subject to the [NASA Open Source Agreement
1.3](../iclr-results/06-co2-physics/ames-reference/NOSA.pdf). The source notices
and pinned reference must accompany these ports.
