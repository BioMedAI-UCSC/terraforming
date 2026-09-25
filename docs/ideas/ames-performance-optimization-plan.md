# Ames performance comparison and optimization plan

## Objective

Produce a reproducible performance comparison between the differentiable JAX
Mars GCM and NASA Ames Mars GCM, while retaining numerical stability and stated
scientific accuracy. The intended final claim is:

> At comparable horizontal and vertical resolution, for a one-Mars-year
> simulation with a defined common set of physical processes and output cadence,
> the JAX simulator runs X times faster than NASA Ames on specified hardware,
> while satisfying the reported numerical-stability and climate-diagnostic
> tolerances.

The value of `X` is an experimental result. Do not choose configurations or
stopping criteria to obtain a predetermined speedup.

## Existing measurements

The current T21, 12-layer, 300 s, float64 JAX configuration completed 668 sols
in approximately 6092 s (1 h 42 min) per case. The separate synchronized warm
benchmark reports approximately 424 simulated sols per wall hour. Compilation,
warm execution and end-to-end execution must remain separate measurements.

The Ames v3.2 default C24 configuration uses:

- 56 vertical layers;
- a 924 s atmospheric/physics timestep;
- four advective substeps, giving a 231 s dynamical timestep;
- a 1848 s radiation calculation interval;
- one vertical remapping per atmospheric timestep.

Thus, the present 300 s JAX step is already comparable to the Ames default
dynamical step. The primary optimization opportunity is multirate physics, not
simply replacing the entire coupled timestep with a much longer step.

## Claim boundaries

Two benchmarks are required because the models do not implement identical
physics.

### Matched-work benchmark

Use the nearest common workload to isolate implementation performance:

- comparable horizontal degrees of freedom;
- matched vertical layer count where practical;
- dry dynamics or a documented common physics subset;
- identical simulated duration;
- identical output cadence;
- matched precision where practical;
- compilation excluded from steady-state timing and reported separately.

T21 has roughly 2048 nodal horizontal points, while Ames C24 has 3456 cubed-sphere
cells. T21/L12 versus C24/L56 is not a matched-resolution comparison. Include a
JAX layer-scaling run, preferably T21/L56, or label the mismatch prominently.

### Representative-model benchmark

Compare the validated full JAX Mars configuration with the Ames default Mars
configuration, each using its normal stable timestep and physics cadence. This
supports a practical throughput claim, but not a claim that both models perform
the same numerical work.

## Benchmark measurements

For every configuration record:

- exact source revision and dirty-tree status;
- input and restart checksums;
- horizontal grid, layers, timestep and physics cadence;
- active physics and disabled physics;
- numeric precision;
- compiler, JAX/XLA version, MPI layout and relevant flags;
- CPU/GPU/TPU model, device count, CPU core count and memory;
- compilation or build/startup time;
- synchronized warm integration time;
- end-to-end time including normal output;
- simulated sols per wall hour and simulated years per day;
- model steps per second;
- column-layer updates per second;
- peak host and accelerator memory;
- checkpoint and output time.

Use a one-sol warm-up, at least three synchronized 10-sol measurements, a 30-sol
stability pilot and a final 668-sol run. Report median, individual repetitions
and dispersion. Do not time queued asynchronous JAX work without a device
synchronization at each timing boundary.

## Phase 1: freeze baselines

- [x] Archive the current T21/L12/dt300/float64 configuration and artifacts.
- [x] Add a performance mode with zero cooldown and explicit checkpoint spacing.
- [ ] Run dry-dynamics and full-physics JAX baselines.
- [ ] Build Ames v3.2.1 with recorded compiler and MPI configuration.
- [ ] Run the Ames C24/L56 default case with matched output cadence.
- [ ] Record native-hardware and, if possible, similar-hourly-cost timings.

The Ames CPU versus JAX GPU comparison is useful if the hardware distinction is
explicit. A cost-normalized comparison should accompany it where possible.

## Phase 2: profile the JAX model

Profile representative 300 s and 600 s executions and attribute synchronized
device time to:

- Dinosaur dry dynamics and the IMEX solver;
- spectral modal/nodal transforms;
- shortwave and longwave radiation;
- surface energy and momentum exchange;
- PBL diffusion and convective adjustment;
- regolith conduction;
- CO2 condensation and sublimation;
- positivity projection and hyperdiffusion;
- diagnostics, host transfers, restart writes and NetCDF output.

Deliver a table with milliseconds per model step, calls per model step,
percentage of integration time and temporary/peak memory. Optimize components
in measured cost order.

## Phase 3: remove measurement overhead

- [x] Set checkpoint cooldown to zero for GPU performance runs.
- [x] Make restart spacing configurable independently of performance mode.
- [ ] Compute frequent scalar diagnostics as device reductions.
- [ ] Transfer only reduced diagnostics during the timed interval.
- [ ] Keep plots and NetCDF creation outside kernel-only timing.
- [ ] Retain a separate end-to-end timing with normal output enabled.
- [ ] Reuse the persistent JAX compilation cache without hiding first-call time.

The current five-second cooldown every ten sols contributes about 335 s over a
668-sol run. Removing it changes measurement overhead, not simulator throughput,
so report both kernel and end-to-end results.

## Phase 4: timestep convergence

Use the current 300 s float64 run as the numerical reference. Test:

| Coupled timestep | Purpose |
|---:|---|
| 300 s | Existing reference |
| 450 s | Conservative increase |
| 600 s | Primary performance candidate |
| 675 s | Near the T21 resolved-diurnal limit |
| 900 s or longer | Daily-mean-insolation experiment only |

Run one sol first, then 30 sols. Compare:

- global mean, minimum and maximum surface temperature;
- mean and spatial surface pressure;
- temperature and zonal-wind latitude/height structure;
- atmospheric plus surface-reservoir CO2 mass;
- total-energy and angular-momentum drift;
- amplitude and phase of the resolved diurnal cycle;
- seasonal phase where the pilot duration permits it;
- finite-state and reservoir-positivity status.

Advance a timestep to the annual run only if it passes predetermined tolerances.
A 600 s timestep should approach a twofold speedup if per-step cost is unchanged.

### Observed 600 s stage-physics failure

The first 30-sol GPU matrix established that T21/L12 with a 600 s timestep and
stage-evaluated full physics is not stable for the requested window. A nonfinite
state was detected at the final callback, before step 4439, corresponding to the
30-sol boundary. Because the original performance run used one 50-sol chunk, this establishes
only that the failure occurred somewhere within the 30-sol integration; it does
not locate the first failing timestep. The 300 s
reference and 450 s stage-evaluated candidate completed, and the 450 s candidate
passed the configured scalar numerical gates.

This is a numerical integration failure: one or more prognostic arrays contain
NaN or infinity. It is not a Matplotlib-cache, SSH, Mamba, or GPU-memory failure.
The geometric moving-terminator bound of about 690 s at T21 is a necessary
sampling limit, not a guarantee that the fully coupled nonlinear radiation,
surface, PBL, convection, regolith and CO2 system remains stable up to that
limit. The empirical stable timestep can be lower.

Do not use 600 or 675 s for an annual performance claim unless a subsequent
configuration completes the pilot and passes the numerical gates. Continue the
matrix because step-held physics and float32 are separate configurations with
different stability behavior. The runner records failed executions in
`failure.json`, marks them red in the plots, and continues with later candidates.
The climate runner now reports the exact pytree state leaves containing nonfinite
values so a repeated failure can be attributed more precisely.

The matrix now uses five-sol checkpoints. Float32 and float64 checkpoint times
are compared on a shared interpolation grid, avoiding false "insufficient
overlapping coverage" failures caused by sub-ulp endpoint differences.

To locate the first failing interval, rerun that configuration without
performance mode, using one-sol checkpoints and no cooldown:

```bash
python scripts/run_gcm3d_ablation.py \
  data/tes/mgs_tes_surface_1deg.nc \
  --output-dir outputs/performance/dt600-stage-diagnostic \
  --config convection --truncation T21 --layers 12 --dt 600 \
  --initial-ls 0 --diurnal --sols 30 --chunk-sols 1 --cooldown-seconds 0 \
  --precision float64 --physics-evaluation stage \
  --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc
```

The runner now accepts the full T21 matrix through 675 s and records precision
and solar longitude in its configuration/diagnostic artifacts. Run a candidate
with, for example:

```bash
rtk proxy .venv/bin/python scripts/run_gcm3d_ablation.py \
  data/tes/mgs_tes_surface_1deg.nc \
  --output-dir outputs/performance/t21-l12-dt600 \
  --config convection --truncation T21 --layers 12 --dt 600 \
  --initial-ls 0 --diurnal --sols 30 --performance-mode \
  --precision float64 \
  --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc
```

Compare it with the frozen reference using the executable numerical gates:

```bash
rtk proxy .venv/bin/python scripts/compare_gcm_performance_runs.py \
  --reference outputs/performance/t21-l12-dt300/convection/checkpoint_diagnostics.csv \
  --candidate outputs/performance/t21-l12-dt600/convection/checkpoint_diagnostics.csv \
  --output outputs/performance/t21-l12-dt600/comparison.json
```

The comparator returns a nonzero exit status when a gate fails. It checks scalar
checkpoint diagnostics and explicitly does not replace three-dimensional field
comparison.

Run the complete pilot matrix with:

```bash
rtk proxy .venv/bin/python scripts/run_gcm_performance_matrix.py \
  data/tes/mgs_tes_surface_1deg.nc \
  --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc \
  --output-dir outputs/performance/timestep-matrix \
  --sols 30 --layers 12 \
  --timesteps 300 450 600 675 \
  --precisions float64 float32 \
  --physics-evaluations stage step
```

The matrix always runs the 300 s float64 stage-evaluated reference first and
evaluates every other run with the numerical gates. A failed candidate is
retained and reported rather than deleted. It writes:

- `matrix.json`: complete machine-readable configuration and results;
- `performance.csv`: flat timing, throughput, speedup and error metrics;
- `performance.png`: presentation-ready four-panel summary;
- `performance.pdf`: vector version of the same figure;
- one `comparison.json` per candidate.

The plots distinguish accepted/reference configurations from failed numerical
gates and show integration time, end-to-end time, simulated sols per wall hour,
speedup, temperature error and pressure error. Integration timing includes JAX
compilation for the invocation; use the existing synchronized warm benchmark
when making a steady-state kernel-throughput claim.

## Phase 5: Ames-style multirate physics

Add cached slow-physics fields and their update counters to the JAX state. Use
fixed `jax.lax.cond` schedules so the rollout remains compiled, pure and
differentiable. Test one cadence change at a time before combining them.

| Process | Initial candidate cadence |
|---|---:|
| Dynamics | 600 s |
| Surface energy | 600 s |
| CO2 mass exchange | 600 s |
| Surface drag | 600-1200 s |
| PBL diffusion | 1200-1800 s |
| Convective adjustment | 1200-1800 s |
| Radiation | 1800 s |
| Prescribed dust interpolation | 1800-3600 s |
| Regolith conduction | 1800 s or an implicit/exact update |

Implementation and validation order:

1. Cache radiation tendencies for three 600 s dynamical steps.
2. Cache prescribed dust forcing or interpolate it at the radiation cadence.
3. Cache PBL and convective tendencies.
4. Cache or replace regolith conduction with a stable implicit/exact update.
5. Combine only the individually accepted schedules.

If held tendencies introduce a measurable diurnal phase bias, test time-centered
or linearly interpolated tendencies. Document that gradients represent the
chosen held-physics approximation.

The first split-cadence level is implemented as
`--physics-evaluation step`. It diagnoses the complete column-physics tendency
once from the step-start state and holds it through the internal IMEX stages,
while dry dynamics continues to be evaluated at every stage. The default
`--physics-evaluation stage` preserves the original reference integration. Both
paths remain pure JAX and differentiable. The timestep matrix tests both modes.
Caching radiation or other slow components across several complete timesteps is
still a separate phase; do not describe step-held physics as an 1800 s radiation
cadence.

## Phase 6: precision study

Benchmark full float64, full float32 and mixed precision. Begin mixed precision
with local radiation and surface calculations in float32 while retaining CO2
mass accounting, global reductions and sensitive spectral state in float64.

Repeat the 30-sol and annual validation for any precision configuration used in
a performance claim. Float32 benefits depend strongly on the GPU: accelerator
model and precision throughput must be reported.

The climate runner implements `--precision float64|float32`; no mixed-precision
kernel has yet passed the validation gates.

## Phase 7: validation gates

Derive final acceptance thresholds from the observed 300-versus-150 s or
300-versus-450 s convergence error where possible. Freeze thresholds before the
final optimized annual result is inspected. Starting candidate gates are:

- no NaNs, infinities, negative reservoirs or atmospheric collapse;
- relative total CO2 drift no greater than `1e-5` over the 30-sol pilot, with
  the reference drift and candidate/reference ratio retained as diagnostics;
- annual global-mean surface-temperature difference below 1 K;
- global-mean pressure difference below 1%;
- seasonal pressure-amplitude difference below 5%;
- seasonal peak timing within 5 degrees of solar longitude;
- zonal-mean wind and temperature errors below declared physical tolerances;
- no material increase in energy or angular-momentum drift.

These are numerical-agreement gates, not observational error bars. Validation
against Ames or observations is a separate scientific comparison.

## Phase 8: final comparison matrix

At minimum run:

| Model | Configuration | Purpose |
|---|---|---|
| Ames | C24/L56 default | Native reference |
| JAX baseline | T21/L12, dt300, float64 | Existing reference |
| JAX optimized | T21/L12, accepted timestep/cadence/precision | Practical speedup |
| JAX layer-matched | T21/L56, accepted settings | Reduce vertical-work mismatch |
| Both dry/common subset | Nearest matched grids/layers | Matched-work comparison |

Present speed and numerical quality together. The performance table must include
hardware, precision, physics cadence, simulated years per day, one-year wall
time and speedup. The companion quality table must include temperature,
pressure, wind, seasonal phase, CO2 drift and stability metrics.

## Claim hierarchy

Use only the strongest claim supported by completed measurements:

1. **Absolute throughput:** the JAX model simulates one Mars year in the measured
   GPU time.
2. **Native-hardware comparison:** the representative JAX configuration is `X`
   times faster in wall time than Ames on the stated GPU-versus-CPU systems.
3. **Matched-work comparison:** at comparable resolution, layers, physics subset
   and output cadence, the JAX implementation is `X` times faster.
4. **Speed with scientific agreement:** the JAX simulator is `X` times faster
   while satisfying the stated numerical gates and matching the selected Ames
   climate diagnostics within reported tolerances.

The fourth claim is the target. Preserve the complete timing records and failed
configurations so timestep, precision and cadence selection remains auditable.

## Expected decision points

- If 600 s fails convergence, retain 300 or 450 s and prioritize multirate
  physics.
- If radiation is not a leading cost, do not implement radiation caching first.
- If float32 violates mass or stability gates, retain float64 or restrict reduced
  precision to validated local kernels.
- If a matched Ames build cannot be executed, publish absolute JAX throughput
  and mark the Ames speedup as unmeasured.
- If T21/L56 eliminates the apparent speed advantage, report the layer-scaling
  result rather than using T21/L12 as a matched-resolution claim.
