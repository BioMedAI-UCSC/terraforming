# Neural components and differentiable experiments

The optional JAX GCM stack now supports explicit trainable parameters, sampled
rollouts, a small column MLP, and replaceable radiation. Two generated-data
examples demonstrate physical parameter recovery and neural radiation training.
No MACDA or terrain download is required. These examples do not establish
observational accuracy on Mars or long-term climate stability.

## Install and run

Run from the repository root in your active Python 3.12+ environment:

```bash
rtk proxy python -m pip install -e './package[gcm3d]'
rtk proxy env JAX_PLATFORMS=cpu python examples/neural/recover_parameter.py
rtk proxy env JAX_PLATFORMS=cpu python examples/neural/train_radiation.py
```

The scripts use the active environment's `python`, including mamba environments.
Remove `rtk proxy` if RTK is unavailable. JAX on CPU is sufficient for the default
examples. GPU timings require a compatible GPU-enabled JAX installation; no GPU
allocation is requested by the scripts. They enable float64 explicitly.

For a short pipeline check:

```bash
rtk proxy env JAX_PLATFORMS=cpu python examples/neural/recover_parameter.py \
  --iterations 80 --steps 12 --output outputs/neural_framework/recovery_smoke
rtk proxy env JAX_PLATFORMS=cpu python examples/neural/train_radiation.py \
  --epochs 20 --fine-tune-epochs 5 --columns 24 --layers 2 \
  --steps 4 --long-steps 8 --coupled-steps 2 \
  --output outputs/neural_framework/radiation_smoke
```

Use `--ames` to train against the bundled Ames correlated-k radiation instead of
the compact multiband reference. This changes the reference and model boundary
convention; its checkpoint is not interchangeable with a compact checkpoint.
The default radiation run uses 96 training columns, 4 layers, 100 local updates,
20 trajectory updates, and evaluation at 12 and 48 column steps. Column timesteps
are 30 seconds, with fixed pressure, dust and illumination within a trajectory.
The global deployment check defaults to just two T21 steps of 30 seconds.

Use distinct `--output` directories for different configurations and `--seed`
values. Scripts are deterministic except for the wall-time-matched continuation
budget and hardware timing. Re-running a directory overwrites its reports; these
are examples, not resumable experiment schedulers.

Outputs include `report.json`, `report.md`, and a plot. The radiation example
also writes three inference checkpoints and `predictions.npz`. Its reports retain
failure counts and represent nonfinite metrics as null. Check `coupled_status`
and the individual `failed` fields, not just whether a report file exists.

## Public interfaces

### Parameterized integration

```python
from src.framework.gcm.learning import make_parameterized_step, rollout

# equation_fn(params) constructs the existing IMEX equation with parameters bound
# into a physical coefficient or a learned component.
step = make_parameterized_step(equation_fn, dt_seconds=30.0, specs=specs)
result = rollout(step, params, initial_state, n_steps=12, save_every=3, remat=True)
final_state = result.final_state
saved_states, step_indices = result.trajectory, result.steps
```

`step(params, state)` and `rollout` can be used with `jax.jit`, `jax.grad` and
`jax.vmap`. Resolution and step counts are static. Samples exclude the initial
state and include the final state, including partial last sampling intervals.
With `save_every=None`, no trajectory is stacked. `remat=True` recomputes each
step during reverse-mode differentiation to trade computation for memory.

Build arrays and static configuration outside the differentiated function. Do
not run `run_maps` reporting, dataset conversion, plotting or checkpoint I/O
inside a training objective. The optional radiation hook is also exposed on
`run_maps` for ordinary deployment, but its reporting wrapper is not a training
kernel.

### Model and normalization

`src.framework.neural.ColumnMLP(input_size, output_size, hidden_sizes=(32, 32))`
provides `init(key)` and `apply(params, features)`. Parameters are tuples of
weight/bias pairs; arrays use a feature-last layout. There is no Flax, Haiku or
Optax requirement. Importing conventional physics does not import these models.

`fit_normalization(features, split="train", weights=None)` returns frozen mean
and scale arrays. It rejects other split labels, invalid values and invalid
weights. The caller is responsible for assigning actual data to disjoint splits;
the split label alone cannot establish provenance.

`save_checkpoint` and `load_checkpoint` use atomic, pickle-free NPZ files.
Loading requires matching feature and physics metadata and checks parameter and
normalization shapes. These are inference checkpoints, not optimizer-resume
files. Custom models can use the callable interface and their own serialization.
Float64 checkpoints require JAX float64 enabled; loading rejects silent dtype
conversion.

### Radiation adapter

```python
from src.framework.neural import NeuralRadiation
from src.framework.physics.gcm import forced_primitive_equations

adapter = NeuralRadiation(
    sigma_boundaries=tuple(coords.vertical.boundaries),
    stefan_boltzmann=forcing.stefan_boltzmann,
    compact_shortwave=not forcing.ames_correlated_k_enabled,
)
component = adapter.bind(params, normalization)
equation = forced_primitive_equations(
    coords, body, forcing, specs=specs, radiation_component=component,
)
```

Require `forcing.co2_radiation_enabled=True`. Omitting `radiation_component`
retains the conventional implementation. The hook replaces that radiation call;
it does not add a second heating term. It is also supported by
`forced_co2_primitive_equations`.

A compatible custom radiation callable has the signature:

```python
component(state, coords, specs, body, forcing,
          *, air_temperature_k=None, surface_pressure_pa=None)
```

It returns `RadiativeFluxDiagnostics`. `two_stream_radiative_fluxes` itself
satisfies this contract and can be supplied to check reference substitution.

Alternatively, supply `apply(params, standardized_features)` to `adapter.bind`.
It must return `4*(layers+1)` raw outputs per column. The adapter retains feature
construction, boundary constraints and conservative flux accounting for that
user-defined neural architecture.

Inputs use air temperature `(layer, x, y)` in kelvin and scalar fields
broadcastable to `(x, y)`. The feature-last vector contains air temperatures,
midlayer pressures in Pa, then surface temperature, surface pressure, visible
and longwave dust optical depth, dust-top height in km, incoming solar W/m²,
solar path factor, albedo and emissivity. `feature_schema` specifies exact order.
Sigma interfaces increase from zero at TOA to one at the surface.

Directional flux outputs are nonnegative W/m². Incoming sunlight, zero downward
TOA thermal radiation, reflected surface sunlight, and surface thermal emission
are imposed exactly. All solar fluxes vanish in darkness. The compact reference
has zero upward solar flux above the surface; the compact adapter preserves that
convention. The Ames adapter allows upward solar flux throughout the atmosphere.

`radiative_flux_diagnostics` computes atmospheric flux convergence and surface
and TOA net input from the same interfaces. `heating_rates` divides by physical
layer/surface heat capacities. `radiation_budget_residual` measures the resulting
instantaneous closure. Small closure residuals do not establish good flux
predictions or integrated stability.

### Controlled column laboratory

`src.framework.neural.columns` provides `initial_column`, `make_column_step`,
and `column_energy`. It integrates the same radiation kernel with bulk sensible
exchange using midpoint RK2. This reduced experiment excludes circulation,
regolith and phase change; it is not a replacement atmospheric GCM. An external
energy accumulator allows a discrete budget check against integrated TOA input.

The recovery example estimates a multiplier on a bulk conductance of 2 W/m²/K.
It does not recover the GCM's stability-dependent aerodynamic coefficient.
The separate T21 evaluation in the radiation example uses the actual GCM
stepper and shared radiation hook.

## Interpreting the demonstrations

Parameter recovery compares three initial guesses against bounded scalar search,
checks gradients at three points and reports held-out trajectories. A scalar
derivative-free method can be faster and more accurate; gradients are useful
capabilities, not a guaranteed winning optimizer.

Radiation compares conventional physics, local fitting, trajectory fine-tuning,
and additional local fitting from the same initial checkpoint. Validation chooses
checkpoints, including the starting checkpoint. The continuation control receives
at least as many updates and approximately the fine-tuning loop's warm wall time,
subject to a 10,000-update cap. Compilation is reported separately; this is a
small illustrative budget control, not a hardware-independent compute match.

Training and test columns are independently generated complete profiles. An
explicit extrapolation split has visible dust in [1, 1.5], versus [0, .8] for
training/interpolation. Do not claim general extrapolation from that one split.
Timings are synchronized single warm calls plus first-call costs; repeat and
aggregate them before making performance claims. Neural trajectories are checked
for finite values and temperatures within 50–400 K, not certified climate
stability. Broad multi-seed and long-horizon studies remain research work.

## Verification

```bash
rtk proxy python -m pytest package/tests/gcm3d/test_neural_framework.py -q
rtk proxy python -m pytest package/tests/gcm3d/test_physics.py \
  package/tests/gcm3d/test_ames_radiation.py package/tests/gcm3d/test_maps.py \
  -m 'not slow' -q
```

Tests cover reference substitution, shared kernel parity, boundary conditions,
energy closure, normalization/checkpoint compatibility, parameter recovery,
weight gradients and updates through column rollouts, and a user-defined model's
parameter gradient through the full GCM. They also cover batching, sampled
trajectories and rematerialization equivalence.
