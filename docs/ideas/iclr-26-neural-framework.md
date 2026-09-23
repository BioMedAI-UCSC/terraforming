# Differentiable planetary physics for neural experiments

Date: 2026-09-23  
Design branch: `iclr-26`  
Status: initial framework extensions and two runnable examples implemented;
publication-scale evaluation remains outstanding.

See the [implementation and run guide](../package/gcm3d/neural-experiments.md)
for concrete APIs, commands, constraints and verification. The design below
records the research direction; improvement and speedup remain hypotheses.

## Research direction

Provide reusable neural components and tools for learning within planetary
simulations, demonstrated by two reproducible examples. The intended users are
researchers estimating physical parameters or developing learned atmospheric
parameterizations. They should be able to supply a component, train it, and
evaluate its coupled behavior without editing the dynamical solver.

The model for this presentation is [JAX MD](https://arxiv.org/html/1912.04232v2):
composable simulation primitives, compatible learned components, and applications
demonstrating their usefulness. Its examples include neural interaction potentials
and differentiation through dynamics. This is an architectural analogy, not a
proposal to use molecular-dynamics algorithms for planetary circulation.

A proposed paper claim is:

> We present a differentiable planetary simulation framework that composes a
> spectral dynamical core with atmospheric and surface physics. A common component
> interface supports physical parameter estimation and learned parameterizations,
> enabling training and evaluation of hybrid models within the same simulation
> environment.

This claim requires the implementation and evidence below. Dinosaur supplies the
existing differentiable spectral dynamical core; automatic differentiation itself
is not a new contribution. The proposed additions are the planetary physics
integration, constrained learned components, reusable training interfaces, and
validated applications. An API wrapper alone would be insufficient evidence.

The branch name does not establish a conference submission year or deadline.
This document proposes a new emphasis alongside the earlier
[submission plan](iclr-submission-plan.md), not a claim that its milestones have
been completed.

## Relationship to the temperature-only experiment

The [temperature-only plan](../neural-temp-only-plan.md),
[implementation recap](../neural-temp-only-implementation-recap.md), and
[runbook](../neural-temp-only-runbook.md) describe an offline postprocessor:

```text
physical forecast -> cached features -> PyTorch temperature correction
```

That pipeline does not train through the physical solver or feed its correction
back into the simulation. Even a successful correction would establish a forecast
postprocessing result rather than the usefulness of simulator gradients. The recap
reports local verification; the full real-data campaign and held-out improvement
remain unestablished.

The proposed workflow instead supports:

```text
trainable physics component -> physical timesteps -> trajectory loss
            ^                                         |
            +--------------- gradients ---------------+
```

Keep the existing pipeline as an optional application. Its provenance and evaluation
practices remain useful. Generated reference data will support the first two new
examples, so MACDA downloads are not prerequisites. Those examples can establish
software functionality and reference-model approximation, not improved accuracy
on real Mars observations.

## Existing foundation and missing capabilities

The current framework already provides:

- `ColumnPhysicsState` and `ColumnPhysicsTendencies` for atmospheric dynamics and
  surface reservoirs in `package/src/framework/physics/gcm.py`.
- `column_primitive_equations(base, parameterization)`, which composes a
  state-to-tendencies callable with Dinosaur dynamics.
- `stepper` and `integrate` in `package/src/framework/gcm/dynamics.py`; integration
  uses `jax.lax.scan` and returns a final state.
- Tests of selected gradients through short rollouts and individual physics
  calculations under `package/tests/gcm3d/`.

These are foundations, not proof that every physical configuration can be trained.
In particular, the map-producing `run_maps` workflow includes setup, reporting and
NumPy/xarray conversion that should remain outside differentiated execution.
Existing short gradient tests do not establish long-horizon gradient quality,
optimization convergence, or stable coupled neural physics.

## Required framework changes

### 1. Optional JAX-native neural components

Add an optional `src.framework.neural` package with a small column MLP as the
reference architecture. Use JAX arrays and explicit parameter pytrees so a model
inside the GCM remains in the same differentiation system. Do not build an
automatic PyTorch-to-JAX bridge for the existing temperature model.

Provide training-only feature normalization with frozen statistics. Document
feature names, order, units, vertical ordering, and the fixed layer count for each
model. Checkpoint metadata must record model configuration, feature schema,
normalization, parameter shapes, and relevant physical configuration; reject
incompatible loads. Checkpoint I/O stays outside JIT and gradient execution.

Researchers may supply another JAX-compatible callable; the solver must not depend
on the reference MLP's internal architecture. Adding this capability must not make
neural dependencies necessary for conventional physics runs.

### 2. Trainable physics composition

Extend the existing composition path to bind explicit parameter pytrees to
physical components. Preserve the current conventional-physics default and
existing state and tendency containers. A learned radiation component replaces
the selected radiation contribution; it must not run alongside and double-count
the conventional contribution it approximates.

The adapter owns conversion between dimensional physical inputs and the solver's
nondimensional state, including nodal/modal transformations where needed. The
network should not have to understand the solver's storage conventions.

The following are illustrative proposed interfaces, not current executable APIs:

```python
params = model.init(key, input_spec)
fluxes = model.apply(params, normalized_column_inputs)
step_fn = make_step(model_config, radiation_component)
next_state = step_fn(params, state)
trajectory = rollout(step_fn, params, initial_state, n_steps, save_every)
loss, grads = jax.value_and_grad(objective)(params)
```

Keep configuration such as resolution and array shapes static. Keep trainable
values explicit, without converting traced arrays to Python floats or NumPy.
The parameter-recovery example uses the same rollout machinery with physical
parameters instead of neural weights.

### 3. Radiation-column adapter

The first learned component approximates the existing radiation calculation.
Inputs include atmospheric temperature and pressure profiles, surface temperature,
dust properties, illumination, and the surface properties required by the chosen
reference configuration. Reference and emulator must use the same configuration.

Predict upward and downward shortwave and longwave fluxes at layer interfaces,
ordered from top of atmosphere to surface and expressed in W/m². Derive layer
heating from net flux convergence and layer heat capacity, and derive surface
energy input from the same boundary fluxes. Reuse the physical implementation's
conversion conventions rather than independently predicting atmospheric and
surface temperature changes.

Enforce nonnegative directional flux magnitudes and the applicable boundary
conditions, including zero solar flux in darkness. Shared flux accounting makes
energy exchanges consistent; it does not guarantee physical accuracy or stable
time integration. Record top-of-atmosphere forcing and boundary exchanges so
budget diagnostics do not incorrectly demand constant energy in a forced system.

Keep other physics conventional. Learned mixing coefficients, arbitrary residual
tendencies, and neural control policies are later extensions, not initial scope.

### 4. Parameterized simulation and training utilities

Build reusable short-rollout functions on the existing stepper and scan, rather
than a second simulation engine. Support sampled trajectories, JAX differentiation,
batching across initial states, and optional rematerialization for memory control.
Saving every state should not be mandatory. File I/O, plotting, progress callbacks
with side effects, and dataset conversion remain outside the training kernel.

Provide local flux/heating losses and trajectory losses with explicit area or
layer-mass weighting, as appropriate. Report field errors, gradient norms,
physical budget residuals, and nonfinite or unstable trajectories. Failed runs
must remain visible in evaluation rather than disappearing from aggregate scores.

Place reusable model application, normalization, physical adapters, rollout
operations and diagnostics in the package. Keep dataset generation policies,
split selection, optimization schedules and reporting commands in thin examples
or notebooks. Freeze validation decisions before inspecting test results.

## Two demonstrations of usefulness

### Example 1: parameter estimation through simulation

Generate short reference trajectories with a known surface-exchange multiplier.
Start the inverse problem from different values and optimize through the simulator.
Begin with a controlled column or short low-resolution configuration in which the
parameter measurably affects the observations; establish sensitivity before fitting.

Compare autodiff gradients with central finite differences at several interior
parameter values. Compare recovery against a bounded derivative-free search using
the same observations and parameter bounds. For a single parameter, direct search
is a strong baseline; do not assume gradients will be faster.

Measure parameter recovery, trajectory error on unseen initial conditions,
objective evaluations, wall time, and budget residuals. Include multiple initial
guesses. Hold out complete initial-condition cases rather than adjacent timesteps.
If parameter sensitivity is weak or recovery is nonunique, report that limitation
instead of interpreting low trajectory error as successful parameter identification.

This verifies inverse-modeling utility. It is neither a neural experiment nor an
observational validation of Mars physics.

### Example 2: neural radiation inside the solver

Generate atmospheric-column inputs and reference fluxes using the existing
radiation implementation over a documented physical domain. Fit the column MLP
locally, then insert it through the public physics interface.

Compare three configurations:

1. Conventional reference radiation.
2. Neural radiation trained on local column outputs.
3. The same locally trained model fine-tuned through short trajectories.

Use the same architecture and initial local checkpoint for the two neural
configurations. Document the additional compute and trajectory supervision used
by fine-tuning; include an additional-local-training control with comparable
compute when attributing a benefit specifically to trajectory training.

Hold out full initial-condition cases and declared regimes, such as dust loads or
illumination ranges. Distinguish interpolation from extrapolation. Evaluate both
short and longer rollouts, with a small low-resolution coupled run required before
claiming solver integration. Do not impose a global training campaign initially.

Report local flux and heating errors, coupled temperature and other affected field
errors, energy/mass budget residuals, and failure rates. Measure component and
whole-simulation runtimes separately, including compilation cost and synchronized
steady-state timings on the same hardware and precision. A small radiation model
may be slower than the current physical implementation or yield negligible total
speedup; both are valid outcomes and must be reported.

Trajectory fine-tuning improving held-out rollouts is a hypothesis. Matching a
reference model does not demonstrate greater real-world physical accuracy.

## Evidence and paper structure

Organize the paper around the component interface, verified numerical behavior,
and applications. Show which parts are inherited from Dinosaur and which are new.
Include a runnable example in which changing the supplied radiation component is
sufficient to switch between conventional and learned physics without editing
the solver.

Four principal figures should support the argument:

| Figure | Evidence |
|---|---|
| Architecture | State, component and gradient paths; inherited versus added capabilities |
| Inverse problem | Parameter recovery and held-out error versus evaluations and wall time |
| Neural component | Approximation error versus measured component and total runtime |
| Coupled behavior | Error growth, budget residuals and failure rates versus rollout length |

Report variability across initial conditions and training seeds for final neural
results, not only the best run. State configuration, hardware, precision, data
generation, splits, stopping rules and artifact locations. Synthetic tests alone
support the framework and approximation claims; real-data forecasting requires
a separate observational evaluation.

## Tests, acceptance and implementation sequence

Required implementation checks:

- Conventional physics remains numerically consistent when no learned component
  is supplied; reference-adapter substitution does not double-count radiation.
- Feature shapes, units, layer order, normalization and checkpoint restoration
  are consistent; incompatible model/configuration combinations fail explicitly.
- Radiation flux divergence matches atmospheric and surface energy accounting;
  night and boundary conditions are respected.
- Parameter and neural-weight gradients through short rollouts are finite and
  agree with finite differences on small deterministic cases away from nonsmooth
  transitions. Check actual parameter updates, not merely gradient existence.
- Batched and individual rollouts agree, and rematerialization preserves results
  and gradients within numerical tolerance.
- Held-out cases are not used for fitting statistics or checkpoint selection;
  unstable runs and computational costs are included in reports.

Deliver in this order:

1. Audit gradients for the selected physical configuration and run the parameter
   recovery demonstration with existing low-level primitives.
2. Extract the parameterized rollout and component interfaces into public package
   APIs, with regression checks and a documented inverse-problem example.
3. Implement the reference column MLP, normalization, checkpointing and radiation
   adapter; establish local approximation behavior.
4. Demonstrate coupled insertion, then compare local and trajectory training.
5. Broaden held-out regimes, seeds and rollout lengths for the paper evaluation.

Software acceptance requires reusable APIs and runnable examples with the checks
above. Scientific claims of speedup, improved rollout accuracy or useful
extrapolation require measured evidence and are not guaranteed acceptance targets.

Provisional engineering estimates, not measured promises:

| Milestone | Estimated effort | Compute expectation |
|---|---|---|
| Small parameter-recovery demonstration | 4–8 hours | Minutes to a few hours |
| Reusable interfaces and documented example | 2–4 days | Modest |
| Neural radiation with coupled checks | 3–7 days | Hours for bounded experiments |
| Convincing multi-experiment evaluation | Several weeks | Resolution and horizon dependent |

These milestones overlap and estimates must be revised after profiling and
gradient checks. A few-hour effort should target the first demonstration, not
promise strong neural results or a publication-ready framework.
