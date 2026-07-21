# Prototype: the terraforming ODE on the dinosaur substrate

**Question this answers.** Can dinosaur be the *backbone* of the engine — i.e.
can our terraforming physics live inside dinosaur's ODE/coordinate/time-stepping
machinery rather than beside it — before we commit to porting the whole engine
off torch?

**Status: GO.** The existing 0-D coupled ODE runs as a first-class dinosaur ODE,
integrated by dinosaur's own stepper, differentiably and batched, with exact
parity to the torch kernel.

## What was built

`package/src/gcm3d/terraforming_ode.py` (torch-free) re-expresses
`Mars.compute_derivatives` — the `dy/dt` for `y = [T, P, M_ice]` — as a
`dinosaur.time_integration.ImplicitExplicitODE`. The 0-D system is non-stiff, so
the implicit side is empty (`implicit_terms = 0`, `implicit_inverse = identity`)
and dinosaur's `imex_rk_sil3` degenerates to its explicit Runge-Kutta tableau.
It is stepped by the **same** stepper and the **same** `src.gcm3d.integrate`
scan the 3-D dry dynamics use. Orbital forcing and the polar reservoirs are held
frozen at an epoch (`ZeroDForcing`); the full port would instead carry `sim_time`
in state and advance the orbit inside `explicit_terms`.

## Results (regression tests: `tests/gcm3d/test_terraforming_ode.py`)

| Property | Result | Why it matters |
|---|---|---|
| Tendency parity vs torch | `max|Δ| = 1.7e-18` (float64) | The physics port is exact, not an approximation. |
| Rollout vs torch RK4 (500×100 s) | agree to `1.6e-8` rel | dinosaur's scheme integrates our physics correctly (differs only at truncation order — a scheme choice). |
| Differentiable through rollout | `jax.grad` matches finite-diff to `1e-5` | Inverse design / world-model training gradients flow through the substrate. |
| Batched via `vmap` + `jit` | 256–1000 trajectories, all finite | The 10^5–10^6 trajectory regime the ICLR work needs — JAX's strength. |

Notably, autodiff gave the correct gradient where a naive finite difference lost
it entirely to catastrophic cancellation (`M_ice ~ 1e16` swamping the loss) — a
concrete point in favour of the differentiable substrate over the current setup.

## What this de-risks (and what it does not)

De-risked: the substrate mapping itself. Our physics *does* fit dinosaur's ODE
abstraction cleanly; there is no impedance mismatch, and the three properties the
engine port must preserve (parity, differentiability, batching) all hold.

Still open for the full port (deliberately out of scope here):
- Carry `sim_time` in state and advance the orbit inside `explicit_terms`
  (drop the frozen-forcing assumption).
- Port `compute_fast_physics` (the FAST relaxation path) and the intervention
  hooks / `TimeController` orchestration.
- Prove `vmap`+`jit` batched throughput meets or beats the torch
  `BatchedController` before retiring the torch path.
- Decide whether 0-D becomes a global-mean *configuration* of the same
  spatial engine (recommended) or a separate cheap tier.

The isolation contract is preserved: `terraforming_ode.py` imports only
JAX/dinosaur; the torch comparison lives in the test (the experiment layer), and
the module is not wired into `src.gcm3d.__init__`, so the torch-only CI is
unaffected.
