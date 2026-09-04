# Prototype: the terraforming ODE on the dinosaur substrate

**Question this answers.** Can dinosaur be the *backbone* of the engine — i.e.
can our terraforming physics live inside dinosaur's ODE/coordinate/time-stepping
machinery rather than beside it — before we commit to porting the whole engine
off torch?

**Status: GO.** The existing 0-D coupled ODE runs as a first-class dinosaur ODE,
integrated by dinosaur's own stepper, differentiably and batched, with exact
parity to the torch kernel.

## What was built

`package/src/celestials/planets/mars/seasonal.py` (torch-free) re-expresses
`Mars.compute_derivatives` — the `dy/dt` for `y = [T, P, M_ice]` — as a
`dinosaur.time_integration.ImplicitExplicitODE`. The 0-D system is non-stiff, so
the implicit side is empty (`implicit_terms = 0`, `implicit_inverse = identity`)
and dinosaur's `imex_rk_sil3` degenerates to its explicit Runge-Kutta tableau.
It is stepped by the same framework integration machinery
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
- ~~Carry `sim_time` in state and advance the orbit inside `explicit_terms`~~
  **Done** — see "Seasonal outputs" below.
- Port `compute_fast_physics` (the FAST relaxation path) and the intervention
  hooks / `TimeController` orchestration.
- Prove `vmap`+`jit` batched throughput meets or beats the torch
  `BatchedController` before retiring the torch path.
- Decide whether 0-D becomes a global-mean *configuration* of the same
  spatial engine (recommended) or a separate cheap tier.

The isolation contract is preserved: `terraforming_ode.py` imports only
JAX/dinosaur; the torch comparison lives in the tests (the experiment layer). The
seasonal API now lives under `src.celestials.planets.mars.seasonal`, so it is
only importable with the `gcm3d` extra and the
torch-only CI is unaffected.

## Seasonal outputs (integrated: orbit advances inside the ODE)

The frozen-epoch assumption is dropped. The seasonal path carries elapsed time
`t` as a 5th state component (`[T, P, M_north, M_south, t]`, the **two-cap**
layout main uses) and rebuilds the orbital forcing from `t` every step — a
line-for-line port of `BatchedController.advance_orbit` + the two-cap
`compute_derivatives`:

| Public symbol (`src.celestials.planets.mars.seasonal`) | Role |
|---|---|
| `SeasonalForcing` | Pure-Python constants **+ Keplerian orbital elements** (no frozen flux/caps). |
| `seasonal_tendency` / `seasonal_ode` | Two-cap kernel with orbit derived from `t`; parity-tested vs torch at arbitrary phase. |
| `run_seasonal` | `jax.lax.scan` rollout, samples every N steps. |
| `SeasonalTrajectory` | Ls-indexed diagnostics + `write_csv` (`sol, ls_deg, temperature_k, pressure_pa, ice_*_kg, solar_flux_wm2`). |
| `solar_longitude` / `solar_flux` | Derive `Ls(t)` and inverse-square flux `S(t)`. |

A one-Mars-year rollout sweeps `Ls` 0→360° with `S` peaking at perihelion
(~713 W m⁻²) and troughing at aphelion (~490 W m⁻²), `T` in 172–277 K, and a
seasonal pressure drift from cap exchange + escape — the CSV columns match the
existing Mars seasonal exports so the same Ls-on-x plotting works unchanged.
Sample: `outputs/gcm3d_seasonal/mars_ls_evolution.csv`.

**Accuracy guard.** The explicit step must resolve the diurnal energy balance
(`h = 2π t / rotation_period`). Near ~1 step/rotation the `T⁴` relaxation aliases
and diverges, so `run_seasonal` **raises** if `dt > rotation_period/8` rather
than emit silently-wrong output; `dt ≤ rotation_period/40` (~2200 s for Mars) is
converged.

**MOLA topography: not required here, and not added.** A 0-D global-mean column
has no horizontal grid, so it has no orography — MOLA elevation cannot enter the
seasonal energy/mass budget. It only matters to the **3-D** dycore's
surface-pressure field, which still uses `flat_orography`; wiring real MOLA there
is a separate 3-D task, out of scope for accurate 0-D Ls outputs.
