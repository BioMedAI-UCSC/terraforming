# ICLR 2027 Workplan — discrete tasks with exact code targets

Companion to [iclr2027-direction.md](iclr2027-direction.md). Tasks are ordered;
each has a definition of done (DoD). Line numbers as of branch `v2Perihelion`
(commit 4742727).

---

## Phase 0 — Differentiability audit results (already done, July 1)

Traced every path from an injection schedule to end-of-run temperature.
The good news: state updates are tensor *reassignments*, not in-place ops, so
the autograd graph survives `TimeController.run()`. The graph is cut in
exactly four places, all in the intervention path:

| # | Location | Problem |
|---|----------|---------|
| B1 | `mars.py:459-462` (`inject`) | `float(kg) * float(g.item()) / float(A.item())` → schedule (the design variable!) is cast to Python float; fresh leaf tensor created. Gradient dead-end. |
| B2 | `controller.py:124` | `self._schedule = {k: float(v) …}` — schedule stored as Python floats before it ever reaches `inject`. |
| B3 | `mars.py:502` (`_recompute_greenhouse_factor`) | `if float(dF.item()) <= 0.0: return` — CPU sync + data-dependent control flow. |
| B4 | `forcing.py:184` (`update_greenhouse_factor`) | same `float(delta_F.item())` pattern. |

Discontinuities that don't cut the graph but make it useless for optimization:

| # | Location | Problem |
|---|----------|---------|
| D1 | `mars.py:308-317` and `mars.py:403-412` | Hard ice-exhaustion gate `torch.where((ice <= 0) & …)` — zero gradient once a cap empties; the CO2-collapse regime (the scientifically interesting one) is invisible to the optimizer. |
| D2 | `mars.py:504` / `forcing.py:200-202` | `GHF.clamp(min=1.0)` — dead gradient when forcing is small; acceptable, document it. |

Memory hazard (not a correctness bug):

| # | Location | Problem |
|---|----------|---------|
| M1 | `time_controller.py:230-243` (`run`) | One Mars year at dt=3600 s = 16,488 steps; keeping the graph across a 50-year rollout (~800k steps) will OOM. Needs gradient checkpointing + sparse snapshotting. |

---

## Phase 1 — Make the intervention path differentiable (~1.5 weeks)

### T1. Tensor-native `inject()`
**Where:** `package/src/celestials/planets/mars.py:446-467`
**Change:** accept `dict[str, float | torch.Tensor]`; compute
`dP = kg * g / A` in tensor arithmetic (no `.item()`, no `torch.tensor(float(...))`).
Coerce float inputs with `torch.as_tensor(kg, dtype=TF_DTYPE, device=self._device)`
so the public API is unchanged for CLI users.
**DoD:** `kg.requires_grad → mars.thermal.greenhouse_factor.requires_grad` is True
after `inject()`. Test in `package/tests/mars/basic/` per testing rule.

### T2. Branch-free greenhouse resync
**Where:** `mars.py:494-505` and `forcing.py:184-202`
**Change:** delete the `float(dF.item()) <= 0` early-returns; compute
`GHF_new = GHF_base * (1 + torch.relu(dF) / F_in_base) ** 0.25` unconditionally
(relu reproduces the old "skip when ≤ 0" semantics differentiably and removes
the GPU→CPU sync from the hot loop).
**DoD:** existing forcing tests pass; new test asserts `d(GHF)/d(kg)` is finite
and positive.

### T3. Tensor schedule in `InterventionController`
**Where:** `package/src/interventions/controller.py:111-137` (`__init__`) and
`controller.py:165-172` (`run` step 1).
**Change:** stop coercing to `float(v)`; hold the schedule as given. Also make
`_cumulative_injected_kg` accumulate tensors. Accept two schedule forms:
constant `dict[str, Tensor]` (current behavior) and per-year
`Tensor[n_years, n_compounds]` + compound-name list (needed for demo 1's
time-varying schedules).
**DoD:** `InterventionController(mars, {"SF6": t}, …)` with
`t.requires_grad_()` yields a final temperature with `grad_fn`.

### T4. New module — differentiable schedule parameterization
**Where (new):** `package/src/interventions/schedule.py`
**Content:** `InjectionSchedule` dataclass: raw parameter tensor
`theta [n_years, n_compounds]`, `rates() -> softplus(theta)` (enforces
non-negativity smoothly — do NOT clamp, clamp kills the gradient at 0),
`total_mass()`, `to(device)`. This is the object the optimizer owns.
**DoD:** unit tests in `package/tests/interventions/test_schedule.py`.

### T5. Smooth the ice-exhaustion gate (opt-in)
**Where:** `mars.py:305-318` (`compute_derivatives`) and `mars.py:400-417`
(`compute_fast_physics`).
**Change:** add `Mars(smooth_gates: bool = False)`; when True replace the hard
`torch.where((ice <= 0) & (dM < 0), 0, dM)` with
`dM * torch.sigmoid(ice / ice_ref)` applied only to the sublimation
(negative) branch, `ice_ref ≈ 1e12 kg` (tunable). Keep the hard gate as
default so existing tests/CLI are untouched.
**DoD:** with `smooth_gates=True`, gradient of final T w.r.t. schedule is
nonzero even in runs where a cap empties; invariance test: both modes agree
to <0.5% when ice ≫ ice_ref. Run `/math-review` on the new gate equation.

### T6. Gradient regression tests + memory management
**Where (new):** `package/tests/engine/test_gradients.py`;
**modify:** `time_controller.py:196-247` (`run`).
**Change to `run`:** add `snapshot_every: int = 1` (record every Nth step —
16k Snapshots/year each holding graph references is the OOM source) and use
`torch.utils.checkpoint.checkpoint` around year-long segments in
`InterventionController.run` (`controller.py:174-177`) behind a
`checkpoint_years: bool = False` flag.
**DoD:** (a) `torch.autograd.gradcheck` (float64, small n_steps) passes on a
3-sol rollout; (b) 20-year FAST-mode rollout with grad fits in <8 GB GPU;
(c) finite-difference vs autograd agreement test on d(T_final)/d(SF6 kg/yr).

---

## Phase 2 — Demo 1: gradient-based intervention design (~2 weeks)

### T7. Objectives module
**Where (new):** `package/src/objectives/__init__.py`, `objectives/climate.py`
**Content:** differentiable losses over a rollout: `target_temperature(history,
T_target=273.15)` (softplus of shortfall, evaluated at annual minima — the
collapse-prone season), `total_injected_mass(schedule)`,
`pressure_floor_penalty`. Composable: `loss = temp + λ·mass`.

### T8. Optimization driver + baselines
**Where (new):** `experiments/demo1_gradient_design/` (top level, next to
`package/`): `optimize.py` (Adam on `InjectionSchedule.theta`, cosine LR,
~200 rollouts), `baselines.py` (CMA-ES via `cma` pip package + random search,
same rollout budget), `config.yaml`, `figures.py`.
**Setup:** 50 Mars-year horizon, FAST accuracy, dt=6 h, compounds
{SF6, CF4, C3F8}; per-year rates are the 150 design variables.
**DoD:** figure: loss vs. #rollouts for {Adam-through-sim, CMA-ES, random};
figure: optimized schedule + resulting T(t), P(t), ice(t). This is the
paper's Figure 2.

### T9. Batched candidate evaluation (supports T8 baselines + paper scaling
section)
**Where:** `package/src/engine/batched_controller.py:53-268` (`BatchedMars`).
**Change:** `BatchedMars` currently stacks planet state but has no intervention
support — add batched `greenhouse_factor` handling and an
`inject_batch(dP: Tensor[B, n_compounds])` so CMA-ES populations and RL
rollouts evaluate as one batch. Benchmark script
`experiments/benchmarks/throughput.py`: planets/sec vs batch size
{1, 8, 64, 512, 4096}, CPU vs GPU, FAST vs ACCURATE → paper's scaling figure.

---

## Phase 3 — Demo 2: MarsGym RL benchmark (~2 weeks)

### T10. Gym environment
**Where (new):** `package/src/envs/mars_env.py` (+ `envs/__init__.py`)
**Content:** `gymnasium.Env` wrapping `InterventionController`:
observation = [T_mean, T_min, T_max, P, M_ice, GHF, ΔF, year/horizon];
action = per-compound injection rates (Box, kg/yr, log-scaled);
step = 1 Mars year; reward = −(objectives from T7); termination on
atmospheric-collapse (P below floor) — the irreversibility hook.
Vectorized variant backed by T9's `BatchedMars` for GPU rollouts.
**DoD:** passes `gymnasium.utils.env_checker`; PPO (CleanRL or SB3) trains;
comparison figure: PPO/SAC final performance & sample count vs the
gradient planner from demo 1.

## Phase 4 — Demo 3 + validation (~2 weeks)

### T11. Learned closure (hybrid neural-physics)
**Where (new):** `package/src/framework/closure.py` + training script in
`experiments/demo3_closure/`.
**Content:** small MLP `f_θ(T, P, M_ice, Ls, F_solar) → correction to dy/dt`,
added inside a `Mars` subclass's `compute_derivatives` (`mars.py:232` is the
override point — architecture already separates physics from integration, so
this is a ~40-line subclass). Train θ by backprop **through the RK4
integrator** so FAST+closure matches ACCURATE trajectories (teacher =
`Accuracy.ACCURATE` runs; later, Mars Climate Database annual cycles).
**DoD:** closure reduces FAST-vs-ACCURATE trajectory RMSE by >5× at equal
wall-clock; ablation: offline (per-step) vs through-integrator training.

### T12. Observational validation section
**Where (new):** `experiments/validation/viking.py`
**Content:** simulate 1 Mars year at Viking Lander 1 & 2 latitudes
(22.5°N — note default `latitude=22.0` in `mars.py:110` is already VL1 — and
48°N); overlay observed annual pressure curve (the 610→900 Pa seasonal CO2
cycle) and diurnal temperature range. Fit/report discrepancies honestly.
**DoD:** validation figure + a table of model-vs-observed seasonal extrema.

## Phase 5 — Paper (Sept 1 →)

- T13. Writing: intro/related work (JAX MD, Brax, DiffTaichi/PhiFlow,
  gradient-based climate tuning), method = Phases 1 sections, results =
  demos 1–3 + scaling + validation.
- T14. Repo release prep: README, license, `pip install`, reproduce scripts.
- T15. Submit by the CfP deadline (confirm dates at iclr.cc — expected
  ~Sept 16–24, 2026).

---

## Known cleanup (not on critical path)

- `interventions/state.py` (`GHGState`) and `forcing.py:35-101`
  (`compute_concentration_ppb`, `delta_F_total`) are the legacy mass-based
  path; `controller.py` docstring says composition is the single source of
  truth. Either delete the legacy path or mark deprecated before release —
  reviewers will read this code.
- `interventions/magneticfield.py` is a 5-line stub — either implement as a
  second intervention type (escape-rate modifier on `mars.py:224
  _ESCAPE_RATE`) for demo breadth, or exclude from the release.
