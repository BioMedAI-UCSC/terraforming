# Master Task List — World Model for Terraforming (ICLR 2027)

Consolidated from the documents under `docs/ideas/` (July 2026):
[benchmark.md](benchmark.md), [iclr2027-direction.md](iclr2027-direction.md),
[iclr2027-workplan.md](iclr2027-workplan.md),
[differentiable-framework-features.md](differentiable-framework-features.md),
[ml-features-jaxmd-analogs.md](ml-features-jaxmd-analogs.md),
[mars-model-landscape.md](mars-model-landscape.md),
[amesgcm-comparison.md](amesgcm-comparison.md),
[adr-language-framework-choice.md](adr-language-framework-choice.md).

**The story (one paper, two layers):**

> Build a **differentiable, GPU-batched Mars terraforming simulator** (the
> substrate), then train a **physics-informed JEPA-style world model** on it
> (the headline ICLR contribution): an action-conditioned latent dynamics
> model of century-scale planetary engineering, with physics-consistency
> regularizers, evaluated on long-horizon prediction, extrapolation to unseen
> intervention regimes, and planning — against gradients-through-the-real-
> simulator and model-free RL baselines.

**Target:** ICLR 2027 — deadline expected ~Sept 16–24, 2026 (confirm at
iclr.cc). **~10 weeks from July 7.** Slip plan: NeurIPS 2027 D&B / ICML 2027.

**Parked for now:** the Mars Atmospheric Foundation Model track
(arXiv:2605.28851 contributions) — revisit post-submission; the world-model
data/training infrastructure built here transfers directly.

Priorities: **[P0]** paper-critical · **[P1]** credibility-critical ·
**[P2]** strengthens the story · **[P3]** stretch. Cross-references
(T#, A#, B#, ML#…) point at the source docs above.

---

## Why the simulator work stays on the critical path

The world model needs the simulator to be:

- **Differentiable** (Phase 1 / T1–T6): the strongest baseline in the planning
  comparison is gradients through the *real* simulator (BPTT); the hybrid
  latent+physics variant needs a differentiable mechanistic step; and
  "world model vs true differentiable dynamics" is the comparison no other
  world-model paper can run — most learned world models have no ground-truth
  gradient oracle. That comparison *is* the novelty.
- **Batched** (D1): dataset generation at 10⁵–10⁶ trajectories and batched
  world-model evaluation both live on `BatchedMars`.
- **Dynamically rich** (B1–B3): bistability, hysteresis, and irreversible
  collapse are what make long-horizon latent prediction hard and interesting.
  A world model that captures regime changes a per-step regressor misses is a
  result; on the current monostable 3-state model there is nothing hard to
  capture.
- **Spatially non-trivial** (B7 banded EBM): a JEPA-style claim over a
  3-dim state is thin; 10–20 latitude bands give fields worth masking and
  predicting in latent space. B7 moves ONTO the critical path (was
  post-submission) — with an early go/no-go (task 3).

---

## TRACK A — Simulator substrate (weeks 1–3, Jul 7 → Jul 25)

### A-1. Differentiable intervention path [P0] (~1.5 weeks)

1. **T1 — Tensor-native `inject()`** (`mars.py:446-467`): remove
   `float()`/`.item()` casts; DoD: gradient flows from injected kg to
   greenhouse factor.
2. **T2 — Branch-free greenhouse resync** (`mars.py:494-505`,
   `forcing.py:184-202`): replace the `dF ≤ 0` early-return with
   `torch.relu(dF)`; DoD: d(GHF)/d(kg) finite and positive.
3. **Go/no-go decision: B7 banded EBM in the paper?** Decide by ~Jul 18
   after sizing (it's the largest single item, ~1–2 weeks). GO → world model
   v1 is spatial (stronger JEPA claim). NO-GO → paper stands on the 0-D
   state + richer physics (B1–B3), framed as "low-dimensional but bistable,
   irreversible dynamics"; spatial version becomes future work.
4. **T3 — Tensor schedules in `InterventionController`**
   (`controller.py:111-137, 165-172`): accept per-year
   `Tensor[n_years, n_compounds]` schedules; stop coercing to float.
5. **T4 — `InjectionSchedule` module** (new `interventions/schedule.py`):
   raw `theta` + softplus rates (never clamp — clamp kills the gradient
   at 0); the object optimizers and policies own.
6. **T5 — Smooth ice-exhaustion gates (opt-in)** (`mars.py:305-318,
   400-417`): sigmoid fade so the collapse regime has usable gradients
   (hard switches make BPTT explode/vanish — this directly protects the
   BPTT baseline); hard gate stays default; `/math-review` the equation.
7. **T6 + A5 — Gradient regression tests + memory management**
   (new `tests/engine/test_gradients.py`; `time_controller.py:196-247`):
   `gradcheck`, finite-difference agreement, `snapshot_every` sparse
   snapshots, per-year `torch.utils.checkpoint`; permanent CI guard so a
   future `float(x.item())` can't silently kill differentiability.
   DoD: 20-year rollout with grad in <8 GB GPU.

### A-2. Batched engine + physics richness [P0/P1] (~1.5 weeks, overlaps)

8. **T9/D1 — Batched interventions in `BatchedMars`**
   (`batched_controller.py:53-268`): batched greenhouse state +
   `inject_batch(Tensor[B, n_compounds])`; DoD: B=64 batch matches 64
   sequential runs to 1e-12. **Blocking for dataset generation.**
9. **B1 — Pressure-dependent CO2 greenhouse** GHF_CO2(P) (gray-gas,
   calibrated to GHF(610 Pa)≈1.02) — without it the model cannot express a
   runaway or threshold.
10. **B2 — Pressure-dependent CO2 frost point** (Clausius–Clapeyron
    replacing the constant 149 K) — atmospheric-collapse risk as P rises.
11. **B3 — Regolith CO2 adsorption reservoir** (4th state variable,
    Zubrin–McKay) — the hysteresis. B1+B2+B3 land **together** (individually
    inert; jointly they create the bistable landscape the world model must
    learn). Verify the two-equilibria structure; `/math-review` required.
12. **B4 — Liquid-water habitability metric** (soft-AND of T>273 K,
    P>611 Pa) — headline objective for planning and reward.
13. **[If GO at task 3] B7 — N-band latitudinal EBM** (10–20 bands,
    diffusive meridional transport, Budyko–Sellers class): state gains a
    band dimension; per-band insolation reuses the existing declination
    math (`mars.py:363-373`); verify equator–pole contrast within ~15 K of
    MCD zonal means.
14. [P2] **C1 — Composable `Intervention` ABC** + **C3 nanoparticle
    aerosols** (Ansari–Kite 2024): only if it strengthens the action space
    cheaply — more intervention types = richer action-conditioned dynamics.

### A-3. Validation (folds into the paper's fidelity section) [P1]

15. **Turn the existing Ames GCM benchmark into a validation figure** —
    nighttime floor to 0.1 K, diurnal mean to ~4 K vs a 3-D GCM is the
    fidelity claim for the substrate.
16. **Close the 22 K daytime-peak gap**: tune surface heat capacity /
    thermal inertia; sanity-check against REMS via the `gale-crater`
    preset; re-run the benchmark.
17. **T12/F1 — Viking validation**: 1 Mars year at VL1/VL2 vs the observed
    ~610↔900 Pa seasonal pressure cycle; honest residual table.
18. **F2 — Conservation diagnostics** (`engine/diagnostics.py`): CO2/energy
    budgets in CI — these are also the physics-informed regularizers for the
    world model (task 25), so build them as reusable differentiable
    functions, not just asserts.
19. [P2] Remaining benchmark follow-ups (exact-coordinate GCM re-run,
    local-time diagnostic, ≥30-sol GCM run, matched boundary conditions) —
    only if the validation section needs strengthening.

---

## TRACK B — The world model (weeks 3–7, Jul 21 → Aug 22)

### B-1. Groundwork (start in parallel with Track A)

20. **Literature review + positioning memo** (`docs/ideas/worldmodel.md`):
    I-JEPA, V-JEPA/V-JEPA 2, Dreamer-v3, TD-MPC2, DINO-WM, physics-informed
    latent models, learned simulators. Define precisely what
    "physics-informed JEPA for terraforming" means here: an
    **action-conditioned latent predictor trained in representation space
    (EMA target encoder, no reconstruction) on simulator rollouts, with
    physics-consistency regularizers** — and write the related-work
    paragraph distinguishing it from (a) GCM emulators and (b) model-based
    RL world models. Decide the name (working: **TerraJEPA / MarsWM**).
21. **ML10 — training & data infrastructure** (`package/src/ml/`):
    trajectory dataset generators covering the *intervention-reachable*
    state space (not just present-day Mars — include collapse and runaway
    regimes), minimal trainer (grad clipping, checkpointing, seeds, metric
    logs), metrics: horizon-N trajectory RMSE, rollout-stability
    (divergence step count), calibration error, conservation-violation
    rate. Resist importing Lightning.
22. **Dataset generation at scale** (needs task 8): 10⁵–10⁶ trajectories
    from `BatchedMars` — randomized initial states, physical parameters,
    and intervention schedules; stratify so bistable/collapse regimes are
    well represented; store with config hashes.

### B-2. Model versions (the paper's method section)

23. **v0 — action-conditioned latent dynamics model** on the 0-D state:
    encoder → z, predictor p(z′ | z, action, Δt) with 1 step = 1 Mars year,
    JEPA-style representation-space loss + EMA target encoder; baselines =
    direct state-space regression and the per-step closure (ML3 per-step
    regime). DoD: beats per-step regression on horizon-50 rollout RMSE.
24. **[If B7 GO] v1 — spatial world model over the banded EBM**: masked
    latent prediction over the latitude dimension (the actual JEPA
    mechanic); action conditioning on latitude-targeted interventions
    (cap darkening vs equatorial GHG becomes a *spatial* action).
25. **Physics-informed variants (the core contribution — each is an
    ablation row):**
    - (a) **Conservation-aware latents**: penalize violation of the F2
      CO2/energy budgets decoded from z (uses task 18's differentiable
      budget functions).
    - (b) **Hybrid rollout**: learned latent model as a residual/correction
      to the differentiable mechanistic step (the ML3 closure idea, lifted
      into latent space) — uniquely possible because the simulator is
      differentiable.
    - (c) **Invariance/monotonicity priors**: more injection ⇒ warmer,
      frost-point consistency — cheap regularizers, honest ablations.
26. **Long-horizon stability study**: 100-year rollouts; the classic
    learned-simulator failure is slow drift — measure divergence step count
    across all variants; through-integrator/rollout training vs per-step
    (curriculum K = 50 → 500, needs T6 checkpointing).
27. **Extrapolation study (the headline table)**: train on one intervention
    regime, test on held-out regimes (higher injection rates, unseen
    compounds, post-collapse states). Hypothesis: physics-informed variants
    degrade gracefully; pure latent models fail exactly where the
    terraforming question lives. Either outcome is a publishable finding.
28. **Hysteresis/bistability capture test** (needs B1–B3): does the world
    model reproduce the two-equilibria structure and the warm-branch
    lock-in? A per-step regressor provably smears the threshold — show it.
29. [P2] **ML9 — deep ensembles via vmap-over-weights** (needs A6): epistemic
    uncertainty bands on projected trajectories; disagreement flags when a
    planned schedule leaves the training distribution.
30. [P2] **A6 — functional pure-step API** `step(state, params, t, dt)`:
    do opportunistically during Track A refactors — unlocks vmap/jacrev,
    ML9, and is the JAX hedge from the ADR; not blocking for v0.

---

## TRACK C — Planning evaluation (weeks 6–9, Aug 10 → Sept 5)

The paper's synthesis: four method families, one protocol, one table.

31. **T7/E1 — `objectives/` module** [P0]: differentiable losses
    (target-T softplus shortfall at annual minima, liquid-water days,
    total injected mass, collapse penalty) — shared verbatim by every
    method below; if they score differently the comparison is meaningless.
32. **Planning in the world model** [P0]: MPC/CEM (and optionally a
    TD-MPC2-style policy) in latent space; report plan quality *and*
    wall-clock per plan.
33. **Gradients through the real simulator** [P0] (slimmed demo-1 / T8):
    Adam on `InjectionSchedule.theta` through batched rollouts; also
    CMA-ES + random search at matched rollout budgets as black-box
    controls. No longer the headline — now the strongest baseline the
    world model must be judged against.
34. **T10/E3 — MarsGym environment** [P1]: gymnasium wrapper (1 step =
    1 Mars year, termination on collapse — the irreversibility hook),
    vectorized on `BatchedMars`; PPO/SAC as the model-free baseline; the
    env itself is a citable benchmark artifact for the community.
35. **E6 — the 4-way comparison protocol** [P0]
    (`experiments/eval_protocol/`): {latent-world-model planning,
    gradients-through-simulator, BPTT closed-loop policy (ML5, if time),
    model-free RL} on identical objectives, horizons, seeds, and a
    perturbation suite (±30 % escape rate, ±20 % thermal inertia; B9 dust
    shocks if available). Questions the table answers: sample efficiency,
    wall-clock per plan, robustness under perturbation, behavior near
    irreversible thresholds. **Report whichever method wins — the
    comparison is the contribution.**
36. [P2] **ML4 — differentiable calibration to Viking data** (~50 lines
    after Track A): fit 6–8 constants by gradient descent, VL1 fit / VL2
    holdout, Laplace uncertainties — cheap, strengthens the "real
    differentiable substrate" claim; include if August is kind.
37. [P3] **ML6 — amortized inverse design**: hypernetwork
    (scenario → schedule); the world-model latent is the natural scenario
    embedding. Follow-up paper unless everything lands early.

---

## TRACK D — Paper & release (Sept 1 → deadline)

38. **Write the paper** [P0]. Framing: *"a physics-informed world model for
    planetary engineering, trained on and evaluated against a differentiable
    GPU-batched Mars simulator."* Lead with the world model + extrapolation
    + planning results; the simulator is the vehicle (ICLR has no systems
    track). Figures: (1) substrate validation (GCM benchmark + Viking),
    (2) world-model horizon-RMSE & stability curves, (3) extrapolation /
    hysteresis-capture table, (4) 4-way planning comparison, (5) scaling
    (planet-years/sec vs batch — run D2 throughput script once, cheap).
    Related work: JEPA family, Dreamer/TD-MPC2, JAX MD/Brax/DiffTaichi,
    GCM emulators — and the contrast with foundation-model-for-Mars
    (Roy et al. 2026): emulators interpolate observations; terraforming
    trajectories are far outside any observational distribution; a
    mechanistic substrate + physics-informed world model extrapolates by
    construction. (One paragraph — the parked track still earns its
    citation.)
39. **F3 — experiment config system** [P1]: every figure regenerates from
    one command (`python -m experiments.run configs/fig2.yaml`); extend the
    existing CLI YAML convention.
40. **F4 — release packaging** [P1]: public repo, README with the 10-line
    "train a world model on Mars" example (verbatim in the paper), clean-venv
    pip install, delete/deprecate the legacy `GHGState` path and the
    `magneticfield.py` stub, docs per repo rules.
41. **Internal review + `/math-review` audit** of every new equation
    (soft gates, B1–B3, banded EBM, conservation losses) before submission.
42. **Submit** [P0]; confirm CfP dates at iclr.cc; slip plan = NeurIPS 2027
    D&B (the benchmark/world-model framing fits D&B even better) or
    ICML 2027.

---

## Parked (post-submission backlog)

- **Mars Atmospheric Foundation Model track** (arXiv:2605.28851): author
  outreach, ARCO-Mars/OpenMARS loaders, CO2 mass-budget benchmark
  contribution, FM-as-teacher / reduced-model-as-prior experiments. The
  ML10 data infra and `to_netcdf()` exporter built for the paper transfer
  directly when this resumes.
- ML7 Bayesian inference + robust design; ML8 learned meridional transport
  (needs B7 validated); B5/B6/B8/B9 physics breadth; C4–C7 intervention
  breadth; A7 adjoint backend; A8 second-order; A9 dtype policy; D4 secular
  mode; D5 serialization; ML1/ML2 pluggable-terms + neural-RT surrogate;
  AmesCAP borrowings beyond what validation needs (MarsCalendar is cheap —
  grab it whenever Ls handling causes a bug).

---

## Timeline at a glance (~10 weeks)

| Weeks | Focus |
|---|---|
| Jul 7–18 | Track A-1 (differentiability T1–T6) + lit review (20) + ML10 infra (21) + **B7 go/no-go (3)** |
| Jul 18–25 | D1 batching (8) + B1–B3 physics (9–11) + dataset generation (22) |
| Jul 25–Aug 8 | World model v0 (23) + physics-informed variants (25); B7 + v1 if GO (13, 24) |
| Aug 8–22 | Stability/extrapolation/hysteresis studies (26–28); objectives (31); planning baselines (32–34) |
| Aug 22–Sep 5 | 4-way comparison (35); validation figures (15–18); calibration if time (36) |
| Sep 5–deadline | Paper, configs, release, math-review, submit (38–42) |

**Biggest risks:** (1) world-model training instability on long horizons —
mitigate with curriculum + truncated BPTT + smooth gates ON; (2) B7 slipping
mid-August — that's what the go/no-go is for, decide early and don't reopen;
(3) "is the 0-D version enough for a JEPA claim" — the no-go framing must
lean on bistability/irreversibility, not spatial structure; (4) reviewer
question "why learn a world model when the simulator is differentiable and
fast?" — prepared answer: amortized planning speed, fine-tunability to
observational data the mechanistic model can't fit, uncertainty
quantification, and the paper *measures* exactly when each wins (task 35).
