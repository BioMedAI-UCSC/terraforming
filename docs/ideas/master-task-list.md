# Master Task List — Differentiable Terraforming Simulator, World Model, and Mars Foundation Model

Consolidated from all documents under `docs/ideas/` (July 2026): the
GCM benchmark results ([benchmark.md](benchmark.md)), the ICLR 2027 direction
and workplan ([iclr2027-direction.md](iclr2027-direction.md),
[iclr2027-workplan.md](iclr2027-workplan.md)), the tiered feature spec
([differentiable-framework-features.md](differentiable-framework-features.md)),
the ML feature guide ([ml-features-jaxmd-analogs.md](ml-features-jaxmd-analogs.md)),
the model landscape ([mars-model-landscape.md](mars-model-landscape.md)),
the AmesGCM comparison ([amesgcm-comparison.md](amesgcm-comparison.md)), and
the language ADR ([adr-language-framework-choice.md](adr-language-framework-choice.md)).

**The two stories this program tells:**

1. **Story 1 — Differentiable simulation for planetary engineering.** Build the
   first differentiable, GPU-batched Mars climate simulator and use gradients
   through it for inverse design of terraforming interventions (ICLR 2027
   paper), then extend it into a **JEPA-style learned world model** for
   terraforming (follow-up research direction).
2. **Story 2 — Mars Foundation Model.** Contribute to the community effort
   toward a Mars Atmospheric Foundation Model
   ([Roy et al. 2026, arXiv:2605.28851](https://arxiv.org/abs/2605.28851)),
   positioning our mechanistic differentiable model as the complementary
   design-oriented counterpart to their data-driven emulator.

Priorities: **[P0]** = paper-critical, **[P1]** = credibility-critical,
**[P2]** = strengthens the story, **[P3]** = stretch / follow-up paper.
Cross-references (T#, A#, B#, …) point at the source docs above.

---

## TRACK 0 — Benchmark follow-ups (from the tform vs. Ames GCM run)

Current result: nighttime floor matches the GCM to 0.1 K, diurnal mean to
~4 K; daytime peak is 22 K too cool and pressure ~50 Pa high.

1. [P1] **Close the daytime-peak gap.** Tune tform surface heat capacity /
   thermal inertia (suspected too large, damping the noon peak); re-run the
   benchmark and re-measure the 22 K discrepancy.
2. [P1] **Ground-truth against REMS.** Run the existing `gale-crater` preset
   against REMS observations to decide whether the GCM or tform is closer to
   reality at the daytime peak.
3. [P2] **Exact-coordinate re-run.** Interpolate GCM output to lat/lon
   (amescap / `runpinterp.csh`) and run tform at the exact matched point
   (equator, 137°E, 0 m) instead of the proxy belt column.
4. [P2] **Add a local-time diurnal diagnostic to the GCM run** (or fold hourly
   snapshots by longitude) so diurnal *phase* becomes comparable — currently
   only amplitude/min/max/mean are.
5. [P2] **Longer, higher-resolution GCM run** (≥30 sols, higher C-number) —
   the 10-sol C24 run is smoke-test resolution; peak and dust field not spun up.
6. [P2] **Isolate physics from boundary conditions:** feed the GCM's local
   albedo and thermal inertia into tform so residuals reflect physics
   differences, not differing surface maps.
7. [P1] **Turn the benchmark into a validation figure** for the paper — a 0-D
   EBM matching a 3-D GCM's nighttime/mean T to a few K is the fidelity claim.

---

## TRACK 1 — Story 1a: Differentiable simulator + ICLR 2027 paper

Target: ICLR 2027, deadline expected ~Sept 16–24, 2026 (confirm at iclr.cc).
Decision already made: stay on PyTorch; build the functional core (A6) as the
JAX hedge.

### Phase 1 — Make the intervention path differentiable (~1.5 weeks) [P0]

8. **T1 — Tensor-native `inject()`** (`mars.py:446-467`): remove
   `float()`/`.item()` casts; DoD: gradient flows from injected kg to
   greenhouse factor.
9. **T2 — Branch-free greenhouse resync** (`mars.py:494-505`,
   `forcing.py:184-202`): replace the `dF <= 0` early-return with
   `torch.relu(dF)`; DoD: d(GHF)/d(kg) finite and positive.
10. **T3 — Tensor schedules in `InterventionController`**
    (`controller.py:111-137, 165-172`): stop coercing to float; accept
    per-year `Tensor[n_years, n_compounds]` schedules.
11. **T4 — `InjectionSchedule` module** (new `interventions/schedule.py`):
    raw `theta` parameter + `softplus` rates (never clamp — clamp kills the
    gradient at 0); the object the optimizer owns.
12. **T5 — Smooth ice-exhaustion gates (opt-in)** (`mars.py:305-318, 400-417`):
    `smooth_gates=True` replaces the hard `torch.where` with a
    sigmoid fade so the CO2-collapse regime is visible to the optimizer;
    hard gate stays default. Run `/math-review` on the gate equation.
13. **T6 — Gradient regression tests + memory management** (new
    `tests/engine/test_gradients.py`; `time_controller.py:196-247`):
    `gradcheck`, finite-difference agreement, `snapshot_every` sparse
    snapshots, per-year `torch.utils.checkpoint`; DoD: 20-year rollout with
    grad in <8 GB GPU.
14. **A5 — permanent gradient-guard suite in CI** so a future `float(x.item())`
    can never silently undo Phase 1.

### Phase 2 — Demo 1: gradient-based intervention design (~2 weeks) [P0]

15. **T7 / E1 — `objectives/` module**: differentiable losses
    (target-temperature softplus shortfall at annual minima, total injected
    mass, pressure-floor/collapse penalty, liquid-water metric once B4 lands);
    shared by demos, RL rewards, and calibration.
16. **T8 / E2 — Optimization driver + baselines**
    (`experiments/demo1_gradient_design/`): Adam on `InjectionSchedule.theta`
    vs CMA-ES vs random search at identical rollout budgets; 50 Mars-year
    horizon, {SF6, CF4, C3F8}, 150 design variables; 5 seeds; report both
    wall-clock and rollout-count axes. **This is the paper's headline figure.**
    Tune the CMA-ES baseline honestly (beatable-strawman is the likeliest
    reviewer attack).
17. **T9 / D1 — Batched interventions in `BatchedMars`**
    (`batched_controller.py:53-268`): batched greenhouse state +
    `inject_batch(Tensor[B, n_compounds])`; DoD: B=64 batch matches 64
    sequential runs to 1e-12.
18. **D2 — Throughput benchmark suite** (`experiments/benchmarks/throughput.py`):
    planet-years/sec vs batch {1…4096}, CPU vs GPU, FAST vs ACCURATE,
    compile on/off — the paper's scaling figure.

### Phase 3 — Demo 2: MarsGym RL benchmark (~2 weeks) [P0/P1]

19. **T10 / E3 — Gymnasium environment** (`envs/mars_env.py`): obs = climate
    state stats, action = per-compound rates, 1 step = 1 Mars year, reward =
    −objectives, termination on atmospheric collapse (the irreversibility
    hook); vectorized variant on batched engine; PPO/SAC baselines vs the
    gradient planner.

### Phase 4 — Demo 3 + validation (~2 weeks) [P0/P1]

20. **T11 / ML3 — Learned closure trained through the integrator**
    (`framework/closure.py`): MLP correction to dy/dt so FAST matches
    ACCURATE (then MCD); the key ablation = through-integrator vs per-step
    training; DoD: >5× trajectory-RMSE reduction at equal wall-clock.
21. **ML4 — Differentiable calibration to Viking data** ⭐ cheap/high-value
    (~50 lines after Phase 1): fit 6–8 physical constants by gradient descent
    against the VL1 annual pressure curve; hold out VL2; Laplace uncertainty
    from the Hessian. Candidate to promote to a headline demo.
22. **T12 / F1 — Observational validation suite** (`experiments/validation/`):
    1 Mars year at VL1/VL2; overlay observed ~610↔900 Pa seasonal pressure
    cycle, diurnal T range, MCD climatology; honest residual table.
23. **F2 — Conservation diagnostics** (`engine/diagnostics.py`): CO2 and
    energy budgets checked in CI (mirror AmesGCM's `testconserv` pattern —
    cite it); doubles as a gradient-sanity harness.

### Phase 5 — Physics credibility upgrades [P1]

Land B1+B2+B3 **together** — individually inert, jointly they close the
terraforming feedback loop.

24. **B1 — Pressure-dependent CO2 greenhouse** GHF_CO2(P) (gray-gas,
    calibrated to GHF(610 Pa)≈1.02 and Marinova/Wordsworth at high P) —
    without it the model cannot express a runaway or threshold.
25. **B2 — Pressure-dependent CO2 frost point** (Clausius–Clapeyron replacing
    the constant 149 K) — captures atmospheric-collapse risk as P rises.
26. **B3 — Regolith CO2 adsorption reservoir** (4th state variable,
    Zubrin–McKay formulation) — the missing hysteresis; verify the
    two-equilibria structure. `/math-review` required.
27. **B4 — Liquid-water habitability metric** (soft-AND of T>273 K and
    P>611 Pa) — the honest headline objective for demos and RL reward.
28. **C1 — Composable `Intervention` ABC** (before adding any new
    intervention): effects container (ΔF, Δalbedo, solar/escape multipliers);
    GHG injection refactored as the first instance, bit-identical.
29. **C3 — Nanoparticle aerosols** (Ansari–Kite 2024, >5000× per-kg
    effectiveness): isomorphic to GHG injection, one afternoon of work, high
    topicality; enables an optimize-gas-vs-particle-mix demo.
30. [P2] **B5 ice–albedo feedback; C4 albedo modification; C5 solar mirrors;
    C6 magnetic shield + B6 coupled escape** (replace the 5-line
    `magneticfield.py` stub or exclude it from release).

### Phase 6 — Architecture & engine [P1/P2]

31. **A6 — Functional pure-step API** `step(state, params, t, dt) -> state`:
    unlocks `torch.func` vmap/jacrev/hessian, replaces ~300 hand-written
    batching lines, and is the JAX-port hedge from the ADR. Do together with
    ML1.
32. **ML1 — Pluggable neural physics terms** (`greenhouse_fn=`, `escape_fn=`
    constructor kwargs) — the "any callable is physics" JAX-MD framework
    claim; default path must stay bit-identical.
33. **D3 — `torch.compile` hardening test** so Phase-1 edits don't silently
    break the compiled fast path.
34. [P2] **A7 adjoint ODE backend (torchdiffeq); A8 second-order derivatives;
    A9 float32/float64 precision policy** (the ADR flags FP64 on consumer
    GPUs as a bigger perf trap than the language choice); **D4 secular mode**
    (10³-yr horizons); **D5 state serialization** (checkpoint/restart, cheap
    env resets).

### Phase 7 — Paper & release (Sept 1 →) [P0]

35. **T13 — Write the paper**: lead with the ML contributions (design,
    benchmark, hybrid), not the simulator; related work anchors = JAX MD,
    Brax, DiffTaichi/PhiFlow, gradient-based climate tuning; include the
    landscape doc's Table-1 comparison and the "why not differentiate an
    emulator" prepared answer.
36. **T14 / F4 — Release packaging**: public repo, README with the 10-line
    optimize-a-schedule example (verbatim in the paper), pip install in a
    clean venv, delete/deprecate the legacy `GHGState` path, docs.
37. **F3 — Experiment config system**: every paper figure regenerates from
    one command (`python -m experiments.run configs/fig2.yaml`).
38. **E6 — Shared evaluation protocol** comparing open-loop gradient
    schedules, closed-loop BPTT policies (ML5), and RL — one table, one
    figure, fixed seeds and perturbation suites.
39. **T15 — Submit**; slip plan = NeurIPS 2027 Datasets & Benchmarks or
    ICML 2027.
40. **Borrow from AmesCAP** along the way: `MarsCalendar` Ls/sol conventions
    (~half a day, kills off-by-a-season bugs), CAP derived-variable
    names/formulas for our analysis layer, a `to_netcdf()` exporter so
    MarsPlot can render our trajectories next to GCM output.

---

## TRACK 2 — Story 1b: A JEPA-style world model for terraforming

New research direction (not yet in the ideas docs): use the differentiable
simulator as the data engine and physics prior for a **physics-informed
JEPA / world model** of terraforming dynamics. Follow-up paper target
(NeurIPS 2027 / ICLR 2028); prerequisites are Track 1 Phases 1–3 (batched
engine, objectives, gym env).

### 2.1 Groundwork

41. **Literature review + positioning memo** (`docs/ideas/`): JEPA family
    (I-JEPA, V-JEPA/V-JEPA 2), latent world models (Dreamer-v3, TD-MPC2,
    DINO-WM), physics-informed latent models, action-conditioned prediction;
    define precisely what "PI-JEPA for terraforming" means here — an
    action-conditioned latent predictor trained on simulator rollouts with
    physics-consistency regularizers.
42. **Decide the observation space**: start with the 0-D state vector
    (trivial encoder, tests the training loop); the interesting version
    arrives with the **B7 N-band latitudinal EBM** (10–20-band fields —
    enough spatial structure for masked/latent prediction to be nontrivial).
    B7 is therefore a shared dependency: schedule it after the ICLR
    submission is safe.
43. **ML10 — training & data infrastructure** (`package/src/ml/`): trajectory
    dataset generators covering the *intervention-reachable* state space (not
    just present-day Mars), minimal trainer with grad clipping and
    checkpointing, horizon-N RMSE / rollout-stability / calibration metrics.
    Shared with Track 1 demo 3.

### 2.2 Build the world model

44. **Dataset generation at scale**: use the batched engine (D1) to produce
    ~10⁵–10⁶ diverse trajectories (randomized initial states, physical
    parameters, intervention schedules — including collapse and runaway
    regimes); store with config hashes per ML10 conventions.
45. **v0 — action-conditioned latent dynamics model** on the 0-D state:
    encoder → latent z, predictor p(z' | z, action, Δt), JEPA-style loss
    (predict in representation space, EMA target encoder, no pixel/state
    reconstruction); baseline against direct state-space regression.
46. **v1 — physics-informed variants** (the research contribution):
    (a) conservation-aware latents — penalize violation of the F2 CO2/energy
    budgets decoded from z; (b) hybrid rollout — learned latent model
    corrected by (or residual to) the differentiable mechanistic step;
    (c) distillation — pretrain the predictor on mechanistic rollouts, then
    fine-tune on MCD/reanalysis so the world model absorbs data the reduced
    model can't fit.
47. **v2 — spatial world model over the banded EBM (after B7)**: masked
    latent prediction over the latitude dimension; learned meridional
    transport (ML8) becomes a special case; teacher = MCD zonal climatology.

### 2.3 Evaluate and explore

48. **Planning in latent space**: MPC/CEM and policy learning (TD-MPC2-style)
    inside the world model; extend the E6 protocol to a 4-way comparison —
    {open-loop gradients through simulator, BPTT policy, model-free RL,
    latent-world-model planning} on identical objectives/horizons/perturbations.
49. **Key research questions to answer (each is a paper section):**
    - Does a latent world model trained on mechanistic rollouts extrapolate
      to unseen intervention regimes better than per-step regression?
    - Where does differentiating the *real* simulator beat planning in a
      *learned* model (gradient quality vs speed vs robustness under chaos)?
    - Do physics-informed regularizers (conservation, invariances) measurably
      improve long-horizon stability — the classic learned-simulator failure?
    - Can the world model's epistemic uncertainty (deep ensembles via
      vmap-over-weights, ML9) flag when an optimized schedule leaves the
      training distribution?
50. **Uncertainty & robustness add-ons** (shared with Track 1 stretch):
    ML7 gradient-based Bayesian inference + robust design against posterior
    ensembles; ML9 deep ensembles; B9 stochastic dust events as the
    perturbation suite.
51. **Stretch — ML6 amortized inverse design**: hypernetwork
    (scenario → schedule) trained across batched scenarios; compare one
    forward pass vs per-instance optimization; the world-model latent is the
    natural scenario embedding.

---

## TRACK 3 — Story 2: Mars Atmospheric Foundation Model (arXiv:2605.28851)

Roy et al. 2026 propose a multi-scale foundation model (MAFM) for the Martian
atmosphere: data = OpenMARS reanalysis + five retrieval suites (TES, MCS,
ACS, NOMAD, EMIRS) + two GCMs (MGCM, MarsWRF); preliminary architectures =
Mars-adapted GraphCast, Mars-adapted Prithvi-WxC, and a Spherical Neural
Operator; downstream tasks include dust storms, frontal systems, low-level
jets, cloud phenomena, frost cycles, global mass-budget forecasting,
downscaling, and reanalysis–observation fusion. Companion resources:
ARCO-Mars cloud-optimized archive (arXiv:2606.21701, EMARS + MACDA +
OpenMARS, MY 24–35) and the PDE-FM Mars emulator paper (arXiv:2602.15004).

### 3.1 Understand and connect

52. **Deep-read the three papers** (2605.28851, 2606.21701, 2602.15004) and
    write an annotated summary in `docs/ideas/` mirroring the AmesGCM
    comparison doc: what MAFM is, where it is strong, and exactly where a
    mechanistic differentiable model is complementary.
53. **Contact the MAFM authors / community** (NASA IMPACT & collaborators):
    present the differentiable-simulator work and propose the complementary
    collaboration angles below; identify their contribution process
    (benchmarks, data loaders, evaluation tasks).
54. **Write the positioning paragraph** for both papers (already drafted in
    mars-model-landscape.md): FMs emulate/interpolate the observed
    distribution; terraforming trajectories (P → 10⁴–10⁵ Pa) are far outside
    any training set; design problems need a small mechanistic model you can
    differentiate through — the two approaches meet at calibration and
    closure training.

### 3.2 Data engineering (shared infrastructure with Tracks 1–2)

55. **Build ARCO-Mars / OpenMARS / EMARS data loaders** under
    `experiments/data/` (fixture-based, no network in tests per repo rule);
    reuse for F1 validation, ML3 stage-2 teacher data, and world-model
    fine-tuning.
56. **MCD v6.1 extraction pipeline** at validation sites (VL1, VL2, Gale) —
    already needed by T11/T12; make it reusable for FM evaluation points.
57. **`to_netcdf()` exporter + CAP-convention variable names** so our
    trajectories are legible to the Mars community and comparable to
    MAFM/GCM output in the same plotting stack.

### 3.3 Concrete contribution candidates (pick 2–3 after author contact)

58. **Contribute a benchmark task: global CO2 mass-budget forecasting.**
    MAFM lists it as a downstream task; our seasonal cap/pressure cycle with
    conservation diagnostics (F2) is a natural low-dimensional evaluation
    target and sanity check for FM outputs — propose our validated seasonal
    curves + Viking/MCD fixtures as an evaluation suite.
59. **Physics-consistency probes for FM evaluation**: use the reduced model
    to score FM forecasts on energy/mass budget closure and frost-point
    consistency — cheap, differentiable, and something a pure-data pipeline
    lacks.
60. **FM-as-teacher for our closure (ML3 stage 3)**: fine-tune/query a
    pretrained Mars FM (Prithvi-WxC-Mars or the PDE-FM emulator) as a
    high-fidelity teacher for the learned closure and the world model — an
    alternative/adjunct to MCD climatology.
61. **Reduced-model-as-prior for the FM**: propose hybrid training in which
    the mechanistic model provides physics regularization or synthetic
    augmentation in regimes with sparse observations (dust-storm years,
    polar night) — the exact inverse of task 60, and the most publishable
    joint result.
62. **Propose "intervention response" as a MAFM downstream task**: once the
    FM exists, evaluating counterfactual forcing scenarios (our intervention
    library, C1–C7) against reduced-model predictions probes FM
    extrapolation — connects terraforming design to their roadmap.
63. **Reproduce/fine-tune one preliminary MAFM baseline** (whichever ships
    code first — GraphCast-Mars, Prithvi-WxC-Mars, or SNO on OpenMARS
    surface variables) to build hands-on credibility before proposing
    contributions.

---

## Sequencing and dependencies

- **Now → mid-September 2026:** Track 1 is the only deadline-bound work.
  Order: Phase 1 (differentiability) → Demo 1 → Demo 2 → Demo 3 +
  calibration + validation → physics upgrades as time allows → paper.
  Track 0 items 1–2 and 7 fold into the validation section.
- **Parallel low-effort:** Track 3 items 52–54 (reading + author contact)
  and 57 (netCDF exporter) can run during Track 1 without endangering it.
- **Post-submission (Oct 2026 →):** Track 2 (world model) becomes primary —
  it reuses the batched engine, objectives, gym env, and ML10 infra built
  for the paper; B7 (banded EBM) is its enabling physics feature.
  Track 3 contributions 58–63 ramp up as the MAFM effort matures.
- **Hard dependencies:** T1–T6 before every demo; C1 before new
  interventions; B1+B2+B3 land together; A6 before vmap/ML9 claims;
  D1 (batching) before RL-vectorized, ML6, ML7, ML9, and world-model data
  generation; B7 before ML8 and world-model v2.
