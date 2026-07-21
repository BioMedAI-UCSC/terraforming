# ML features — the JAX MD analogs for a terraforming framework

Companion to [differentiable-framework-features.md](differentiable-framework-features.md)
(sections E1–E6 are the baseline ML layer; this doc goes deeper on the
JAX-MD-style "neural networks and simulation compose freely" capabilities).

JAX MD's ML story has three legs, each demoed in the paper:

| JAX MD capability | Their demo | Our analog |
|---|---|---|
| Neural functions are first-class physics (any callable can be the energy function, including a GNN trained on DFT) | GNN potential inside MD | Neural terms inside the climate RHS (ML1–ML3, ML8) |
| Meta-optimization *through* a whole trajectory | Particle-packing design | Intervention design (workplan T8) + model calibration to spacecraft data (ML4) |
| Agents/control inside simulation | Flocking | Neural intervention policies trained by BPTT (ML5–ML6) |

Each feature below has a collapsible **implementation guide** with concrete
steps, data requirements, and verification criteria.

---

## ML1. Pluggable neural physics terms (the "any callable" API)

The single most JAX-MD-like feature. Every term in the RHS becomes replaceable
by an `nn.Module` with the same signature:

```python
mars = Mars(
    greenhouse_fn = NeuralGreenhouse(...),   # default: analytic GHF
    escape_fn     = None,                    # default: MAVEN constant
    transport_fn  = None,                    # (after N-band EBM lands)
)
```

<details>
<summary><b>Implementation guide</b></summary>

**Why this is the framework claim:** JAX MD's core design is that
`simulate.nvt(energy_fn, ...)` accepts *any* function — hand-written
Lennard-Jones or a GNN — and everything (integration, gradients, vmap)
composes. Our equivalent: the climate RHS is currently a monolith; making its
terms injectable callables is what turns "a Mars model in PyTorch" into "a
framework."

**Steps:**
1. Inside `compute_derivatives` (`mars.py:232-332`) and
   `compute_fast_physics` (`:337-430`), extract each physical term into a
   named method with a uniform signature `(state, t) -> Tensor`:
   `_greenhouse_term`, `_escape_term`, `_sublimation_term`, `_insolation_term`.
2. Add constructor kwargs (`greenhouse_fn=None, escape_fn=None, ...`); when
   provided, the callable replaces the default method. Type-annotate the
   protocol in a new `framework/neural.py` (term signatures + an
   `nn.Module` base class per term with shape assertions).
3. Since callables may be `nn.Module`s with parameters, `Mars` (or the
   controller) must expose `parameters()` so optimizers/trainers find them —
   simplest: make the term container an `nn.ModuleDict`.
4. Document the contract: terms must be pure w.r.t. the passed state (no
   hidden `self` state reads) so batching/vmap stay valid — this overlaps
   with feature A6 and should be done together.

**Effort:** ~2–3 days of careful refactoring + tests (the physics is
untouched; the risk is accidental behavior change — pin with bit-identity
tests against the pre-refactor model).

**Verify:** (a) Default-callable run is bit-identical to current code.
(b) A dummy `nn.Module` term receives gradients through a 1-year rollout.
(c) The four demo lines above appear verbatim in the paper's API figure.
</details>

## ML2. Neural radiative transfer surrogate (the "GNN potential" analog)

JAX MD's flagship trick: train a network on expensive high-fidelity physics
(DFT), then run cheap dynamics with it. Our version: the linear optically-thin
ΔF in `forcing.py:104-134` is only valid below ~1000 ppb — exactly the regime
serious terraforming leaves. Train an MLP `(partial pressures, P, T) → ΔF`
on band-model / correlated-k radiative transfer calculations, plug it in via
ML1. The framework then stays valid into the thick-atmosphere regime *and*
remains differentiable — the analytic formula becomes the ablation baseline.

<details>
<summary><b>Implementation guide</b></summary>

**Data (the real work):** you need ΔF targets across the regime the linear
model can't reach. Options, in order of preference:
1. Marinova et al. (2005) published forcing curves per compound (digitize
   their figures — they go well beyond the linear regime and are the source
   your `compounds.py` efficiencies already come from).
2. Generate your own grid with an open 1-D RT code (e.g. a gray/band model,
   or `climt`/petitRADTRANS-class tools) over
   (P_CO2 ∈ [0.6, 200] hPa, T ∈ [150, 300] K, per-compound ppb ∈ [0, 10⁶]).
3. Fallback: fit the saturating functional form F = a·ln(1 + C/C₀) per
   compound (Wordsworth-style) — not a neural net, but fixes the same
   validity problem and is a fine ablation point.

**Model:** small MLP (3 hidden layers × 64, softplus activations so output is
smooth), inputs log-transformed (pressures span 6 decades), output ΔF in
W/m². Enforce ΔF(0 trace gas) = 0 by construction (multiply output by a
smooth mask) so the untouched-Mars baseline stays exact.

**Steps:** dataset builder in `experiments/data/rt_grid.py` → train with the
ML10 trainer → wrap as `NeuralGreenhouse(nn.Module)` conforming to ML1's term
protocol → swap in via `greenhouse_fn=`.

**Effort:** ~1 week, dominated by data generation/digitization.

**Verify:** (a) Matches the linear model within 5 % below 500 ppb (their
common validity region). (b) Saturates (sub-linear) at high concentration —
plot vs Marinova curves. (c) End-to-end: a high-injection optimization run
with the neural term chooses visibly different (more realistic) schedules
than the linear term — that comparison *is* the paper figure.
</details>

## ML3. Neural closure trained through the integrator

Already specced (workplan T11 / features E4): MLP correction to dy/dt trained
so FAST matches ACCURATE, then so the model matches MCD climatology. The
JAX-MD-specific point worth demonstrating: **through-integrator training**
(gradients via BPTT over the rollout) versus per-step supervised regression —
through-integrator training is what a differentiable simulator uniquely
enables, and it should win on long-horizon stability.

<details>
<summary><b>Implementation guide</b></summary>

**Architecture:** `MarsWithClosure(Mars)` overriding `compute_derivatives`:
`dy/dt = physics(y, t) + scale * f_θ(features)` with
`features = [T/200, log P/610, M_ice/M₀, sin Ls, cos Ls, F_solar/F₀]`
(normalize everything; unnormalized planetary magnitudes will destroy
training). `scale` chosen so the correction is ≤ ~10 % of typical physics
magnitudes — the closure must stay a *correction*, which is also your answer
to "why not a pure emulator."

**Two training regimes to compare (the ablation is the result):**
1. *Per-step regression:* dataset of (y, t, dy_dt_teacher) pairs sampled from
   ACCURATE trajectories; minimize per-step MSE. Standard, but errors
   compound autoregressively at rollout time.
2. *Through-integrator:* unroll FAST+closure for K steps (K ≈ 50–500,
   curriculum from short to long), minimize trajectory MSE vs teacher,
   backprop through the rollout (needs A4 checkpointing for large K).

**Teacher data:** stage 1 = ACCURATE-mode runs (free, exact state access);
stage 2 = MCD v6.1 annual climatology at 2–4 sites (real-fidelity target;
observation operator maps model state to MCD variables).

**Effort:** ~1 week after Phase 1 + ML10 exist.

**Verify:** (a) Horizon-N RMSE curves: through-integrator flat vs per-step
diverging as N grows — the paper figure. (b) >5× RMSE reduction vs uncorrected
FAST at equal wall-clock. (c) Closure magnitude histogram confirms it stayed
a correction (report it — reviewers will ask).
</details>

## ML4. Differentiable calibration / system identification  ⭐ cheap, high value

The direct analog of JAX MD's meta-optimization demo, but with *real data*:
treat the hand-set physical constants as learnable parameters and fit them by
gradient descent so a 1-year rollout matches the Viking annual pressure curve
and diurnal temperature range. This is 4D-Var-style data assimilation, ~50
lines once Phase 1 lands. Strong candidate to *replace or join* a weaker demo
in the paper.

<details>
<summary><b>Implementation guide</b></summary>

**Which parameters:** thermal inertia (`mars.py:60`), polar cap fraction
(`:64`), diurnal swing amplitude (`:65`), tide amplitude/phase (`:66-67`),
escape rate (`:61`), plus greenhouse τ₀ once B1 lands. Six to eight scalars —
a tiny, well-conditioned problem, which is why this demo is cheap.

**Steps:**
1. Add a `learnable: bool` path in `setup_properties` (`mars.py:169-227`):
   register the `self._*` cache as `nn.Parameter`s (in log-space for positive
   quantities) instead of plain tensors.
2. Observation loss: simulate 1 Mars year at VL1/VL2 (dt = 1 h, FAST or
   ACCURATE); loss = MSE against (a) Viking daily-mean pressure series
   (public PDS data — build a fixture per repo testing rules) and
   (b) observed diurnal T range. Weight terms by observational variance.
3. Optimize with LBFGS (small parameter count → quasi-Newton shines);
   ~50–200 rollouts.
4. Uncertainty: Hessian at the optimum (feature A8) → Laplace-approximate
   parameter covariance → report constants as value ± σ.

**Effort:** 2–3 days after Phase 1. Genuinely ~50 lines of new logic plus the
data fixture.

**Pitfalls:** identifiability — tide phase and diurnal amplitude can trade
off; check the Hessian conditioning and freeze redundant parameters rather
than reporting garbage precision. Don't calibrate and validate on the same
year/site: fit VL1, hold out VL2.

**Verify:** (a) Calibrated model beats hand-set constants on the held-out
site (table). (b) Recovered constants are physically plausible (thermal
inertia within published Mars ranges). (c) Optimization trace figure:
loss vs iteration alongside data overlay before/after — visually the most
persuasive figure in the paper.
</details>

## ML5. Neural intervention policies via BPTT (analytic policy gradients)

Closed-loop control: `π_θ(observation) → injection rates`, unrolled through
the simulator, θ updated by backprop — the Brax/SHAC recipe, and the flocking
demo's serious cousin. Compare against (a) the open-loop optimized schedule
(demo 1) and (b) PPO on the gym env (demo 2). This turns demos 1+2 into one
coherent comparison: {open-loop + gradients, closed-loop + gradients,
closed-loop + RL}.

<details>
<summary><b>Implementation guide</b></summary>

**Architecture:** MLP policy (2×64, tanh), input = the E3 observation vector
(normalized), output = per-compound rates via softplus (same positivity
treatment as `InjectionSchedule`). One policy step per Mars year, physics
integrated between steps — so a 50-year rollout is a 50-step recurrent
computation over the simulator.

**Training loop:** sample a batch of B initial-condition/parameter
perturbations (needs D1 batching) → unroll policy+sim 50 years → loss = mean
E1 objective → backprop through everything → Adam on θ. Gradient variance
through long chaotic-ish rollouts is the known failure mode of this recipe;
mitigations in order: shorter curriculum horizons, truncated BPTT (detach
every K years), smooth-gates mode ON (A3 — hard switches make BPTT gradients
explode/vanish).

**The comparison protocol (E6):** all three methods share objectives, horizon,
and an evaluation suite of *unseen* perturbations (±30 % escape rate, ±20 %
thermal inertia, dust shocks if B9 exists). Hypotheses to test: closed-loop
beats open-loop under perturbation (robustness); BPTT beats PPO on rollout
count by 10–100× (sample efficiency); PPO may still win if BPTT gradients are
too noisy — *report whichever happens*, the comparison is the contribution.

**Effort:** ~1 week after D1 + E1 + A4.

**Verify:** (a) Policy generalizes to held-out perturbations. (b) Comparison
table regenerates from one command. (c) Gradient-variance diagnostic logged
(so the truncation choice is justified, not folklore).
</details>

## ML6. Amortized inverse design

Train a hypernetwork `(initial state, target spec) → full schedule` across a
distribution of batched scenarios. At inference: instant design without
per-case optimization. Stretch goal — include only if Phases 1–3 land early.

<details>
<summary><b>Implementation guide</b></summary>

**Setup:** scenario distribution = (initial P ∈ [400, 1200] Pa, ice inventory
±50 %, target T ∈ [250, 280] K, mass-budget weight λ ∈ [0.1, 10]). Network:
MLP (4×128) mapping the scenario vector to `theta` of an `InjectionSchedule`
(so output shape = n_years × n_compounds; keep n_years fixed). Training:
sample scenarios in batches, decode schedules, batched rollout (D1), minimize
mean objective — i.e., exactly demo 1's loss, but amortized over a
distribution instead of solved per-instance.

**Baseline comparison:** per-instance Adam (demo 1) from scratch vs one
hypernetwork forward pass, on held-out scenarios: report objective gap vs
speedup (should be ~10⁴×). Also useful as a warm-start: hypernetwork output +
5 Adam steps often closes the gap — a practical middle ground worth a row in
the table.

**Effort:** ~1 week, but only *after* demo 1 is solid (it reuses every piece).

**Verify:** held-out-scenario objective within X % of per-instance
optimization (report X); warm-start row; failure analysis at distribution
edges (where amortization degrades — honest limitation paragraph).
</details>

## ML7. Gradient-based Bayesian inference & robust design

Differentiable likelihood ⇒ gradient-based samplers work: run Langevin/NUTS
over uncertain physics conditioned on observations, then optimize schedules
against the *posterior ensemble* instead of point estimates. "Robust
terraforming design under parameter uncertainty" is a framing no forward-only
model can touch.

<details>
<summary><b>Implementation guide</b></summary>

**Two stages:**
1. *Inference:* posterior over uncertain constants (escape rate, cap
   fraction, regolith exchange constants once B3 lands, greenhouse τ₀ from
   B1) given Viking/MCD observations — i.e., the Bayesian upgrade of ML4.
   Likelihood = Gaussian around the observation loss; priors = published
   ranges. Sampler: hand-rolled SGLD (~30 lines, batched chains via D1) or
   preconditioned Langevin; NUTS via Pyro if dependencies are acceptable.
   Each likelihood evaluation is a 1-year rollout — batched chains make this
   tractable (64 chains × 1000 steps ≈ 64k rollout-years ≈ minutes on GPU
   in FAST mode).
2. *Robust design:* draw K ≈ 64–256 posterior parameter samples, optimize one
   schedule (or ML5 policy) against the *expected* loss over the batched
   ensemble — a single vmap/batch dimension away from demo 1 once D1 exists.

**The result figure:** trajectory fan (posterior-ensemble spread) under
(a) the point-estimate-optimized schedule vs (b) the robust schedule —
robust should sacrifice mean performance for tail safety (fewer collapse
trajectories). Quantify with CVaR or collapse-probability.

**Effort:** ~1.5 weeks; stretch / follow-up paper unless August is generous.

**Verify:** SGLD posterior marginals overlap ML4's Laplace approximation
(consistency check); robust schedule reduces collapse probability on held-out
posterior samples; chains pass R-hat < 1.05.
</details>

## ML8. Learned meridional transport (the literal GNN analog)

Once the N-band latitudinal EBM exists (feature B7), the diffusive heat
transport between bands is a stencil operator — replace/correct it with a
small graph or conv network over the latitude chain, trained against MCD
zonal-mean climatology. The closest structural match to "GNN inside the
simulation loop."

<details>
<summary><b>Implementation guide</b></summary>

**Prerequisite:** B7 (banded EBM). Do not start this before B7 is validated
in its plain-diffusion form.

**Model:** the latitude chain is a path graph; a 1-D conv with kernel 3
(equivalently a GNN message pass to neighbors) mapping
`[T_bands, insolation_bands, sin/cos Ls] → transport flux per interface`.
Enforce two structural constraints *by construction*: (a) conservation —
predict interface *fluxes* and take divergence, never per-band tendencies;
(b) zero flux at an isothermal state — subtract the mean or gate on ∇T.
These two constraints are the difference between a physical closure and an
unstable curve-fit.

**Training:** teacher = MCD zonal-mean seasonal T climatology (12–36 Ls
bins × bands). Train through-integrator (ML3 regime 2) over seasonal
rollouts so the closure learns transport that produces the right *cycle*,
not just the right snapshot.

**Effort:** ~1 week after B7. Follow-up-paper material within the ICLR
timeline.

**Verify:** (a) banded model + learned transport matches MCD zonal seasonal
cycle better than tuned constant-D diffusion (RMSE table); (b) conservation
diagnostic (F2) holds exactly; (c) stable in 100-year rollouts (the classic
learned-closure failure is slow drift — test it explicitly).
</details>

## ML9. Deep ensembles via `vmap` over weights

With the functional core (A6), `torch.func.vmap` over *closure network
weights* runs a deep ensemble of hybrid models in one batched rollout —
epistemic uncertainty bands on every projected terraforming trajectory.

<details>
<summary><b>Implementation guide</b></summary>

**Recipe:** train M ≈ 8–16 ML3 closures from different seeds/data shuffles
(cheap — the closure is tiny); `torch.func.stack_module_state` to get stacked
parameters; `vmap(functional_rollout, in_dims=(0, None))` over the weight
axis → M trajectories in one batched call. This is exactly the
`vmap`-over-models pattern the functional core exists for; with the OO API it
would need M separate Python rollouts.

**Where it shows up in the paper:** every projected-trajectory figure gains a
shaded epistemic band (ensemble spread) alongside the aleatoric band from
parameter ensembles (ML7) — distinguish the two in the caption; conflating
them is a common review complaint. Also a decision rule: if the ensemble
disagrees strongly along an optimized trajectory, the schedule has left the
closure's training distribution — flag and re-train (connects to the ML2
"extrapolation" argument).

**Effort:** 2–3 days once A6 + ML3 exist.

**Verify:** vmapped ensemble matches M sequential rollouts exactly; band
widens in state-space regions far from training data (sanity plot).
</details>

## ML10. Training & data infrastructure

Shared plumbing for ML2–ML9: trajectory dataset generation, a small `Trainer`,
standard eval metrics.

<details>
<summary><b>Implementation guide</b></summary>

**Modules (`package/src/ml/`):**
- `datasets.py` — generators: (a) ACCURATE-mode trajectory sampler (states +
  derivatives at random phase/season/composition points — cover the
  *intervention-reachable* state space, not just present-day Mars, or every
  learned component will extrapolate immediately); (b) MCD extraction →
  fixture files (no network in tests, per repo rule); (c) Viking PDS pressure
  series → fixture.
- `trainer.py` — minimal loop: loss fn, optimizer, LR schedule, gradient
  clipping (BPTT through rollouts *will* need it), checkpointing, seed
  control, CSV/JSON metrics log. Resist importing Lightning — the repo's
  dependency footprint is part of the release story.
- `metrics.py` — horizon-N trajectory RMSE (the standard learned-simulator
  metric), rollout-stability (divergence step count), calibration error for
  UQ features, conservation-violation rate (ties to F2).

**Conventions:** every trained artifact saved with its config hash + data
fingerprint; `experiments/data/` holds generated datasets out of the package;
all randomness through one seeded generator.

**Effort:** 2–3 days, before ML2/ML3 start (they consume it immediately).

**Verify:** one end-to-end smoke test: generate tiny dataset → train 50 steps
→ metrics file written → resume from checkpoint reproduces losses.
</details>

---

## Priority within the paper timeline

- **Must (paper-critical):** ML3 (demo 3), ML4 (calibration — consider
  promoting to headline demo), ML5 (unifies demos 1–2), ML10 (enables the rest).
- **Should (if August goes well):** ML1 API refactor (it's the *framework*
  claim), ML2, ML9.
- **Stretch / follow-up paper:** ML6, ML7, ML8.

Dependencies: ML1 before ML2/ML8; A6 (functional core) before ML9, helps ML7;
B7 (EBM) before ML8; D1 (batching) before ML6/ML7/ML9.
