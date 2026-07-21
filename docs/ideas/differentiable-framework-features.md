# Feature spec — a fully functional differentiable Mars terraforming framework

Companion to [iclr2027-workplan.md](iclr2027-workplan.md) (which sequences the
paper-critical subset). This is the complete feature inventory, tiered by
priority. P0 = the framework doesn't deserve the word "differentiable" without
it; P1 = needed for credible terraforming science; P2 = breadth that makes it
a *framework* rather than one model; P3 = nice-to-have.

Each feature has a collapsible block: **What** (the deliverable), **Why**
(what breaks or is impossible without it), **How** (implementation route with
file targets), **Verify** (how you know it's done).

---

## A. Differentiable core

<details>
<summary><b>A1 (P0). Tensor-native intervention path</b> — fix the four autograd graph breaks</summary>

**What:** Remove every `.item()` / `float()` cast between an injection
schedule and the simulated temperature, so `d(T_final)/d(kg injected)` exists.

**Why:** The schedule is the design variable of the whole paper. Today it is
cast to Python float twice before physics ever sees it, so autograd sees a
constant. Nothing else in the framework matters until this works.

**How:** Four edits (details in workplan Phase 1):
1. `mars.py:459-462` (`inject`) — replace
   `torch.tensor(float(kg) * float(g.item()) / float(A.item()))` with tensor
   arithmetic `dP = torch.as_tensor(kg, dtype=TF_DTYPE, device=self._device) * g / A`.
2. `controller.py:124` — stop coercing the schedule dict with `float(v)`;
   hold whatever the caller passed (float or tensor).
3. `mars.py:502` (`_recompute_greenhouse_factor`) — delete
   `if float(dF.item()) <= 0.0: return`; compute unconditionally with
   `torch.relu(dF)` (same semantics, no CPU sync, no branch).
4. `forcing.py:184` (`update_greenhouse_factor`) — same fix.

**Verify:** Test: build `t = torch.tensor(1e9, requires_grad=True)`, run
`InterventionController(mars, {"SF6": t}).run(2)`, assert the final
temperature has a `grad_fn`, call `backward()`, assert `t.grad` is finite and
positive. Add to `package/tests/interventions/`.
</details>

<details>
<summary><b>A2 (P0). Differentiable schedule objects</b> — <code>InjectionSchedule</code> as the optimizer-owned parameter</summary>

**What:** New module `package/src/interventions/schedule.py`: a class holding
a raw parameter tensor `theta` of shape `[n_years, n_compounds]`, exposing
`rates() = softplus(theta)` and `total_mass()`.

**Why:** Optimizers need one flat tensor of unconstrained parameters.
Physical rates must be ≥ 0 — but if you enforce that with `clamp(min=0)`,
the gradient dies exactly at 0 and the optimizer can never turn a compound
back on. `softplus` keeps the constraint *and* the gradient. Per-year shape
makes time-varying strategies (front-load vs ramp) representable — a constant
dict cannot express "stop injecting after year 20".

**How:** Plain dataclass wrapping `nn.Parameter`; a compound-name list maps
columns to registry entries; `to(device)`; optional per-compound upper bounds
via scaled sigmoid. `InterventionController` (feature A1 change 2) accepts it
alongside the legacy dict form.

**Verify:** Unit tests: rates are ≥ 0 everywhere, gradient flows to `theta`,
round-trip through one `run()` epoch optimizes a trivial objective (increase
T) in <20 Adam steps.
</details>

<details>
<summary><b>A3 (P0). Smooth switching physics</b> — opt-in soft gates so the optimizer can see through regime changes</summary>

**What:** `Mars(smooth_gates=True)` replaces hard, gradient-dead switches with
smooth equivalents. Primary target: the ice-exhaustion gates at
`mars.py:308-317` (RK4 path) and `mars.py:403-412` (fast path).

**Why:** `torch.where((ice <= 0) & (dM < 0), 0, dM)` is correct physics but
its gradient w.r.t. anything is zero once a cap empties. The CO2-collapse /
cap-depletion regime is *the* scientifically interesting one — if gradients
vanish there, the optimizer is blind exactly where it matters. Hard `clamp`
floors (`GHF ≥ 1`, `P ≥ 0`) have the same dead-zone problem when the floor is
reachable.

**How:** Multiply the sublimation (negative) branch by
`torch.sigmoid(ice / ice_ref)` with `ice_ref ≈ 1e12 kg`: when ice ≫ ice_ref
the gate ≈ 1 (physics unchanged); as ice → 0 the flux fades smoothly instead
of snapping off. Keep the hard gate as default so existing tests, CLI and
validation runs are bit-identical. Run `/math-review` on the gate equation
per repo rule.

**Verify:** (a) Agreement test: hard vs smooth modes differ <0.5 % while
ice ≫ ice_ref. (b) Gradient test: with `smooth_gates=True`, d(T_final)/d(rate)
is nonzero in a run where the north cap empties; with the hard gate it is
zero (document this as the counterexample).
</details>

<details>
<summary><b>A4 (P0). Memory-feasible backprop through long rollouts</b> — checkpointing + sparse snapshots</summary>

**What:** Make a 100-Mars-year rollout with gradients fit in <8 GB GPU.

**Why:** One Mars year at dt = 3600 s is 16,488 steps; `run()`
(`time_controller.py:230-243`) appends a `Snapshot` of cloned tensors *every
step*, and every clone keeps a reference into the autograd graph. A 50-year
design problem is ~800k steps — naive BPTT will OOM long before that.

**How:** Two independent mechanisms:
1. `snapshot_every: int = 1` parameter on `run()` — record every Nth step.
   For optimization you typically need only annual statistics, so N ≈ 100–700.
2. `torch.utils.checkpoint.checkpoint` around each one-year segment inside
   `InterventionController.run` (`controller.py:174-177`), behind
   `checkpoint_years: bool = False`. Memory becomes O(years + steps-per-year)
   instead of O(total steps), at the cost of one extra forward per year on the
   backward pass.

**Verify:** Benchmark script asserting peak `torch.cuda.max_memory_allocated`
for a 20-year FAST rollout with grad < 8 GB; gradient equality test:
checkpointed vs non-checkpointed gradients match to 1e-10 on a 2-year run.
</details>

<details>
<summary><b>A5 (P0). Gradient test harness</b> — never lose differentiability silently again</summary>

**What:** `package/tests/engine/test_gradients.py` — a permanent test suite
guarding the gradient path.

**Why:** Graph breaks are silent: code still runs, results look right, and
`.grad` is just `None` or stale. One innocent `float(x.item())` in a future
PR would undo Phase 1 without any test failing today.

**How:** Three test classes:
1. `torch.autograd.gradcheck` on a 3-sol rollout (float64, small dt) — exact
   Jacobian check of the composed step function.
2. Finite-difference vs autograd on d(T_final)/d(SF6 kg/yr) over 1 year,
   `rel_tol 1e-4`.
3. Smoke guard: end-to-end run, assert `T_final.grad_fn is not None` and
   every schedule parameter receives a nonzero `.grad`.

**Verify:** Suite runs in CI (<10 s per test per repo testing rule; keep
horizons tiny).
</details>

<details>
<summary><b>A6 (P1). Functional pure-step API</b> — the one real architecture decision</summary>

**What:** Alongside the current object-oriented API, expose
`step(state, params, t, dt) -> state` where `state` is an immutable tensor
container (TensorDict / NamedTuple) — no hidden mutation.

**Why:** Mutable-attribute updates (`self.thermal.surface_temperature = …`)
are *compatible* with autograd but *incompatible* with `torch.func`
transforms: `vmap` (free batching — would replace ~300 hand-written lines in
`batched_controller.py`), `jacrev`/`jacfwd` (one-call sensitivity analysis,
feature E5), `vmap` over network weights (deep ensembles, ML doc), and clean
`torch.compile(fullgraph=True)`. This is exactly the design choice that made
JAX MD composable.

**How:** The boundary already exists: `pack_state`/`unpack_state`
(`planet.py:120-151`) define what the evolving state is. Extract the body of
`compute_derivatives`/`compute_fast_physics` into pure functions of
`(state, params, t)`; the OO methods become thin wrappers that pack, call,
unpack. Params (constants cached as `self._*` in `mars.py:213-227`) become an
explicit NamedTuple so calibration (ML4) and vmap-over-params work.

**Verify:** `torch.func.vmap(step)` over a batch of 64 states matches 64
sequential OO steps exactly; `jacrev` of one step returns the 3×3 Jacobian.
</details>

<details>
<summary><b>A7 (P1). Adjoint / ODE-solver backend</b> — O(1)-memory gradients for very long horizons</summary>

**What:** Optional ACCURATE-mode backend using `torchdiffeq.odeint_adjoint`
instead of the hand-rolled RK4 loop.

**Why:** Even with checkpointing (A4), BPTT memory grows with rollout length.
The adjoint method recomputes the trajectory backwards and needs constant
memory — the standard tool for 10³-year horizons. But adjoints through
stiff/switching dynamics (frost-point gates) can be *inaccurate*; the
BPTT-vs-adjoint accuracy comparison on this system is itself a useful paper
paragraph.

**How:** The RHS is already the right shape (`compute_derivatives(y)`); wrap
it as `f(t, y)`, feed to `odeint_adjoint`. Requires A6-style purity (the RHS
currently reads `self.elapsed_time` and `self.orbital_angle` — time-dependence
must come in through `t`, which also fixes a latent RK4 inconsistency: substep
evaluations at `t + dt/2` currently see the *start-of-step* orbital state).

**Verify:** Gradient agreement BPTT vs adjoint on a 1-year run (loose tol,
~1e-3); memory profile flat vs horizon length.
</details>

<details>
<summary><b>A8 (P2). Second-order derivatives</b> — Hessians for calibration confidence and curvature-aware optimizers</summary>

**What:** Support `torch.autograd.grad(create_graph=True)` /
`torch.func.hessian` through short rollouts.

**Why:** The Hessian at a calibration optimum (ML4) gives parameter
uncertainties (Laplace approximation); HVPs enable Newton-CG / trust-region
optimizers that converge in far fewer rollouts than Adam when the schedule
space is small.

**How:** Mostly falls out of A6; the work is auditing for ops with missing
double-backward (none expected in this op set) and adding a test.

**Verify:** `torch.func.hessian` of a 10-sol loss w.r.t. 3 schedule params
matches finite differences of gradients.
</details>

<details>
<summary><b>A9 (P2). Precision policy</b> — float32 throughput mode, float64 verification mode</summary>

**What:** Make `TF_DTYPE` (in `src/constants`) a runtime choice; document
which activities use which.

**Why:** Everything currently runs float64, which halves-to-quarters GPU
throughput vs float32. RL rollouts and CMA-ES populations don't need float64;
`gradcheck` and validation do. Long secular integrations (10³ years of small
increments) genuinely need float64 — so it must stay available, not be
globally switched.

**How:** Thread dtype through planet construction (it already flows from
`TF_DTYPE` everywhere); add `dtype=` to `Mars.__init__` mirroring `device=`.
Watch the accumulation of `elapsed_time` in float32 over long runs (switch to
a float64 scalar clock regardless of state dtype).

**Verify:** float32 vs float64 1-year trajectories agree to ~1e-4 relative;
throughput benchmark shows the expected speedup.
</details>

## B. Physics completeness (terraforming credibility)

<details>
<summary><b>B1 (P1). Pressure-dependent CO2 greenhouse</b> — without it there is no outgassing feedback at all</summary>

**What:** Replace the constant CO2 greenhouse contribution
(`greenhouse_factor=1.02` set at construction, `mars.py:107`) with a
differentiable function GHF_CO2(P).

**Why:** The entire terraforming premise is a positive feedback: warming
releases CO2 (caps/regolith) → thicker atmosphere → *stronger greenhouse* →
more warming. In the current model the last link is missing — pressure can
triple and the greenhouse effect never changes, so the model literally cannot
express a runaway or a threshold. Every published scenario (Zubrin–McKay,
McKay–Toon–Kasting) hinges on this curve.

**How:** Gray-gas form: optical depth τ = τ₀·(P/P₀)^n (n ≈ 1 for
pressure-broadened CO2), GHF = (1 + τ)^(1/4), calibrated so GHF(610 Pa) ≈
1.02 (today's ~5 K greenhouse) and GHF(2 bar) matches Marinova/Wordsworth
radiative-convective results. Lives next to the trace-GHG forcing in
`_recompute_greenhouse_factor` (`mars.py:494`) so the two compose:
GHF_total = GHF_CO2(P) · (1 + ΔF_trace/OLR)^(1/4). Smooth in P → autodiff-safe.

**Verify:** GHF(610 Pa)≈1.02; monotone increasing; reproduce the classic
Zubrin–McKay equilibrium diagram (T_eq vs P S-curve with two stable branches)
once B2+B3 land. `/math-review` the derivation.
</details>

<details>
<summary><b>B2 (P1). Pressure-dependent CO2 frost point</b> — caps must respond to a thickening atmosphere</summary>

**What:** Replace the constant `MARS_CO2_FROST_POINT = 149 K` (`mars.py:62`,
baked into `_Q_out_pole` at `:221-222`) with Clausius–Clapeyron
T_frost(P) = B / ln(A/P).

**Why:** 149 K is the frost point at ~610 Pa *today*. As the atmosphere
thickens, CO2 condenses at *higher* temperatures — the polar cold-trap
strengthens and can abort a terraforming attempt (atmospheric collapse). With
a constant frost point the model overstates how easy warming is, in exactly
the scenarios the paper optimizes.

**How:** Standard CO2 sublimation curve constants (A ≈ 1.23×10¹² Pa,
B ≈ 3168 K, from the Fanale/James parameterizations); `_Q_out_pole` becomes a
function εσT_frost(P)⁴ evaluated per step instead of a cached constant.
Smooth in P; a few extra ops per step.

**Verify:** T_frost(610 Pa) ≈ 148–150 K; T_frost(1 bar) ≈ 195 K; existing
seasonal-cap tests still pass at current-Mars pressure.
</details>

<details>
<summary><b>B3 (P1). Regolith CO2 adsorption reservoir</b> — the fourth state variable and the missing hysteresis</summary>

**What:** Add M_regolith with temperature-dependent exchange (desorb when
warm, adsorb when cold), following the Zubrin–McKay (1993) formulation.

**Why:** The regolith likely holds more exchangeable CO2 than the caps; it is
the main reservoir a warming intervention taps, and it creates the celebrated
hysteresis (once warmed past a threshold, Mars stays warm without further
forcing). Reviewers who know the terraforming literature will look for it,
and it makes the optimization landscape genuinely interesting (bang-bang
strategies become optimal).

**How:** State vector grows to `[T, P, M_ice, M_reg]`: touch
`pack_state`/`unpack_state` (`planet.py:120-151`), `compute_derivatives`
(`mars.py:232`), fast path (`:337`), `BatchedMars.pack_state`
(`batched_controller.py:124`), and Snapshot. Exchange law: equilibrium
loading M_eq(T, P) ∝ P^γ · exp(T_d/T) with first-order relaxation toward it;
constants from Zubrin–McKay / Kieffer–Zent. B1+B2+B3 must land together —
individually each is inert; jointly they close the feedback loop.

**Verify:** Reproduce the two-equilibria structure: from cold start the model
stays cold; after a sufficiently large forcing pulse it settles onto the warm
branch and *remains* there when the pulse is removed. `/math-review` required.
</details>

<details>
<summary><b>B4 (P1). Liquid-water habitability diagnostics</b> — measure the thing terraforming is actually for</summary>

**What:** Differentiable indicator for "liquid water possible":
T > 273.15 K *and* P > 611 Pa (triple point), plus a "liquid-water
days per year" annual metric.

**Why:** "Raise mean temperature" is a proxy; the real target is surface
liquid water stability. It needs *both* conditions simultaneously — a warm
sol under 600 Pa still boils water away. This is also the natural headline
objective for demo 1 and reward for the gym env.

**How:** Soft-AND of two sigmoids:
`σ((T−273.15)/w_T) · σ((P−611)/w_P)`, integrated over each year in
`objectives/climate.py` (feature E1). The `Water` dataclass already carries
`liquid_mass`/`vapour_mass` fields that are currently inert — wire the
indicator to them or document them as reporting-only.

**Verify:** Metric is 0 for present-day Mars, rises monotonically along a
warming trajectory, gradient nonzero on both sides of each threshold.
</details>

<details>
<summary><b>B5 (P2). Ice–albedo feedback</b> — the second classic feedback, a few lines</summary>

**What:** Albedo becomes a smooth function of cap extent instead of the
constant `radiation.albedo`.

**Why:** Shrinking bright caps lower planetary albedo → more absorption →
more warming. It's the second-most-famous climate feedback, cheap to add, and
strengthens the nonlinearity story. Currently albedo never changes.

**How:** `α = α_ground + (α_ice − α_ground) · f(M_ice/M_ref)` with a smooth
saturating f; applied wherever `(1.0 - s.radiation.albedo)` appears
(`mars.py:285, 299-300, 375, 397-398`).

**Verify:** Warming run shows albedo declining; feedback strength (ΔT with
vs without) reported; still bounded in [α_ground, α_ice].
</details>

<details>
<summary><b>B6 (P2). Coupled escape physics</b> — give the magnetic-shield intervention a lever</summary>

**What:** Replace the constant `_ESCAPE_RATE` (0.2 kg/s MAVEN value,
`mars.py:61,224`) with a parameterization f(B_shield, EUV, T_upper).

**Why:** The magnetic-field intervention (`interventions/magneticfield.py`,
currently a 5-line stub) has nothing to act on: escape is a constant, so a
shield changes nothing. Also, escape should scale up as the atmosphere
thickens and warms — a (small) headwind on terraforming that the model
currently ignores.

**How:** Keep it simple and cited: escape = R_MAVEN · g(B) · (P/P₀)^a ·
(F_EUV/F₀)^b, with g(B) a smooth suppression factor (published estimates for
an L1 dipole suggest large suppression of ion escape). Feeds
`compute_derivatives` dP/dt (`mars.py:326-330`) and fast path (`:425-430`).

**Verify:** B=0 reproduces MAVEN rate; shield-on run shows reduced secular
pressure loss; escape remains negligible vs seasonal terms at present Mars
(sanity: it is ~0.2 kg/s ≈ 0.02 Pa per millennium — document that this term
matters only on 10³–10⁶ yr horizons, which is honest and worth a sentence in
the paper).
</details>

<details>
<summary><b>B7 (P2). N-band latitudinal energy-balance model</b> — the biggest fidelity jump that stays differentiable</summary>

**What:** Upgrade 0-D → 1-D: 10–20 latitude bands, each with its own T (and
optionally ice), coupled by diffusive meridional heat transport
(Budyko–Sellers class).

**Why:** The 0-D model cannot represent equator-pole contrast, so polar caps
see hacked-in insolation (`cos_zenith_N = sin(δ)` at `mars.py:296`) and
"where to intervene" is meaningless. A banded EBM is a *standard, respected*
model class, fully differentiable, and makes latitude-targeted interventions
(cap darkening vs equatorial GHG) a real design variable. It also unlocks the
learned-transport GNN feature (ML doc, ML8).

**How:** State tensors gain a leading band dimension `[N_bands]`;
insolation per band uses the existing declination math (`mars.py:363-373`
already computes the daily-mean insolation integral — apply it per band
latitude); transport term D·∇²T via a tridiagonal stencil. Batching then
gives `[B, N_bands]` — shape change only, autodiff carries over. This is the
largest single work item in the spec (~1–2 weeks); schedule it only after
Phase 2 of the workplan is safe.

**Verify:** Annual-mean equator-pole temperature contrast within ~15 K of MCD
zonal means; global mean matches the 0-D model when D → large; conservation
diagnostic (F2) holds per band.
</details>

<details>
<summary><b>B8 (P3). Milankovitch option</b> — orbital-element drift for paleo/stability studies</summary>

**What:** Time-varying obliquity and eccentricity (Laskar solutions) for
10⁴–10⁶ year integrations.

**Why:** Mars's obliquity wanders chaotically (15°–35°+); long-term stability
of a terraformed state against obliquity cycles is a genuinely novel question
the framework could answer cheaply. Not needed for the paper's 50–100 yr
horizons.

**How:** Make `axial_tilt`/`eccentricity` interpolated functions of
`elapsed_time` (differentiable table lookup) inside `advance_orbit`
(`planet.py:173`); pair with the secular integrator (D4).

**Verify:** Reproduce published insolation-vs-obliquity curves at the poles.
</details>

<details>
<summary><b>B9 (P3). Stochastic dust events</b> — robustness stress-test for optimized schedules</summary>

**What:** Optional random global-dust-storm forcing: episodic albedo/opacity
shocks with seasonal climatology.

**Why:** Dust storms dominate Martian interannual variability; a schedule
optimized on a clean model may be fragile. Reparameterized noise keeps the
expectation differentiable, enabling *robust* (expected-loss) optimization —
a nice extra result, not paper-critical.

**How:** Sample storm onset/intensity via reparameterized distributions
(Gumbel-softmax timing, log-normal intensity) applied as a transient Δalbedo
+ ΔF_IR; seeded per repo testing rules.

**Verify:** Ensemble statistics (storm frequency/season) roughly match the
observed climatology; robust-optimized schedule outperforms clean-optimized
schedule under storm ensembles.
</details>

## C. Intervention library

<details>
<summary><b>C1 (P0). Composable <code>Intervention</code> ABC</b> — stop hard-coding interventions into Mars</summary>

**What:** Abstract base class: `forcing(state, t) -> InterventionEffect`
(fields: ΔF_radiative, Δalbedo, solar_multiplier, escape_multiplier, mass
fluxes). A planet/controller takes a *list* of interventions and sums effects.

**Why:** GHG injection is currently three Mars-specific methods
(`mars.py:436-505`) — adding a second intervention type means editing the
planet class again. The library framing (and the paper's "framework" claim)
needs interventions to be plug-ins with a common interface, individually
testable and freely combinable (mirrors + GHG + shield simultaneously).

**How:** `interventions/base.py` defining the ABC and effect container;
refactor GHG injection into `GHGInjection(Intervention)` as the first
instance (existing methods become its implementation); `TimeController.evolve`
(`time_controller.py:134`) applies summed effects before the physics step.
Everything parameterized by tensors so A1's guarantees extend automatically.

**Verify:** GHG-only run through the new interface is bit-identical to the
legacy path; a two-intervention run shows additive forcing; gradients flow to
both interventions' parameters.
</details>

<details>
<summary><b>C2 (P1). Time-varying GHG schedules</b> — (= workplan T3/T4)</summary>

**What:** Per-year tensor rates `[n_years, n_compounds]` accepted by the
controller; constant-dict form kept for compatibility.

**Why:** Optimal strategies are almost certainly time-varying (front-load to
kick the feedback, then taper); a constant schedule can't express them, and
150 free parameters vs 3 is what makes the gradient-vs-CMA-ES comparison
compelling.

**How:** See A2 and workplan T3; controller indexes the schedule row by year
inside its loop (`controller.py:165-172`).

**Verify:** Covered by A2/A5 tests plus one integration test with a ramp
schedule.
</details>

<details>
<summary><b>C3 (P1). Nanoparticle aerosols</b> — the 2024 state-of-the-art warming method, one afternoon of work</summary>

**What:** `NanoparticleRelease(Intervention)`: release rate → airborne
particle burden with ~10-year lifetime decay → ΔF.

**Why:** Ansari–Kite (Sci. Adv. 2024) showed engineered nanorods warm Mars
>5000× more effectively per kg than the best gases — it's now *the* reference
intervention, and it's topical. Mathematically it's isomorphic to GHG
injection (release → decaying reservoir → forcing), so the marginal cost is
tiny while the demo value (optimize gas-vs-particle mix) is high.

**How:** Reuse the compound-registry pattern: add a particle "compound" with
its RF efficiency per burden and 10-yr lifetime; forcing enters through the
same ΔF pathway (`forcing.py`). Cite their efficiency numbers.

**Verify:** Reproduce their headline order-of-magnitude: sustained ~30 L/s
release class warming of ~30 K (within reduced-order tolerance); decay test:
burden halves in ~7 years post-shutoff.
</details>

<details>
<summary><b>C4 (P2). Albedo modification</b> — cap darkening / regional aerogel</summary>

**What:** `AlbedoModification(Intervention)`: scheduled Δalbedo, optionally
cap-targeted (couples to B5/B7).

**Why:** Oldest proposal in the literature (dark dust on caps to trigger
sublimation); with B7 it becomes latitude-targeted and genuinely
optimizable ("darken which band, when?").

**How:** Effect container field `Δalbedo` already exists after C1; magnitude
parameterized in albedo units with a deployment-mass cost model for the
objective.

**Verify:** Cap-darkening run sublimates the cap faster; gradient flows to
the Δalbedo schedule.
</details>

<details>
<summary><b>C5 (P2). Solar flux augmentation</b> — orbital mirrors as a solar multiplier</summary>

**What:** `SolarMirror(Intervention)`: multiplicative factor ≥ 1 on
`radiation.solar_flux`, with mirror-area cost model.

**Why:** One line of physics but an important *qualitatively different* lever
(shortwave in vs longwave trapping) — the optimizer choosing between mirror
area and GHG mass is a nice results figure.

**How:** `solar_multiplier` field in the C1 effect container, applied after
`advance_orbit` sets flux (`planet.py:230`); cost = f(area) in `objectives/`.

**Verify:** Multiplier 1.0 is identity; equilibrium T scales as
multiplier^(1/4) in the analytic limit.
</details>

<details>
<summary><b>C6 (P2). Magnetic shield</b> — replace the stub, coupled through B6</summary>

**What:** `MagneticShield(Intervention)`: L1 dipole field strength →
escape-rate suppression (needs B6).

**Why:** Completes the classic intervention triad and replaces the 5-line
`magneticfield.py` stub which would look bad in a public release. Honest
framing: matters on millennial horizons, negligible over 50 years — say so.

**How:** `escape_multiplier` effect field → B6's g(B); parameter is dipole
moment or field-at-Mars in tesla.

**Verify:** Shield-on halts secular pressure decline in a 10³-yr secular-mode
run (D4).
</details>

<details>
<summary><b>C7 (P3). Volatile delivery</b> — impulsive imports with differentiable timing</summary>

**What:** `VolatileDelivery(Intervention)`: discrete events adding mass to P
(e.g., N2/NH3 from redirected objects) or ice inventory.

**Why:** Completes the intervention taxonomy (import vs mobilize vs retain);
scientifically interesting because *timing* relative to the feedback state
matters — but discrete timing is the hardest thing to differentiate, hence P3.

**How:** Represent each event as mass m_i (differentiable trivially) at time
t_i, smoothed with a narrow Gaussian kernel in time so ∂/∂t_i exists;
document the kernel-width bias.

**Verify:** Total delivered mass conserved; gradient w.r.t. both m_i and t_i
finite; kernel-width sensitivity reported.
</details>

## D. Engine & scale

<details>
<summary><b>D1 (P0). Batched interventions</b> — populations, RL, and UQ all need this</summary>

**What:** `BatchedMars` (`batched_controller.py:53`) currently stacks
planetary state but knows nothing about greenhouse factors or injection. Add
batched GHF state and `inject_batch(dP: Tensor[B, n_compounds])`.

**Why:** Every consumer of scale — CMA-ES populations (E2 baselines),
vectorized gym envs (E3), posterior ensembles (ML7), throughput figures (D2)—
evaluates *B different schedules* at once. Without batched interventions each
candidate is a separate Python-loop rollout and the GPU sits idle.

**How:** Stack `greenhouse_factor` and composition partial pressures with a
leading batch dim like the existing state stacking (`:82-122`); batched
`_recompute_greenhouse_factor` is elementwise so it vectorizes trivially once
A1 removes the `.item()` branch. Long-term this whole class is replaced by
`vmap` over A6's pure step — build only what Phase 2 needs.

**Verify:** B=64 batched run matches 64 sequential runs to 1e-12; throughput
scales sublinearly in B on GPU.
</details>

<details>
<summary><b>D2 (P1). Throughput benchmark suite</b> — the paper's scaling figure</summary>

**What:** `experiments/benchmarks/throughput.py`: planet-years/second vs
batch size {1, 8, 64, 512, 4096}, CPU vs GPU, FAST vs ACCURATE,
compile on/off.

**Why:** JAX MD's credibility rested partly on scaling curves. "10⁴
simultaneous century-scale Mars simulations on one GPU" is the quantitative
version of the framework claim, and reviewers expect the figure.

**How:** Straight benchmarking script with warmup, `torch.cuda.synchronize`
timing, JSON output + plotting; runs on D1.

**Verify:** Reproducible numbers (±10 % across runs); figure generated by one
script per repo reproducibility conventions (F3).
</details>

<details>
<summary><b>D3 (P1). <code>torch.compile</code> hardening</b> — keep the fast path alive through the refactors</summary>

**What:** CI smoke test that the compiled path (`time_controller.py:125-129`)
still compiles without graph breaks after Phase 1 changes.

**Why:** Graph breaks don't error — they silently fall back to eager and
quietly destroy the CUDA-graph speedup. The Phase 1 edits (A1, A3) touch
exactly the code `torch.compile` traces. Without a guard you won't notice
until the benchmark numbers mysteriously regress in August.

**How:** Test with `torch._dynamo.config` fullgraph assertion on one FAST and
one ACCURATE step; mark `@pytest.mark.slow` if compile time is an issue.

**Verify:** Test fails if someone reintroduces a data-dependent Python branch
in the hot path.
</details>

<details>
<summary><b>D4 (P2). Time-scale splitting / secular mode</b> — 10³-year horizons at ~1 step per year</summary>

**What:** Third accuracy mode: analytically orbit-averaged insolation,
integrating only the slow variables (mean T, P, reservoirs) with dt ≈ 1 Mars
year.

**Why:** Millennial questions (escape, shield value, obliquity stability,
hysteresis mapping) are absurd at 16k steps/year. The fast/slow structure is
already half-present (FAST mode); an annual-mean mode completes the ladder
FAST-diurnal → secular.

**How:** The annual-mean insolation integral has a closed form given (a, e,
tilt, latitude) — the daily-mean machinery at `mars.py:363-373` extends to
orbit averaging; frost/regolith exchange uses annual-mean T with a
seasonal-amplitude correction term. `/math-review` the averaging derivation.

**Verify:** 100-year secular run matches annual means of the FAST run within
2 K / 20 Pa; 10⁴-year run completes in seconds.
</details>

<details>
<summary><b>D5 (P2). State serialization</b> — checkpoint/restart and env resets</summary>

**What:** `Planet.state_dict()` / `load_state_dict()` covering evolving state
+ composition + intervention baselines (`_baseline_ghf`, `_baseline_olr`).

**Why:** Long experiments need restart; gym envs need cheap exact resets
(currently reset = rebuild the whole planet); reproducibility packaging (F3)
needs saved end states.

**How:** Mirror the `nn.Module` convention over the five property dataclasses
+ time/orbit scalars; version the schema.

**Verify:** Round-trip test: save mid-run, restore into a fresh planet,
trajectories continue identically.
</details>

## E. ML / optimization layer

*(deep-dive on E-features and beyond: [ml-features-jaxmd-analogs.md](ml-features-jaxmd-analogs.md))*

<details>
<summary><b>E1 (P0). <code>objectives/</code> module</b> — differentiable losses worth optimizing</summary>

**What:** `package/src/objectives/climate.py`: target-temperature loss
(softplus shortfall at the annual *minimum* — the collapse-prone season),
liquid-water days (B4), total injected mass, collapse penalty (soft indicator
on P below floor), per-intervention cost models. Composable weighted sums.

**Why:** Demos, RL rewards, and calibration all need loss functions that are
(a) differentiable, (b) physically meaningful, (c) shared — if demo 1 and the
gym env score differently, their comparison (ML5) is meaningless.

**How:** Pure functions over Snapshot lists / trajectory tensors; annual
reductions use `min`/`softmin` consistently (document which); every loss unit-
tested for gradient flow and sign.

**Verify:** Gradient-flow tests per loss; monotonicity sanity (more injection
→ lower temperature-shortfall loss, higher mass loss).
</details>

<details>
<summary><b>E2 (P0). Gradient optimization driver + baselines</b> — the demo-1 engine</summary>

**What:** `experiments/demo1_gradient_design/`: Adam/LBFGS loop over
`InjectionSchedule.theta`; CMA-ES (`cma` package) and random search at the
same rollout budget; loss-vs-rollouts and schedule/trajectory figures.

**Why:** The headline claim is that gradients through the simulator beat
black-box search. That claim needs a scrupulously fair comparison: identical
objective, identical rollout budget, tuned baselines (report the CMA-ES
population/sigma sweep — a beatable-strawman baseline is the most likely
reviewer attack on the whole paper).

**How:** Config-driven (F3): 50-year horizon, FAST, dt=6 h, {SF6, CF4, C3F8},
per-year rates = 150 params; 5 seeds each; wall-clock *and* rollout-count
axes (gradients cost ~2× forward each — show both accountings honestly).

**Verify:** Deterministic under seed; figures regenerate from one command;
gradient method converges to feasible schedules that black-box methods don't
match at equal budget (or report otherwise — either result is publishable
with the memory/accuracy analysis).
</details>

<details>
<summary><b>E3 (P1). Gymnasium environment</b> — terraforming as a long-horizon control benchmark</summary>

**What:** `package/src/envs/mars_env.py`: obs = [T stats, P, M_ice, GHF, ΔF,
year/horizon]; action = per-compound rates (Box, log-scaled); 1 step = 1 Mars
year; reward = −E1 losses; termination on collapse. Vectorized variant on D1.

**Why:** The RL community lacks benchmarks with *irreversibility* (collapse
is absorbing, escaped atmosphere is gone) and decade-delayed rewards; that's
the benchmark contribution. Also the substrate for the ML5 comparison.

**How:** Thin wrapper over `InterventionController` + D5 resets; passes
`gymnasium.utils.env_checker`; PPO/SAC baselines via CleanRL or SB3 with
config-pinned hyperparameters.

**Verify:** env-checker clean; PPO learns a nontrivial policy (beats
zero-action and random baselines); vectorized env matches serial env
step-for-step under fixed seeds.
</details>

<details>
<summary><b>E4 (P1). Neural closure hook</b> — (= workplan T11; see ML doc ML3)</summary>

**What:** `Mars` subclass adding an MLP correction to `compute_derivatives`
(override point `mars.py:232`), trained through the integrator.

**Why:** The hybrid neural-physics demo — shows NNs and the simulator compose,
the core JAX MD message.

**How/Verify:** See workplan T11 and ML doc ML3 (including the
through-integrator vs per-step training ablation).
</details>

<details>
<summary><b>E5 (P2). Sensitivity & UQ toolkit</b> — Jacobians and ensembles as one-liners</summary>

**What:** `jacrev`-based sensitivities of outcomes w.r.t. all physical
constants; batched parameter-ensemble sweeps; tornado-plot helper.

**Why:** "Which physical uncertainty dominates the projected outcome?" is the
question a mission designer actually asks; autodiff answers it in one call
where a GCM needs a finite-difference campaign. Cheap, differentiating (no
Mars model offers it), and it feeds the robust-design story (ML7).

**How:** Needs A6 (params as explicit NamedTuple); `torch.func.jacrev` over a
rollout-summary function; ensemble sweeps via D1/vmap.

**Verify:** Autograd sensitivities match finite differences; tornado plot for
d(T_final)/d(each constant) generated by one script.
</details>

<details>
<summary><b>E6 (P2). Learned policies vs planners protocol</b> — one comparison table, three method families</summary>

**What:** Shared evaluation protocol comparing {open-loop gradient schedule
(E2), closed-loop BPTT policy (ML5), closed-loop RL (E3)} on identical
objectives, horizons, and perturbation suites.

**Why:** This is the paper's scientific synthesis — without a shared
protocol the demos are three disconnected anecdotes. Perturbation suite
(parameter noise, B9 dust events if available) tests the open-vs-closed-loop
hypothesis.

**How:** `experiments/eval_protocol/`: fixed seeds, fixed eval scenarios,
one results table + one figure.

**Verify:** Every number in the paper's comparison table regenerates from one
command.
</details>

## F. Validation, data, tooling

<details>
<summary><b>F1 (P1). Observational validation suite</b> — the fidelity statement</summary>

**What:** `experiments/validation/`: simulate 1 Mars year at Viking Lander
sites (VL1 22.5°N — note the default `latitude=22.0` at `mars.py:110` is
already VL1 — and VL2 48°N); overlay the observed annual pressure cycle
(~610↔900 Pa CO2 seasonal swing), diurnal temperature range, and MCD
climatology; report residuals in a table.

**Why:** A reduced-order model claiming terraforming relevance must show it
gets *present-day* Mars right first. This section pre-empts the "toy model"
review and doubles as the target data for calibration (ML4).

**How:** Viking pressure data is public (PDS); MCD point queries via its web
interface or API; fixtures per testing rule (no network in tests).

**Verify:** Seasonal pressure extrema within ~15 %; honest residual table in
the paper (discrepancies *stated*, not hidden).
</details>

<details>
<summary><b>F2 (P1). Conservation diagnostics</b> — physics CI</summary>

**What:** Runtime-checkable budgets: total CO2 (atmosphere + caps + regolith)
constant up to escape ∫dt; energy-balance residual bounded.

**Why:** Reduced models rot silently — a sign error in a new exchange term
(B3!) produces plausible-looking trajectories that leak mass. Conservation
checks catch whole bug classes, and "mass-conserving by construction, verified
in CI" is a sentence reviewers like. Also doubles as a gradient-sanity
harness (conserved quantities should have ~zero gradient w.r.t. allocation
parameters).

**How:** `engine/diagnostics.py`: budget functions over Snapshot series;
pytest assertions on every physics test run; optional runtime warning hook in
`TimeController.run`.

**Verify:** Deliberately broken exchange term (test fixture) trips the check.
</details>

<details>
<summary><b>F3 (P2). Experiment config system</b> — every figure from one command</summary>

**What:** YAML → (planet config, interventions, objective, optimizer/agent,
seeds) so each paper figure is `python -m experiments.run configs/fig2.yaml`.

**Why:** ICLR reproducibility expectations + your own sanity in September
when figures need regenerating after a physics fix. The CLI already has a
YAML config pattern (`cli/config_loader.py`) — extend the convention rather
than inventing a second one.

**How:** Dataclass-validated configs (mirror `cli/models.py` style), seed
pinning, output dir with config hash, figure scripts consuming saved arrays
(never re-simulating inside plotting code).

**Verify:** Fresh-clone test: every figure regenerates bit-identically (up to
GPU nondeterminism, documented).
</details>

<details>
<summary><b>F4 (P2). Docs & release packaging</b> — the open-source half of the framework claim</summary>

**What:** Public-repo readiness: README with the 10-line "optimize a
terraforming schedule" example (the JAX MD trick — API snippets verbatim in
the paper), install path, license, docs per the repo documentation rule,
legacy-code removal (`interventions/state.py` GHGState path,
`magneticfield.py` stub if C6 doesn't land).

**Why:** JAX MD's impact came as much from the usable package as the paper;
reviewers *will* click the anonymized repo link. Dead code and stubs cost
more credibility than missing features.

**How:** Follow `docs/` structure rules; API examples doubling as doctests;
`pip install` verified in a clean venv; anonymized release for submission.

**Verify:** Clean-machine install + README example runs in <5 min;
`grep -r GHGState` returns nothing (or a deprecation shim with a removal
date).
</details>

---

## Suggested build order

1. **P0 row (A1–A5, C1, D1, E1, E2)** — after this it *is* a differentiable
   terraforming framework, minimal but real. ~3 weeks.
2. **P1 physics (B1–B4) + A6 functional core + F1/F2** — after this the
   science is defensible and `torch.func` transforms work. ~3 weeks.
3. **P1/P2 breadth (C3–C6, B5–B7, E3–E6, D2–D4)** — after this it's a
   framework with an intervention library, an RL benchmark, and a UQ story.
   Scope to whatever fits before the writing freeze.

Dependency notes: C1 before C3–C7 (don't add interventions twice); A6 before
E5 and before any `vmap` claims; B1+B2+B3 together (they form the feedback
loop — individually they're inert); D1 before E3-vectorized and E2 baselines.
