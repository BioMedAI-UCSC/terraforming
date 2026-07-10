# Dinosaur → Mars workplan — a differentiable Mars GCM plugged into this framework

Exact plan for adapting the NeuralGCM dynamical core
([neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur), Apache-2.0,
active — v1.3.6 Jun 2026) into a differentiable 3-D Mars model (`mars3d`),
and for wiring it into this repo. Companion to
[master-task-list.md](master-task-list.md) (this is the "v2: 3-D
differentiable" item, expanded). Timing: **post-ICLR submission** (Oct 2026 →),
~8–10 person-weeks. Template for the coupling pattern: JCM v1.0
(EGUsphere 2025-6266 — Dinosaur as a dependency + swappable physics interface).

**Strategy in one line:** depend on `dinosaur` as a pip package; shim one file
(`scales.py`, the hardcoded Earth constants); subclass one module
(`primitive_equations.py`, to add the CO₂ mass source the fixed-mass spectral
core lacks); write the Mars physics and the terraforming bridge ourselves.
No wholesale fork — forks of active repos rot.

Dinosaur module map (verified from the repo): `scales.py` (Earth constants —
RADIUS, GRAVITY_ACCELERATION, OMEGA, KAPPA=2/7), `primitive_equations.py` /
`primitive_equations_states.py` (dycore), `spherical_harmonic.py` /
`associated_legendre.py` / `fourier.py` (transforms), `sigma_coordinates.py`,
`time_integration.py` + `leapfrog_utils.py` (semi-implicit stepping),
`filtering.py` (hyperdiffusion), `held_suarez.py` (validation case),
`radiation.py` (incoming solar), `coordinate_systems.py`, `units.py`.

---

## Phase 0 — De-risk spike (2–3 days)

**D0.1 Reproduce upstream.** Pin `dinosaur` at the current release; run the
Held–Suarez and baroclinic-instability notebooks unchanged on our hardware
(JAX CPU + one GPU run).
*DoD:* both notebooks reproduce; one `jax.grad` through a short rollout works.

**D0.2 Mars-constants smoke test.** Shim module `mars3d/scales.py` exporting
Mars values with dinosaur's names:

| Constant | Earth (dinosaur) | Mars |
|---|---|---|
| RADIUS | 6.37122e6 m | **3.3895e6 m** (matches `mars.py`) |
| GRAVITY_ACCELERATION | 9.80616 | **3.72076** |
| OMEGA | 7.292e-5 s⁻¹ | **7.0779e-5 s⁻¹** (2π/88,775.244 s — nearly Earth's) |
| KAPPA (R/cp) | 2/7 (air) | **≈0.257** (CO₂: R=188.9 J kg⁻¹ K⁻¹, cp≈735) |
| Reference T profile (semi-implicit linearization) | ~250–300 K | **~140–220 K** (retune) |

Run a dry Mars Held–Suarez analog (published Mars-HS forcing profiles exist —
planetMPAS and UM idealized-Mars papers are the reference targets) at T21/L20
for 30 sols.
*DoD:* stable integration; a westerly midlatitude jet structure appears;
gradient of mean T w.r.t. a forcing parameter is finite through ≥1 sol.
**This spike is the go/no-go for the whole plan.**

## Phase 1 — Constants & boundary done right (~1 week)

**D1.1 Upstream PR** to neuralgcm/dinosaur parameterizing `scales.py`
(planet-constants factory instead of module constants). Fallback if declined:
keep the shim + version pin, documented.
**D1.2 Topography.** MOLA orography → truncation-filtered spherical-harmonic
coefficients → dycore lower boundary (same path their ERA5 notebook feeds
Earth orography).
*DoD:* dry Mars core with MOLA topography stable for 100 sols at T42/L25;
upstream PR opened; surface-pressure field shows the Hellas/Olympus contrast
(the −50 Pa problem from our 0-D benchmark, now resolved by construction).

## Phase 2 — The CO₂ mass cycle (~2 weeks — the genuinely new numerics)

Spectral cores assume fixed total dry mass; Mars condenses ~25 % of its
atmosphere seasonally. This is the one dycore-internal change:

**D2.1** Subclass the primitive-equations module: add a surface mass
source/sink S(θ_s, p_s) to the ln(p_s) tendency, active where surface T <
T_frost(p) (Clausius–Clapeyron — the same B2 form as the 0-D model), with
latent heating into the lowest layer.
**D2.2** New 2-D grid-space prognostic: surface frost reservoir M_frost(λ, φ)
(no spectral transform — it's a surface field, sidestepping ringing).
**D2.3** Port the PR #33 ledger discipline to 3-D: global ∫(p_s/g)dA + ΣM_frost
+ escape·t conserved; extend `engine/diagnostics.py`'s budget-residual pattern.
*DoD:* an annual T42 run **reproduces the Viking-shaped seasonal pressure
cycle emergently** (the same 610↔900 Pa target the 0-D model chases by
calibration); mass budget closes to float tolerance; `/math-review` on the
source-term derivation.

## Phase 3 — Mars physics package (~3–4 weeks)

`mars3d/physics/` behind a JCM-style column interface
(`physics(column_state) → tendencies`); **our schemes, not SPEEDY's**
(SPEEDY is Earth moist physics — none of it transfers):

- **D3.1 Radiation:** two-stream, CO₂ 15 µm band + dust opacity (gray to
  start); validation targets and band data from the Ames `Rad/` scheme
  (Track E task 49 synergy — same distillation targets as ML2).
- **D3.2 Surface:** energy balance with TES albedo + thermal-inertia maps,
  frost albedo when M_frost > 0 (smooth gate — reuse the tanh pattern from
  the smooth-gates PR).
- **D3.3 Dust:** 1–2 tracers, **grid-space** transport (spectral advection of
  sharp fields rings/goes negative), simple lifting + sedimentation,
  radiative coupling. This is the fidelity jump the 0-D model can never make.
- **D3.4 Diffusive boundary layer.**
*DoD:* zonal-mean T within ~10–15 K of MCD climatology; thermal tides now
**emergent** — compare amplitude/phase against our prescribed 30 Pa overlay
(closing the loop on the tide work); the standing `vs_gcm.py` benchmark
harness gains a `mars3d` column beside tform-0D and Ames.

## Phase 4 — Plugging into the terraforming framework (~1–2 weeks)

How `mars3d` connects to this repo — the integration contract:

**D4.1 Repo mechanics.** New uv-workspace member `mars3d/` (JAX deps isolated
from `package/`'s torch); its own CI job. **Strict framework boundary:** JAX
inside `mars3d`, PyTorch inside `package/` — they meet only at the experiment
layer via zero-copy DLPack (`torch.utils.dlpack` ↔ `jax.dlpack`) or numpy.

**D4.2 Interventions plug in through the C1 effect container.** The
`Intervention` ABC's effect fields map 1:1 onto 3-D inputs:
ΔF_radiative → column forcing in D3.1's radiation; Δalbedo → the surface
albedo *field* (latitude-targeted cap darkening becomes real, not a 0-D
hack); GHG/nanorod injections → well-mixed absorber concentrations;
escape_multiplier → the (tiny) escape sink. One intervention definition, two
backends.

**D4.3 Objectives evaluate on both backends.** E1 losses consume named
diagnostics (T_surf, p_s, frost mass, liquid-water indicator); on `mars3d`
the habitability metric becomes *spatial* — "fraction of surface area where
liquid water is stable," the honest version of the headline objective.

**D4.4 The model ladder.** tform-0D (design/optimization, ≪1 s/year, batched)
← calibrated against → `mars3d` (differentiable mid-fidelity, ~minutes/year)
← validated against → Ames GCM / MCD (reference, CPU-hours). `mars3d`
rollouts become: (a) the teacher for the 0-D learned closure (ML3 stage 2),
(b) the observation space for the spatial world model — **supersedes the B7
banded EBM** as JEPA v2's substrate if timing aligns, (c) the verification
tier for schedules optimized on the 0-D model before anyone cites them.

**D4.5 ML4-style calibration transfers.** `mars3d` is differentiable, so the
gradient-calibration-to-Viking/REMS demo runs on it unchanged in spirit —
at a fidelity tier where beating Ames at held-out lander sites (Track E
task 44) is a genuinely strong claim.

## Phase 5 — Deliverables & stretch

- **Paper candidate:** "A differentiable Mars general circulation model" —
  nobody occupies this niche (NeuralGCM is Earth; MAFM is data-driven); it is
  also the strongest possible artifact for the parked foundation-model track.
- Gradient-through-dust-storm sensitivity, spatial intervention design
  (where to darken, where to inject), BPTT policies over fields.

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Upstream rejects the scales PR | Shim + pin (already the Phase-0 mechanism); revisit each release |
| Semi-implicit reference profile unstable at Mars temperatures | Retune reference T in D0.2 — known knob, not research |
| Dust tracer numerics (ringing/negativity) | Grid-space transport from day one (D3.3) |
| JAX↔torch friction at the bridge | DLPack only at the experiment layer; no cross-framework imports in cores |
| Scope creep into the ICLR window | Hard gate: nothing here starts before submission except the 1-day D0 spike, which is allowed as a background curiosity |

## Dependencies on current work

Everything this plan reuses lands on `main` first: PR #33's ledger discipline
(→ D2.3), B2 frost point (→ D2.1), C1 intervention ABC (→ D4.2), the
`vs_gcm.py` harness (→ D3 DoD), A6 functional core (makes the 0-D physics
portable for cross-checks).
