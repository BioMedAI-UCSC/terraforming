# Strategy — implementing NeuralGCM's Dinosaur dycore into terraforming

Definitive strategy for building a **differentiable 3-D Mars GCM (`mars3d`)** on
[neuralgcm/dinosaur](https://github.com/neuralgcm/dinosaur) (Apache-2.0, active —
v1.3.6, Jun 2026) and plugging it into this framework. Grounded in Dinosaur's
actual API (verified from source). Companion to
[master-task-list.md](master-task-list.md) (the "v2: 3-D differentiable" item).
Timing: **post-ICLR submission** (Oct 2026 →), ~8–10 person-weeks. Coupling
template: JCM v1.0 (Dinosaur-as-dependency + swappable physics).

---

## 1. The decision (settled)

**Depend, don't fork. Subclass, don't rewrite. Bridge, don't merge frameworks.**

- **Dinosaur is a pip dependency** (`jax, jaxlib, numpy, pint, xarray` — clean,
  Python ≥3.10). The repo is active; a hard fork would rot. We pin a version.
- **`mars3d/` is a new JAX workspace member**, isolated from `package/` (torch).
  The two meet only at the experiment layer via DLPack — never cross-import.
- **Three surgical touch points** into Dinosaur; everything else is unchanged
  upstream code or new Mars code we own.

## 2. What Dinosaur gives us (verified from source)

**Prognostic state** (`primitive_equations.State`): `vorticity`, `divergence`,
`temperature_variation` (anomaly from a reference T profile), `log_surface_pressure`,
`tracers: Mapping[str, Array]`, `sim_time`. Two things matter for Mars:
- `log_surface_pressure` → surface pressure is **positive by construction** (no
  clamp, no dead gradient — contrast our 0-D `P.clamp(min=0)`).
- `tracers` dict → **dust drops straight in** as a tracer.

**The coupling seam** (`time_integration.ImplicitExplicitODE`): the whole model
is `∂x/∂t = explicit_terms(x) + implicit_terms(x)`, with a semi-implicit split —
fast gravity waves go in `implicit_terms` (solved via `implicit_inverse(state,
dt)`), everything else (advection **+ all physics**) in `explicit_terms`.
`PrimitiveEquationsSigma` is the concrete sigma-coordinate solver we build on.
**Our Mars physics plugs in by contributing tendencies to `explicit_terms`** —
this is the entire integration contract.

**Hooks already present** (no new numerics needed to reach them):
- `nodal_log_pressure_tendency(...)` — where the **CO₂ frost source/sink** goes.
- `orography_tendency()` — **topography already supported**; feed it MOLA.
- `coriolis_parameter`, `T_ref` — set from Mars constants.

**Infrastructure that transfers verbatim** (planet-agnostic, zero changes):
`spherical_harmonic` (transforms), `coordinate_systems` (nodal/modal duality,
resolution change, sharding — reviewed in depth), `sigma_coordinates`,
`time_integration`, `filtering` (hyperdiffusion), `held_suarez` (validation).

**The only Earth-hardcoded file:** `scales.py` — module constants `RADIUS`,
`GRAVITY_ACCELERATION`, `OMEGA`, `KAPPA=2/7`. This is the whole "Earth-ness" of
the dycore. One shim (or upstream PR) fixes it.

## 3. The three touch points

| # | Touch point | Dinosaur target | Effort |
|---|---|---|---|
| **T-A** | Planetary constants | `scales.py` — shim to Mars values, or upstream a planet-parameterized factory | days |
| **T-B** | Mars physics tendencies | subclass `PrimitiveEquationsSigma`; add our physics to `explicit_terms` | weeks (Phase 3) |
| **T-C** | CO₂ mass cycle | extend `nodal_log_pressure_tendency` with a frost source/sink + a 2-D frost reservoir + latent heat into `temperature_variation` | ~2 weeks (Phase 2, the one novel bit) |

Everything else = new code in `mars3d/` (physics schemes, interventions bridge,
objectives) that never modifies Dinosaur.

## 4. Phased plan

### Phase 0 — Go/no-go spike (2–3 days)
Pin Dinosaur; reproduce the Held–Suarez notebook. Add `mars3d/scales.py` with
Mars constants (RADIUS 3.3895e6, GRAVITY 3.72076, OMEGA 7.0779e-5 ≈ Earth's,
KAPPA≈0.257 for CO₂) and retune `T_ref` to Mars temps (~140–220 K, for the
semi-implicit linearization). Run a dry Mars Held–Suarez analog at T21/L20 for
30 sols. **DoD:** stable; a midlatitude jet forms; `jax.grad` of mean-T w.r.t. a
forcing parameter is finite through ≥1 sol. **This spike is the whole go/no-go.**

### Phase 1 — Constants & boundary (~1 week)
Upstream a `scales.py` parameterization PR (fallback: shim + pin). Feed MOLA
topography → truncation-filtered spherical-harmonic coefficients →
`orography_tendency`. **DoD:** 100-sol stable run at T42/L25; surface-pressure
field shows the Hellas/Olympus contrast (resolves the −50 Pa 0-D benchmark bias
by construction).

### Phase 2 — CO₂ mass cycle (~2 weeks — the genuinely new numerics)
Spectral cores assume fixed dry-air mass; Mars condenses ~25% seasonally.
- **T-C.1** Extend `nodal_log_pressure_tendency` with source/sink S(θ_surf, p_s)
  where T_surf < T_frost(p) (Clausius–Clapeyron — same B2 form as the 0-D model).
- **T-C.2** New 2-D grid-space prognostic `M_frost(λ,φ)` (a surface field — no
  spectral transform, so no ringing), with latent heat into the lowest layer's
  `temperature_variation` tendency.
- **T-C.3** Port PR #33's ledger discipline to 3-D: ∫(p_s/g)dA + Σ M_frost +
  escape·t conserved; extend the `engine/diagnostics.py` residual pattern.
**DoD:** an annual T42 run reproduces the **Viking-shaped seasonal pressure cycle
emergently** (the 610↔900 Pa target the 0-D model chases by calibration); budget
closes to float tolerance; `/math-review` on the source term.

### Phase 3 — Mars physics package (~3–4 weeks)
`mars3d/physics/` — **our schemes, not JCM's SPEEDY** (SPEEDY is Earth moist
physics; none transfers). Each contributes tendencies via `explicit_terms`:
- **Radiation:** two-stream, CO₂ 15 µm band + dust opacity (gray first);
  validation/band targets from the Ames `Rad/` scheme (shares ML2 distillation).
- **Surface:** energy balance with TES albedo + thermal-inertia maps; frost
  albedo when M_frost>0 (smooth tanh gate — reuse the smooth-gates pattern).
- **Dust:** 1–2 `tracers`, **grid-space** transport (spectral advection of sharp
  dust rings/goes negative), lifting + sedimentation, radiative coupling.
- **Boundary layer:** diffusive.
**DoD:** zonal-mean T within ~10–15 K of MCD; thermal tides now **emergent** —
compare amplitude/phase against our prescribed 30 Pa overlay (closes that loop);
`mars3d` joins the standing `vs_gcm.py` benchmark beside tform-0D and Ames.

### Phase 4 — Plug into terraforming (~1–2 weeks)
- **Repo:** `mars3d/` uv-workspace member, JAX deps isolated, own CI job.
  **Framework boundary is strict:** JAX in `mars3d`, torch in `package/`, meeting
  only at the experiment layer via zero-copy DLPack (`jax.dlpack ↔ torch.utils.dlpack`).
- **Interventions** plug in through the **C1 `Intervention` effect container**:
  ΔF_radiative → column forcing in Phase-3 radiation; Δalbedo → the surface
  albedo *field* (latitude-targeted cap darkening becomes real); GHG/nanorod →
  well-mixed absorber concentrations; escape_multiplier → the escape sink. One
  intervention definition, two backends (0-D torch, 3-D JAX).
- **Objectives** (E1) evaluate on both backends via named diagnostics; on
  `mars3d` the habitability metric becomes **spatial** ("fraction of surface area
  where liquid water is stable").
- **Model ladder:** tform-0D (design/optimization, ≪1 s/yr, batched) ←calibrated→
  `mars3d` (differentiable mid-fidelity, ~min/yr) ←validated→ Ames/MCD. `mars3d`
  rollouts become (a) the ML3 closure teacher, (b) the **spatial world-model
  substrate — supersedes B7's banded EBM** if timing aligns, (c) the verification
  tier for schedules optimized on the 0-D model.

### Phase 5 — Deliverables
Paper candidate: **"A differentiable Mars general circulation model"** — an
unoccupied niche (NeuralGCM is Earth; MAFM is data-driven) and the mechanistic
counterpart the foundation-model track lacks. Plus: gradient-through-dust-storm
sensitivity, spatial intervention design, BPTT policies over fields.

## 5. Dependencies on current work (must land on `main` first)

| Needs | Feeds |
|---|---|
| PR #33 pressure ledger + `diagnostics.py` | Phase 2 CO₂-cycle conservation (T-C.3) |
| B2 pressure-dependent frost point | Phase 2 source term (T-C.1) |
| PR #31 smooth-gates tanh pattern | Phase 3 frost-albedo gate |
| C1 `Intervention` ABC | Phase 4 intervention bridge |
| `vs_gcm.py` benchmark harness (Track E) | Phase 3 DoD |

## 6. Risks & mitigations

| Risk | Mitigation |
|---|---|
| Upstream rejects the `scales.py` PR | Shim + version pin (the Phase-0 mechanism anyway) |
| Semi-implicit unstable at Mars temps | Retune `T_ref` in Phase 0 — a known knob, not research |
| Dust tracer ringing / negativity | Grid-space transport from day one (Phase 3) |
| JAX↔torch friction | DLPack only at the experiment layer; no cross-framework imports in cores |
| Two frameworks in one repo | Accepted cost; the A6 functional-core work keeps 0-D physics portable for cross-checks; ICLR/0-D stack must not migrate |
| Scope creep into the ICLR window | Hard gate: nothing here starts before submission except the 1-day Phase-0 spike |

## 7. One-paragraph summary

Build `mars3d` as a JAX workspace member that **depends on** Dinosaur, **shims**
`scales.py` to Mars constants, **subclasses** `PrimitiveEquationsSigma` to add
Mars physics through `explicit_terms` and a CO₂ frost source/sink through
`nodal_log_pressure_tendency`, and **bridges** to the existing torch framework
via DLPack at the experiment layer — reusing the intervention ABC, objectives,
and conservation-diagnostics discipline already built for the 0-D model. Result:
the first differentiable Mars GCM, a mid-fidelity rung between tform-0D and the
Ames GCM, and the spatial substrate for the world-model track.
