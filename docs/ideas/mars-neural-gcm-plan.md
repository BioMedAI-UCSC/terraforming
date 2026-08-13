# Mars Neural GCM — a NeuralGCM-style hybrid model for Mars

> **Goal.** Build a Mars equivalent of NeuralGCM (Kochkov et al., *Nature* 2024,
> `s41586-024-07744-y`): a **differentiable Mars dynamical core** coupled to a
> **neural-network learned-physics** module, trained end-to-end by backprop
> through the solver. Reuse the JCM (`jax-gcm`) integration we already wired into
> `package/`, and keep the door open to **differentiable terraforming** (inverse
> design of interventions) as a later extension.

---

## 1. What NeuralGCM actually is (the template)

NeuralGCM splits an atmospheric model into two halves and makes the whole thing
differentiable:

1. **Differentiable dynamical core** — solves the primitive equations (spectral,
   in JAX, via the `dinosaur` library). This is *known physics kept as
   equations*: advection, pressure, gravity waves, the resolved fluid dynamics.
2. **Neural-network "learned physics"** — a per-column network that ingests the
   atmospheric column state and outputs **tendencies** (heating, moistening,
   drag). It replaces the traditional hand-coded parameterizations (radiation,
   convection, clouds, boundary layer) with a single learned correction.

**Training:** the two halves are coupled and the model is *unrolled* for many
steps; the loss compares the rollout to **ERA5 reanalysis**, and gradients flow
**through the differentiable dynamical core** back into the NN weights (online /
rollout training, not offline column fitting). This is what buys stability and
skill simultaneously.

**Result:** competitive with operational NWP at 1–15 day forecasts, orders of
magnitude cheaper, and stable for multi-decade climate runs with realistic
climatology and emergent phenomena.

**Why it maps onto our stack:** JCM wraps the *same* `dinosaur` dycore, exposes a
`Physics` interface with a learned/carry hook, ships `flax.nnx` NN modules and
GRU radiation emulators, and has demonstrated end-to-end gradients through
`model.run`. The architecture is already here — the work is Mars-specific.

---

## 2. Architecture — Mars Neural GCM

```
                 ┌────────────────────────────────────────────┐
                 │        MarsGCM (Planet interface)           │  ← package seam
                 │   TimeController → Accuracy.GCM → step_gcm  │    (already built)
                 └───────────────────┬────────────────────────┘
                                     │ drives
                 ┌───────────────────▼────────────────────────┐
                 │              jcm.model.Model                │
                 │  ┌────────────────────┐  ┌───────────────┐  │
   KNOWN PHYSICS │  │ Differentiable      │  │ Hybrid        │  │ NEURAL PHYSICS
   (equations)   │  │ Mars dynamical core │  │ Physics =     │  │ (learned)
                 │  │ dinosaur + Mars     │  │  baseline     │  │
                 │  │  constants/patches  │  │  + NN correction │
                 │  └────────────────────┘  └───────────────┘  │
                 └─────────────────────────────────────────────┘
                                     ▲
                                     │ trained by backprop through the solver
                 ┌───────────────────┴────────────────────────┐
                 │   Mars reanalysis / climatology targets     │
                 │   (OpenMARS / EMARS / MCD)                  │
                 └─────────────────────────────────────────────┘
```

Two tracks, developed in order: **(A) the physics** (a working differentiable
Mars dynamical core with a simple baseline physics), then **(B) the neural part**
(learned correction trained against data). A is the prerequisite; B is the paper.

---

## 3. Track A — the physics (differentiable Mars dynamical core)

This is the `marsphys` work from `differentiable-terraforming-framework.md`,
scoped to "what NeuralGCM keeps as equations."

### 3.1 Make the dynamical core Mars-correct
- **Constants** (`set_constants`): gravity 3.72, radius 3.39e6, Ω = 2π/sidereal
  day, CO₂ `cpd`/`akap`, `p0 ≈ 610 Pa`, `solc` from Mars orbit. (JCM Patch 3:
  thread constants instead of the Earth singleton.)
- **CO₂ mass tendency** (JCM Patch 1) — **the non-negotiable one.** Mars condenses
  ~25–30 % of its atmosphere onto the poles each year; Earth GCMs assume dry-mass
  conservation. Without a surface-pressure/mass tendency the seasonal cycle is
  impossible. This is the single biggest deviation from NeuralGCM's Earth setup.
- **Gravity leak** (JCM Patch 2) — fix Earth-gravity in the state bridge.
- **MOLA topography** as `TerrainData` — Mars's 3.6× spatial pressure range is
  topographic; the Held-Suarez `from_coords` flat terrain is not enough.
- **Orbital insolation**: e = 0.0934, obliquity 25.19°, Ls_perihelion 251°.

### 3.2 Baseline "known physics" kept as equations
Composed as a `ComposablePhysics([...])` (the Held-Suarez pattern):
- `co2_cycle` — condensation/sublimation + latent heat + mass tendency.
- `mars_radiation` — grey/two-stream CO₂ + dust (analytic, differentiable).
- `regolith` — subsurface thermal inertia coupling.
- `dust` — prescribed opacity τ(Ls, lat) for the baseline (later: learned).

**Gate A:** dry Mars stable for 1 Mars-year; seasonal surface-pressure cycle
matches Viking Lander 1/2 within tolerance (reuse the existing validation
harness). This is the credibility milestone *before* any ML.

---

## 4. Track B — the neural part (learned Mars physics)

### 4.1 The learned-physics module
Implement JCM's `Physics` interface with a `flax.nnx` network (mirroring
`nn_emulator.py`'s `DenseWeights`/`GRUWeights`):

```python
class MarsLearnedPhysics(nnx.Module):        # implements jcm Physics protocol
    def initial_carry_state(self, coords) -> PhysicsCarryState:
        ...   # per-column recurrent/stochastic memory (NeuralGCM-style)

    def compute_tendencies(self, state, forcing, terrain, prev_physics_data):
        # inputs: column T, u, v, tracers, p_s, insolation, dust, surface maps
        # outputs: PhysicsTendency (dT, du, dv, d tracers) + new carry
        ...
```

- **Per-column** network (like NeuralGCM): operates on each vertical column,
  applied across the grid — cheap and resolution-flexible.
- **Carry state** gives it memory → supports stochastic/recurrent corrections
  and sub-timestep processes.

### 4.2 Hybrid, not pure-NN (the Mars-data-scarcity adaptation)
NeuralGCM had 40+ years of hourly ERA5. **Mars has far less data** — this is the
central risk and it changes the design:

> **Learn the residual, not the whole thing.** Keep the Track-A baseline physics
> as a differentiable prior, and have the NN learn only the **correction** to it:
> `tendency = baseline_physics(state) + NN(state, carry)`.

This is more data-efficient and more stable than a pure learned physics, and it
degrades gracefully where data is thin. What the NN should absorb:
- **Dust radiative heating** — the dominant, highly variable Mars process (global
  dust storms); hardest to hand-model → best ML target.
- **PBL turbulence / regolith coupling** corrections.
- **Radiative-transfer** corrections beyond the grey baseline.
- **CO₂-cycle** timing/spatial corrections.

### 4.3 Training (backprop through the solver)
- **Targets (Mars-ERA5 analog)** — evaluate, in priority order:
  - **OpenMARS** (Holmes et al. 2020) — reanalysis of MGS/MRO, MY24–32.
  - **EMARS** (Greybush et al.) — ensemble reanalysis assimilating TES/MCS.
  - **Mars Climate Database (MCD)** — LMD GCM climatology (not obs, but dense and
    complete; good for pre-training / where reanalysis is absent).
  - *(All need availability/format/licence verification — see risks.)*
- **Encoder/decoder** between observation space and the spectral model state
  (regrid + variable mapping), as in NeuralGCM.
- **Loss**: rollout MSE (+ spectral / energy terms) against targets, unrolled N
  steps; gradients flow through `model.run` (proven: `jax.vjp` works).
- **Curriculum**: 1-step → short → long rollouts; the demonstrated
  `02_optimization_example` gradient path is the mechanism.
- **Pre-train on MCD climatology, fine-tune on OpenMARS/EMARS** to stretch the
  scarce reanalysis.

**Gate B:** hybrid model beats the Track-A baseline on held-out Mars reanalysis
(forecast skill + climatology), and stays stable over a full Mars year.

---

## 5. Terraforming extension (later)

Once the hybrid model is differentiable *and* fast, terraforming becomes an
**inverse-design** problem — the headline downstream result:

- Parameterize interventions (polar albedo change, PFC injection from the
  existing `interventions/compounds.py`, orbital mirrors) as differentiable
  forcings/`PhysicsTerm`s.
- Backprop through multi-year rollouts to **optimize an intervention schedule**
  toward a target climate (e.g. maximize equatorial habitable area, minimize PFC
  mass). The NN physics makes each rollout cheap; the differentiability makes the
  optimization tractable — neither is possible with the torch 0-D box model.

---

## 6. How it plugs into the existing framework

- The learned physics is just a `Physics` passed to `Model(physics=...)`.
- The `Model` is driven behind `MarsGCM` via the `Accuracy.GCM` seam already
  built in `package/` — so `TimeController`, `Snapshot`, the CLI, and plotting
  are unchanged; the 3-D→scalar reduction still feeds the existing diagnostics.
- The torch 0-D box model stays as a fast baseline under `FAST`/`ACCURATE`.

---

## 7. Phasing & GO/NO-GO gates

| Phase | Deliverable | Gate |
|---|---|---|
| 0 | JCM Patches 1–3; Mars constants; MOLA terrain | dry Mars stable 1 yr |
| 1 | Baseline `ComposablePhysics` (co2/radiation/regolith) | **Gate A**: seasonal p_s vs Viking |
| 2 | Data pipeline: OpenMARS/EMARS/MCD → spectral grid + encoder/decoder | reanalysis loads, regrids, round-trips |
| 3 | `MarsLearnedPhysics` (nnx) as residual correction; 1-step training | gradients flow, loss decreases |
| 4 | Rollout curriculum training | **Gate B**: beats baseline on held-out data, stable 1 yr |
| 5 | Inverse-design interventions (terraforming) | optimize a schedule end-to-end |

---

## 8. Risks (ranked)

1. **Data scarcity — the central risk.** Mars reanalysis is sparse and short vs
   ERA5. Mitigation: residual/hybrid design, MCD pre-training, heavy physics
   prior. If reanalysis proves inaccessible, fall back to MCD-only (learn a fast
   emulator of MCD) — weaker paper, still valid.
2. **CO₂ mass conservation** (Patch 1) — blocks the whole seasonal cycle; must
   land first and be verified.
3. **Training stability** through long differentiable rollouts (checkpointing,
   memory) — NeuralGCM's hardest engineering; JCM's `checkpoint_terms` helps.
4. **Compute** — JIT-compiled runs are fast per step (~0.3 s/day at T30 on CPU),
   but rollout training over a Mars year × batches wants a GPU.
5. **Dust variability** — global dust storms are episodic and hard to predict;
   may need stochastic learned physics (the carry state).

---

## 9. Immediate next actions

1. Land JCM Patches 1–3 + Mars constants; get `MarsGCM` stepping with a
   Mars-Held-Suarez placeholder physics (de-risks the pipeline with zero
   boundary data).
2. Stand up the baseline `co2_cycle` term with the mass tendency → hit **Gate A**.
3. In parallel, spike the **data pipeline**: can we obtain and regrid one
   OpenMARS/EMARS/MCD field onto `get_speedy_coords()`? This is the true
   feasibility test for the neural half.
```
