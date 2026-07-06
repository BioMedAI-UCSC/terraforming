# ICLR 2027 Paper Direction

**Target:** ICLR 2027 — full-paper deadline expected ~September 16–24, 2026
(based on ICLR 2026's Sept 19 abstract / Sept 24 paper pattern; confirm at
https://iclr.cc when the CfP posts). Working budget: **~11 weeks** from July 1.

**Template paper:** [JAX MD: A Framework for Differentiable Physics](https://arxiv.org/abs/1912.04232)
(Schoenholz & Cubuk, NeurIPS 2020).

---

## The JAX MD recipe

The paper we are emulating is a *framework paper*. Its structure:

1. **One core claim** — physics simulation made end-to-end differentiable and
   composable with ML, so entire trajectories can be optimized through.
2. **Scalability primitives** — spatial partitioning, vectorization; benchmark
   curves showing throughput on a single GPU.
3. **Three demo applications** — each a short, self-contained result showing a
   capability that is impossible or painful without the framework
   (GNN-in-the-loop, meta-optimization through a trajectory, multi-agent).
4. **Open source** with a clean API that appears verbatim in the paper.

## Working title / thesis

> **Differentiable Planetary Engineering: gradient-based design of
> terraforming interventions on a GPU-batched reduced-order Mars simulator.**

Terraforming is a *design/control problem over century-scale, irreversible
dynamics*. Because the simulator is differentiable, intervention schedules
(greenhouse-compound release rates, albedo modification, magnetic shielding)
can be optimized by backpropagating through decades of simulated climate —
and the same environment doubles as a long-horizon control benchmark for RL.

That framing matters for venue fit: ICLR has no systems/frameworks track, so
the paper must lead with the **ML contributions** (differentiable design,
benchmark, hybrid neural-physics) and present the framework as the vehicle.

## Asset inventory (what exists, July 2026)

- Reduced-order coupled ODE model of Mars: state = (T_surf, P_surf, M_ice),
  orbital mechanics with eccentric anomaly / L_s, solar zenith flux model,
  CO2 frost condensation/sublimation, MAVEN-calibrated escape, greenhouse
  forcing ([mars.py](../../package/src/celestials/planets/mars.py), ~500 loc).
- Engine: RK4 + fast analytic modes ([time_controller.py](../../package/src/engine/time_controller.py)),
  stacked-tensor GPU batching ([batched_controller.py](../../package/src/engine/batched_controller.py)).
- Interventions: compound registry + radiative forcing ΔF → greenhouse factor
  ([forcing.py](../../package/src/interventions/forcing.py),
  [compounds.py](../../package/src/interventions/compounds.py)), magnetic field stub.
- CLI, mkdocs site, strong test discipline.

## Gap analysis vs the JAX MD bar

| JAX MD ingredient | Status here | Work needed |
|---|---|---|
| Differentiability actually used | **Absent** — PyTorch backend but zero `backward()`/`requires_grad` anywhere; snapshots likely `.item()` and in-place state updates will cut the graph; CO2 frost point is a hard branch | Gradient-safe simulation path + smoothed (soft-threshold) phase transitions + `gradcheck` tests |
| Scalability story | Batched controller exists, no numbers | Throughput benchmark: planets/sec vs batch size, CPU vs GPU, both integrators |
| Demo applications | None | 3 demos (below) |
| Validation | Constants from NASA fact sheet; no comparison to published models/observations | Reproduce Viking/MSL annual T & P cycles; cite McKay/Zubrin, Wordsworth for terraforming physics |
| Open source + clean API | Repo exists, private | Public release with paper |

## The three demos (paper sections 4.1–4.3)

1. **Gradient-based intervention design.** Optimize a time-varying
   super-greenhouse-gas release schedule (kg/s of SF6/PFC mix) to reach a
   target (e.g. sustained T > 273 K at the collapse-prone season) at minimum
   total mass. Compare adjoint/backprop gradients vs CMA-ES and random search
   on wall-clock and sample efficiency. *This is the headline result.*
2. **Terraforming as a long-horizon control benchmark ("MarsGym").**
   Gym-style wrapper; the interesting properties are irreversibility
   (atmospheric escape, CO2 collapse), multi-decade delayed reward, and
   cheap batched rollouts on GPU. Baselines: PPO/SAC vs the differentiable
   planner from demo 1.
3. **Hybrid neural-physics.** Either (a) a learned closure: NN correction term
   trained so the fast reduced-order mode matches RK4 (or Mars Climate
   Database data), trained *through* the integrator; or (b) autodiff
   sensitivity analysis / uncertainty quantification over batched parameter
   ensembles. Pick (a) if time allows — it is the stronger ML result.

## Timeline (11 weeks)

| Weeks | Milestone |
|---|---|
| Jul 1–14 | Differentiability audit & repair: functional state (no in-place/`.item()` in grad path), smooth frost-point transition, `torch.autograd.gradcheck` tests, differentiable intervention parameterization |
| Jul 15–31 | Demo 1 complete with baselines + figures |
| Aug 1–14 | Demo 2 (gym wrapper + RL baselines); start demo 3 |
| Aug 15–31 | Demo 3, scaling benchmarks, observational validation section |
| Sep 1–14 | Writing, figures, repo cleanup for release, internal review |
| Sep 15–24 | Buffer → submit |

## Risks and mitigations

- **"Is this ML?" rejection risk.** Lead with demos 1–3, not the simulator.
  Related-work anchors: JAX MD, Brax, PhiFlow/DiffTaichi, gradient-based
  climate-model tuning, science-of-deep-learning benchmarks.
- **Physics credibility.** A 3-state model will draw fire from anyone who
  knows Mars. Frame explicitly as a *reduced-order energy-balance model*
  (a respected class), validate against observed annual cycles, and cite the
  terraforming literature for parameter choices. The math-review skill in
  this repo should audit every equation before submission.
- **Gradients through discontinuous physics.** CO2 condensation and escape
  are switching processes; smoothing them differentiably is itself a
  presentable contribution, not just plumbing.
- **Slip plan.** If demos slip past early September: NeurIPS 2027 D&B track
  or ICML 2027 — the D&B framing fits a framework paper even better.

> **Discrete task breakdown with exact file/line targets:**
> [iclr2027-workplan.md](iclr2027-workplan.md)
> **Background vs Mars GCMs (Ames, MarsWRF, LMD PCM) and positioning:**
> [mars-model-landscape.md](mars-model-landscape.md)
> **Full framework feature spec (tiered P0–P3):**
> [differentiable-framework-features.md](differentiable-framework-features.md)
> **ML features (JAX MD analogs):**
> [ml-features-jaxmd-analogs.md](ml-features-jaxmd-analogs.md)
> **ADR — PyTorch vs JAX/Julia/C++:**
> [adr-language-framework-choice.md](adr-language-framework-choice.md)
> **NASA AmesGCM/CAP comparison & what to borrow:**
> [amesgcm-comparison.md](amesgcm-comparison.md)

## Immediate next actions

1. Run a 1-day spike: `requires_grad_()` on an intervention magnitude, roll
   30 sols, call `backward()` — inventory everything that breaks.
2. Decide demo 3 variant (learned closure vs UQ).
3. Confirm ICLR 2027 CfP dates when published.
