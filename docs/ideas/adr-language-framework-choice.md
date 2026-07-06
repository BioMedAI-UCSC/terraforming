# ADR: PyTorch/Python vs JAX vs Julia vs C++

**Status:** decided — stay on PyTorch through the ICLR 2027 submission; adopt
a JAX-shaped architecture (functional core, feature A6) so a later port is
mechanical. Revisit after submission.

**Date:** July 2026. **Context:** ~2,200 lines of working PyTorch (tests,
CLI, docs), 4 line-level fixes away from end-to-end differentiability,
~11 weeks to the deadline.

---

## What the workload actually is

Not molecular dynamics: a **tiny stiff-ish ODE (3–4 states, ~20 after the
banded EBM) integrated for very many steps (16,488/Mars-year at dt=1 h),
batched over designs/parameters, with BPTT**. This profile drives everything:

- Per-step kernels are microscopic → in eager PyTorch, **kernel-launch and
  Python-loop overhead dominates**, not FLOPs. This is PyTorch's worst case
  and JAX's best case (`jit` + `lax.scan` compiles the entire rollout into
  one XLA program with zero Python in the loop).
- **Batching rescues PyTorch**: at B=4096, every per-step op works on [B]
  tensors, so launch overhead amortizes across the batch. Single-sim latency
  will stay mediocre; batched throughput (the paper's scaling figure) will be
  good. The framework story is batched anyway.
- `torch.compile(mode="reduce-overhead")` + CUDA graphs (already wired in,
  `time_controller.py:125-129`) recovers much of the rest; experimental
  `torch.while_loop`/scan higher-order ops exist if needed later.

## Options

### Stay: PyTorch + Python  ✅ chosen
- **For:** zero rewrite cost; all workplan Phase-1 fixes are line-level;
  team knowledge; RL baselines (SB3/CleanRL) native; `torch.func` provides
  vmap/jacrev/hessian once the functional core (A6) lands; largest reviewer
  familiarity.
- **Against:** eager loop overhead (mitigated by batching + compile); no
  true `scan` on the stable API; BPTT memory needs manual checkpointing (A4)
  where JAX has `jax.checkpoint`/adjoint conveniences.

### Rewrite: JAX
- **For:** technically the best fit for this exact workload (jit+scan whole
  rollouts, native vmap/grad/checkpoint, `diffrax` for adjoint ODEs);
  maximal thematic alignment with the JAX MD template.
- **Against:** 2–3 weeks of rewrite + revalidation out of an 11-week budget,
  spent reproducing results we already have; every test, the CLI, and the
  batched controller need porting; the paper's claims (differentiability,
  batched design, closures) are framework-agnostic — **no reviewer accepts or
  rejects this paper because of the autodiff framework**. Rewrite risk is the
  single most likely way to miss the deadline.

### Rewrite: Julia (SciML)
- **For:** honestly the best-in-class ecosystem for *small differentiable
  ODEs* — DifferentialEquations.jl's stiff solvers, adjoint sensitivities,
  event handling (frost-point switching!) are exactly our physics.
- **Against:** wrong audience for an ICLR framework paper (reviewer pool and
  citation graph live in Python); RL tooling weaker; two-language project if
  the ML layer stays in Python. Would be the right call if the target were
  a SciML/JuliaCon venue instead.

### Rewrite: C++ (+ bindings)
- **Against, decisively:** we'd hand-build autodiff or bolt on Enzyme; dev
  velocity collapses; the model is 3 ODEs — there is nothing for C++ to be
  fast *at* that batching doesn't already solve. No credible upside for this
  project. (A C++ core is how you'd productionize a mature framework years
  from now, not how you get a paper by September.)

## Decision & hedge

1. **PyTorch through submission.** All P0/P1 work proceeds as planned.
2. **Build A6 (functional pure-step core) early** — `step(state, params, t,
   dt) -> state` with explicit params. This is simultaneously (a) what
   unlocks `torch.func` transforms in PyTorch, and (b) a design that is
   line-for-line portable to JAX (`pack_state`/`unpack_state` already define
   the pytree boundary). The hedge costs nothing because A6 is on the
   critical path anyway.
3. **Optional stretch, only if Phases 1–3 land early:** port the ~200-line
   pure step to JAX behind the same tests and add a JAX column to the
   throughput figure. A cross-framework benchmark *strengthens* the paper;
   a mid-flight migration endangers it.
4. **Revisit after submission.** If the framework grows (banded EBM, learned
   transport, 10³-year adjoint runs), JAX becomes the natural v2 target and
   the A6 core makes that a port, not a rewrite.

## One perf trap bigger than the language choice

Default dtype is float64 (`TF_DTYPE`). On consumer NVIDIA GPUs FP64 runs at
**1/32–1/64 of FP32 throughput** (fine on A100/H100). Feature A9 (dtype
policy) will likely buy more speed than any framework migration — measure it
before attributing slowness to PyTorch.
