# One-Mars-year surface albedo experiment

The full comparison uses 200,586 steps at 295.91748 s: 668.62 sols, just over
one Mars orbital period. A 365-sol run would cover only part of a Mars year.
It retains T21/L12, current full physics, the same initial state construction,
2,048 spatial albedo controls, and the one-sol study's albedo bounds.

The primary objective is the area-weighted surface temperature averaged over
every integration step of the year. End-of-year temperature is also reported for
each evaluated candidate. No spinup interval is discarded. This is a seasonal
control experiment from a common cold start, not an equilibrium calculation or
observational calibration. Annual means and final-day temperatures are distinct
objectives; compare methods within the same declared objective.

The runner first audits derivatives at 5 and 30 sols, then audits the annual
objective and compares gradient ascent, diagonal CMA-ES and uniform changes.
Each method has an initial budget of 16 forward calls. Gradient ascent also
requires backward passes. A seeded, unit-norm direction is checked at three
central-difference scales (1e-3, 1e-4, 1e-5). At least two scales must agree with
the directional AD result within 2% relative / 1e-6 absolute tolerance. This
does not certify all phase-boundary sensitivities or climate response.

Block rematerialization keeps integration block boundaries and recomputes
intermediate states during differentiation. No gradients are truncated or
detached between blocks. Each block has 300 steps, with an exact final remainder.

An invalid annual gradient is recorded and prevents gradient optimization;
the forward-only baselines still run. Failure of a shorter audit does not erase
its artifacts or skip the requested annual audit. Every optimization evaluation
saves progress and the current best field. No elapsed-time budget or deadline
stops a running workload.

The new source is archived by the existing Nautilus runtime. It reuses the
completed spatial study's dependency environment and verified input PVC, writes
the resolved package versions, and never changes the old results. The Job uses a
manual retention finalizer to block cluster-injected completed-Job cleanup.

This addresses a longer-horizon albedo test. It does not satisfy the separate
request for calibration across multiple continuous multiyear windows. A usable
annual sensitivity map and a practical gradient advantage are outcomes to test,
not assumed acceptance criteria.
