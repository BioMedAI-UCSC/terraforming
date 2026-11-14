# Reviewer completion evidence

`compound-sources.json` records the primary IPCC table, source hashes, and the
published Marinova gray-fit equations and Table 1 data. `compound-audit.json`
checks all seven registry rows. These are Earth-reference trace-gas coefficients.
Their use for Mars or large concentrations is not validated. The published gray
fits do not reproduce every Table 1 warming within the reported fit errors;
the audit records this difference without tuning the coefficients.

`radiation/` contains a fresh current-physics seed-0 run and independent repeat.
Both use 100 initial updates, 20 trajectory updates, and 20 continued-local
updates. All 60 numerical arrays and report metrics agree at relative tolerance
1e-10 and absolute tolerance 1e-12. Only measured execution times and output
directory names are excluded. Physical timestep settings are compared.
Checkpoint metadata is compared as JSON, so key order has no effect.

The CPU is an Apple M3 Pro. The original reports keep the unavailable model
field; a separate hardware record gives the host query used to identify it.
Each saved source file matches the hash recorded at execution. The snapshot was
copied after execution and verified against those hashes. `manifest.json`
lists hashes and sizes of the archived files.

At 48 steps, air-temperature RMSE is:

| Test | Initial local | Trajectory continuation | Local continuation |
| --- | ---: | ---: | ---: |
| Interpolation | 0.6244153 K | 0.5704848 K | 0.5941652 K |
| High dust | 1.8504062 K | 1.7036034 K | 1.6491594 K |

These are generated-kernel, single-seed results. They do not establish observed
Mars skill or a general advantage for trajectory training.

Recheck from the repository root with the package environment:

```sh
PYTHONPATH=package python scripts/audit_ghg_coefficients.py --output /tmp/compound-audit.json
python scripts/verify_radiation_repeat.py \
  iclr-results/07-review-completion/radiation/radiation-seed0 \
  iclr-results/07-review-completion/radiation/radiation-repeat0 \
  --output /tmp/radiation-repeat-verification.json
```

The long GPU campaign is defined in `experiments/reviewer-validation/`.
Queued jobs are not scientific results. No long-run acceptance gate is marked
complete by this evidence bundle.
