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

The original GPU campaign in `experiments/reviewer-validation/` completed.
Its source is `d9247914c7bb71c8611b00921c9295bf83d48b24`. Its annual results
are the baseline before the surface mass-transport correction, not results
from that correction. The full histories, configurations, initial arrays,
reports and checksums are in the
[public evidence release](https://github.com/BioMedAI-UCSC/terraforming/releases/tag/reviewer-current-physics-20261114).
The three `*-manifest.json` files identify each archive and every member.

At exactly shared times over one adopted Mars year (59,356,800 seconds),
300/150-second runs have 0.110466 K time-weighted RMS separation in global-mean
surface temperature and 4.262006 Pa in global-mean surface pressure. These
global statistics do not measure local field error. Sampled peak Mach is
1.10483, 1.50529 and 3.15847 at 450, 300 and 150 seconds respectively.
All runs finish with finite values, but these wind extremes prevent a claim
of physically stable annual climate. The T170/L40 default-physics run finishes
2,000 steps at 150 seconds (3.38 sols); it does not test full-physics T170
equilibrium. Projected mass drift is reported separately from raw mass,
energy and momentum corrections.

The paired geographic winter control, source
`ae06cda6b5766fd2afccb5ae15f1f37a5fab7c6d`, runs 1,332,000 seconds at Ls270.
Initial arrays, MOLA/TES/dust inputs and forcing match; only atmospheric CO2
phase change differs. Final worst model frost deficit decreases from
9.464689 K to 5.150525 K, a 4.314165 K (45.58%) reduction. Final maximum wind
increases from 175.668958 to 188.268557 m/s. This is mixed mechanism evidence,
not observed skill. `polar-analysis/` contains checked geographic fields,
maps with shared color scales, and the complete summary. Reproduce after
extracting `polar-control.zip`:

```sh
rtk proxy .venv/bin/python scripts/analyse_polar_control.py /path/to/extracted/polar-control \
  --output /tmp/polar-analysis
```

The five-sol forecast preflight at that source failed at 1e-3/1e-4 finite
differences, although all three controls pass at 1e-5. The original checks,
provenance and frozen Python sources are retained in
`multiday-preflight-failed.zip`. No fitted five-sol skill is claimed.

The surface mass-transport correction and conservation tests are documented in
`docs/ames-surface-mass-transport.md`. New annual and five-sol campaigns use
source `361bff80d9101009eb8b12a0f29215ce58811135`. The earlier source
`1f8fc20e2dc02e7606f1eb48c3b60338551b0f5d` has the mass-transport correction
but inconsistent frost capacity for radiation; its runs remain intermediate
controls. Evaluate the complete correction before claiming improved annual
stability or predictive accuracy.
