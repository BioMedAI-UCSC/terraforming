# ICLR result archive

This folder is a curated copy of the paper-facing results already produced in
the repository. Originals remain under `outputs/`; nothing was moved or deleted.
Large raw NetCDF trajectories, restart checkpoints and downloaded source archives
are intentionally excluded. The copied JSON/CSV/PNG/NPZ artifacts retain the
reported metrics, checksums, configurations and plots needed to interpret them.

The archive is organized by claim:

| Folder | Contents | Appropriate paper use |
|---|---|---|
| `01-mola-topography-comparisons` | MOLA/reference comparison reports, seasonal maps, topology/pressure plots, budget diagnostics | Topography and cross-reference figures; model/reference limitations must remain explicit |
| `02-ablations` | Short paper ablation suite, three Mars-year GPU partitions, audit metadata | Physics sensitivity, timestep and one-year stability evidence |
| `03-benchmarks` | Ames comparisons, MCD diagnostics, reference-comparison summaries, `performance-timing/` throughput/wall-clock measurements, and ICLR smoke manifests | Benchmark, performance and reproducibility tables |
| `04-neural-experiments` | Radiation, atmospheric-control and parameter-recovery reports | Framework neural-capability demonstrations |
| `05-paper-docs` | Protocol, limitations, architecture, neural-framework and experiment notes | Methods and claim boundaries |

## Recommended primary results

Use the complete `02-ablations/short-paper-suite/` report for the compact
physics-ablation table. The `mars-year-gpu1`, `mars-year-gpu2` and
`mars-year-gpu3` directories are separate complete one-year runs/partitions with
the same full case repeated; preserve their run identity and do not concatenate
their repeated `full` rows as independent samples. The year protocol is a
transient from a restart, not an equilibrated climate or observational forecast.

Use `01-mola-topography-comparisons/report.json` and `metrics.csv` for the
seasonal physical/Ames/MCD/ARCO comparison diagnostic. It records the pinned
MOLA SHA-256 and source provenance. The status explicitly says this is not
observational validation. `03-benchmarks/physical-vs-ames/` contains additional
benchmark JSON with the same limitation.

The strongest neural capability artifact is
`04-neural-experiments/radiation/report.json`, which compares local fitting,
trajectory fine-tuning and continued local fitting. The atmospheric-control
report demonstrates gradients through a bounded neural tendency. The
parameter-recovery report validates gradients and inverse optimization but is not
itself a neural result. These generated-data experiments do not establish Mars
forecast skill.

## Neural result snapshot

The recorded default atmospheric-control smoke run used T21, two layers, three
30-second steps, one seed and a synthetic `-0.2 K` target shift. It reduced target
RMSE from approximately `0.174 K` to `0.020 K`, remained finite, and respected a
`0.002 K/s` maximum heating bound. This is an integration demonstration. It has
no held-out initial state, observational target, MOLA topography or long-horizon
stability result.

The recorded default radiation run reports local column error, trajectory error,
held-out interpolation/extrapolation splits, energy closure, coupled T21 checks,
failure counts and synchronized first/warm timings. The continued-local control
performed better than trajectory fine-tuning in that run; do not describe the
trajectory objective as superior without new evidence.

## Provenance and interpretation rules

- Read each artifact's `report.json`, `config.json`, `environment.json`,
  `input-manifest.json` and `source-hashes.json` together.
- `benchmark.json` files labeled model/reference comparison are not observational
  truth. MCD and ARCO entries are diagnostics with their stated source and
  temporal/vertical reductions.
- RMSE values in ablation CSV files are relative to the run's full case, not
  independent truth validation. The full row is zero by construction.
- A finite trajectory demonstrates numerical completion for that configuration;
  it does not establish climate realism, equilibrium or extrapolation.
- Smoke runs are retained to show reproducibility and command wiring, but should
  not be mixed with paper-scale runs in aggregate tables.

The source paths copied into this folder are listed in `manifest.json`
(per-claim manifests live inside their subfolders, e.g.
`03-benchmarks/manifest.json`).

## Performance/timing benchmark

`03-benchmarks/performance-timing/` records synchronized warm-call plus
first-call GPU timings (`timing.json`/`timing.csv`): the forward trajectory
workload and the loss-and-three-forward-tangents workload, with median seconds,
simulated-sols-per-wall-hour and compiled temporary bytes. External simulator
timings are marked `not_measured`; no external speedup is claimed.
