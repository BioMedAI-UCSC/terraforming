# Run the temporary aerosol integration branch on an SSH GPU server

Branch: `integration/aerosol-gpu-experiments`. This combines open PRs #55 and
#111–#114, the merged GCM fixes on `main`, and merged experiment-tool PRs
#108–#110 from their stack base. Original PR branches are unchanged.
`validation/aerosols/integration-sources.json` records the integrated revisions.
No manuscript or PDF is changed and no simulated outcome is prescribed.

## Set up the server

Use a Linux NVIDIA GPU machine with a driver supported by the pinned JAX CUDA 12
wheels, Python 3.12+, `uv`, Git, and enough disk for the environment and checkpoints.
Start with one GPU and 32 GB host RAM; neither a peak-memory requirement nor a
runtime benchmark has been established for these experiments. Float64 throughput
matters, so A100/H100-class hardware is preferable for long runs. A consumer GPU
can be used for short pilots but may have much lower float64 throughput.

```bash
git clone --branch integration/aerosol-gpu-experiments https://github.com/BioMedAI-UCSC/terraforming.git
cd terraforming
bash scripts/setup_aerosol_gpu.sh
```

The setup installs locked project dependencies and then adds CUDA support to the
same JAX version. Subsequent commands deliberately use `.venv/bin/python`:
`uv sync` or ordinary `uv run` can restore the lockfile's CPU-only installation.
If your server requires a different CUDA build, install the matching extras for
the same JAX version and verify `jax.devices()` before running.

## First run: measure a short pilot

Planning only writes configurations and does not import JAX or run a model.
This example uses about one hour of spinup and one hour of intervention, solely
to check installation, validity gates, outputs, and wall-clock cost.

```bash
.venv/bin/python scripts/aerosol_paper_experiments.py plan \
  --output outputs/aerosol-quick --sols 0.04 --spinup-sols 0.04

tmux new -s aerosol
CUDA_VISIBLE_DEVICES=0 bash scripts/run_aerosol_gpu.sh outputs/aerosol-quick \
  --case loading_2mg > outputs/aerosol-quick/job.log 2>&1
```

Detach with Ctrl-B then D; reconnect with `tmux attach -t aerosol`.
Alternatively, start the wrapper using `nohup ... &`. `timing.json` records wall
time including compilation, diagnostics, checkpoints and output. Measure a longer
pilot before extrapolating annual cost: compilation dominates very short runs.

## Run the paper comparisons

```bash
.venv/bin/python scripts/aerosol_paper_experiments.py plan \
  --output outputs/aerosol-10sol --sols 10 --spinup-sols 10 \
  --truncation T21 --layers 12 --dt 60 --material aluminium

CUDA_VISIBLE_DEVICES=0 bash scripts/run_aerosol_gpu.sh outputs/aerosol-10sol \
  > outputs/aerosol-10sol/job.log 2>&1
```

Each of the following experiments runs matched control, passive-particle and
radiatively active cases from exactly the same spun-up climate. Experiments run
sequentially, while one directory lock prevents concurrent jobs from corrupting
the shared spinup/restarts.

| Experiment | Initial particles | Release | Paper comparison |
|---|---|---|---|
| `loading_2mg` | 2 mg/m² globally | None | ~4 K at the reference loading; our loading subsequently settles |
| `release_2l_s` | None | 2 L/s throughout | ~4 K assuming one-year effective lifetime |
| `release_60l_s` | None | 60 L/s throughout | >35 K assuming one-year effective lifetime |
| `shutdown_2l_s` | None | 2 L/s, stopped halfway through | Track removal and response after shutdown |

Use `--case release_2l_s` (repeat `--case` to select several) to run a subset.
Use a separate planned directory with `--material iron` for the metal comparison.
The paper indicates similar volume fluxes for iron; it does not establish exact
equivalence of warming between materials. Graphene is not run because validated
input optics are not available in this implementation.

The dotted inventory curves use **one Earth year by default**, an explicit
reference convention. Plan another suite with `--lifetime-year mars` to compare
that alternative convention. The paper's rounded release and loading estimates
are separate references: the script does not force them to agree by silently
changing density or lifetime. The one-box curves assume `dM/dt = release - M/tau`;
the actual GCM computes transport and settling and does not impose this lifetime.

The 2 mg/m² initial layer is roughly 288,000 tonnes globally, already present at
the fork. It is a loading-response experiment, not a simulation of manufacturing
that inventory. The source experiments build their inventories from zero.

## Resume, stop, and inspect results

Rerun the same wrapper command to continue from the latest complete checkpoints.
Completed cases are skipped. SIGINT/SIGTERM request a graceful stop at the next
step boundary. SIGKILL or a scheduler hard time limit can only preserve the last
written checkpoint; leave time for graceful shutdown. Source commit, working-tree
diff and suite changes are rejected on resume. Keep the output directory at its
original path, since checkpoint configuration includes the physical-start path.

```bash
.venv/bin/python scripts/aerosol_paper_experiments.py plot --output outputs/aerosol-10sol
```

Plotting works on completed and partial histories and performs no simulation.

- `paper_comparison.png`: active-minus-control warming, simulated loading versus
  the assumed-lifetime curve, passive-control temperature check, and material residuals.
- `paper_comparison.csv` / `.json`: final warming/loading, duration, reference
  assumptions, raw transport-error diagnostic and paired time-mean warming.
- `annual_warming.png`, once complete years exist: the latest three complete
  annual warming means compared with the paper references. Partial years do not count.
- Each experiment: `history.ndjson`, `configuration.json`, `restart.npz`,
  `summary.json`, `results.nc`, `maps.png`, and `warming.png` when available.
- `runtime.json`: source revision, dirty-diff hash, JAX version and actual GPU model.
- `spinup/physical-start.npz`: shared climate fork; its existence does not establish
  equilibration. `spinup/history.ndjson` records its evolution.

Copy the output directory back using `scp -r` or `rsync` for analysis.

## Longer runs and interpretation

Ten sols is a transient screening run. It cannot confirm the paper's climate
warming or one-year lifetime. An example longer plan, after inspecting the pilot:

```bash
.venv/bin/python scripts/aerosol_paper_experiments.py plan \
  --output outputs/aerosol-multiyear --sols 2675 --spinup-sols 1338 --dt 60
CUDA_VISIBLE_DEVICES=0 bash scripts/run_aerosol_gpu.sh outputs/aerosol-multiyear \
  --case loading_2mg --case release_2l_s --case release_60l_s \
  > outputs/aerosol-multiyear/job.log 2>&1
```

This requests about two Mars years of spinup and four of intervention, **not a
certification that either interval is sufficient**. Check seasonal convergence
of the control, complete annual warming means in the history, and timestep/grid
sensitivity before interpreting climate effects. A Mars year at dt=60 s costs
about one million steps per climate, or three million per matched experiment.
All four experiments cost about twelve million climate steps per Mars year,
before spinup. Measure throughput on your server rather than assuming a GPU time.

Inventory-safe wind/temperature damping is explicitly enabled with the same
timescale convention as the merged Mars-map runner. It leaves pressure and
particle tracers unfiltered. This setting is archived in configurations and the
physical-start contract; new suite starts are required for different settings.

This remains a flat-terrain, emissivity-1 screening model with fixed gas
composition. Atmospheric CO2 condensation and full dust/cloud/removal feedbacks
are not validated. The runner may stop at its temperature, pressure, loading or
transport-error validity gates. Inspect the saved failure diagnostics instead of
weakening gates to obtain a warming number. The ~4 K and >35 K lines are reference
assumptions from [§3.3.1, pp. 24–26](https://arxiv.org/pdf/2604.02242), not pass/fail
thresholds or claimed reproductions of the paper.
