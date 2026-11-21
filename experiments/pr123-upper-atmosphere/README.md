# First five-sol upper-atmosphere evidence

Ticket #129, part of #127. Baseline source is immutable commit
`69d6587cef02e5b17a82e28422b67aa3d5be55ad`.

The completed CPU T21 case reached 1,920 steps. Its five-sol top mean zonal
wind RMS is 12.468694 m/s and mean temperature is 155.980018 K. T42 failed
the finite-state guard before the first 384-step checkpoint. Its only saved
diagnostic is the finite initial state; no restart was written before failure.
All 70 local files match remote hashes. T21 restart equality passed for all
59 arrays, and total CO2, water and dust inventory drift stayed below
2e-12 relative. The measured two-process host memory peak was
37,891,866,624 bytes.

`validation/pr123-five-sol-baseline.json` records common-pressure profiles,
budgets, source/input hashes and the failed T42 case.
`validation/pr123-nautilus-baseline.json` records completion and resource scope;
`validation/pr123-baseline-artifact-hashes.json` records every downloaded file.
The successful CUDA T21 one-sol probe used a TITAN RTX with 24,576 MiB,
peaked at 4,456 MiB and sampled 98% utilization during computation.
Its actual concurrency was one, as recorded in `validation/pr123-gpu-probe.json`.

The public experiment runner uses T21/T42, 56 layers, a 231-second step,
one-sol checkpoints, and the released reference physics. The CPU matrix runs
two cases at once. It compares all restart arrays after a T21 run that restarts
at one sol and an uninterrupted T21 run. Both finish at 1,920 steps.

The existing two-step CPU preflight passed T21, T42 and exact T21 restart.
Its verified files are in `outputs/pr123-validation/nautilus-cli-final` in the
main workspace. The completed five-sol results must pass the same checks.

Run the archived manifests in context `nautilus`, namespace `razvanlab`:
Use fresh job, ConfigMap and output names for a rerun of a completed case.

```sh
rtk kubectl --context nautilus -n razvanlab apply -f experiments/pr123-upper-atmosphere/five-sol-cpu.json
rtk kubectl --context nautilus -n razvanlab apply -f experiments/pr123-upper-atmosphere/gpu-probe.json
rtk kubectl --context nautilus -n razvanlab apply -f experiments/pr123-upper-atmosphere/native-coordinates.json
```

The manifests need the existing prepared PVCs. They do not install packages
on GPU resources. The init container verifies and extracts the CPU-prepared
archives. These contain the pinned Python environment, source, and inputs:

| Archive on `mars-merge-bdc152a-1120` | SHA-256 |
| --- | --- |
| `/work/pr123/runtime-79fd84a7e4d8597593a66be67dedfe80c9eabe14.tar.gz` | `96b7c46885b41acc1da5ca12255273b6baffbef81897374373a28f711a04ffd6` |
| `/work/pr123/69d6587cef02e5b17a82e28422b67aa3d5be55ad/source.tar.gz` | `d3103cac555fe277d580eeedb81768eafa6b93dfc76299cd8016cba1e7753234` |

Results go to `/work/pr123/<source>/five-sol-cpu`,
`/work/pr123/<source>/gpu-probe`, and `/work/pr123/native-coordinates`.
The GPU probe measures T21 memory and utilization before choosing more GPU
concurrency. Its sizing record does not prove that concurrent T42 fits.
The CPU matrix remains useful if no GPU is available. The older full GPU matrix
has not been stopped. Its requested 64 GiB, eight CPUs and GPU memory above
40,000 MiB excluded nodes before any experiment started.

The native metadata reader uses saved `ak`, `bk`, pressure coordinates,
`delz`, `delp` and the first `[0, 5]` time bounds. It keeps the input namelist
and variable attributes. `native_metadata.py` is the source embedded in its
immutable ConfigMap, alongside the verified archived baseline manifest.

Saved native `ak` and `bk` are zero placeholders. The analysis instead sums
saved mean `delp` from the saved fixed top pressure. The resulting surface
pressure matches saved mean `ps` within 0.00047 Pa. Saved `phalf` uses millibars;
`delp` uses pascals and `delz` uses meters. The native GNU build enables
`MARS_GCM` and omits `SECDAY86400`; its included `mars_constants.fh` selects
88,704 seconds per day, matching the Dinosaur solar day. The prepared executable,
source, input and build records are checked on CPU resources by
`native-calendar.json`.

`native-damping-metadata.json` reads the original native startup logs and
namelists. C24 reports `da_min_c=0.26624905E+11 m²` with `k_split=3`,
`n_split=8`; C48 reports `0.65910968E+10 m²` with `k_split=6`, `n_split=8`.
The area values have eight printed significant digits. The archived log and
namelist hashes identify their source. Startup also prints the actual acoustic
and remapping steps: 9.625/77 seconds for C24, and 4.8125/38.5 seconds for C48.
These are preserved in `validation/pr123-native-damping-metadata-v2.json`.

After downloading and verifying the files, run:

```sh
rtk proxy .venv/bin/python experiments/pr123-upper-atmosphere/analyse.py \
  outputs/pr123-validation/five-sol-cpu/runs \
  outputs/pr123-validation/native-topwind/runs \
  outputs/pr123-validation/native-coordinates/runs \
  validation/pr123-five-sol-comparison.json
```

The report gives common-pressure mean-field comparisons and their covered
area. It reports the RMS of the mean zonal wind. It does not treat a checkpoint
maximum as a time mean. Native layer heights come from saved mean thickness;
the Dinosaur height coordinate is an isothermal reference approximation.
Interpolation of mean fields differs from time-averaging fields interpolated
at each step. These five-sol runs do not establish climate convergence.
