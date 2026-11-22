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
was suspended before scheduling after a fresh unbound-container check. Its
requested 64 GiB, eight CPUs and GPU memory above 40,000 MiB excluded nodes
before any experiment started. Its Job and ConfigMap remain preserved.

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

## Completed GPU controls

All entries below use the public runner, 56 layers, 231-second steps and
1,920 steps. The quantities are spatial RMS of the five-sol mean zonal wind
and the area mean of the five-sol mean top temperature.

| Resolution | Initial pressure | Native top D2 | Wind RMS (m/s) | Temperature (K) |
| --- | --- | --- | ---: | ---: |
| T21 | native | off | 12.526755 | 155.957801 |
| T21 | balanced | off | 10.674408 | 159.642904 |
| T21 | native | on | 4.390471 | 170.405967 |
| T21 | balanced | on | 4.354305 | 170.466290 |
| T42 | native | on | 6.093892 | 170.551098 |
| T42 | balanced | on | 5.240840 | 170.687820 |

Native C24 gives 3.964194 m/s and 168.960053 K. Native C48 gives
4.361770 m/s and 169.236322 K. T42 with native pressure and no D2 first
fails at step 21. The separate pressure-only CPU matrix also completed both
resolutions. It gives T21 10.678014 m/s and 159.646625 K, and T42 9.145087 m/s
and 156.131333 K. Pressure balance alone avoids the observed early T42 failure,
but leaves the large cold bias. T42 pressure-only uses CPU; the D2 controls
use GPU, so that T42 comparison also changes execution backend. The matched
T21 pressure-only CPU/GPU means differ by 0.00361 m/s and 0.00372 K.

`validation/pr123-five-sol-pressure-only.json` records the full CPU profiles,
budgets, source and input hashes. All 95 downloaded files match remote SHA-256
values in `validation/pr123-pressure-only-artifact-hashes.json`. The exact
T21 full/resumed comparison passes all 59 saved arrays. The host cgroup peak
was 43,808,333,824 bytes, including concurrent cases, compilation and file cache;
the 353 resource samples cover part of the job. This CPU run predates the
eight-core affinity bound used for later GPU launchers.

Source pins are `69d6587cef02e5b17a82e28422b67aa3d5be55ad` for baseline,
`43fc88bf8f6654ddca5b7e5400a0186a46156e9d` for pressure-only and
`09e8c10500072fa06b51596bc50350d535ac6c5f` for D2 controls. The D2 option
uses native startup area and acoustic step to set the two top coefficients.
The spectral operator differs from the FV3 operator. These controls do not
change the default runner physics.

`validation/pr123-five-sol-d2-native.json` and
`validation/pr123-five-sol-d2-balanced.json` contain the full two-resolution
reports, inventories and common-pressure profiles. For example, native-pressure
T42 D2 gives 156.128 K at 0.03 Pa, compared with native C48 at 161.246 K.
The large top cold bias is removed, but a colder layer remains below it.

The CPU baseline and GPU native-pressure D2 T21 cases pass exact equality of
all 59 final restart arrays. The pressure-only GPU pair and balanced-D2 GPU
case fail their strict independent full-run/restarted-run comparison. They
already differ at step 384, before the restart is loaded. The failure thus
does not isolate the save/load operation. The failed assertions and the
pre-reload comparisons are preserved; tolerances were not relaxed.

`replay_t42_failure.py` uses the public `build_step` entrypoint. It verifies
saved step 19 to 20, then reproduces step 21 as the first nonfinite step.
`validation/pr123-t42-first-invalid.json` preserves the last finite extrema,
the first bad fields and coordinates, and source/input hashes. Field order
does not identify a causal kernel after global spectral propagation.

The native-pressure T42 D2 job peaks at 18,126 MiB on a 24,576 MiB GPU and
5,482,639,360 bytes of host memory. GPU utilization reaches 100%; the mean
of 473 one-second samples is 46.24%, including compilation and output I/O.
These samples exclude runtime extraction. T21 pairs use two processes;
T42 uses one process per GPU. Launcher metadata records affinity and CPU quota.

Only unbound jobs were replaced after fresh node/container checks. Their
Jobs and ConfigMaps remain archived with the suspension reason. Replacement
init containers copy prepared archives to local scratch, verify the same hashes,
extract locally, then remove only their own cached tar after successful extraction.
The final T42 balanced job requests four CPUs and 16 GiB with eight-CPU/64-GiB
limits, based on measured GPU host memory. Assigned cases were left unchanged.

The archived GPU recipes are `paired-t21-gpu.json`, `d2-t21.json`,
`d2-t42-native.json` and `d2-t42-balanced.json`. `d2-source.json` records CPU
source preparation. Use new Job, ConfigMap and output names before a rerun.
Actual downloaded files remain under `outputs/pr123-validation` in the main
workspace; tracked `validation/pr123-*-artifact-hashes.json` files identify them.

`gpu-common-checkpoint.json` runs `gpu_restart.py` for native-D2 and balanced-D2
T21 states from saved step 384. Each worker builds one 384-step executable
through public `build_step`. It compares repeat execution from the same input,
execution after public save/load, and continuation from a saved step-768 state
to step 1,152. A changed-temperature negative control checks that the exact
comparator rejects changed state. These checks isolate serialization and repeat
execution within one compiled executable. Independent cold starts and separate
compilations remain distinct checks.

The first probe attempt failed its setup dtype guard before any model advance:
averaging setup had been applied only to one input. Both save/load roundtrip
files were already byte-identical to the original checkpoint, including metadata.
The failed Job, ConfigMap, logs and seven hash-verified files remain preserved.
The inherited 24-GB requirement was widened to 16 GB after a fresh unbound-job check;
the measured full T21 pair peaked at 9,606 MiB.

The next executed attempt passes same-executable repeat, loaded-input advance,
and midpoint roundtrip for both cases. Its extra continuation stops before
execution because setup makes the counter an integer, while the completed state
stores a float. The function was compiled for integer input and rejects float
input. All ten files and the failed aggregate Job remain preserved in
`validation/pr123-gpu-restart-extended-attempt3.json`. The final probe obtains
the public averaging wrapper without replacing the existing checkpoint state.
This keeps the saved counter dtype unchanged across all compiled calls.

The final Job completed successfully for both cases. All five comparisons pass
exactly for 58 numeric state leaves, and the changed-temperature negative control
is detected. Each saved NPZ has 59 arrays, including JSON metadata; the input
roundtrip files are byte-identical to their original checkpoints. All 14 final
files match remote hashes. `validation/pr123-gpu-common-checkpoint.json` records
the checks, source/input hashes, runtime versions and limits of the result.

The final probe used a Tesla V100-SXM2-32GB. Its 146 resource samples record
9,176 MiB peak GPU memory, 92% maximum GPU utilization and 4,828,659,712 bytes
peak host memory. Samples cover compilation and the checks, and exclude archive
staging. The earlier independent cold-run comparison failures remain recorded;
this test establishes save/load continuity with one common executable.
