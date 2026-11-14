# Five-sol held-out accuracy experiment

The campaign first runs a paired fifteen-sol winter control at Ls270 with
MOLA terrain, TES surface fields, seasonal dust, and the same initial state.
Only atmospheric CO2 phase change is switched on/off. Both cases archive raw
budgets, diagnostics, restarts, and final geographic NetCDF fields. This tests
the condensation mechanism under geographic forcing; it is not an observed
climate benchmark. It does not replace the held-out forecast test below.

This experiment tests predictive error beyond the two-hour pilot. It uses five
daily-mean leads from sixteen ten-sol ARCO-MACDA windows. All four seasons in
MY32 train the three physical controls. MY33 selects the checkpoint. All four
seasons in MY34 and MY35 provide eight test windows. Test errors do not select
parameters, bounds, or stopping conditions.

Each predicted daily mean averages twelve evenly spaced model samples. Their
centres match the reference daily means. The 295.91748-second timestep places
every sample at an exact integration step. The initial daily-mean state and
soil history use days 0/1 only; reference days 2--6 are forecast targets. The
forecast starts after the final sample used by the initial daily mean is
available. Target samples span the following five sols. No initialization
sample overlaps a target sample.

Initialization treats daily means as estimates of instantaneous conditions.
Soil uses linearly interpolated past daily surface means, without a resolved
diurnal history. Dust is fixed at its initial observed value. Input hashes,
source DOIs, license attribution, and these limits are recorded in the manifest.
The original daily-data staging did not record its remote source revision;
frozen file hashes identify the actual reference data used.

Prepare locally from the existing staged references:

```sh
rtk proxy .venv/bin/python scripts/prepare_multiday_windows.py \
  --daily data/arco_macda --inputs outputs/review/inputs-current \
  --output outputs/review/multiday-windows
```

Stage the common inputs at `/work/reviewer-inputs` and the generated windows at
`/work/reviewer-multiday-windows`. Use `apps/nautalis/jobs.py` with this experiment
folder, a pinned source commit, and an A100. Jobs and transfer pods have no
elapsed-time limit. Failed attempts keep separate traces. A completed run must
report every test field and window, training endpoints versus validation
selection, bound hits, and uncertainty. Neither a larger gain nor a successful
annual climate is assumed in advance.
