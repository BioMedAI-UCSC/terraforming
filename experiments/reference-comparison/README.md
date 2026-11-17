# Matched reference comparison

Run the released present-Mars cloud setup for 100 sols on both frameworks.
The native CPU job runs C24 and C48 in parallel. The Dinosaur GPU job runs T21
and T42 in parallel on one 80 GB A100. These are two resolution comparisons;
the grids do not contain the same cells or use the same transport scheme.

Both use the pinned release, the same MOLA16 terrain file, TES/GRS maps,
MY30 dust scenario, polar water inputs, 170 K initial air, released latitude
surface/soil temperatures, 56 atmospheric levels and 231-second physics steps.
Shared inputs have fixed SHA-256 checks. The native C48 case doubles its inner
dynamics remapping frequency as required by the released resolution rule.
Dinosaur retains its sigma grid, arithmetic centers, spectral filtering and
zero-pressure top. Native FV3 retains hybrid interfaces and a finite top.
These numerical differences must appear in the comparison report.

The released visible-dust scenario uses `dgdm_type=1` and
`dust_map_scale=3.67`: the mapped input optical depth is multiplied by
`3.67/2.75`. The reference runner uses this conversion and visible channel 1.
The generic scenario loader keeps its infrared conversion default of `1/2.75`.
Earlier Dinosaur benchmark revisions through `a9a10cf` omitted the visible
scale. Their climate outputs cannot support a matched-forcing comparison.

The generic slim image supplies GNU Fortran, MPI and netCDF. Two CPU preparation
jobs fetch exact source commits, apply the released patches, compile the native
model and install the locked Python/CUDA environment on persistent volumes.
Four uppercase preprocessor directives are lowercased for GNU; compiler paths,
source-list regular expressions and one missing input namelist terminator are
adapted. Fortran physics statements are unchanged. No setup runs on a GPU.
Future builds also keep static fields out of FMS's dynamic output-time gate.
That gate uses an undefined time for static fields and wrote zero coordinates
and maps in the frozen baseline. Parallel zero-step probes at C24 and C48 restored
those exports while all 44 restart files at each resolution remained byte-identical. Existing runs and
their runtime caches are unchanged.
On GNU/Linux, `HAVE_GETTID` selects the system thread-ID function supported by
the image's glibc, as FMS's source interface requires.

First render and submit preparation using the immutable implementation commit:

```sh
python experiments/reference-comparison/deploy.py --name reference-100sol-v1 \
  --ref IMPLEMENTATION_COMMIT --phase prepare --submit
```

The input PVC must contain `inputs/reference` with the seven frozen NetCDF
files listed in `scripts/run_reference_comparison.py`. Set `--input-node` to
the node holding that RWO input volume. Each framework gets a separate 100 GiB
work volume. The preparation jobs share only the read-only input volume.
Use `--pvc-prefix PREVIOUS_NAME` to reuse existing framework volumes for a
new preparation revision. Source-specific caches and previous outputs remain
separate on those volumes.

After both CPU jobs succeed, submit the two compute jobs:

```sh
python experiments/reference-comparison/deploy.py --name reference-100sol-v1 \
  --ref IMPLEMENTATION_COMMIT --phase run --submit
```

Submission checks that preparation succeeded for the same source and image.
Use `--framework dinosaur` in both phases to prepare and run only the two
Dinosaur resolutions when the native experiments are already running. Use a
new name and volume for a corrected revision so previous attempts stay intact.
Compute jobs recheck input hashes, native sources/executable or installed
framework source before running. Each resolution writes separate logs, source
identity, runtime records and outputs. Failed attempts remain available.
Jobs have no time limits or automatic timed deletion.

Monitor actual GPU memory and utilization. Each JAX process has preallocation
disabled and an allocator fraction of 0.45. Record initialization, compilation,
steady computation and output costs separately. A 100-sol cold start checks
stability and early seasonal response; it does not establish an equilibrated
climate or a speedup. Compare native and Dinosaur outputs on common physical
coordinates and report each grid, numerical adaptation, precision and hardware.

## Recover frozen grid metadata

The completed C24 baseline has valid saved surface fields but zero coordinates
in its original `grid_spec` files. Use a separate grid sidecar for this frozen
run. The script checks the exact grid source hashes and supported controls,
calls the unchanged native geometry routines, and checks that all six faces
cover the sphere. It rejects alternate grids and existing output paths.

```sh
python scripts/reconstruct_native_grid.py \
  --source /path/to/frozen/GFDL_atmos_cubed_sphere \
  --input-namelist /path/to/native/outputs/input.nml \
  --output outputs/native-c24-grid.npz
```

The NPZ contains corner longitude/latitude in radians and cell solid angles
in steradians. Axes are `(tile, y, x)` in native tile order. It also records
source, routine, compiled-code and input-namelist hashes. Native NetCDF time
uses `no_calendar`; open these files with `decode_times=False` and retain the
numeric Mars-day axis.

Recorded [C24 surface checks](../../iclr-results/03-benchmarks/reference-comparison/native-c24-surface-checks.json)
cover 1,200 saved times on each tile: 8,294,400 pressure/temperature values are
finite and contain no fill values. The recovered area-weighted final pressure
matches the native log within 6e-8 Pa. This is a saved-surface-field check;
CO2/water inventories, 3-D energy budgets and paired climate agreement require
their own analysis. The original outputs remain unchanged.
