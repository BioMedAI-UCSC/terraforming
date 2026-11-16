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

The generic slim image supplies GNU Fortran, MPI and netCDF. Two CPU preparation
jobs fetch exact source commits, apply the released patches, compile the native
model and install the locked Python/CUDA environment on persistent volumes.
Four uppercase preprocessor directives are lowercased for GNU; compiler paths,
source-list regular expressions and one missing input namelist terminator are
adapted. Fortran physics statements are unchanged. No setup runs on a GPU.
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
