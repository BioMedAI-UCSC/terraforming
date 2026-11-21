# Source-derived top divergence diffusion

Balanced initialization removes a false-wave trigger but does not close the
coupled climate gap. The local five-sol balanced run still gives top mean zonal
wind RMS 10.543688 m/s, versus native C24/C48 3.964194/4.361770 m/s. Its top mean
temperature is 159.583729 K, versus native 168.96/169.24 K. The native-initialized
T42/L56 run also fails its finite-state guard before the first sol. That failed
case is retained in the [baseline evidence](../validation/pr123-five-sol-baseline.json).

## Source and coefficient

Pinned FV3's first two layers use second-order divergence damping with
`d2_bg_k1=0.2`, `d2_bg_k2=0.04`. The released setup has `dddmp=0`,
`do_vort_damp=False` and `n_sponge=0`. The latter retains these overrides.
`sw_core.F90` adds `d2*da_min_c*divergence` to the kinetic-energy potential before
the momentum gradient. `fv_control.F90` reports the physical diffusion rate as
the background coefficient times minimum corner area divided by the advective
substep. The source links and branch conditions are in the
[initial diagnosis](pr123-upper-diagnosis.md#native-top-damping).

The [native startup evidence](../validation/pr123-native-damping-metadata-v2.json)
contains the actual area and substep lines, namelist hashes and source-log hashes.
Its SHA256 is `fca173f9388d6bc69cff30d1d41884b726412fe2e4aa6e6dc615630aa8cbc79b`.

| Native case | Printed minimum corner area (m²) | Advective substep (s) | Top diffusivity (m²/s) | Second-layer diffusivity (m²/s) |
| --- | ---: | ---: | ---: | ---: |
| C24 | 26,624,905,000 | 9.625 | 553,244,779.2207792 | 110,648,955.84415585 |
| C48 | 6,591,096,800 | 4.8125 | 273,915,711.16883117 | 54,783,142.233766235 |

These areas have eight printed significant digits. They are runtime corner
areas, not average cell areas or reconstructed triangular dual areas. Using
the latter would give a different coefficient.

## Explicit adaptation

`build_step(..., divergence_diffusivity_m2_s=profile)` accepts a finite,
nonnegative per-layer physical diffusivity. It applies the exact spectral
solution of `d(divergence)/dt = nu*Laplacian(divergence)` after dry dynamics and
existing hyperdiffusion, before tracer transport and column physics. A spherical
harmonic of degree `l` decays by `exp(-nu*dt*l*(l+1)/radius²)`. Vorticity,
temperature and pressure are not filtered by this operation. Updated winds
change subsequent conservative tracer fluxes; tracer fields need not stay equal.
No dissipated heat is added, matching the source's disabled top `d_con_k` term.

The CLI opt-in is `--native-top-divergence-damping`. T21 selects the recorded
C24 rates and T42 selects C48. Other resolutions reject this preset. Rates stay
fixed in physical units when changing the Dinosaur time step. The selected
profile, native source revision, cell area, substep and provenance hash enter
restart identity. The default remains disabled.

This is a leading-operator adaptation. FV3 applies its metric-dependent finite
volume gradient during each substep, while Dinosaur uses a spectral exponential
after its dry time step. It does not make the sigma core equivalent to FV3's
hybrid core. Native explicit per-substep factors converge toward the exponential
under substep refinement. On T21, maximum factor error over all retained degrees
is 0.0114270, 0.0056758, 0.0028284 and 0.0014118 for refinements 1, 2, 4 and 8.

## Controlled results

The [recorded replay](../validation/pr123-upper-damping.json) starts from the exact
balanced two-sol checkpoint. It records source, checkpoint and configuration
hashes. Reproduce it with:

```sh
rtk proxy env PYTHONPATH=package .venv/bin/python scripts/diagnose_upper_atmosphere.py \
  outputs/pr123-validation/cli-balanced-five-sol-mac/reference AmesGCM/data \
  --restart outputs/pr123-validation/cli-balanced-five-sol-mac/reference/checkpoints/restart_000000768.npz \
  --native outputs/pr123-validation/native-topwind/runs --damping-matrix \
  --output /private/tmp/upper-damping.json
```

| Initial pressure / D2 | Dry one-sol top vector RMS (m/s) | Top mean temperature (K) | Top minimum temperature (K) |
| --- | ---: | ---: | ---: |
| Native / off | 183.021935 | 134.029322 | -3.008995 |
| Native / on | 16.808481 | 170.328946 | 161.869839 |
| Balanced / off | 3.07e-9 | 170.000000 | 169.999999997 |
| Balanced / on | 1.73e-10 | 170.000000 | 169.9999999999 |

The negative-temperature native/off case is a failed physical state, despite
remaining finite. It is retained rather than silently discarded.

| From balanced two-sol checkpoint, after 32 full-physics steps | D2 off | D2 on |
| --- | ---: | ---: |
| Top divergent vector RMS (m/s) | 79.004908 | 19.727451 |
| Degree 1–3 divergent vector RMS (m/s) | 45.032824 | 19.332218 |
| Top mean temperature (K) | 154.686128 | 158.600964 |
| Top minimum temperature (K) | 91.889137 | 101.627818 |
| CO2/gas inventory relative drift | -3.23e-15 | -2.12e-14 |
| Water inventory relative drift | -6.16e-14 | -6.16e-14 |
| Dust inventory relative drift | 8.74e-12 | 8.68e-12 |

All transported tracers remain nonnegative. These checks use a 1e-10 material
gate and report the actual drift rather than claiming exact conservation.

Earlier eight-step probes show that removing vertical diffusion has negligible
effect and removing top Rayleigh drag increases the motion. Dry dynamics grows
and cools more than full physics. At matched 1,848 seconds, changing the T21 step
from 231 to 132 to 66 seconds gives top vector RMS 89.8871, 89.8830 and 89.8779 m/s.
Short-window time refinement does not remove this motion. It does not establish
long-run T42 stability.

## Transport splitting limit

The [constant-tracer probe](../validation/pr123-uniform-tracer-probe.json) uses
flat terrain, 170 K, uniform pressure and mixing ratio 1e-4, with a degree-three
divergent wind of 20 m/s top RMS. It uses the actual C24 diffusivities and no
forcing. It compares the production filter before transport with an external
filter after transport. The latter changes no transport code and produces the
same wind and temperature trajectory, so the concentration difference isolates
the split.

| Wind profile, after 32 steps | D2 off: maximum relative uniformity error | D2 before transport | D2 after transport | Maximum before/after concentration difference |
| --- | ---: | ---: | ---: | ---: |
| First two layers | 0.00482795 | 0.00301231 | 0.00304381 | 0.00005907 |
| All layers | 0.00584639 | 0.00584632 | 0.00584638 | 0.00015604 |

The existing finite-volume tracer flux and spectral carrier update do not
preserve a uniform mixing ratio exactly. The added placement difference is at
most 0.016% here after 7,392 seconds, while total inventory error stays below
3e-14 and concentrations stay nonnegative. Moving the filter after transport
does not improve uniformity in these cases. This supports keeping the current
placement; it does not establish native tracer-transport fidelity. The source
applies damping within its dynamics substeps, while this adaptation splits it
at a full Dinosaur step. Longer tracer-profile and substep-convergence checks
remain necessary.

Reproduce this separate probe by replacing `--damping-matrix` in the command
above with `--uniform-tracer-probe` and selecting a new output path. This probe
does not compare native time means.

## Checks and remaining work

Tests through public `build_step` check independently known mode-decay factors,
decreasing divergent kinetic norm, angular momentum, material inventories,
nonnegative tracers, untouched nonfiltered fields, invalid profile rejection,
derivatives and exact save/load continuation. Existing hyperdiffusion behavior
and the balanced one-sol rest check also pass.
The decay factor is cast to the dry output dtype, so the new filter cannot
promote that field in a mixed-precision scan. The current public Mars core
already promotes float32 input to float64 under enabled x64; no dedicated
float32-output regression is claimed at that interface.

The controls support this source-derived treatment; they do not establish a
matched climate. The [native validation](physics-parity.md) passed all 48
validators, including 24 coupled-column cases, with current source hashes.
Five-sol T21/T42 comparisons, native pressure-aware means, time-step convergence
and full CI remain required. The frozen-sigma pressure and sampling differences
remain.
