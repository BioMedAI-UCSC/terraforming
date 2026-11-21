# PR #123 upper-atmosphere diagnosis

The source cold-start pressure is not in discrete balance with Dinosaur's
truncated terrain. A dry, initially resting MOLA atmosphere develops strong
upper winds and cooling within one sol. The same core retains rest with an
exponential pressure field built from its resolved terrain. The runner now
offers this explicit `sigma-balanced` adaptation and retains native pressure
initialization as its default. No physical damping coefficient changed.

## Reproduce

From a checkout with the archived outputs and released Ames inputs:

```sh
rtk proxy env PYTHONPATH=package .venv/bin/python scripts/diagnose_upper_atmosphere.py \
  outputs/pr123-validation/cli-five-sol-mac/reference AmesGCM/data \
  --native outputs/pr123-validation/native-topwind/runs \
  --reference-physics --output /private/tmp/upper-replay.json
```

Add `--assert-mean-ratio 2` to reproduce the red archived-mean check. It reports
12.494631 m/s versus native C24/C48 3.964194/4.361770 m/s and exits with an
assertion failure. The factor two is a diagnostic alarm, not a scientific
acceptance tolerance. This assertion checks the archived mean. A new five-sol
mean is required to validate a climate correction; a one-step replay cannot
turn an old saved mean green.

The script reports all changed archived source hashes and verifies each released
input hash before replay. Only the runner's initialization source has changed;
the archived dynamics and physics kernels still match. The replay uses the
public `build_step` interface and keeps the original files unchanged.
The [machine-readable result](../validation/pr123-upper-diagnosis.json) records
the checkpoint, configuration, mean-field, diagnostic and native archive hashes.

## Ranked hypotheses and one-factor controls

1. **Different top divergence damping.** If the damping difference drives the
   excess, a source-equivalent divergence treatment must change the relevant
   modes, without selecting a coefficient only to lower wind maxima.
2. **Frozen sigma pressure coordinates.** If this changes upper dynamics and
   sampling, top pressure must vary over terrain in Dinosaur and stay fixed in
   the native hybrid grid. A layer-number comparison then samples different air.
3. **Initial terrain imbalance.** If it launches the large motion, the released
   cold start must produce a strong initial response with terrain, before
   radiation or tracer feedback is present.
4. **Held radiation and dynamics coupling.** If stale heating drives the
   immediate response, refreshing radiation at the same state and clock must
   change that response. A small one-step difference does not exclude a
   slower feedback.

The initial saved top vector RMS is 105.124870 m/s. The divergent component is
94.110451 m/s, about 80% of the squared vector RMS. This is an instantaneous
state; native archives here contain time means. They cannot supply a native
instantaneous-wind reference.

| One 231-second step | Top vector RMS (m/s) | Top mean temperature (K) |
| --- | ---: | ---: |
| Dry dynamics, original terrain | 105.863069 | 152.357801 |
| Same state, flat lower boundary | 105.732085 | 152.357594 |
| Original terrain, no hyperdiffusion | 105.980689 | 152.357801 |
| Full reference physics, held radiation | 105.577744 | 152.476000 |
| Same full physics, refreshed radiation | 105.577726 | 152.481024 |

Each dry comparison changes one `build_step` argument. The radiation comparison
changes only `radiation_initialized`. Both radiation clocks are advanced by
231 seconds because the archived checkpoint lies on a radiation update
boundary. Without this shared clock shift both cases refresh, which would
make the test ineffective. The full physics cases use identical original
surface maps, material fields, radiation settings and terrain.

The evolved-state flat replay retains the original pressure and temperature.
It is not a balanced flat-world experiment and cannot rule out accumulated
terrain pressure-gradient errors. The separate cold-start evolution rebuilds
the released hydrostatic pressure for each lower boundary, starts each at
170 K with zero winds, removes tracers and surface feedback, and runs each
matching dry core for one sol (384 steps).

The released 170 K resting cold start produces 0.665383 m/s top vector RMS
after one dry step. It reaches 183.021935 m/s and 134.029322 K after one sol.
The matched flat cold start stays at 170 K with 3.05e-9 m/s top vector RMS.
Changing only the source log-weighted pressure correction to an exponential
field over the unfiltered terrain reduces one-sol winds to 39.727038 m/s and
keeps top temperature at 165.167069 K. Using exponential pressure over the
same truncated terrain as the core reduces winds to 3.07e-9 m/s and retains
170 K. These controls separate the log-weighted correction from the spectral
terrain mismatch. They reproduce upper winds and cooling without radiation,
tracers or surface physics.
Removing forcing and surface updates from the evolved state preserves the
large top motion. Removing terrain or changing diffusion does not remove it
in one step. The warmed dry replay takes about 0.02 seconds; compilation takes
about 1.3 seconds on the local CPU.

## Native top damping

Pinned FV3 `dyn_core.F90` at
[`29654ea`, lines 695–733](https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/29654ea167575fa73b65f7f2ceb85f1682884ddd/model/dyn_core.F90#L695)
proves that `n_sponge=0` enables the special top damping. Only a single-layer
model or `n_sponge<0` bypasses it. Top layers one and two set `nord_k=0`,
which selects second-order divergence damping, with the released
`d2_bg_k1=0.2` and `d2_bg_k2=0.04`. The third-layer condition requires
`d2_bg_k2>0.05` and does not apply here.

The corresponding
[`sw_core.F90`, lines 1286–1370](https://github.com/NOAA-GFDL/GFDL_atmos_cubed_sphere/blob/29654ea167575fa73b65f7f2ceb85f1682884ddd/model/sw_core.F90#L1286)
forms a corner divergence with cubed-sphere metric factors. It multiplies this
by minimum corner-cell area and a coefficient that depends on background
damping, divergence and the dynamics substep. It then adds the result to the
kinetic-energy potential used by the momentum update. Thus 0.2 is not a
Rayleigh decay time or a per-physics-step wind multiplier.

Dinosaur currently applies fourth-power Laplacian diffusion (eighth order) at
every layer after the time step. The missing native second-order top treatment
is a confirmed scheme difference. Selecting a spectral coefficient requires
an explicit mapping of native substeps and cell areas to spectral resolution,
then convergence and energy checks. Copying 0.2 into a spectral decay factor
would not preserve the native operation. This diagnosis does not make that
conversion or claim that damping alone explains the five-sol temperature bias.

## Pressure coordinates

The saved surface pressure spans 272.789 to 1285.120 Pa. At that same pressure:

| Pressure location | Minimum (Pa) | Maximum (Pa) |
| --- | ---: | ---: |
| Dinosaur arithmetic top full level | 0.001262 | 0.005944 |
| Native top interface | 0.001319 | 0.001319 |
| Native interface below top layer | 0.006521 | 0.006521 |
| Dinosaur positive radiation top interface | 0.000510 | 0.002404 |

`vertical_grid.sigma_boundaries` explicitly freezes the released hybrid grid
at its reference column and sets the dynamics top boundary to zero.
`radiation_interfaces` retains a positive top fraction and scales it with
surface pressure. Native `pressure_interfaces` retains fixed A coefficients
aloft. Replacing the physics pressure array with native interfaces would
make it inconsistent with the sigma core's layer mass, transport and pressure
gradient. A consistent correction reaches the dynamical core, its pressure
gradient, vertical motion, remapping, radiation coupling and restart geometry.
It is not an isolated diagnostic pressure change.

## Checks and limits

The replay asserts finite output, deterministic repeated steps, nonnegative
local tracer values and bitwise equality after save/load continuation. Dry
CO2/gas, water and dust-related material drift is below 4.5e-15. The derivative
of next-step divergence norm with respect to input divergence scale is
282.4155378534; the centered finite difference is 282.4155378580.

The new one-sol T21/L56 regression runs synthetic terrain with both resolved
and discarded modes through the runner initializer and public `build_step`.
The original initialization fails this check. The explicit balanced treatment
keeps 170 K and negligible winds, retains native mean gas mass and passes an
exact save/load continuation check. Native and balanced initialization retain
reverse derivatives; three-step full-physics gradient and restart checks cover
both native temperature branches and the balanced isothermal branch.

Select the treatment with `--initial-pressure-scheme sigma-balanced`. Its
restart identity is `sigma-balanced-native-mass`; native default identity stays
`released-hydro-eq`. The treatment uses the isothermal hydrostatic equation
`ln(p_s) = constant - g*z/(R*T)` with the same terrain cutoff as the core.
Its constant preserves the native projected initial mean pressure, 702.948121 Pa
for this T21/MOLA input. It changes the initial spatial pressure distribution,
so it is an explicit Dinosaur adaptation, not native initialization parity.

No native kernel changed. No arbitrary top drag, diffusion coefficient or wind
cap was introduced. The archived climate mean remains red. Matched five-sol
T21/T42 treatment runs and pressure-aware native statistics must establish the
effect on climate means before selecting a new reference default. Initial
balance removes a confirmed false-wave trigger; the separate pressure-coordinate
and damping scheme differences remain.
