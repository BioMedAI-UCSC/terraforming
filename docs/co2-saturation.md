# CO2 saturation after each physical step

NASA Ames `co2_condense` computes the predicted air temperature after the other
physics. It forms CO2 ice and releases latent heat when that prediction lies
below the frost point. The reference is
[`update_mars_atmos.F90`](https://github.com/nasa/AmesGCM/blob/ba47cada5f9eb89c75c4f10d06dbae6e7f9ecb26/atmos_param_mars/update_mars_atmos.F90#L71).
Our old one-hour relaxation allowed large cold deficits to persist.

The default now follows the per-step saturation principle. The framework keeps
its transported ice and finite settling, rather than copying the reference's
immediate deposition assumption. It is a conservative sigma-grid adaptation,
not a port of the full Ames microphysics or FV3 dynamical core.

Each step first advances dynamics, surface exchange, radiation and existing
cloud settling. Saturation then acts on that predicted state:

1. Cold gas forms ice and heats the air. Warm ice can return to gas.
2. The changed gas mass changes surface pressure and sigma-layer masses.
3. A donor-cell remap transports ice, total energy and horizontal momentum
   onto those new layer masses. Cumulative integrals keep storage linear in
   the layer count.
4. A triangular heat solve includes gas and ice heat capacities, ice height,
   latent energy and kinetic energy lost when different winds mix.

Up to twelve conservative sweeps solve the pressure/temperature coupling, with
a 1e-8 K equilibrium tolerance. A transported numerical budget limits net
condensation to ten percent of the original gas mass in a step. This expresses
the Ames mass-loss guard in gas-mass units. If that guard is exhausted, the
remaining cold deficit stays visible in diagnostics. It is not filled by an
external heat source. These are solver iterations and mass bounds, not job
runtime limits.

The pure column tests reach the frost point to within 1e-6 K while conserving
gas plus ice mass, resolved energy and horizontal momentum at relative
tolerance 2e-12. They cover cold, warm and mixed columns, nonuniform layers,
distinct horizontal columns, complete sublimation, and the mass-loss guard.
A reverse-mode derivative agrees with centred finite differences at three
perturbation sizes under the existing strict tolerance.

Spectral projection and positivity repair follow the nodal solve. They can
introduce residual deficits and budget changes. The runner records the
saturation stage separately from settling and global pressure projection;
column conservation alone does not establish whole-model energy closure.
Annual local fields and held-out forecasts must be checked after this change.
Settling and saturation project only their physical increments. The existing
spectral flow is preserved when the exchanged ice mass is negligible.

`CO2Forcing.atmospheric_saturation_adjustment` defaults to `True`. Set it to
`False` to reproduce the finite-rate operator, whose relaxation time remains
`atmospheric_condensation_timescale_s=3600`. The reviewer runner exposes
`--no-atmospheric-saturation-adjustment` for that controlled comparison.
The five-sol campaign runs matched fifteen-sol winter cases for saturation,
finite-rate atmospheric phase change, and surface-only phase change before
calibration. Input states and forcing stay identical across these controls.
