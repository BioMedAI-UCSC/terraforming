# Solar forcing in the Mars GCM

The GCM computes solar geometry for each horizontal grid cell from its own
simulation clock and Mars orbital/rotation constants. These fields drive surface
and atmospheric radiation; see [column physics](../../package/gcm3d/implementation.md).

## Keplerian season and distance

Mean anomaly advances uniformly: `M(t) = M0 + 2*pi*t/T_orb`. Six Newton iterations
solve `M = E - e*sin(E)`, then `r = a*(1-e*cos(E))` and true anomaly determine
solar longitude `Ls = true_anomaly + Ls_perihelion`. Solar irradiance is
`S_1AU*(AU/r)**2`.

Solar longitude is measured from northern spring equinox, not perihelion. Set an
epoch with `mean_anomaly_for_ls`; do not treat an Ls value as the mean anomaly or
insert it directly into a perihelion-centered orbital-distance formula.

## Illumination and optical path

Declination is `asin(sin(obliquity)*sin(Ls))`. Diurnal illumination uses local
longitude and the rotating hour angle to obtain positive cosine of zenith;
daily-mean mode integrates illumination across daylight. Polar day/night are
handled by the geometry. `solar_slant_path_enabled` applies a zenith-dependent
direct-solar optical path, distinct from the projected incident energy.

The resolved radiation solver computes absorption/scattering through layers with
CO₂ and prescribed visible dust, then surface reflection. A fixed scalar
transmittance is not the correlated-k solver. Albedo may be scalar or a TES field.
Visible and infrared dust remain separate in the evolving Ames climatology.

## Entry points and stability

Mars `radiative_forcing()` is diurnal by default; `tform mars maps` instead defaults
to daily mean. The browser uses its request's sampling and adjusts timestep to
retain duration when enforcing `dt <= rotation_period/(2*n_lon)`. Passing that
guard is necessary, but finiteness and dynamical stability still need checks.

Use `insolation_sampling`, season and temporal-sampling metadata when matching
references. Daily means and instantaneous snapshots are different observables.
See [reference comparisons](../../cli/reference-comparison.md).

## Related guides

- [GCM quickstart](../../getting-started/quickstart.md).
- [Radiation implementation](../../package/gcm3d/implementation.md).
- [General solar radiation](../solar-radiation.md) and [orbital mechanics](../orbital-mechanics.md).
- [Deprecated global-mean Mars equations](../../architecture/global-mean-mars.md).
