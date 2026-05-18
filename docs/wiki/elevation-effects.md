# Elevation Effects on Mars Temperature and Pressure

This page audits what the current model does with elevation and documents the
remaining gap for terrain-aware temperature and pressure modelling.

## What is currently modelled

`package/src/celestials/planets/mars.py` accepts an initialization parameter:

```python
Mars(elevation_m=0.0)
```

During construction, it applies a hydrostatic pressure correction:

```text
P(z) = P_ref * exp(-z / H)
H = 11100 m
```

where positive `z` is elevation above the reference surface. This means an
initial condition at high elevation starts with lower pressure, and an initial
condition in a basin starts with higher pressure.

Examples for `P_ref = 610 Pa`:

| Elevation | Pressure multiplier | Initial pressure |
|---:|---:|---:|
| `-7000 m` | `1.879` | `~1146 Pa` |
| `0 m` | `1.000` | `610 Pa` |
| `5000 m` | `0.637` | `~389 Pa` |
| `21000 m` | `0.151` | `~92 Pa` |

## What is absent

The elevation correction is an initialization-only scalar adjustment. The
runtime does not currently model:

- A stored `elevation_m` state variable after initialization.
- A pressure lapse-rate field or spatial pressure gradients over terrain.
- A temperature lapse rate or altitude-dependent thermal tendency.
- MOLA DEM lookup by latitude/longitude.
- Slope/aspect shadowing, terrain horizon effects, or elevation-dependent
  atmospheric optical depth.
- Coupling between local pressure, greenhouse factor, CO2 frost point, boiling
  limits, or sublimation rates beyond the initialized surface pressure.
- Batched elevation propagation in `package/src/engine/batched_controller.py`
  beyond whatever pressure each `Mars` instance already has at batching time.

## Expected physical influence

Elevation should matter most through atmospheric column mass. In a hydrostatic
atmosphere, pressure decreases approximately exponentially with height:

```text
dP/dz = -rho * g
P(z) = P0 * exp(-z / H)
H = R_specific * T / g
```

For a CO2-dominated Mars atmosphere, `H ~= 11 km` is a reasonable present-day
scale-height approximation near the model's default temperature. The impact is
large because Mars terrain spans many kilometers:

- High terrain such as Tharsis or Olympus Mons should have much lower surface
  pressure, reduced atmospheric shielding, lower column greenhouse effect, and
  more volatile loss/sublimation stress.
- Low basins such as Hellas should have higher pressure, greater atmospheric
  column mass, more shielding, and higher likelihood of transient liquid-water
  stability for the same temperature.
- If a temperature lapse rate were introduced, a first-order dry-adiabatic CO2
  estimate would be `Gamma = g / c_p`, roughly a few K/km. A production model
  should use Mars Climate Database or GCM-derived lapse rates instead of a
  universal constant.

## Implementation gap

The model has a useful first-order pressure initialization:

```text
surface_pressure_initial = surface_pressure_ref * exp(-elevation_m / 11100)
```

but it is not yet a terrain-aware climate model. To close the gap, the next
implementation should add explicit terrain inputs and derive local pressure and
temperature from `lat`, `lon`, `Ls`, local time, and MOLA-referenced elevation.
The minimum viable extension is:

1. Persist `elevation_m` on `Mars`.
2. Add a documented `pressure_at_elevation(P_ref, z, T=None)` helper.
3. Optionally derive `elevation_m` from MOLA for `(latitude, longitude)`.
4. Add an altitude temperature adjustment behind an explicit parameter or data
   source, rather than silently changing current baseline temperatures.

## References

- Smith, D. E. et al. (2001), MOLA experiment summary, doi: <https://doi.org/10.1029/2000JE001364>
- Allison, M. and McEwen, M. (2000), Mars solar longitude convention, doi: <https://doi.org/10.1016/S0032-0633(99)00092-6>
