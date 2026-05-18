# Orbital Mechanics and Solar Flux Audit

This page documents the Mars orbital convention used by the runtime model and
the resulting seasonal top-of-atmosphere solar flux cycle.

## Runtime convention

The Mars implementation stores `orbital_angle` as true anomaly-like angle
relative to perihelion:

```text
nu = orbital_angle
Ls = nu + 251 deg
```

where `Ls` is areocentric solar longitude. In the Allison and McEwen (2000)
Mars calendar convention, `Ls = 0 deg` is northern vernal equinox and Mars is
near perihelion at `Ls ~= 251 deg`. The runtime therefore maps:

```text
nu = 0 deg      -> Ls = 251 deg  -> perihelion
nu = 180 deg    -> Ls = 71 deg   -> aphelion
```

The heliocentric distance uses the Kepler ellipse:

```text
r(nu) = a * (1 - e^2) / (1 + e * cos(nu))
```

and the normal-incidence solar flux at Mars is:

```text
F_Mars = S_1AU * (1 AU / r)^2
```

## Elliptical orbit diagram

```text
                         aphelion
                      Ls ~= 71 deg
                     r ~= a(1 + e)
                      F_TOA minimum
                            *
                         .     .
                      .           .
        Sun *------*-----------------*------ Mars orbit
              perihelion
             Ls ~= 251 deg
             r ~= a(1 - e)
             F_TOA maximum
```

Mars's eccentricity (`e = 0.0934`) produces a large seasonal solar-flux
contrast. With the current constants (`a = 2.279392e11 m`, `S_1AU = 1361 W/m2`):

| Season point | Runtime angle | `Ls` | Distance (AU) | Normal TOA flux (W/m2) |
|---|---:|---:|---:|---:|
| Perihelion | `0 deg` | `251 deg` | `~1.381` | `~713` |
| Aphelion | `180 deg` | `71 deg` | `~1.666` | `~491` |

This is a `~45%` increase from aphelion to perihelion. The model therefore
matches the intended Ls-based seasonal pattern: solar flux peaks near southern
summer (`Ls ~= 251 deg`) and bottoms near northern summer (`Ls ~= 71 deg`).

## Current approximation

`Planet.advance_orbit()` increments `orbital_angle` at constant mean motion but
then uses that angle as if it were true anomaly in the ellipse. This preserves
the correct flux extrema and Ls phase, but it does not solve Kepler's equation.
The documented consequence is a seasonal timing phase error of roughly
`5-10` sols for Mars.

## Constants audit

| Quantity | Runtime value | Audit result |
|---|---:|---|
| Solar constant `S_1AU` | `1361 W/m2` | Consistent with Kopp and Lean (2011), which gives `1360.8 +/- 0.5 W/m2`. |
| Mars albedo `alpha` | `0.25` | Reasonable global baseline against TES/MGS albedo products from Christensen et al. (2001); regional values vary substantially. |
| Atmospheric transmittance `tau_atm` | `0.55` | Treated as an effective thin/dusty-atmosphere parameter, consistent with the simplified solar-radiation docs; it is not a full radiative-transfer model. |

## References

- Allison, M. and McEwen, M. (2000), Planetary and Space Science, doi: <https://doi.org/10.1016/S0032-0633(99)00092-6>
- Kopp, G. and Lean, J. L. (2011), Geophysical Research Letters, doi: <https://doi.org/10.1029/2010GL045777>
- Christensen, P. R. et al. (2001), Journal of Geophysical Research, doi: <https://doi.org/10.1029/2000JE001370>
- Haberle, R. M. et al. (1993), Journal of Geophysical Research, doi: <https://doi.org/10.1029/92JE02679>
