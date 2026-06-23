# Elevation, Pressure, and Temperature on Mars

Mars has extreme relief: deep impact basins sit several kilometres below the
reference datum, while Tharsis and Olympus Mons rise far above it. Because the
atmosphere is thin, these elevation differences strongly affect local surface
pressure and can also influence expected surface temperature, frost stability,
and volatile exchange.

---

## Pressure and the barometric formula

For an isothermal atmosphere in hydrostatic balance, pressure decreases
approximately exponentially with height:

$$
P(z) = P_0 \exp\left(-\frac{z}{H}\right)
$$

where:

| Symbol | Meaning | Unit |
|--------|---------|------|
| $P(z)$ | Surface pressure at elevation $z$ | Pa |
| $P_0$ | Reference pressure at datum | Pa |
| $z$ | Elevation above the reference datum | m |
| $H$ | Atmospheric scale height | m |

The scale height is:

$$
H = \frac{R_\text{specific} T}{g}
$$

For present-day Mars, a CO2-dominated atmosphere at about $210\,\text{K}$ gives
a scale height near $11\,\text{km}$. The current simplified package model uses
$H = 11\,100\,\text{m}$ when applying the initial elevation correction.

---

## Example pressure effects

Using $H = 11\,100\,\text{m}$:

| Terrain example | Elevation $z$ | $P(z) / P_0$ | Interpretation |
|----------------|---------------|--------------|----------------|
| Low basin | $-7\,\text{km}$ | $1.88$ | About 88% higher pressure than datum |
| Datum plain | $0\,\text{km}$ | $1.00$ | Reference pressure |
| High plateau | $+5\,\text{km}$ | $0.64$ | About 36% lower pressure than datum |
| Olympus-scale summit | $+21\,\text{km}$ | $0.15$ | About 85% lower pressure than datum |

This is why low regions such as Hellas Basin can have substantially higher
surface pressure than high volcanic terrain, even under the same global
atmospheric mass.

---

## Temperature influence

Elevation can also influence temperature. A dry adiabatic estimate gives:

$$
\Gamma_d = \frac{g}{c_p}
$$

For CO2 under Martian conditions this is roughly $4-5\,\text{K km}^{-1}$.
Interpreted as a simple lapse-rate estimate, a site $5\,\text{km}$ above datum
could be tens of kelvin colder than a comparable datum site, while a deep basin
could be warmer.

This is only a first-order guide. Real near-surface Mars temperatures also
depend on local time, season, dust, slope, albedo, thermal inertia, winds, and
boundary-layer stability. The near-surface lapse rate can weaken or invert,
especially at night.

---

## Physical consequences

Elevation-driven pressure and temperature differences affect:

- CO2 and H2O frost stability.
- Sublimation and condensation rates.
- The probability of transient liquid-water conditions under warmer scenarios.
- Atmospheric density, which affects heat transport and aerodynamic drag.
- Site comparisons between basins, plains, crater floors, highlands, and
  volcanic summits.

A model that ignores elevation can still represent broad global tendencies, but
it will miss important local behaviour at named sites.

---

## What tform currently models

The package accepts `elevation_m` when creating a `Mars` instance. It uses that
value once to correct the initial pressure:

$$
P_\text{initial} = P_\text{ref}\exp\left(-\frac{\text{elevation_m}}{11\,100}\right)
$$

That corrected pressure then enters the model state as surface pressure.
Batched simulations inherit the corrected value because they stack the
initialized state of each `Mars` instance.

---

## What tform does not yet model

The current package implementation does not yet include:

- MOLA DEM lookup from latitude and longitude.
- Elevation retained as an explicit evolving state or diagnostic field.
- Temperature-dependent pressure scale height updates.
- Temperature lapse-rate correction with altitude.
- Slope, aspect, horizon shading, or local terrain illumination.
- Terrain-dependent thermal inertia, albedo, or dust/ice surface class.

For now, named-location simulations only reflect elevation if `elevation_m` is
provided explicitly, and they only reflect it through the initial pressure
correction.

---

## Implementation

The current elevation pressure correction is applied during `Mars`
initialization. See [`src.celestials`](../../api/celestials.md) for the Mars
implementation and [Mars Architecture](../../architecture/mars.md) for the
model-scope notes.
