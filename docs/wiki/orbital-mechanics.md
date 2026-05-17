# Orbital Mechanics

Planetary orbits are described by [Keplerian orbital elements](https://en.wikipedia.org/wiki/Orbital_elements). For a planet on an elliptical orbit, the key elements are the semi-major axis $a$, orbital eccentricity $e$, and the true anomaly $\nu$ (the angle from perihelion). In tform the true anomaly is parameterised through the **solar longitude** $L_s$, which is the standard convention in Mars science ([Allison & McEwen, 2000](https://doi.org/10.1016/S0032-0633(99)00092-6)).

---

## Orbital distance

The instantaneous Sun–planet distance $r$ as a function of solar longitude $L_s$ follows directly from the [vis-viva equation](https://en.wikipedia.org/wiki/Vis-viva_equation) and the geometry of an ellipse ([Wikipedia: Elliptic orbit](https://en.wikipedia.org/wiki/Elliptic_orbit)):

$$
r(L_s) = \frac{a\,(1 - e^2)}{1 + e\cos L_s}
$$

| Symbol | Meaning | Unit |
|--------|---------|------|
| $r$ | Sun–planet distance | AU |
| $a$ | Semi-major axis | AU |
| $e$ | Orbital eccentricity | dimensionless |
| $L_s$ | Solar longitude (true anomaly from perihelion) | rad |

For Mars: $a = 1.524\,\text{AU}$, $e = 0.0934$ ([NASA Mars Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html)). Mars's high eccentricity causes a ~19% variation in solar flux between perihelion ($L_s = 251°$) and aphelion ($L_s = 71°$), driving strong seasonal asymmetry.

---

## Solar longitude and Martian seasons

$L_s$ runs from $0°$ (northern spring equinox) to $360°$. Key events:

| $L_s$ | Event |
|-------|-------|
| $0°$ | Northern spring equinox |
| $90°$ | Northern summer solstice |
| $180°$ | Northern autumn equinox |
| $251°$ | **Perihelion** (closest to Sun, southern summer) |
| $270°$ | Northern winter solstice |

Because perihelion coincides with southern summer, the southern hemisphere receives more intense (but shorter) summers than the north — a major driver of the asymmetric CO₂ polar cap exchange.

---

## Solar zenith angle

The solar zenith angle $\theta_z$ is the angle between the local vertical and the direction to the Sun. It determines how much of the solar beam is intercepted per unit horizontal area ([Wikipedia: Solar zenith angle](https://en.wikipedia.org/wiki/Solar_zenith_angle)):

$$
\cos\theta_z = \sin\phi\sin\delta + \cos\phi\cos\delta\cos h
$$

| Symbol | Meaning |
|--------|---------|
| $\phi$ | Geographic latitude |
| $\delta$ | Solar declination (function of $L_s$ and axial tilt) |
| $h$ | Hour angle (local solar time) |

At $\theta_z = 0°$ the Sun is directly overhead; at $\theta_z = 90°$ it is on the horizon and the direct flux is zero.

---

## Axial tilt and declination

The solar declination $\delta$ oscillates with the axial tilt $\varepsilon$ over the course of the year:

$$
\delta = \varepsilon \sin L_s
$$

Mars has $\varepsilon = 25.19°$ ([NASA Mars Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html)), slightly larger than Earth's $23.45°$, contributing to pronounced seasonal temperature swings.

---

## Implementation

Orbital distance and zenith angle are computed in [`src.framework.orbital`](../api/framework.md) (`OrbitalParameters.distance_from_sun`) and used by the Mars physics model to set the instantaneous solar flux at each timestep.
