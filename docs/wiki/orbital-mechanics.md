# Orbital Mechanics

Planetary orbits are described by [Keplerian orbital elements](https://en.wikipedia.org/wiki/Orbital_elements). For a planet on an elliptical orbit, the key elements are the semi-major axis $a$, orbital eccentricity $e$, mean anomaly $M$, eccentric anomaly $E$, and true anomaly $\nu$ (the angle from perihelion).

For Mars, seasons are conventionally described by the **areocentric solar longitude** $L_s$, following the Allison & McEwen (2000) convention ([doi:10.1016/S0032-0633(99)00092-6](https://doi.org/10.1016/S0032-0633(99)00092-6)). $L_s = 0°$ is northern spring equinox. It is not measured from perihelion. In the current simplified Mars model, true anomaly is related to solar longitude by:

$$
\nu = L_s - L_{s,\text{perihelion}}
$$

with $L_{s,\text{perihelion}} \approx 251°$.

---

## Orbital distance

The instantaneous Sun–planet distance $r$ follows from the geometry of an ellipse ([Wikipedia: Elliptic orbit](https://en.wikipedia.org/wiki/Elliptic_orbit)):

$$
r(\nu) = \frac{a\,(1 - e^2)}{1 + e\cos \nu}
$$

| Symbol | Meaning | Unit |
|--------|---------|------|
| $r$ | Sun–planet distance | AU |
| $a$ | Semi-major axis | AU |
| $e$ | Orbital eccentricity | dimensionless |
| $\nu$ | True anomaly measured from perihelion | rad |

For Mars: $a = 1.524\,\text{AU}$, $e = 0.0934$ ([NASA Mars Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html)). Mars's high eccentricity causes perihelion flux to be about 45% higher than aphelion flux, driving strong seasonal asymmetry.

---

## Orbital time propagation

True anomaly does not advance uniformly in time for an eccentric orbit. The uniformly advancing quantity is mean anomaly:

$$
\frac{dM}{dt} = \frac{2\pi}{T_\text{orbital}}
$$

At each timestep, the model converts the current true anomaly $\nu$ to eccentric anomaly $E$:

$$
E = 2 \tan^{-1}\!\left(
    \sqrt{\frac{1-e}{1+e}} \tan\frac{\nu}{2}
\right)
$$

then computes mean anomaly:

$$
M = E - e\sin E
$$

The timestep advances $M$ by the mean motion:

$$
M_{t+\Delta t} = M_t + \frac{2\pi}{T_\text{orbital}}\Delta t
$$

The updated eccentric anomaly is recovered by solving Kepler's equation:

$$
M = E - e\sin E
$$

and then converted back to true anomaly:

$$
\nu = 2 \tan^{-1}\!\left(
    \sqrt{\frac{1+e}{1-e}} \tan\frac{E}{2}
\right)
$$

This implements Kepler's second law: Mars moves faster near perihelion and slower near aphelion.

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
