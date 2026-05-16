# Solar Radiation

Solar radiation is the primary energy input to any planetary climate system. This page covers the geometry and physics of how solar energy reaches a planetary surface, from the top of atmosphere down to the ground. These relations are applied in the Mars model but hold generically for any planet.

---

## Solar constant and the inverse-square law

The solar irradiance at a distance $r$ from the Sun follows the [inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law):

$$
F_\odot(r) = \frac{S_{1\,\text{AU}}}{r^2}
$$

where $S_{1\,\text{AU}} = 1361\,\text{W\,m}^{-2}$ is the [solar constant](https://en.wikipedia.org/wiki/Solar_constant) — the total solar irradiance at 1 AU ([Kopp & Lean, 2011](https://doi.org/10.1029/2010GL045777)) — and $r$ is in AU.

At Mars's semi-major axis ($a = 1.524\,\text{AU}$), the time-averaged flux is approximately $\bar{F} \approx 586\,\text{W\,m}^{-2}$, dropping to $\sim 493\,\text{W\,m}^{-2}$ at aphelion and rising to $\sim 718\,\text{W\,m}^{-2}$ at perihelion.

---

## Top-of-atmosphere flux

The top-of-atmosphere (TOA) incident shortwave flux on a horizontal surface depends on the solar zenith angle $\theta_z$ ([Wikipedia: Air mass (solar energy)](https://en.wikipedia.org/wiki/Air_mass_(solar_energy))):

$$
F_\text{TOA}(\theta_z) = \frac{S_{1\,\text{AU}}}{r^2}\,\max\!\bigl(0,\,\cos\theta_z\bigr)
$$

The $\max(0,\cdot)$ term enforces that there is no downward solar flux below the horizon ($\theta_z > 90°$).

---

## Atmospheric transmittance

A fraction of TOA radiation is absorbed and scattered by the atmosphere before reaching the surface. The surface incident shortwave flux is:

$$
F_\text{sfc} = F_\text{TOA} \cdot \tau_\text{atm}
$$

where $\tau_\text{atm} \in [0, 1]$ is the effective atmospheric transmittance. For current Mars (thin, dusty atmosphere), a representative value is $\tau_\text{atm} \approx 0.55$ ([Haberle et al., 1993](https://doi.org/10.1029/92JE02679)).

Dust storms substantially reduce $\tau_\text{atm}$, causing surface cooling even while the upper atmosphere warms — a key phenomenon in Martian meteorology.

---

## Surface albedo and absorbed flux

The surface reflects a fraction $\alpha$ of incident shortwave radiation ([Wikipedia: Albedo](https://en.wikipedia.org/wiki/Albedo)). The net absorbed shortwave flux is:

$$
F_\text{abs} = F_\text{sfc}\,(1 - \alpha)
$$

Mars's global mean albedo is $\alpha \approx 0.25$, though it varies from $\sim 0.10$ in dark basaltic regions to $\sim 0.45$ over bright dust deposits and polar caps ([Christensen et al., 2001](https://doi.org/10.1029/2000JE001368)).

---

## Numerical example at Mars baseline

Using $S_{1\,\text{AU}} = 1361\,\text{W\,m}^{-2}$, $r \approx 1.38\,\text{AU}$ (near $L_s = 0°$), $\tau_\text{atm} = 0.55$, $\alpha = 0.25$:

| $\theta_z$ | $F_\text{TOA}$ | $F_\text{sfc}$ | $F_\text{abs}$ |
|-----------|---------------|---------------|---------------|
| $0°$ | $713.2\,\text{W\,m}^{-2}$ | $392.3\,\text{W\,m}^{-2}$ | $294.2\,\text{W\,m}^{-2}$ |
| $30°$ | $617.7\,\text{W\,m}^{-2}$ | $339.7\,\text{W\,m}^{-2}$ | $254.8\,\text{W\,m}^{-2}$ |
| $60°$ | $356.6\,\text{W\,m}^{-2}$ | $196.1\,\text{W\,m}^{-2}$ | $147.1\,\text{W\,m}^{-2}$ |
| $90°$ | $\approx 0$ | $\approx 0$ | $\approx 0$ |

---

## Implementation

These equations are computed at each timestep in the Mars model. See [Solar Flux — Mars](mars/solar-flux.md) for the Mars-specific derivation, and [`src.framework.orbital`](../api/framework.md) for the orbital distance calculation.
