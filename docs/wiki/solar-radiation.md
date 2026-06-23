# Solar Radiation

Solar radiation is the primary energy input to any planetary climate system. This page covers the geometry and physics of how solar energy reaches a planetary surface, from the top of atmosphere down to the ground. These relations are applied in the Mars model but hold generically for any planet.

---

## Solar constant and the inverse-square law

The solar irradiance at a distance $r$ from the Sun follows the [inverse-square law](https://en.wikipedia.org/wiki/Inverse-square_law):

$$
F_\odot(r) = \frac{S_{1\,\text{AU}}}{r^2}
$$

where $S_{1\,\text{AU}} = 1361\,\text{W\,m}^{-2}$ is the [solar constant](https://en.wikipedia.org/wiki/Solar_constant) — the total solar irradiance at 1 AU ([Kopp & Lean, 2011](https://doi.org/10.1029/2010GL045777)) — and $r$ is in AU.

At Mars's semi-major axis ($a = 1.524\,\text{AU}$), the flux at mean orbital distance is approximately $586\,\text{W\,m}^{-2}$, dropping to $\sim 490\,\text{W\,m}^{-2}$ at aphelion and rising to $\sim 713\,\text{W\,m}^{-2}$ at perihelion.

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

The simplified Mars model uses a representative global shortwave albedo $\alpha \approx 0.25$. Real Martian surface albedo varies strongly by terrain and season, with darker basaltic regions absorbing more sunlight and bright dust or polar deposits reflecting more. This value should be treated as an effective model parameter unless tied to a specific Bond-albedo or surface-albedo data product.

---

## Numerical example at Mars baseline

Using $S_{1\,\text{AU}} = 1361\,\text{W\,m}^{-2}$, $r \approx 1.558\,\text{AU}$ (near $L_s = 0°$), $\tau_\text{atm} = 0.55$, $\alpha = 0.25$:

| $\theta_z$ | $F_\text{TOA}$ | $F_\text{sfc}$ | $F_\text{abs}$ |
|-----------|---------------|---------------|---------------|
| $0°$ | $560.9\,\text{W\,m}^{-2}$ | $308.5\,\text{W\,m}^{-2}$ | $231.4\,\text{W\,m}^{-2}$ |
| $30°$ | $485.7\,\text{W\,m}^{-2}$ | $267.1\,\text{W\,m}^{-2}$ | $200.3\,\text{W\,m}^{-2}$ |
| $60°$ | $280.4\,\text{W\,m}^{-2}$ | $154.2\,\text{W\,m}^{-2}$ | $115.7\,\text{W\,m}^{-2}$ |
| $90°$ | $\approx 0$ | $\approx 0$ | $\approx 0$ |

---

## Implementation

These equations are computed at each timestep in the Mars model. See [Solar Flux — Mars](mars/solar-flux.md) for the Mars-specific derivation, and [`src.framework.orbital`](../api/framework.md) for the orbital distance calculation.
