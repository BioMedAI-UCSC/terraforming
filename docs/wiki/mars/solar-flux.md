# Solar Flux at Mars

This page derives the solar flux model used at each timestep in the Mars climate simulation. The general theory is in [Solar Radiation](../solar-radiation.md); this page applies it to Mars-specific conditions with calibrated numerical values.

---

## Orbital distance

Mars's distance from the Sun at solar longitude $L_s$ ([Allison & McEwen, 2000](https://doi.org/10.1016/S0032-0633(99)00092-6)):

$$
r(L_s) = \frac{a\,(1 - e^2)}{1 + e\cos L_s}
$$

With $a = 1.524\,\text{AU}$ and $e = 0.0934$:

| $L_s$ | Event | $r$ (AU) |
|--------|-------|----------|
| $0°$ | N. spring equinox | $1.517$ |
| $71°$ | Aphelion | $1.666$ |
| $180°$ | N. autumn equinox | $1.517$ |
| $251°$ | Perihelion | $1.381$ |

---

## Top-of-atmosphere incident flux

The TOA incident shortwave flux on a horizontal surface ([Wikipedia: Solar irradiance](https://en.wikipedia.org/wiki/Solar_irradiance)):

$$
F_\text{TOA} = \frac{S_{1\,\text{AU}}}{r(L_s)^2}\,\max\!\bigl(0,\,\cos\theta_z\bigr)
$$

where $S_{1\,\text{AU}} = 1361\,\text{W\,m}^{-2}$ ([Kopp & Lean, 2011](https://doi.org/10.1029/2010GL045777)) and $\theta_z$ is the solar zenith angle. The normal-incidence TOA flux at $L_s \approx 0°$ is:

$$
F_\text{normal} = \frac{1361}{1.517^2} \approx 591.5\,\text{W\,m}^{-2}
$$

At perihelion ($r = 1.381\,\text{AU}$) this rises to $\approx 713\,\text{W\,m}^{-2}$, a 21% increase.

---

## Surface incident flux

The Martian atmosphere (thin, dusty CO₂) transmits a fraction $\tau_\text{atm}$ of the TOA flux to the surface ([Haberle et al., 1993](https://doi.org/10.1029/92JE02679)):

$$
F_\text{sfc} = F_\text{TOA} \cdot \tau_\text{atm}
$$

The baseline value $\tau_\text{atm} = 0.55$ is representative of moderate dust opacity ($\tau_\text{dust} \approx 0.5$). During global dust storms, $\tau_\text{atm}$ can drop below $0.2$.

---

## Reflected shortwave

The surface reflects a fraction $\alpha$ of incident shortwave ([Wikipedia: Albedo](https://en.wikipedia.org/wiki/Albedo)):

$$
F_\text{refl} = \alpha \cdot F_\text{sfc}
$$

Mars's global mean albedo is $\alpha \approx 0.25$, though regional values range from $0.10$ (dark basalt) to $0.45$ (bright dust, polar caps) ([Christensen et al., 2001](https://doi.org/10.1029/2000JE001368)).

---

## Zenith-angle reference table

At $L_s \approx 0°$ ($F_\text{normal} = 591.5\,\text{W\,m}^{-2}$), $\tau_\text{atm} = 0.55$, $\alpha = 0.25$:

| $\theta_z$ | $F_\text{TOA}$ | $F_\text{sfc}$ | $F_\text{abs}$ |
|-----------|---------------|---------------|---------------|
| $0°$ | $591.5\,\text{W\,m}^{-2}$ | $325.3\,\text{W\,m}^{-2}$ | $244.0\,\text{W\,m}^{-2}$ |
| $30°$ | $512.1\,\text{W\,m}^{-2}$ | $281.7\,\text{W\,m}^{-2}$ | $211.3\,\text{W\,m}^{-2}$ |
| $60°$ | $295.8\,\text{W\,m}^{-2}$ | $162.7\,\text{W\,m}^{-2}$ | $122.0\,\text{W\,m}^{-2}$ |
| $90°$ | $\approx 0$ | $\approx 0$ | $\approx 0$ |

At perihelion with $r = 1.381\,\text{AU}$ the $0°$ values increase to $F_\text{TOA} \approx 713\,\text{W\,m}^{-2}$, matching the baseline used in the evolve-1hr diagnostic.

---

## Implementation

This model is computed at each integration timestep in the Mars ODE. See [`src.celestials`](../../api/celestials.md) for the implementation and [`src.framework.orbital`](../../api/framework.md) for the orbital distance calculation.
