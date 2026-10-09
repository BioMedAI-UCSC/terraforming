# Mars

Mars is a terrestrial planet with a thin CO₂ atmosphere, extreme diurnal and seasonal temperature swings, a low-pressure surface environment, and extensive polar CO₂ and water ice deposits. It is the primary target of the tform simulator.

---

## Physical constants

All values from the [NASA Mars Fact Sheet](https://nssdc.gsfc.nasa.gov/planetary/factsheet/marsfact.html):

| Property | Symbol | Value | Unit |
|----------|--------|-------|------|
| Mass | $M$ | $6.4171 \times 10^{23}$ | kg |
| Mean radius | $R$ | $3\,389\,500$ | m |
| Surface gravity | $g$ | $3.721$ | m s⁻² |
| Sidereal rotation period | — | $88\,642.66$ | s (~24h 37m) |
| Orbital period | — | $686.97$ | Earth days |
| Semi-major axis | $a$ | $1.524$ | AU |
| Orbital eccentricity | $e$ | $0.0934$ | — |
| Axial tilt | $\varepsilon$ | $25.19°$ | — |

---

## Current atmospheric state

Mars today has a surface pressure of approximately $636\,\text{Pa}$ — less than 1% of Earth's $101\,325\,\text{Pa}$. The atmosphere is $\sim 95\%$ CO₂ with trace N₂, Ar, and O₂ ([Mahaffy et al., 2013](https://doi.org/10.1126/science.1237966)):

| Species | Volume mixing ratio |
|---------|-------------------|
| CO₂ | 95.32% |
| N₂ | 2.60% |
| Ar | 1.93% |
| O₂ | 0.13% |
| CO | 0.08% |

Mean surface temperature is approximately $210\,\text{K}$ ($-63°\text{C}$), with diurnal swings of 60–100 K and seasonal swings driven by CO₂ cap cycling.

The implementation uses `MARS_ROTATION_PERIOD = 88775.244 s` as its model sol
and `MARS_ORBITAL_PERIOD = 59356800 s` (about 668.62 model sols). These differ
from the sidereal rotation value in the fact-sheet table; use the model constants
when reproducing its clock and timestep settings.

---

## Mars in tform

The [3-D GCM](../../package/gcm3d/README.md) is the primary model. It evolves
spatial circulation, atmospheric/surface temperature, pressure and frost over
MOLA terrain. Start with [GCM setup](../../getting-started/quickstart.md) and
read [validation limits](../../package/gcm3d/validation.md). The torch global-mean
model and its injection workflows are deprecated.

| Topic | Page |
|-------|------|
| Solar flux and zenith angle model | [Solar Flux](solar-flux.md) |
| Spatial dynamics and column physics | [Climate Model](climate-model.md) |
| Deprecated global-mean GHG injection | [GHG Interventions](interventions.md) |
| API reference | [Mars API](../../api/celestials.md) |
| Architecture | [Mars Architecture](../../architecture/mars.md) |

---

## Further reading

- [NASA Mars Exploration — Facts](https://mars.nasa.gov/all-about-mars/facts/)
- [Wikipedia: Mars](https://en.wikipedia.org/wiki/Mars)
- [Wikipedia: Atmosphere of Mars](https://en.wikipedia.org/wiki/Atmosphere_of_Mars)
- [Haberle, R.M. (1998). Early Mars climate models. *Journal of Geophysical Research: Planets*, 103(E12).](https://doi.org/10.1029/98JE01388)
