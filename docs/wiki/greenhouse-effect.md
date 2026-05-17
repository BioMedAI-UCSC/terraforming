# Greenhouse Effect

The greenhouse effect is the mechanism by which atmospheric gases trap outgoing thermal infrared radiation and re-emit part of it back toward the surface, raising surface temperatures above the bare-rock radiative equilibrium. It is fundamental to understanding both the current Martian climate and the warming trajectory under any terraforming scenario.

---

## Blackbody radiation and the Stefan-Boltzmann law

Any body at temperature $T$ emits thermal radiation. For a perfect blackbody, the total power emitted per unit area is given by the [Stefan-Boltzmann law](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_law) ([Stefan, 1879](https://en.wikipedia.org/wiki/Josef_Stefan); [Boltzmann, 1884](https://en.wikipedia.org/wiki/Ludwig_Boltzmann)):

$$
F = \sigma T^4
$$

where $\sigma = 5.670 \times 10^{-8}\,\text{W\,m}^{-2}\,\text{K}^{-4}$ is the [Stefan-Boltzmann constant](https://en.wikipedia.org/wiki/Stefan%E2%80%93Boltzmann_constant).

Real surfaces are characterised by emissivity $\varepsilon \in [0, 1]$, so the actual upwelling IR flux is:

$$
F_\text{IR,up} = \varepsilon\,\sigma\,T^4
$$

For bare Martian regolith, $\varepsilon \approx 0.95$ ([Putzig & Mellon, 2007](https://doi.org/10.1016/j.icarus.2007.01.022)).

---

## Radiative equilibrium temperature

Without an atmosphere, a planet's equilibrium temperature $T_\text{eq}$ is set by the balance between absorbed solar radiation and emitted thermal IR:

$$
T_\text{eq} = \left[\frac{S_{1\,\text{AU}}(1-\alpha)}{4\,\sigma\,r^2}\right]^{1/4}
$$

The factor of 4 accounts for the ratio of the planet's cross-sectional area (which intercepts sunlight) to its total surface area (which radiates). For Mars, this gives $T_\text{eq} \approx 210\,\text{K}$, close to the observed mean surface temperature of about $210\,\text{K}$ — Mars has a very weak natural greenhouse effect today ([Haberle, 1998](https://doi.org/10.1029/98JE01388)).

---

## The greenhouse effect

Real atmospheres absorb some fraction of the upwelling surface IR and re-emit it both upward and downward. The downwelling component adds energy back to the surface, raising $T$ above $T_\text{eq}$. The effective downwelling IR flux from the atmosphere is:

$$
F_\text{IR,down} = \varepsilon_\text{atm}\,\sigma\,T^4
$$

where $\varepsilon_\text{atm}$ is the effective atmospheric IR emissivity ($0$ = transparent, $1$ = fully opaque). The net surface energy budget then includes both fluxes:

$$
F_\text{net} = F_\text{abs} + F_\text{IR,down} - F_\text{IR,up}
$$

In tform, the greenhouse amplification is represented by a single dimensionless factor $\gamma \geq 1$ that scales the effective absorbed solar flux:

$$
\frac{dT}{dt} = \frac{F_\text{net}}{C_\text{eff}}
$$

---

## Radiative forcing from greenhouse gases

When new greenhouse gases are added to an atmosphere, the change in net downward radiative flux at the tropopause is called the **radiative forcing** $\Delta F$ (W m⁻²). Positive $\Delta F$ warms the surface ([IPCC AR6, Chapter 7](https://www.ipcc.ch/report/ar6/wg1/chapter/chapter-7/)).

For a gas at concentration $C$ (in ppb), the forcing scales approximately linearly at low concentrations:

$$
\Delta F = \eta \cdot C
$$

where $\eta$ is the **radiative forcing efficiency** (W m⁻² ppb⁻¹). This is the quantity tabulated by [Marinova et al. (2005)](https://doi.org/10.1029/2004JD005027) for Mars-specific conditions (where the absence of water-vapour overlap bands and the thinner CO₂ column make $\eta$ differ significantly from Earth IPCC values).

The total forcing from a mixture of gases is the sum over all species:

$$
\Delta F_\text{total} = \sum_i \eta_i \cdot C_i
$$

---

## Global warming potential

The [Global Warming Potential (GWP)](https://en.wikipedia.org/wiki/Global_warming_potential) of a gas is its time-integrated forcing relative to CO₂ over a 100-year horizon. For terraforming, GWP is less relevant than atmospheric lifetime: long-lived gases (CF₄ at >50,000 yr, SF₆ at 3,200 yr) are strongly preferred because injected quantities remain effective for geological timescales.

---

## Further reading

- [Pierrehumbert, R.T. (2010). *Principles of Planetary Climate*. Cambridge University Press.](https://doi.org/10.1017/CBO9780511780783)
- [Sagan, C. & Mullen, G. (1972). Earth and Mars: Evolution of Atmospheres and Surface Temperatures. *Science*, 177(4043).](https://doi.org/10.1126/science.177.4043.52)
- [Marinova, M.M., McKay, C.P., & Hashimoto, H. (2005). Radiative-convective model of warming Mars with artificial greenhouse gases. *Journal of Geophysical Research: Planets*, 110(E3).](https://doi.org/10.1029/2004JD005027)
