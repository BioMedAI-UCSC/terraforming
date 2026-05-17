# GHG Interventions on Mars

Warming Mars to temperatures compatible with liquid water requires raising the surface temperature by at least $\sim 60\,\text{K}$ (from $\sim 210\,\text{K}$ to $\sim 273\,\text{K}$). The most practical near-term pathway is injecting synthetic super-greenhouse gases (GHGs) into the Martian atmosphere — a strategy first rigorously analysed by [Marinova, McKay, & Hashimoto (2005)](https://doi.org/10.1029/2004JD005027) and [McKay et al. (1991)](https://doi.org/10.1038/352489a0).

---

## Why super-GHGs on Mars?

The natural Martian greenhouse effect is weak: the thin CO₂ atmosphere produces only $\sim 5\,\text{K}$ of warming above the bare-rock equilibrium, compared to $\sim 33\,\text{K}$ on Earth. This is because:

1. Mars's CO₂ atmosphere is 170× thinner than Earth's by mass, so it absorbs very little outgoing IR.
2. There is no water-vapour feedback — Mars's water inventory is locked in ice.
3. The dominant IR window ($8$–$12\,\mu\text{m}$) is essentially transparent on Mars.

Super-GHGs are effective precisely because they absorb strongly in this window region ([Pierrehumbert & Gaidos, 2011](https://doi.org/10.1088/2041-8205/734/1/L13)).

---

## Radiative forcing model

The forcing from a gas at atmospheric concentration $C_i$ (ppb by volume) is:

$$
\Delta F_i = \eta_i \cdot C_i
$$

where $\eta_i$ (W m⁻² ppb⁻¹) is the **Mars-specific radiative forcing efficiency** from [Marinova et al. (2005)](https://doi.org/10.1029/2004JD005027). These differ from IPCC Earth values because the absence of water-vapour overlap bands and the thinner Martian CO₂ column change the spectral windows available for absorption.

The total forcing from all injected species:

$$
\Delta F_\text{total} = \sum_i \eta_i \cdot C_i
$$

---

## Concentration from injected mass

Given an annual injection rate $\dot{m}_i$ (kg yr⁻¹) of species $i$ with molar mass $M_i$ (kg mol⁻¹), the steady-state atmospheric concentration (in ppb) after accounting for atmospheric loss with lifetime $\tau_i$ (yr) is:

$$
C_i^\text{steady} = \frac{\dot{m}_i\,\tau_i}{M_\text{atm}/M_i} \times 10^9
$$

where $M_\text{atm} = P \cdot 4\pi R^2 / g$ is the total atmospheric mass ($\approx 2.5 \times 10^{16}\,\text{kg}$ at current Martian pressure, [Mahaffy et al., 2013](https://doi.org/10.1126/science.1237966)).

The transient concentration at time $t$ under continuous injection evolves as:

$$
\frac{dC_i}{dt} = \frac{\dot{m}_i}{M_\text{atm}/M_i} \times 10^9 - \frac{C_i}{\tau_i}
$$

---

## Compound registry

| Compound | Formula | Lifetime $\tau$ (yr) | Notes |
|----------|---------|---------------------|-------|
| CF₄ | Tetrafluoromethane | >50,000 | Strongest long-lived candidate; GWP 6,630 ([IPCC AR6](https://www.ipcc.ch/report/ar6/wg1/)) |
| SF₆ | Sulfur hexafluoride | 3,200 | GWP 23,500; spectral absorption in 8–10 µm window |
| C₂F₆ | Hexafluoroethane | 10,000 | GWP 11,100 |
| NF₃ | Nitrogen trifluoride | 500 | GWP 17,400 |
| C₃F₈ | Octafluoropropane | 2,600 | — |
| CHF₃ | Trifluoromethane | 228 | — |
| CH₂F₂ | Difluoromethane | 5.2 | Short-lived |
| CH₄ | Methane | 12 | Low GWP; synergistic with CF₄ in some spectral bands |
| N₂O | Nitrous oxide | 114 | GWP 273 ([IPCC AR6](https://www.ipcc.ch/report/ar6/wg1/)) |

Lifetimes from [IPCC AR5 Annex II](https://www.ipcc.ch/report/ar5/wg1/) and [Ravishankara et al. (1993)](https://doi.org/10.1126/science.259.5092.194). Mars atmospheric lifetimes may differ from Earth values due to the different UV environment and the absence of OH radical sinks.

---

## Greenhouse factor update

After each year of simulation, the greenhouse factor $\gamma$ is updated based on the total accumulated forcing:

$$
\gamma_\text{new} = \gamma_\text{base} \cdot \left(1 + \frac{\Delta F_\text{total}}{F_\text{ref}}\right)
$$

This $\gamma$ is then fed into the temperature ODE for the next year's integration (see [Climate Model](climate-model.md)).

---

## Terraforming phase 1 target

A commonly cited first milestone is reaching $\sim 273\,\text{K}$ mean surface temperature and $\sim 1000\,\text{Pa}$ surface pressure — conditions under which liquid water is stable at low elevations and the CO₂ caps begin to fully sublime, releasing additional CO₂ that further amplifies the greenhouse effect through a positive feedback ([McKay et al., 1991](https://doi.org/10.1038/352489a0)).

[Marinova et al. (2005)](https://doi.org/10.1029/2004JD005027) estimated that injecting $\sim 300\,\text{ppb}$ of CF₄ into the Martian atmosphere would provide $\sim 25\,\text{W\,m}^{-2}$ of additional forcing — enough to initiate significant polar cap sublimation, releasing a much larger CO₂ reservoir that then provides additional forcing through the cap feedback.

---

## Implementation

GHG compounds are registered in [`src.interventions.compounds`](../../api/interventions.md). Concentration calculations are in `src.interventions.forcing`, and the annual injection schedule is managed by `src.interventions.controller`. See the [Interventions API](../../api/interventions.md) for the full interface.
