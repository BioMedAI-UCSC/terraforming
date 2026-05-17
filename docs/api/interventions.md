# Interventions API

The `src.interventions` module models super-greenhouse gas (GHG) injection and its radiative effects on the Martian climate. Radiative forcing efficiencies are calibrated to Mars conditions (Marinova et al. 2005) — they differ from Earth IPCC values because Mars lacks water-vapour overlap bands and has a much thinner CO₂ column.

## Radiative Forcing Model

The net greenhouse amplification from injected GHGs:

$$
\Delta F_\text{total} = \sum_i \eta_i \cdot C_i
$$

where $\eta_i$ is the Mars-specific radiative forcing efficiency (W m⁻² ppb⁻¹) and $C_i$ is the concentration in ppb.

The updated greenhouse factor fed into the temperature ODE:

$$
\gamma_\text{new} = \gamma_\text{base} \cdot \left(1 + \frac{\Delta F_\text{total}}{F_\text{ref}}\right)
$$

## Supported Compounds

| Compound | Lifetime (yr) | Notes |
|----------|---------------|-------|
| CF4 | >50,000 | Extremely long-lived |
| SF6 | 3,200 | High GWP |
| C2F6 | 10,000 | — |
| NF3 | 500 | — |
| C3F8, CHF3, CH2F2, CH3F | varies | HFC/PFC family |
| CH4 | 12 | Short-lived, synergistic |
| N2O | 114 | — |
| C3H8, CH3Cl | varies | — |

See `list_compounds()` for current values loaded at runtime.

## GHG Compounds Registry

::: src.interventions.compounds

## Radiative Forcing

::: src.interventions.forcing

## Intervention Controller

::: src.interventions.controller
