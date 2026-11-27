# Interventions API

!!! warning "Deprecated global-mean workflow"
    This page documents the deprecated torch model or its dependent intervention
    workflow. Use the [3-D GCM](../package/gcm3d/README.md) for new simulations.


The `src.interventions` module tracks greenhouse-gas injection in the deprecated
global-mean workflow. Its efficiencies and Earth-year lifetimes use IPCC AR6
WG1 Table 7.SM.7 as explicit Earth-reference proxies. Their transfer to Mars
and to large concentrations is not validated.

## Radiative Forcing Model

The net greenhouse amplification from injected GHGs:

$$
\Delta F_\text{total} = \sum_i \eta_i \cdot C_i
$$

where $\eta_i$ is the Earth-reference trace-gas radiative efficiency
(W m⁻² ppb⁻¹) and $C_i$ is molar concentration in ppb. Stored species column
weights are converted to mole counts before concentration is calculated.

The updated greenhouse factor fed into the temperature ODE:

$$
\gamma_\text{new} = \gamma_\text{base} \cdot \left(1 + \frac{\Delta F_\text{total}}{F_\text{ref}}\right)^{1/4}
$$

## Supported Compounds

| Compound | Lifetime (yr) | Notes |
|----------|---------------|-------|
| CF4 | 50,000 | Earth reference |
| SF6 | 1,000 | Earth reference |
| C2F6 | 10,000 | — |
| NF3 | 569 | Earth reference |
| C3F8 | 2,600 | Earth reference |
| C4F10 | 2,600 | Earth reference |
| C6F14 | 3,100 | Earth reference |

See `list_compounds()` for current values loaded at runtime.

## GHG Compounds Registry

::: src.interventions.compounds

## Radiative Forcing

::: src.interventions.forcing

## Intervention Controller

::: src.interventions.controller
