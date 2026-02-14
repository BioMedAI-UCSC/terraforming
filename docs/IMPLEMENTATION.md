# Mars Terraforming Simulation - Implementation Spec

This file defines implementation details for each phase in `docs/PLAN.md`.
For every phase: (1) input, (2) output, (3) modeling approach + equations + parameters, (4) data needed, (5) assumptions, (6) pre-requisites.

## Global notation
- Time: `t` (years unless stated)
- Mars radius: `R_M`
- Surface area: `A_M = 4 * pi * R_M^2`
- Atmospheric species mass: `M_i(t)` for species `i`
- Surface pressure: `P(t) = g_M * sum_i M_i(t) / A_M`
- State vector: `x(t)` (all coupled states)

---

## 1) Baseline Mars system
1. **Input**
   - Raw present-day Mars datasets (topography, atmosphere, radiation, ice, dust, regolith).
2. **Output**
   - Calibrated initial state `x(0)` + uncertainty bounds `Sigma_0`.
3. **Modeling approach (equations + parameters)**
   - Data assimilation / state estimation:
     - `x_hat = argmin_x (y - Hx)^T R^{-1} (y - Hx) + (x - x_b)^T B^{-1} (x - x_b)`
   - Parameters: `H` (observation operator), `R` (obs covariance), `B` (background covariance), `x_b` (background state).
4. **Data needed**
   - MOLA DEM, MCD (Mars Climate Database), MAVEN loss/radiation products, ice maps, dust climatology.
5. **Assumptions**
   - Datasets are temporally harmonized to one reference epoch.
   - Bias corrections are stable over simulation horizon.
6. **Pre-requisites**
   - None (foundation phase).

## 2) Resources and capability
1. **Input**
   - Mission constraints: budget, launch cadence, compute, staffing.
2. **Output**
   - Capacity envelope `C_cap(t)` for simulation and deployment throughput.
3. **Modeling approach (equations + parameters)**
   - Throughput constraint model:
     - `u_k(t) <= min(C_power_k(t), C_mass_k(t), C_labor_k(t), C_compute_k(t))`
   - Parameters: capacities per subsystem `C_*`, task demand coefficients `d_{k,*}`.
4. **Data needed**
   - Cost curves, launch vehicle specs, workforce assumptions, hardware availability.
5. **Assumptions**
   - Capacity is piecewise-constant over planning windows.
   - No hard geopolitical/supply-chain disruptions in baseline scenario.
6. **Pre-requisites**
   - Phase 1 baseline for realistic demand estimates.

## 3) Infrastructure enablers
1. **Input**
   - Capacity envelope + candidate infrastructure designs.
2. **Output**
   - Time series for available power, mined mass, and processed materials.
3. **Modeling approach (equations + parameters)**
   - Stock-flow ODEs:
     - `dE/dt = G_E(t) - L_E(t) - D_E(t)` (energy stock)
     - `dQ_ore/dt = m_dot_mine(t) - m_dot_proc(t)` (ore stock)
     - `dQ_mat/dt = eta_proc * m_dot_proc(t) - m_dot_use(t)` (usable materials)
   - Parameters: generation `G_E`, losses `L_E`, degradation `D_E`, processing efficiency `eta_proc`.
4. **Data needed**
   - Solar/nuclear generation profiles, ore grade maps, process efficiencies, maintenance intervals.
5. **Assumptions**
   - First-order degradation approximates equipment aging.
   - Transport delays can be represented as fixed lag or buffer stock.
6. **Pre-requisites**
   - Phases 1-2.

## 4) Magnetic shield program
1. **Input**
   - Solar wind conditions, shield architecture (equatorial superconducting loop), material limits.
2. **Output**
   - Shield design point + effectiveness `f_shield(t)` reducing atmospheric escape.
3. **Modeling approach (equations + parameters)**
   - Pressure-balance requirement:
     - `P_ram = rho_sw * v_sw^2`
     - `P_mag = B_mp^2 / (2 * mu_0)`
     - Shield condition: `P_mag >= P_ram`
   - Dipole field scaling:
     - `B(r) = B_0 * (R_M / r)^3`
   - Superconductor geometric constraint (compact form):
     - `B_0 / B_c ~= (pi * d * a^2) / (8 * R_M^3)`
     - where `a` = loop radius, `d` = bundle radius, `B_c` = critical field.
   - Cryogenic power:
     - `Q_dot = (k * A / L) * DeltaT`
   - Escape reduction coupling:
     - `E_i,eff = E_i,base * (1 - f_shield)`
   - Key parameters: `rho_sw`, `v_sw`, `mu_0`, `B_c`, `T_c`, `k`, `A`, `L`, `DeltaT`, uptime `U`.
4. **Data needed**
   - Solar wind statistics at Mars orbit, superconducting material properties, thermal insulation performance, failure rates.
5. **Assumptions**
   - Equatorial loop geometry is baseline architecture.
   - `f_shield` represented as scenario parameter or function of shield uptime/performance.
   - Non-superconducting fallback is modeled as high-power penalty scenario.
6. **Pre-requisites**
   - Phases 1-3.

## 5) Atmospheric mass budget
1. **Input**
   - Volatile sources, import plans, shield effectiveness, regolith exchange coefficients.
2. **Output**
   - `M_i(t)`, `P(t)`, net accumulation rates per species.
3. **Modeling approach (equations + parameters)**
   - Species-wise mass balance:
     - `dM_i/dt = S_i(t) + I_i(t) - E_i,eff(t) - R_i(M_i,T,...) - C_i(...)`
   - Pressure mapping:
     - `P(t) = g_M * sum_i M_i / A_M`
   - Parameters: source `S_i`, import `I_i`, escape `E_i`, regolith sink/source `R_i`, chemical conversion `C_i`.
4. **Data needed**
   - MAVEN-derived loss rates, volatile inventories, outgassing curves, import logistics, regolith sorption data.
5. **Assumptions**
   - Global box model (well-mixed atmosphere) for first implementation.
   - Species coupling through `C_i` is reduced-order.
6. **Pre-requisites**
   - Phases 1, 3, 4.

## 6) Climate warming model
1. **Input**
   - Atmospheric composition trajectories, GHG injection schedules, albedo/dust scenarios.
2. **Output**
   - Global mean temperature trajectory `T(t)` and climate forcing budget.
3. **Modeling approach (equations + parameters)**
   - Zero-D energy balance model:
     - `C_T * dT/dt = F_solar * (1 - alpha) / 4 + F_GHG(M_i) + F_dust - OLR(T, M_i)`
   - Typical parameterization:
     - `OLR ~= A_olr + B_olr * T`
   - Parameters: heat capacity `C_T`, albedo `alpha`, forcing coefficients for super-GHG and dust.
4. **Data needed**
   - Spectral forcing coefficients, dust optical depth statistics, albedo maps, thermal inertia estimates.
5. **Assumptions**
   - First-order global mean model before spatial GCM refinement.
   - Forcing parameterization remains valid over scenario ranges.
6. **Pre-requisites**
   - Phases 1, 5.

## 7) Hydrology and melting
1. **Input**
   - Temperature/pressure trajectories, cryosphere inventory, terrain and permeability.
2. **Output**
   - Liquid water inventory, melt rates, runoff/infiltration fluxes, stable water zones.
3. **Modeling approach (equations + parameters)**
   - Water-phase mass balances:
     - `dW_ice/dt = -Melt(T,P) + Freeze(T,P) + Deposition - Sublimation`
     - `dW_liq/dt = Melt - Freeze - Evap(T,P) - Infiltration + Runon - Runoff`
   - Simple melt law example:
     - `Melt = k_m * max(0, T - T_m(P))`
   - Parameters: melt coefficient `k_m`, permeability `K`, evaporation coefficients.
4. **Data needed**
   - Ice distribution/thickness, topography, regolith hydraulic properties, phase diagram constraints.
5. **Assumptions**
   - First implementation uses regional boxes/catchments, not full 3D hydro.
   - Subsurface hydrology represented by effective parameters.
6. **Pre-requisites**
   - Phases 1, 5, 6.

## 8) Atmospheric composition evolution
1. **Input**
   - Mass-budget outputs, oxygen production plans, chemical conversion yields.
2. **Output**
   - Time series of composition fractions `x_i(t)` and partial pressures `p_i(t)`.
3. **Modeling approach (equations + parameters)**
   - Mole-fraction dynamics:
     - `x_i = M_i / sum_j M_j`
     - `dx_i/dt = (1/M_tot) * (dM_i/dt - x_i * dM_tot/dt)`
   - Partial pressure:
     - `p_i = x_i * P`
   - Parameters: conversion efficiencies (for O2 pathways), leakage factors, control schedules.
4. **Data needed**
   - ISRU process specs, electrolysis/sabatier yields, storage/transport losses.
5. **Assumptions**
   - Atmosphere remains well mixed on modeled timescales.
   - Controlled injection schedules are executable as planned.
6. **Pre-requisites**
   - Phases 3, 5, 6, 7.

## 9) Photochemistry and ozone
1. **Input**
   - Composition trajectories, UV flux, atmospheric temperature profile assumptions.
2. **Output**
   - Ozone column depth trajectory and UV-at-surface attenuation estimate.
3. **Modeling approach (equations + parameters)**
   - Reduced Chapman-like chemistry:
     - `d[O]/dt = J1[O2] - k1[O][O2][M] + J3[O3] - k3[O][O3]`
     - `d[O3]/dt = k1[O][O2][M] - J3[O3] - k3[O][O3]`
   - Parameters: photolysis rates `J1, J3`, reaction rates `k1, k3`, third-body concentration `[M]`.
4. **Data needed**
   - Solar UV spectrum at Mars, reaction cross-sections and rates, dust/aerosol attenuation priors.
5. **Assumptions**
   - Start with 1D column chemistry and effective mixing.
   - Heterogeneous chemistry on dust is parameterized.
6. **Pre-requisites**
   - Phases 6, 8.

## 10) Soil conditioning and biosphere readiness
1. **Input**
   - Soil chemistry maps, water availability, atmospheric conditions.
2. **Output**
   - Soil readiness index and crop viability envelopes by region.
3. **Modeling approach (equations + parameters)**
   - Contaminant decay/removal:
     - `dC_perc/dt = -k_rem * C_perc + S_perc`
   - Nutrient dynamics:
     - `dN_avail/dt = I_N - U_N - L_N`
   - Composite readiness score:
     - `S_soil = w1*f_pH + w2*f_nutrients + w3*f_toxicity + w4*f_water`
   - Parameters: removal rate `k_rem`, nutrient fluxes, weights `w_i`.
4. **Data needed**
   - Perchlorate concentrations, mineralogy, pH/salinity, nutrient profiles, irrigation quality.
5. **Assumptions**
   - Biogeochemical complexity represented by reduced indices for planning stage.
6. **Pre-requisites**
   - Phases 7, 8, 9.

## 11) Settlement suitability
1. **Input**
   - Outputs from all prior phases + risk and logistics constraints.
2. **Output**
   - Ranked candidate settlement zones and robustness scores.
3. **Modeling approach (equations + parameters)**
   - Multi-criteria score:
     - `S_site = sum_k w_k * z_k`
   - Robustness under uncertainty:
     - `R_site = Pr(S_site >= S_min)` from Monte Carlo over uncertain inputs.
   - Parameters: criterion weights `w_k`, normalized metrics `z_k`, threshold `S_min`.
4. **Data needed**
   - Terrain hazards, resource proximity, climate reliability, infrastructure distance/cost.
5. **Assumptions**
   - Weighting scheme is stakeholder-defined and scenario-dependent.
   - Correlations across risk factors can be sampled with simplified copulas.
6. **Pre-requisites**
   - Phases 1-10.

---

## Cross-phase implementation requirements
- **State coupling:** all phase models exchange state via `x(t)` and shared forcing/control inputs.
- **Uncertainty:** each key parameter has prior distribution; run Monte Carlo ensembles per scenario.
- **Validation:** compare intermediate outputs against observed Mars constraints and reject non-physical runs.
- **Versioning:** store equation set, parameters, and datasets by scenario tag (`baseline`, `optimistic`, `conservative`).

