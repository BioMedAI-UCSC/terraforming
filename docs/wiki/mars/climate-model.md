# Mars Climate Model

The climate model integrates a coupled system of three ordinary differential equations (ODEs) describing the evolution of surface temperature $T$, atmospheric pressure $P$, and polar CO₂ ice mass $M_\text{ice}$ over time. This page derives each equation and explains the physical meaning of every term.

---

## State vector

At each timestep the simulation tracks:

$$
\mathbf{y}(t) = \begin{pmatrix} T(t) \\ P(t) \\ M_\text{ice}(t) \end{pmatrix}
$$

where $T$ is in K, $P$ in Pa, and $M_\text{ice}$ in kg. All three are coupled — pressure affects the greenhouse factor, ice mass affects pressure through sublimation, and temperature drives sublimation.

---

## Temperature ODE

The surface energy balance governs $dT/dt$ ([Haberle, 1998](https://doi.org/10.1029/98JE01388); [Pierrehumbert, 2010](https://doi.org/10.1017/CBO9780511780783)):

$$
\frac{dT}{dt} = \frac{F_\text{net}}{C_\text{eff}}
$$

The net surface energy flux $F_\text{net}$ is:

$$
F_\text{net} = F_\text{sfc} - F_\text{refl} + F_\text{IR,down} - F_\text{IR,up} + \Delta F_\text{GHG}
$$

Expanding each term:

**Absorbed shortwave (solar):**

$$
F_\text{sfc} - F_\text{refl} = \frac{S_{1\,\text{AU}}}{r(L_s)^2}\,\max\!\bigl(0,\cos\theta_z\bigr)\cdot\tau_\text{atm}\cdot(1 - \alpha)
$$

**Downwelling thermal IR (greenhouse back-radiation):**

$$
F_\text{IR,down} = \varepsilon_\text{atm}\,\sigma\,T^4
$$

**Upwelling thermal IR (surface emission):**

$$
F_\text{IR,up} = \varepsilon_\text{sfc}\,\sigma\,T^4
$$

**Radiative forcing from injected GHGs** (zero at baseline, see [GHG Interventions](interventions.md)):

$$
\Delta F_\text{GHG} = \sum_i \eta_i \cdot C_i
$$

**Effective heat capacity** $C_\text{eff}$ (J m⁻² K⁻¹) is the thermal inertia of the surface layer. The baseline value is $C_\text{eff} = 2.0 \times 10^6\,\text{J\,m}^{-2}\,\text{K}^{-1}$, consistent with fine-grained Martian regolith ([Putzig & Mellon, 2007](https://doi.org/10.1016/j.icarus.2007.01.022)).

---

## Pressure ODE

Atmospheric pressure evolves through CO₂ cap sublimation/deposition and any injected gases:

$$
\frac{dP}{dt} = \frac{dP}{dt}\bigg|_\text{sublimation} + \frac{dP}{dt}\bigg|_\text{injection}
$$

The sublimation term couples to $M_\text{ice}$: when $T$ rises above the CO₂ frost point ($\approx 148\,\text{K}$ at Martian pressures, [Wikipedia: Carbon dioxide (data page)](https://en.wikipedia.org/wiki/Carbon_dioxide_(data_page))), CO₂ ice sublimes and raises $P$.

---

## CO₂ ice mass ODE

The polar CO₂ cap mass evolves through sublimation and condensation:

$$
\frac{dM_\text{ice}}{dt} = -k_\text{sub}(T, P) + k_\text{dep}(T, P)
$$

where the sublimation rate $k_\text{sub}$ increases with temperature and decreases with surface pressure (Le Chatelier's principle), and the deposition rate $k_\text{dep}$ applies when the surface reaches the CO₂ frost point. The current CO₂ ice reservoir is approximately $M_\text{ice} \approx 3 \times 10^{15}\,\text{kg}$ ([Byrne & Ingersoll, 2003](https://doi.org/10.1126/science.1080895)).

---

## Greenhouse factor

Rather than tracking $\varepsilon_\text{atm}$ directly, the model uses a dimensionless greenhouse factor $\gamma \geq 1$ that amplifies the effective absorbed solar flux and modifies the net IR balance. It is updated each year based on the current atmospheric composition (see [GHG Interventions](interventions.md)):

$$
\gamma_\text{new} = \gamma_\text{base} \cdot \left(1 + \frac{\Delta F_\text{total}}{F_\text{ref}}\right)
$$

where $F_\text{ref}$ is a normalisation flux and $\Delta F_\text{total}$ is the cumulative radiative forcing from all injected species.

---

## Numerical integration

The ODE system is integrated using **4th-order Runge-Kutta** (RK4) ([Wikipedia: Runge-Kutta methods](https://en.wikipedia.org/wiki/Runge%E2%80%93Kutta_methods)) at hourly timesteps ($\Delta t = 3600\,\text{s}$) in `Accuracy.ACCURATE` mode. RK4 achieves $O(\Delta t^4)$ local truncation error, making it accurate enough for century-scale simulations without adaptive step-size control.

The fast mode uses **reduced-order analytic updates** — a relaxation scheme that gives physically consistent trajectories at much lower computational cost, useful for ensemble runs.

---

## Baseline outputs (1-hour step at $L_s = 0°$, $\theta_z = 60°$)

| Diagnostic | Value |
|-----------|-------|
| $F_\text{TOA}$ | $356.6\,\text{W\,m}^{-2}$ |
| $F_\text{sfc}$ | $196.1\,\text{W\,m}^{-2}$ |
| $F_\text{IR,up}$ | $22.1\,\text{W\,m}^{-2}$ |
| $\Delta T$ per hour | $+0.116\,\text{K}$ |
| $P$ | $610\,\text{Pa}$ (unchanged, no sublimation) |

---

## Implementation

The ODE right-hand side is implemented in [`src.celestials.planets.mars — Mars.compute_derivatives`](../../api/celestials.md). The integrator is in [`src.engine — TimeController`](../../api/engine.md).
