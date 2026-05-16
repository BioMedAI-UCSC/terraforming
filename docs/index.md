# Terraforming

**tform** is a physics-based Mars terraforming simulation framework. It models the evolution of surface temperature, atmospheric pressure, and polar ice mass under solar forcing, greenhouse gas interventions, and orbital mechanics.

---

## Core state vector

The simulator integrates a coupled ODE system over time:

$$
\frac{d}{dt}\begin{pmatrix}T \\ P \\ M_\text{ice}\end{pmatrix} = f\!\left(T, P, M_\text{ice},\, L_s,\, r,\, \text{GHG}\right)
$$

where $T$ is mean surface temperature (K), $P$ is atmospheric pressure (Pa), and $M_\text{ice}$ is polar CO₂ ice mass (kg).

---

## Modules

| Package | Description |
|---------|-------------|
| `src.framework` | Abstract planet, atmosphere, orbital mechanics base classes |
| `src.celestials` | Mars implementation — solar flux, climate ODE, polar cap model |
| `src.engine` | RK4 and fast-path integrators, batched simulation controller |
| `src.interventions` | GHG compound registry, radiative forcing, injection scheduler |

---

## Quick links

- [What is tform?](getting-started/what-is-tform.md) — full feature overview
- [Installation](getting-started/installation.md) — uv setup on any OS
- [Quickstart](getting-started/quickstart.md) — run your first simulation
- [CLI Reference](cli/commands.md) — all flags and experiment types
- [GHG Interventions](wiki/mars/interventions.md) — radiative forcing model
- [API Reference](api/framework.md) — full Python interface
