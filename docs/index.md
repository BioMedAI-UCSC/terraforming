# Terraforming

**Terraforming** is the hypothetical process of deliberately modifying a planet's atmosphere, temperature, surface topography, or ecology to make it habitable for Earth life. The core challenge is an energy balance problem: enough heat must be retained by the atmosphere to sustain liquid water and breathable pressures at the surface.

**tform** is a physics-based simulation framework for modelling terraforming processes. It provides a generic planetary state model — tracking how a planet's temperature, pressure, and volatile reservoirs evolve over time under external forcings — and implements planet-specific physics on top of that foundation. Mars is the first and primary target.

---

## Framework design

A planet in tform is described by a **state vector** of thermodynamic and atmospheric quantities that evolve continuously under physical forcing:

$$
\mathbf{y}(t) = \bigl(T,\; P,\; \ldots\bigr)
$$

The framework defines how that state changes — balancing incoming solar radiation, outgoing thermal emission, greenhouse retention, and any engineered interventions — without prescribing the planet-specific constants. Each planet subclass supplies its own orbital parameters, atmospheric composition, and physical constants, while inheriting the integration infrastructure.

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
