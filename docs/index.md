# Terraforming

A physics-based Mars climate simulation framework for studying planetary-scale terraforming interventions.

---

## What It Is

**Terraforming** is a Python framework that models Mars atmospheric and thermal evolution under greenhouse gas injection scenarios. It couples:

- A **coupled ODE system** for surface temperature, atmospheric pressure, and cryosphere mass
- A **radiative forcing model** calibrated to Martian conditions (Marinova et al. 2005)
- A **4th-order Runge-Kutta integrator** (or reduced-order fast mode) with GPU support via PyTorch
- A **CLI tool** (`tform`) for running simulations from YAML configs or built-in presets

The core equation governing temperature evolution:

$$
\frac{dT}{dt} = \frac{F_\odot(1 - \alpha)}{4} \cdot \gamma - \sigma T^4 + \Delta F_\text{GHG}
$$

where $\gamma$ is the greenhouse amplification factor and $\Delta F_\text{GHG}$ is the net radiative forcing from injected super-greenhouse gases.

---

## Quick Start

```bash
# Install
pip install terraforming-cli

# Run a baseline Mars simulation (1 Martian year)
tform mars run --preset current-mars --type year

# Run a terraforming intervention (50-year GHG injection)
tform mars run --preset terraforming-phase1 --type intervention --sols 18262
```

---

## Package Layout

| Module | Role |
|--------|------|
| `src.constants` | Physical constants as `torch.Tensor` scalars |
| `src.framework` | Abstract `Planet` base class and state dataclasses |
| `src.celestials.planets.mars` | Mars-specific physics implementation |
| `src.engine` | RK4 and reduced-order time integrators |
| `src.interventions` | GHG compound registry, forcing calculations, injection controller |

---

## Navigation

- **[Getting Started](getting-started/installation.md)** — install and run your first simulation
- **[CLI Reference](cli/commands.md)** — all `tform` commands and flags
- **[Physics](PLAN.md)** — terraforming plan, mathematical models, and function derivations
- **[Architecture](architecture/planet.md)** — system design and module responsibilities
- **[API Reference](api/framework.md)** — auto-generated from source docstrings
