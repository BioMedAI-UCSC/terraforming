# Mars API

The `src.celestials` package provides the concrete `Mars` class — a full implementation of the abstract `Planet` interface with NASA-calibrated constants and Mars-specific ODE physics.

## Module layout

`planets/mars/` is split along a backend boundary:

| Module | Imports | Role |
|--------|---------|------|
| `mars/constants.py` | stdlib only | `MarsConstants`, `MARS` — plain-`float` physical constants |
| `mars/planet.py` | `torch` | `Mars`, `constants_to_tensors`, `composition_to_tensors` |
| `mars/__init__.py` | — | re-exports both, so import sites are unchanged |

`constants.py` deliberately imports neither `torch` nor `jax`, so the same numbers can be read by the PyTorch box model, a JAX GCM (`jcm`), or plain NumPy analysis. `planet.py` owns the one-way conversion into tensors.

## `MarsConstants`

**Purpose**: immutable, framework-neutral bundle of Mars's physical constants
**Location**: `package/src/celestials/planets/mars/constants.py`

Every base field is a plain `float` in SI units, with the unit encoded in the field name. The canonical instance is `MARS`. The dataclass is frozen, so variants are made with `dataclasses.replace`:

```python
from dataclasses import replace
from src.celestials import MARS, Mars

thick_cap = replace(MARS, polar_cap_fraction=0.08)
mars = Mars(constants=thick_cap)     # the MARS default is untouched
```

### Base fields

| Field | Value | Unit |
|-------|-------|------|
| `mass_kg` | $6.4171 \times 10^{23}$ | kg |
| `radius_m` | $3.3895 \times 10^6$ | m |
| `gravity_m_s2` | $3.72076$ | m s⁻² |
| `solar_day_s` | $88\,775.244$ | s (one sol, noon-to-noon) |
| `sidereal_day_s` | $88\,642.663$ | s (one rotation w.r.t. the stars) |
| `semi_major_axis_m` | $2.279392 \times 10^{11}$ | m (1.5237 AU) |
| `eccentricity` | $0.0934$ | — |
| `orbital_period_s` | $5.93568 \times 10^7$ | s (~687 Earth days) |
| `axial_tilt_deg` | $25.19$ | deg |
| `ls_perihelion_deg` | $251.0$ | deg |
| `surface_emissivity` | $0.95$ | — |
| `thermal_inertia_j_k_m2` | $6.0 \times 10^4$ | J K⁻¹ m⁻² |
| `diurnal_swing_amp_k` | $50.0$ | K |
| `scale_height_m` | $11\,100$ | m |
| `co2_frost_point_k` | $149.0$ | K |
| `co2_latent_heat_j_kg` | $5.7 \times 10^5$ | J kg⁻¹ |
| `polar_cap_fraction` | $0.04$ | — (calibrated, see below) |
| `thermal_tide_pa` | $30.0$ | Pa |
| `thermal_tide_phase_rad` | $-0.7\pi$ | rad |
| `maven_escape_rate_kg_s` | $0.2$ | kg s⁻¹ |
| `co2_cp_j_kg_k` | $769.8$ | J kg⁻¹ K⁻¹ |
| `composition_pa` | CO₂ 580, N₂ 15, Ar 12, O₂ 0.8, CO 0.4 | Pa |

!!! warning "Two rotation periods, and the difference matters"
    `solar_day_s` drives the diurnal cycle in the 0-D model. `sidereal_day_s` is the one that yields Ω for a dynamical core's Coriolis term. Using the solar day for Ω gives $7.0776 \times 10^{-5}$ — 0.15 % low.

`polar_cap_fraction` is a project calibration, not a measurement: 0.01 gave only a ~6 % seasonal pressure swing, whereas the Viking Landers observe ~25–30 % (Hess et al. 1980; Tillman et al. 1993). 0.04 reproduces ~27 % at a stable ~5.9 mb mean.

### Derived properties

Computed on access, so they can never drift out of sync with their base values.

| Property | Formula | Value |
|----------|---------|-------|
| `axial_tilt_rad` | $\pi\theta/180$ | $0.4397$ rad |
| `ls_perihelion_rad` | $\pi L_s/180$ | $4.3808$ rad |
| `rotation_rate_rad_s` | $2\pi / T_\text{sidereal}$ | $7.0882 \times 10^{-5}$ rad s⁻¹ |
| `co2_gas_constant_j_kg_k` | $R_\text{univ} / M_{\mathrm{CO_2}}$ | $188.92$ J kg⁻¹ K⁻¹ |
| `kappa` | $R / c_p$ | $0.2454$ (Earth: $2/7 = 0.2857$) |
| `surface_area_m2` | $4\pi R^2$ | $1.4441 \times 10^{14}$ m² |
| `semi_major_axis_au` | $a / \mathrm{AU}$ | $1.5237$ |
| `solar_constant_w_m2` | $S_0 / a^2$ | $586.2$ W m⁻² |
| `mean_solar_flux_w_m2` | $S_0 / (a^2\sqrt{1-e^2})$ | $588.8$ W m⁻² |
| `total_composition_pa` | $\sum p_i$ | $608.2$ Pa |

### `as_jcm_overrides() -> dict[str, float]`

Maps these constants onto `jcm.constants.PhysicalConstants` field names, for driving a JAX GCM.

**Returns**: a plain `dict` of keyword overrides — this module imports nothing, so the caller applies them.

```python
import jcm.constants as jc
from src.celestials import MARS

jc.set_constants(**MARS.as_jcm_overrides())
```

Only fields whose Earth defaults are *wrong for Mars* are overridden: `rearth`, `omega`, `grav`, `cpd`, `akap`, `solc`, `alhs`, `p0`, `p0s1_bg`. Universal constants (Stefan-Boltzmann, von Kármán, Boltzmann) are left alone. Water-vapour constants (`rv`, `eps`, `alhc`) are also left at their Earth values — they are physically meaningless in a dry CO₂ atmosphere, and overriding them would imply a moist Mars the radiation code does not model.

---

## Constants → tensors

### `constants_to_tensors(constants=MARS, *, device=None, dtype=TF_DTYPE)`

**Location**: `package/src/celestials/planets/mars/planet.py`

The sole torch-ward crossing for Mars's scalar constants.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `constants` | `MarsConstants` | `MARS` | bundle to convert |
| `device` | `str \| torch.device \| None` | `None` | allocation device; `None` = CPU |
| `dtype` | `torch.dtype` | `TF_DTYPE` (float64) | element type |

**Returns**: `dict[str, torch.Tensor]` of 0-d tensors keyed by the legacy `MARS_*` names, so the result drops into anywhere the old module-level tensors were used. Angles are returned in **radians**; `MARS_ROTATION_PERIOD` is the **solar** day.

```python
>>> t = constants_to_tensors(device="cpu")
>>> float(t["MARS_GRAVITY"])
3.72076
```

### `composition_to_tensors(constants=MARS, *, device=None, dtype=TF_DTYPE)`

Same parameters. **Returns**: `dict[str, torch.Tensor]` of partial pressures in Pa, keyed by species symbol. Returns a fresh dict on each call, since callers mutate composition during a run.

### Module-level `MARS_*` tensors

`planet.py` still exports CPU tensors under the historical names (`MARS_MASS`, `MARS_RADIUS`, …), built from `MARS` via `constants_to_tensors`. Existing import sites are unaffected:

```python
from src.celestials import Mars, MARS_ORBITAL_PERIOD   # unchanged
```

---

## Mars Class

::: src.celestials
    options:
      members:
        - Mars
      show_root_heading: true
      show_root_full_path: false
