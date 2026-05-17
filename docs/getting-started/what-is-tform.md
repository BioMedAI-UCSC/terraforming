# What is tform?

**tform** is a physics-based Mars climate simulation framework for studying how the Martian atmosphere, temperature, and cryosphere evolve over time — and how targeted engineering interventions can drive the planet toward habitability.

It is built for researchers and engineers who want to run rigorous, reproducible simulations of planetary-scale terraforming scenarios, from a single Martian day to centuries of greenhouse gas injection.

---

## What it simulates

At its core, tform integrates a coupled system of ordinary differential equations (ODEs) describing:

| State variable | Symbol | Unit |
|---------------|--------|------|
| Surface temperature | $T$ | K |
| Atmospheric pressure | $P$ | Pa |
| Polar CO₂ ice mass | $M_\text{ice}$ | kg |

These evolve under solar forcing, atmospheric radiative transfer, orbital mechanics, and optional greenhouse gas injections.

---

## Supported experiment types

### `sol` — Single Martian day
Simulates one full diurnal cycle (~24.6 hours) at a given latitude and longitude. Tracks the temperature response as the sun rises, peaks, and sets. Useful for understanding local climate conditions at a specific site.

### `year` — One Martian year
Runs a full Martian year (~687 Earth days) with realistic seasonal cycles driven by orbital eccentricity ($e = 0.0934$) and axial tilt ($25.19°$). Captures CO₂ cap sublimation/deposition and pressure seasonality.

### `multi` — Multi-latitude sweep
Runs three canonical latitudes simultaneously: 45°N, equator (0°), and 40°S — all at 137°E longitude. Shows the latitudinal temperature and pressure gradient across a Martian year.

### `spots` — Landmark sites
Runs four geographically important Martian sites in a single call, with elevation-corrected initial conditions:

| Site | Latitude | Elevation |
|------|----------|-----------|
| Olympus Mons | 18.65°N | +21 km |
| Elysium Mons | 25.02°N | +14 km |
| Hellas Basin | 42.4°S | −7 km |
| South Polar Cap | 90°S | +2 km |

### `intervention` — GHG injection campaign
Multi-year simulation of super-greenhouse gas injection (SF6, CF4, C2F6, and others). Tracks the radiative forcing accumulation, greenhouse factor evolution, and resulting temperature and pressure trajectory over decades to centuries.

---

## Supported forcings and controls

### External forcings passed to the physics engine

| Forcing key | Description |
|-------------|-------------|
| `solar_radiation` | Zenith angle, transmittance, TOA override, surface energy |
| `solar_wind` | Electron density, magnetic field, wind speed, proton density |
| `cosmic_radiation` | GCR flux, SEP flux, TOA dose rate, atmospheric shielding |
| `giant_planets_gravity` | Jupiter and Saturn effective gravitational accelerations |
| `mars_moons_gravity` | Phobos and Deimos direct and tidal accelerations |

### Atmosphere composition controls

Species column tendencies ($\text{kg m}^{-2}\,\text{s}^{-1}$) can be applied to any atmospheric species:

`O2`, `N2`, `H2`, `H`, `CO2`, `CO`, `O3`, `Ar`, `He`, `super_ghg`, `Ne`, `Kr`, `Xe`

### Water and cryosphere controls

| Control | Description |
|---------|-------------|
| `water_ice_tendency_kg_m2_s` | Add/remove bulk solid water |
| `water_liquid_tendency_kg_m2_s` | Add/remove liquid water |
| `water_phase_change_tendency_kg_m2_s` | Transfer between ice and liquid |
| `polar_ice_h2o_tendency_kg_m2_s` | Polar H₂O cap tendency |
| `polar_ice_co2_tendency_kg_m2_s` | Polar CO₂ cap tendency |

### Soil controls

- `soil_compound_tendency_mass_fraction_per_s` — gradual regolith chemistry change
- `soil_compound_delta_mass_fraction` — direct per-step override

### GHG interventions

| Compound | Atmospheric lifetime |
|----------|---------------------|
| CF₄ | >50,000 yr |
| SF₆ | 3,200 yr |
| C₂F₆ | 10,000 yr |
| NF₃ | 500 yr |
| CHF₃, CH₂F₂, CH₃F, C₃F₈ | varies |
| CH₄ | 12 yr |
| N₂O | 114 yr |

---

## Integration modes

| Mode | Method | When to use |
|------|--------|-------------|
| `accurate` | 4th-order Runge-Kutta | Science runs, publications |
| `fast` | Reduced-order analytic updates | Parameter sweeps, interactive exploration |

---

## Built-in presets

11 ready-to-run configurations: `current-mars`, `gale-crater`, `early-mars`, `terraforming-phase1`, `equatorial`, `polar`, `olympus-mons`, `elysium-mons`, `hellas-basin`, `south-polar-cap`, `landmark-spots`.

See [CLI Presets](../cli/presets.md) for full details.
