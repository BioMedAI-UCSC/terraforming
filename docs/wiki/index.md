# Wiki

This wiki is the knowledge base behind the tform simulator. It explains the physics, mathematics, and scientific sources behind every model in the framework — written so that a reader unfamiliar with a topic can follow a chain of references to reach a complete understanding.

---

## Structure

**Generic systems** (this level) cover concepts that apply across all planets:

- [Orbital Mechanics](orbital-mechanics.md) — Keplerian orbits, distance as a function of solar longitude
- [Solar Radiation](solar-radiation.md) — TOA flux, zenith angle geometry, inverse-square law
- [Greenhouse Effect](greenhouse-effect.md) — radiative forcing, Stefan-Boltzmann law, greenhouse factors

**Planet-specific sections** go into the physical constants, calibrated models, and intervention scenarios for each body:

- [Mars](mars/index.md) — baseline state, climate model, atmospheric composition, GHG interventions

Future planets (Venus, the Moon, Europa) will each get their own subsection here.

---

## How to read this wiki

- All equations are written in block math where they stand alone, and inline math $\text{like this}$ when referenced in prose.
- Every non-obvious constant or formula links to its source (paper, NASA fact sheet, or Wikipedia).
- Code references link to the corresponding API page rather than repeating implementation details here.
