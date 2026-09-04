# gcm3d data-staging & validation scripts

These scripts stage the external datasets the framework/Mars 3-D GCM needs and
validate its output. They are one-time setup / offline tools — never imported by
the package at runtime. Each verifies a SHA-256 so a wrong or truncated download
fails loudly rather than silently loading corrupt data.

| Script | Purpose | Produces |
|--------|---------|----------|
| `stage_mola.py` | Download the MGS **MOLA MEGDR** 4-px/° global topography (`megt90n000cb.img`) from the NASA PDS Geosciences Node; verify SHA-256 + size. | `data/mola/meg004/megt90n000cb.img` (used as the dycore's lower boundary). |
| `stage_ames_co2_radiation.py` | Decode the bundled Ames 12-band Fortran CO₂ k-tables into a compact JAX `.npz` asset (nearly pure-CO₂ mixture, H₂O mass fraction 1e-7). | `package/src/framework/physics/ames_co2_12band.npz` (bundled; consumed by `ames_radiation.py`). |
| `stage_tes_surface.py` | Stage independent MGS/**TES** albedo + thermal inertia to a 1° NetCDF. | Surface boundary file for `run_maps(surface_properties_path=…)`. |
| `benchmark_mcd.py` | Fetch **Mars Climate Database v6.1** maps (global diurnal mean; the MCD web service exposes two free dimensions) and compare a `gcm3d` MOLA-map NetCDF against them. | Area-weighted bias / RMSE / correlation report (uses `mcd.py`). |
| `run_gcm3d_ablation.py` | Restartable Mars-year (668-sol) soil spin-up with incremental PBL/physics ablations — a climate experiment, not a short map. | Restart files (`restart.py`) + diagnostics. |

## Typical setup

```bash
python scripts/stage_mola.py            # required for any 3-D Mars map
python scripts/stage_tes_surface.py     # optional: spatial albedo / thermal inertia
# ames_co2_12band.npz is committed; re-run stage_ames_co2_radiation.py only to regenerate it
```

Provenance and integrity constants (source URLs, SHA-256, size) live alongside the
loaders in `topography.py` and `ames_radiation.py`; see
[the gcm3d implementation notes](../package/gcm3d/implementation.md) and
[the physics-limitations ledger](../ideas/gcm3d-physics-limitations.md).
