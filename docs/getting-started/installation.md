# Install the Mars GCM

The primary simulation stack is Python 3.12+, JAX and Dinosaur. The repository
uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for dependencies.
Install uv using its platform instructions, then clone the repository:

```bash
rtk proxy git clone https://github.com/BioMedAI-UCSC/terraforming
cd terraforming
rtk proxy uv sync --all-packages --extra gcm3d --dev
```

Commands here run from the repository root. RTK is optional for readers; remove
`rtk proxy` if it is unavailable. `--all-packages` installs both the physics
package and CLI; `--extra gcm3d` installs the optional GCM dependencies.
Use the installed environment explicitly so another sync does not remove extras.

## Verify the installation

```bash
rtk proxy .venv/bin/python -c 'import jax; from src.framework.gcm import BodyConstants; print(jax.devices())'
rtk proxy .venv/bin/python -m cli.main mars maps --help
```

CPU JAX is sufficient for first maps and generated neural examples. A CUDA GPU
requires a compatible JAX CUDA plugin and driver; the GCM extra alone does not
select a GPU. Check `jax.devices()` on the intended machine before long workloads.
The GPU cache and `--require-gpu` experiment commands reject CPU fallback.

## Stage terrain and boundary data

```bash
rtk proxy .venv/bin/python scripts/stage_mola.py
```

MOLA terrain is required for Mars map runs and verified against a pinned checksum.
The Ames correlated-k radiation table is bundled. TES surface and Ames dust data
are separate inputs for richer configurations; see
[data staging](../scripts/gcm3d-data.md). Existing `outputs/` experiment bundles
referenced in runbooks are not supplied by a fresh checkout.

For ARCO-MACDA streaming, install its extra while retaining GCM dependencies:

```bash
rtk proxy uv sync --all-packages --extra gcm3d --extra arco --dev
```

For the calibration application:

```bash
rtk proxy uv pip install -e ./apps/mars-calibration
```

## First run and browser

Continue to the [quickstart](quickstart.md). The prebuilt UI is served by the CLI;
Node is needed only to modify/build the frontend. Source builds use `npm install`
and `npm run build` from `ui/`, producing assets under `cli/static/`.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Missing Dinosaur/JAX | Repeat sync with `--extra gcm3d`; use the project's Python |
| Missing or corrupt MOLA | Rerun terrain staging; inspect its path and checksum |
| Only CPU devices on a GPU host | Check matching JAX plugin and driver before launching |
| Missing xarray/Zarr streaming dependencies | Retain both `gcm3d` and `arco` extras |
| Diurnal timestep rejected | Use a timestep below `rotation_period/(2*n_lon)` and check stability |

!!! warning "Deprecated model"
    The torch global-mean model is deprecated. Its dependencies remain in the
    package for existing APIs; installation does not make it the recommended workflow.
