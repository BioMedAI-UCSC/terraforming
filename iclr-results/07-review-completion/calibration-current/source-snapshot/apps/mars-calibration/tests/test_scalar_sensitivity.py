import importlib.util
from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np


def test_input_freezer_accepts_fresh_restart(tmp_path):
    import json
    import os
    import subprocess
    import sys
    repo = Path(__file__).resolve().parents[3]
    for relative in ("data/arco_macda/macda_my32_ls045_10sol_daily_t21_l12.nc",
                     "data/tes/mgs_tes_surface_1deg.nc", "data/ames/fv3betaout1/ames_surface_reference.nc",
                     "data/mola/meg004/megt90n000cb.img", "fresh/restart.npz"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    environment = dict(os.environ, PYTHONPATH=str(repo / "apps/mars-calibration"))
    subprocess.run([sys.executable, str(repo / "apps/mars-calibration/prepare_inputs.py"),
                    "--repo", str(tmp_path), "--restart", "fresh/restart.npz",
                    "--output", str(tmp_path / "inputs")], env=environment, check=True)
    assert (tmp_path / "inputs/restart.npz").read_bytes() == b"fresh/restart.npz"
    manifest = json.loads((tmp_path / "inputs/manifest.json").read_text())
    assert manifest["files"]["restart.npz"]["source"] == "fresh/restart.npz"


def test_area_means_and_their_derivatives():
    path = Path(__file__).resolve().parents[3] / "scripts/run_scalar_sensitivity.py"
    spec = importlib.util.spec_from_file_location("scalar_sensitivity", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = SimpleNamespace(specs=SimpleNamespace(dimensionalize=lambda *args: SimpleNamespace(magnitude=2.)),
        coords=SimpleNamespace(horizontal=SimpleNamespace(to_nodal=lambda x: x)), weights=jnp.asarray([.25, .75]))
    def evaluate(x):
        field = x * jnp.asarray([[[[2., 6.]]]])
        state = SimpleNamespace(surface_temperature=field, co2_ice=field,
            dynamics=SimpleNamespace(log_surface_pressure=jnp.log(field)))
        return module.diagnostics(model, state)
    np.testing.assert_allclose(evaluate(1.), [[5., 10., 10.]])
    np.testing.assert_allclose(jax.jacfwd(evaluate)(1.), [[5., 10., 10.]])
