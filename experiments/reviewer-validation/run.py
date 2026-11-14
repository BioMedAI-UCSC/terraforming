"""Run the registered reviewer checks with the generic Nautilus runtime."""
import os
from pathlib import Path
import subprocess

source = Path(os.environ["EXPERIMENT_SOURCE_DIR"])
work = Path(os.environ["EXPERIMENT_WORK_DIR"])
run = Path(os.environ["EXPERIMENT_RUN_DIR"])
python = work / "venv/bin/python"
if not python.exists():
    subprocess.run(["uv", "venv", "--python", "3.12", str(work / "venv")], check=True)
subprocess.run(["uv", "pip", "install", "--python", str(python),
    "-e", str(source / "package"), "-e", str(source / "apps/mars-calibration")], check=True)
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source / "package"), str(source / "apps/mars-calibration")]),
    MPLCONFIGDIR=str(work / "matplotlib"), JAX_PLATFORMS="cuda", XLA_FLAGS="",
    JAX_ENABLE_X64="true", XLA_PYTHON_CLIENT_PREALLOCATE="false")
inputs = Path("/work/reviewer-inputs")
if not (inputs / "manifest.json").is_file():
    raise FileNotFoundError("Upload the frozen current-physics input bundle to /work/reviewer-inputs first")
for case, extra in (("ultra", []), ("timestep-450", ["--dt", "450"]),
                    ("timestep-300", ["--dt", "300"]), ("timestep-150", ["--dt", "150"])):
    subprocess.run([str(python), str(source / "scripts/run_reviewer_stability.py"),
        "ultra" if case == "ultra" else "timestep", "--inputs", str(inputs),
        "--output", str(run / "studies" / case), "--require-gpu", "--resume", *extra],
        cwd=source, env=env, check=True)
