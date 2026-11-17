"""Test conservative surface mass transport against the archived annual baseline."""
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
    MPLCONFIGDIR=str(work / "matplotlib"), JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true",
    XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_FLAGS="", MPLBACKEND="Agg")
for dt in (450, 300, 150):
    subprocess.run([str(python), str(source / "scripts/run_reviewer_stability.py"), "timestep",
        "--inputs", "/work/reviewer-inputs", "--output", str(run / "studies" / f"timestep-{dt}"),
        "--dt", str(dt), "--require-gpu", "--resume"], cwd=source, env=env, check=True)
subprocess.run([str(python), str(source / "scripts/analyse_reviewer_timesteps.py"),
    str(run / "studies"), "--output", str(run / "timestep-comparison.json")],
    cwd=source, env=env, check=True)
subprocess.run([str(python), str(source / "scripts/analyse_reviewer_fields.py"),
    str(run / "studies"), "--output", str(run / "full-field-comparison")],
    cwd=source, env=env, check=True)
