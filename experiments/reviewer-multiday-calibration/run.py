"""Fit five-sol daily-mean trajectories with whole-year held-out evaluation."""
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
    XLA_PYTHON_CLIENT_PREALLOCATE="false", XLA_FLAGS="")
# Isolate atmospheric phase change under identical MOLA/TES/dust winter forcing.
years = 15. * 88775.244 / 59356800.
for name, extra in (("clouds-on", []), ("surface-only", ["--no-atmospheric-condensation"])):
    subprocess.run([str(python), str(source / "scripts/run_reviewer_stability.py"), "timestep",
        "--inputs", "/work/reviewer-inputs", "--output", str(run / "polar-control" / name),
        "--dt", "300", "--years", str(years), "--ls", "270", "--resume", "--require-gpu", *extra],
        cwd=source, env=env, check=True)
attempt = run / f"fit-attempt-{len(list(run.glob('fit-attempt-*'))):03d}"
subprocess.run([str(python), str(source / "scripts/run_differentiable_experiments.py"), "calibration",
    "--inputs", "/work/reviewer-inputs", "--manifest", "/work/reviewer-multiday-windows/windows.json",
    "--output", str(attempt), "--starts", "3", "--updates", "500", "--max-evaluations", "300",
    "--gtol", "1e-6", "--ftol", "1e-12", "--require-gpu"], cwd=source, env=env, check=True)
