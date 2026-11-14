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
calibration = run / "calibration"
if not (calibration / "complete.json").exists():
    # Each interrupted optimization keeps its trace; a retry uses a separate attempt.
    attempt = run / f"calibration-attempt-{len(list(run.glob('calibration-attempt-*'))):03d}"
    subprocess.run([str(python), str(source / "scripts/run_differentiable_experiments.py"), "calibration",
        "--inputs", str(inputs), "--manifest", "/work/reviewer-windows/windows.json",
        "--output", str(attempt), "--starts", "3", "--updates", "500", "--max-evaluations", "300",
        "--gtol", "1e-6", "--ftol", "1e-12", "--require-gpu"], cwd=source, env=env, check=True)
    calibration.symlink_to(attempt, target_is_directory=True)
for case, extra in (("ultra", []), ("timestep-450", ["--dt", "450"]),
                    ("timestep-300", ["--dt", "300"]), ("timestep-150", ["--dt", "150"])):
    subprocess.run([str(python), str(source / "scripts/run_reviewer_stability.py"),
        "ultra" if case == "ultra" else "timestep", "--inputs", str(inputs),
        "--output", str(run / "studies" / case), "--require-gpu", "--resume", *extra],
        cwd=source, env=env, check=True)
subprocess.run([str(python), str(source / "scripts/analyse_reviewer_timesteps.py"),
    str(run / "studies"), "--output", str(run / "timestep-comparison.json")],
    cwd=source, env=env, check=True)
