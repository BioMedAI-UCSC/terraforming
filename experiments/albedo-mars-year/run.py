"""Audit long-horizon albedo derivatives, then run a one-Mars-year comparison."""
import hashlib
import json
import os
from pathlib import Path
import subprocess

source, work, run = [Path(os.environ[k]) for k in
    ("EXPERIMENT_SOURCE_DIR", "EXPERIMENT_WORK_DIR", "EXPERIMENT_RUN_DIR")]
# Reuse the same pinned dependency environment as the completed one-sol study.
python = Path("/work/experiments/experiments/rebuttal-54586/6ddddaff06a70618527c99d77b5973ae4c23eef1/20261115-v1/venv/bin/python")
if not python.is_file():
    python = work / "venv/bin/python"
    if not python.exists():
        subprocess.run(["uv", "venv", "--python", "3.12", str(python.parent.parent)], check=True)
    subprocess.run(["uv", "pip", "install", "--python", str(python), "-e", str(source / "package"),
        "-e", str(source / "apps/mars-calibration"), "cma==4.4.0"], check=True)
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source / "package"), str(source / "apps/mars-calibration")]),
    JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true", XLA_FLAGS="", XLA_PYTHON_CLIENT_PREALLOCATE="false",
    MPLCONFIGDIR=str(work / "matplotlib"), MPLBACKEND="Agg")
inputs = Path("/work/reviewer-inputs")
expected = json.loads(Path("/work/staged-sha256.json").read_text())
for path in inputs.iterdir():
    if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected[str(path)]:
        raise ValueError(f"Input hash mismatch: {path}")
with (run / "resolved-packages.txt").open("w") as stream:
    subprocess.run(["uv", "pip", "freeze", "--python", str(python)], stdout=stream, check=True)
year_steps = int(subprocess.check_output([str(python), "-c",
    "import math;from src.celestials.planets.mars import MARS_BODY_3D as b;"
    "from src.celestials.planets.mars.gcm import radiative_forcing;"
    "print(math.ceil(radiative_forcing().orbital_period_s/(b.solar_day_seconds/300)))"],
    cwd=source, env=env, text=True))
tasks = [("audit-5sol", 1500, True), ("audit-30sol", 9000, True),
         ("annual-optimization", year_steps, False)]
statuses = {}
for name, steps, audit_only in tasks:
    folder = run / name
    folder.mkdir(exist_ok=True)
    attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))):03d}"
    args = [str(python), str(source / "scripts/run_rebuttal_experiments.py"), "spatial",
        "--inputs", str(inputs), "--output", str(attempt), "--steps", str(steps),
        "--objective", "mean", "--audit-directions", "1", "--evaluations", "16",
        "--seed", "0", "--require-gpu", *(["--audit-only"] if audit_only else [])]
    print(f"Starting {name}, {steps} steps, output {attempt}", flush=True)
    with (folder / f"{attempt.name}.log").open("w") as stream:
        result = subprocess.run(args, cwd=source, env=env, stdout=stream, stderr=subprocess.STDOUT)
    statuses[name] = dict(returncode=result.returncode, output=str(attempt))
    (run / "status.json").write_text(json.dumps(statuses, indent=2)+"\n")
    print(f"Finished {name}: exit {result.returncode}", flush=True)
raise SystemExit(0 if all(s["returncode"] == 0 for s in statuses.values()) else 2)
