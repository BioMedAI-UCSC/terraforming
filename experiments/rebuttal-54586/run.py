"""Run independently scheduled rebuttal tasks from one immutable source."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

p = argparse.ArgumentParser()
p.add_argument("task", choices=("projected", "unprojected", "spatial", "radiation", "heating", "stability"))
a = p.parse_args()
source, work, run = [Path(os.environ[key]) for key in
    ("EXPERIMENT_SOURCE_DIR", "EXPERIMENT_WORK_DIR", "EXPERIMENT_RUN_DIR")]
python = work / "venv/bin/python"
if not python.exists():
    subprocess.run(["uv", "venv", "--python", "3.12", str(work / "venv")], check=True)
subprocess.run(["uv", "pip", "install", "--python", str(python), "-e", str(source / "package"),
    "-e", str(source / "apps/mars-calibration"), "cma==4.4.0"], check=True)
env = dict(os.environ, PYTHONPATH=os.pathsep.join([str(source / "package"), str(source / "apps/mars-calibration")]),
    JAX_PLATFORMS="cuda", JAX_ENABLE_X64="true", XLA_FLAGS="", XLA_PYTHON_CLIENT_PREALLOCATE="false",
    MPLCONFIGDIR=str(work / "matplotlib"), MPLBACKEND="Agg")
inputs, references = Path("/work/reviewer-inputs"), Path("/work/rebuttal-references")
manifest = json.loads(Path("/work/staged-sha256.json").read_text())
for path, digest in manifest.items():
    if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
        raise ValueError(f"Staged input hash mismatch: {path}")
failures = []

def execute(name, script, arguments):
    folder = run / name
    if (folder / "success.json").exists():
        return
    attempt = folder / f"attempt-{len(list(folder.glob('attempt-*'))):03d}"
    attempt.parent.mkdir(parents=True, exist_ok=True)
    with (folder / f"{attempt.name}.log").open("w") as log:
        print(f"Starting {name}: {attempt}", flush=True)
        result = subprocess.run([str(python), str(source / script), *arguments, "--output", str(attempt)],
            cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT)
    (folder / f"{attempt.name}-status.json").write_text(json.dumps(dict(returncode=result.returncode)))
    if result.returncode:
        failures.append(name)
        print(f"Failed {name}: exit {result.returncode}; retained {attempt}", flush=True)
        return None
    (folder / "success.json").write_text(json.dumps(dict(output=str(attempt))))
    print(f"Completed {name}: {attempt}", flush=True)
    return attempt

if a.task in ("projected", "unprojected"):
    output = execute(a.task, "scripts/run_rebuttal_experiments.py", ["climate", "--inputs", str(inputs),
        "--years", "2", "--require-gpu", *(["--no-projection"] if a.task == "unprojected" else [])])
    if output is None:
        if failures:
            raise SystemExit(1)
        output = Path(json.loads((run / a.task / "success.json").read_text())["output"])
    subprocess.run([str(python), str(source / "scripts/analyse_rebuttal_climate.py"), str(output),
        "--references", str(references), "--inputs", str(inputs)], cwd=source, env=env, check=True)
elif a.task == "spatial":
    for seed in range(3):
        execute(f"spatial-seed{seed}", "scripts/run_rebuttal_experiments.py", ["spatial", "--inputs", str(inputs),
            "--steps", "300", "--evaluations", "64", "--seed", str(seed), "--require-gpu"])
elif a.task == "radiation":
    for seed in range(3):
        execute(f"radiation-seed{seed}", "examples/neural/train_radiation.py", ["--seed", str(seed),
            "--epochs", "100", "--fine-tune-epochs", "20", "--continuation-updates", "20",
            "--continuation-budget", "updates", "--coupled-steps", "3000"])
elif a.task == "heating":
    for seed in range(3):
        execute(f"heating-seed{seed}", "scripts/run_differentiable_experiments.py", ["neural",
            "--inputs", str(inputs), "--manifest", "/work/reviewer-windows/windows.json",
            "--seed", str(seed), "--updates", "50", "--require-gpu"])
else:
    studies = run / "studies"
    studies.mkdir(exist_ok=True)
    for dt in (450, 300, 150):
        output = execute(f"timestep-{dt}", "scripts/run_reviewer_stability.py", ["timestep", "--inputs", str(inputs),
            "--dt", str(dt), "--years", "1", "--require-gpu"])
        if output is None and not failures:
            output = Path(json.loads((run / f"timestep-{dt}" / "success.json").read_text())["output"])
        link = studies / f"timestep-{dt}"
        if output is not None and not link.exists():
            link.symlink_to(output, target_is_directory=True)
    if not failures:
        subprocess.run([str(python), str(source / "scripts/analyse_reviewer_timesteps.py"), str(studies),
            "--output", str(run / "timestep-comparison.json")], cwd=source, env=env, check=True)
        subprocess.run([str(python), str(source / "scripts/analyse_reviewer_fields.py"), str(studies),
            "--output", str(run / "field-comparison")], cwd=source, env=env, check=True)
(run / "campaign-task.json").write_text(json.dumps(dict(task=a.task, failed=failures), indent=2)+"\n")
raise SystemExit(bool(failures))
