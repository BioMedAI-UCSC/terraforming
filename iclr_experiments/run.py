"""Pinned runtime checkout, installation, staging and sequential paper experiments."""
import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import urllib.request

REPOSITORY = "https://github.com/BioMedAI-UCSC/terraforming.git"
REVISIONS = [
    "2f4825ce0f3e1ac51238ca929ceb357c96a60c04",  # main baseline
    "fb8ee5d4c265e9554d5baff19d3a996999f79cfc",  # #115
    "d57f32910113b4a5ff738ede941990e0f980a79b",  # #116
    "318617a073cf268e6b3d28140eccae006e18cfbb",  # #117
]
SOURCE_ID = hashlib.sha256("\n".join(REVISIONS).encode()).hexdigest()[:16]
from experiment_tasks import experiments, INPUTS, WINDOWS


def command(args, **kwargs):
    print("+", shlex.join(map(str, args)), flush=True)
    subprocess.run(list(map(str, args)), check=True, **kwargs)


def setup(work, gpu):
    source = work / "source" / SOURCE_ID
    marker = source / ".paper-source.json"
    if not marker.exists():
        if source.exists():
            source.rename(source.with_name(source.name + ".partial-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
        source.mkdir(parents=True, exist_ok=True)
        command(["git", "init", source])
        command(["git", "-C", source, "config", "user.email", "paper-runtime@localhost"])
        command(["git", "-C", source, "config", "user.name", "Paper runtime"])
        for revision in REVISIONS:
            command(["git", "-C", source, "fetch", REPOSITORY, revision])
        command(["git", "-C", source, "checkout", "--detach", REVISIONS[0]])
        # Fixed dates make the merged source identity reproducible across pods.
        env = dict(os.environ, GIT_AUTHOR_DATE="2026-11-05T00:00:00Z", GIT_COMMITTER_DATE="2026-11-05T00:00:00Z")
        for revision in REVISIONS[1:]:
            command(["git", "-C", source, "merge", "--no-edit", revision], env=env)
        marker.write_text(json.dumps({"repository": REPOSITORY, "revisions": REVISIONS}, indent=2))
    if json.loads(marker.read_text())["revisions"] != REVISIONS:
        raise RuntimeError("Source provenance mismatch")
    os.chdir(source)
    if 'EXPERIMENT_RUN_DIR' in os.environ:
        prepared=Path(os.environ['EXPERIMENT_RUN_DIR'])/'prepared'/SOURCE_ID
        prepared.mkdir(parents=True,exist_ok=True)
        if (source/'outputs').is_symlink():
            if (source/'outputs').resolve()!=prepared.resolve():
                raise RuntimeError('Prepared-output identity mismatch')
        elif (source/'outputs').exists():
            raise RuntimeError('Existing prepared outputs must not be overwritten')
        else:
            (source/'outputs').symlink_to(prepared,target_is_directory=True)
    os.environ["PYTHONPATH"] = f"{source}/package:{source}:{source}/apps/mars-calibration"
    command(["uv", "sync", "--frozen", "--all-packages", "--extra", "gcm3d", "--extra", "arco", "--no-dev"])
    python = source / ".venv/bin/python"
    if gpu:
        command(["uv", "pip", "install", "--python", python, "jax[cuda12]==0.11.0"])
        os.environ["JAX_PLATFORMS"] = "cuda"
        command([python, "-c", "import jax; print(jax.devices()); assert jax.default_backend() == 'gpu'"])
    else:
        os.environ["JAX_PLATFORMS"] = "cpu"
    return source, python


def stage(python):
    commands = [
        "scripts/stage_mola.py", "scripts/stage_tes_surface.py",
        "scripts/stage_arco_macda.py --mars-year 32 --center-ls 45 --sols 10 --truncation T21 --layers 12 --output data/arco_macda/macda_my32_ls045_10sol_daily_t21_l12.nc",
        "scripts/neural_temp_data.py --run-dir outputs/paper/native --mola data/mola/meg004/megt90n000cb.img",
        "scripts/prepare_differentiable_windows.py --native-run outputs/paper/native --output outputs/paper/windows --per-split 4",
    ]
    progress = Path(".paper-stage-steps")
    progress.mkdir(exist_ok=True)

    def step(name, args, exclusive_output=None):
        marker = progress / name
        if marker.exists():
            return
        if exclusive_output and Path(exclusive_output).exists():
            path = Path(exclusive_output)
            path.rename(path.with_name(path.name + ".partial-" + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")))
        command([python, *args])
        marker.write_text("complete")

    for index, text in enumerate(commands):
        step(str(index), shlex.split(text), "outputs/paper/windows" if index == 4 else None)
    for year in (34, 35):
        for ls in (45, 135, 225, 315):
            step(f"arco-{year}-{ls}", ["scripts/stage_arco_macda.py", "--mars-year", year, "--center-ls", ls, "--sols", 10, "--truncation", "T21", "--layers", 12, "--output", f"data/arco_macda/macda_my{year}_ls{ls:03d}_10sol_daily_t21_l12.nc"])
    raw = Path("data/ames/fv3betaout1/03340.atmos_average_zstd.nc")
    raw.parent.mkdir(parents=True, exist_ok=True)
    if not raw.exists():
        partial = raw.with_suffix(".partial")
        urllib.request.urlretrieve("https://data.nas.nasa.gov/legacygcm/fv3betaout1/fv3betaout1/" + raw.name, partial)
        partial.replace(raw)
    step("ames", ["scripts/stage_ames_surface_reference.py", raw, raw.parent / "ames_surface_reference.nc"])
    step("inputs", ["apps/mars-calibration/prepare_inputs.py", "--repo", ".", "--restart", "outputs/paper/windows/my32_q0_s0-restart.npz", "--output", INPUTS], INPUTS)
    Path(".paper-staged").write_text(SOURCE_ID)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", nargs='?', default='all', choices=["stage", "all", "list", *experiments()])
    parser.add_argument("--work", type=Path, default=Path(os.environ.get("EXPERIMENT_WORK_DIR", "/work")))
    parser.add_argument("--run-id", default=os.environ.get("EXPERIMENT_RUN_ID", "paper-v1"))
    args = parser.parse_args()
    if args.task == "list":
        print(json.dumps(experiments(), indent=2))
        return
    if not args.run_id or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_" for c in args.run_id):
        parser.error("run-id must contain only letters, digits, dash or underscore")
    args.work.mkdir(parents=True, exist_ok=True)
    with (args.work / ".paper.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        source, python = setup(args.work, args.task != "stage")
        if Path('.paper-staged').exists() and Path('.paper-staged').read_text()!=SOURCE_ID:
            raise RuntimeError('Input source identity mismatch')
        if args.task == "stage":
            if not Path(".paper-staged").exists():
                stage(python)
            return
        if not Path(".paper-staged").exists():
            stage(python)
        root = Path(os.environ["EXPERIMENT_RUN_DIR"]) / "tasks" / SOURCE_ID if "EXPERIMENT_RUN_DIR" in os.environ else args.work / "outputs" / SOURCE_ID / args.run_id
        root.mkdir(parents=True, exist_ok=True)
        (root / "source.json").write_text((source / ".paper-source.json").read_text())
        if not (root / "source.tar").exists():
            command(["git", "archive", "--output", root / "source.tar", "HEAD"])
        with (root / "environment.txt").open("w") as handle:
            command(["uv", "pip", "freeze", "--python", python], stdout=handle)
            command([python, "-c", "import jax; print(jax.__version__, jax.devices())"], stdout=handle)
        failed = []
        for name, commands in experiments().items():
            if args.task not in ("all", name):
                continue
            status = root / f"{name}.json"
            if status.exists() and json.loads(status.read_text())["returncode"] == 0:
                continue
            attempt = root / name / datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
            if name == "climate":
                previous = sorted((root / name).glob("*/results/convection/restart.npz"))
                if previous:
                    attempt = previous[-1].parents[2]
                    commands = [text + " --resume" for text in commands]
            attempt.mkdir(parents=True, exist_ok=True)
            rc = 0
            with (attempt / "run.log").open("a", buffering=1) as log:
                try:
                    climate = ""
                    if name == "seasonal":
                        climate_status = json.loads((root / "climate.json").read_text())
                        if climate_status["returncode"] != 0:
                            raise RuntimeError("Climate must complete before seasonal evaluation")
                        climate = str(Path(climate_status["attempt"]) / "results")
                        (attempt / "results").mkdir()
                    for text in commands:
                        command([python, *shlex.split(text.format(output=attempt / "results", climate=climate))], stdout=log, stderr=subprocess.STDOUT)
                except (subprocess.CalledProcessError, RuntimeError, FileNotFoundError) as error:
                    print(str(error), file=log)
                    rc = error.returncode if isinstance(error, subprocess.CalledProcessError) else 1
            status.write_text(json.dumps({"returncode": rc, "attempt": str(attempt), "commands": commands, "scientific_acceptance": "inspect experiment reports; process exit alone is insufficient"}, indent=2))
            if rc:
                failed.append(name)
        if failed:
            raise RuntimeError(f"Failed experiments: {failed}; inspect per-attempt run.log")


if __name__ == "__main__":
    main()
