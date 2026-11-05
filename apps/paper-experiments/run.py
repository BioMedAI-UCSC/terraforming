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
INPUTS = "outputs/paper/inputs"
WINDOWS = "outputs/paper/windows/windows.json"


def experiments():
    tasks = {}
    for seed in range(3):
        for suffix in ([str(seed), "repeat"] if seed == 0 else [str(seed)]):
            name = f"radiation-{suffix}"
            tasks[name] = [f"examples/neural/train_radiation.py --output {{output}} --seed {seed} --epochs 100 --fine-tune-epochs 20 --continuation-updates 20 --continuation-budget updates"]
    tasks["column-recovery"] = ["examples/neural/recover_parameter.py --output {output} --iterations 120 --steps 24 --seed 0"]
    tasks["control"] = ["examples/neural/optimize_atmospheric_control.py --output {output} --iterations 40 --steps 8 --layers 2 --target-shift-k -0.2 --seed 0"]
    driver = f"scripts/run_differentiable_experiments.py"
    tasks["preflight"] = [f"{driver} calibration --inputs {INPUTS} --manifest {WINDOWS} --output {{output}} --validate-only"]
    tasks["gradients"] = [f"{driver} gradients --inputs {INPUTS} --output {{output}} --steps 1 4 16 32 74 148 --epsilons 1e-3 1e-4 1e-5 --require-gpu"]
    tasks["scalar-sensitivity"] = [f"scripts/run_scalar_sensitivity.py --inputs {INPUTS} --output {{output}} --steps 74 296 1480 --dt 300 --epsilon 1e-3 --require-gpu"]
    for seed in range(3):
        tasks[f"calibration-{seed}"] = [f"{driver} calibration --inputs {INPUTS} --manifest {WINDOWS} --output {{output}} --optimizer lbfgsb --starts 3 --updates 500 --max-evaluations 1000 --gtol 1e-6 --ftol 1e-12 --seed {seed} --require-gpu"]
        tasks[f"neural-{seed}"] = [f"{driver} neural --inputs {INPUTS} --manifest {WINDOWS} --output {{output}} --updates 50 --seed {seed} --require-gpu"]
    tasks["short-ablations"] = [f"{driver} ablations --inputs {INPUTS} --manifest {WINDOWS} --output {{output}} --cases no_pbl no_convection no_regolith half_timestep --require-gpu"]
    for name, task, config in [("synthetic", "recovery", "paper-experiment.json"), ("timing", "benchmark", "paper-experiment.json"), ("annual", "ablations", "paper-ablations-year.json")]:
        tasks[name] = [f"scripts/run_mars_paper_experiments.py --inputs {INPUTS} --config apps/mars-calibration/{config} --output {{output}} --task {task}"]
    tasks["climate"] = ["scripts/run_gcm3d_ablation.py data/tes/mgs_tes_surface_1deg.nc --output-dir {output} --config convection --truncation T21 --layers 12 --dt 300 --initial-ls 0 --diurnal --sols 2006 --chunk-sols 5 --cooldown-seconds 0 --co2-lw-scale 1 --dust-lw-scale 1 --surface-exchange-multiplier 1 --ames-dust-reference data/ames/fv3betaout1/ames_surface_reference.nc"]
    tasks["seasonal"] = [
        "scripts/evaluate_seasonal_convergence.py {climate}/convection/checkpoint_diagnostics.csv --start-sol 668 --output {output}/convergence.json",
    ]
    for year, start, end in [("year2", 668, 1337), ("year3", 1337, 2006)]:
        tasks["seasonal"].append(f"scripts/export_mars_seasonal_checkpoints.py {{climate}}/convection/checkpoints data/tes/mgs_tes_surface_1deg.nc data/ames/fv3betaout1/ames_surface_reference.nc --output-dir {{output}}/{year} --targets 45,135,225,315 --half-width 5 --evaluation-start-sol {start} --evaluation-end-sol {end} --dt 300 --diurnal")
    tasks["seasonal"] += [
        "scripts/evaluate_seasonal_field_convergence.py {output}/year2 {output}/year3 --output-dir {output}/field-convergence",
        "scripts/benchmark_ames.py {output}/year3 data/ames/fv3betaout1/ames_surface_reference.nc --output-dir {output}/ames --evaluation-start-sol 1337",
        "scripts/evaluate_arco_seasonal.py {output}/year3 data/arco_macda --output-dir {output}/arco --mars-years 34,35",
    ]
    for ls in (45, 135, 225, 315):
        tasks["seasonal"].append(f"scripts/benchmark_mcd.py {{output}}/ames/model_ls{ls:03d}.nc --output-dir {{output}}/mcd-ls{ls:03d} --ls {ls}")
    return tasks


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
    parser.add_argument("task", choices=["stage", "all", "list", *experiments()])
    parser.add_argument("--work", type=Path, default=Path("/work"))
    parser.add_argument("--run-id", default="paper-v1")
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
        if args.task == "stage":
            if not Path(".paper-staged").exists():
                stage(python)
            return
        if not Path(".paper-staged").exists():
            raise RuntimeError("Run the CPU staging job first")
        root = args.work / "runs" / SOURCE_ID / args.run_id
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
