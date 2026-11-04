"""Plan, run/resume, and plot matched aerosol experiments on a GPU server.

Planning uses only the standard library. No simulation runs without `run`.
Paper numbers are reference assumptions, never imposed temperature targets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import signal
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "package")]
SOL_S = 88775.244
MARS_YEAR_S = 59356800.0
EARTH_YEAR_S = 365.25 * 86400
PAPER_URL = "https://arxiv.org/pdf/2604.02242"


def write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def git_output(*args):
    result = subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True, text=True)
    return result.stdout.strip() if result.returncode == 0 else "unknown"


def plan(args):
    if not (math.isfinite(args.sols) and args.sols > 0 and
            math.isfinite(args.spinup_sols) and args.spinup_sols > 0):
        raise ValueError("experiment and spinup sols must be finite and positive")
    if not math.isfinite(args.dt) or not 0 < args.dt <= 300:
        raise ValueError("dt must be in (0, 300] seconds")
    if not 2 <= args.layers <= 40:
        raise ValueError("layers must be between 2 and 40")
    output = args.output.resolve()
    if (output / "suite.json").exists():
        raise ValueError("suite already exists; use run to resume or choose another output directory")
    duration = math.ceil(args.sols * SOL_S / args.dt) * args.dt
    spinup_duration = math.ceil(args.spinup_sols * SOL_S / args.dt) * args.dt
    # Same inventory-safe wind/temperature damping convention as Mars maps.
    tau = .1 * SOL_S * 42 / int(args.truncation[1:])
    common = dict(material=args.material, paired=True, mode="active", dt_s=args.dt,
                  n_layers=args.layers, truncation=args.truncation, duration_s=duration,
                  hyperdiffusion_tau_seconds=tau)
    cases = []
    for name, burden, rate, stop, warming in (
        ("loading_2mg", 2e-6, 0., duration, 4.),
        ("release_2l_s", 0., 2., duration, 4.),
        ("release_60l_s", 0., 60., duration, 35.),
        ("shutdown_2l_s", 0., 2., duration / 2, None),
    ):
        source = [] if rate == 0 else [dict(start_s=0., stop_s=stop, rate=rate, unit="L/s")]
        cfg = dict(planet=dict(surface_temperature=210., surface_pressure=610., initial_ls_deg=0.),
                   intervention=dict(aerosol=dict(common, initial_burden_kg_m2=burden, sources=source)))
        cases.append(dict(name=name, configuration=cfg, expected_warming_k=warming,
                          expectation_kind="greater_than" if rate == 60 else "approximately",
                          reference_lifetime_s=EARTH_YEAR_S if args.lifetime_year == "earth" else MARS_YEAR_S,
                          interpretation="transient comparison; reference warming is not an imposed target"))
    manifest = dict(schema="aerosol-gpu-suite-v1", paper=PAPER_URL,
                    paper_section="3.3.1, pages 24–26",
                    assumptions=dict(metal_loading_for_4k_kg_m2=2e-6,
                                     aluminium_release_for_4k_l_s=2.,
                                     aluminium_release_for_over_35k_l_s=60.,
                                     effective_lifetime_s=cases[0]["reference_lifetime_s"],
                                     lifetime_year_convention=args.lifetime_year,
                                     lifetime_is_simulated=False,
                                     graphene="not run: source-backed graphene optics are unavailable",
                                     caveat="paper release numbers are rounded; loading and rate are separate comparisons"),
                    spinup_s=spinup_duration, duration_s=duration, cases=cases,
                    source_commit=git_output("rev-parse", "HEAD"),
                    baseline="flat terrain, emissivity 1, fixed gas composition; spinup is not proof of equilibration")
    write_json(output / "suite.json", manifest)
    for case in cases:
        write_json(output / case["name"] / "input.json", case["configuration"])
    steps = math.ceil(duration / args.dt)
    print(f"Plan saved: {output / 'suite.json'}")
    print(f"Spinup: {spinup_duration / SOL_S:.3f} sols; each experiment: {duration / SOL_S:.3f} sols")
    print(f"{len(cases)} sequential matched experiments, {steps:,} steps per case; "
          f"{steps * 3 * len(cases):,} climate case-steps plus spinup")
    print("No simulation executed. Run a short suite first to measure throughput and validity.")


def _already_finished(output, duration):
    summary = output / "summary.json"
    return summary.exists() and read_json(summary).get("status") == "completed" and \
        read_json(summary)["elapsed_seconds"] >= duration - 1e-6


def run(args):
    import fcntl
    # A shared spinup/restart must not be written by two GPU jobs at once.
    with (args.output.resolve() / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("another process is already running this suite") from None
        _run(args)


def _run(args):
    import jax
    jax.config.update("jax_enable_x64", True)
    devices = jax.devices()
    if not args.allow_cpu and not any(device.platform == "gpu" for device in devices):
        raise RuntimeError("JAX cannot see a GPU. Install CUDA-enabled JAX, or explicitly pass --allow-cpu.")
    from cli.aerosols import run as run_aerosols
    from cli.models import AerosolConfig, PlanetConfig
    from src.interventions.aerosols import checkpoint
    from src.interventions.aerosols.pilot import prepare
    from src.interventions.aerosols.starting import save_physical_start
    import dataclasses

    output = args.output.resolve()
    manifest = read_json(output / "suite.json")
    if manifest.get("schema") != "aerosol-gpu-suite-v1":
        raise ValueError("unsupported suite manifest")
    selected = args.case or [case["name"] for case in manifest["cases"]]
    if set(selected) - {case["name"] for case in manifest["cases"]}:
        raise ValueError("unknown case selection")
    runtime = dict(commit=git_output("rev-parse", "HEAD"),
                   working_tree=git_output("status", "--porcelain"),
                   source_diff_sha256=hashlib.sha256(git_output("diff", "HEAD", "--binary").encode()).hexdigest(),
                   python=sys.version, platform=platform.platform(), jax=jax.__version__,
                   devices=[dict(platform=d.platform, kind=d.device_kind, id=d.id) for d in devices],
                   x64=bool(jax.config.x64_enabled),
                   suite_sha256=hashlib.sha256((output / "suite.json").read_bytes()).hexdigest())
    previous_runtime = output / "runtime.json"
    if previous_runtime.exists():
        previous = read_json(previous_runtime)
        if any(previous.get(key) != runtime[key] for key in ("commit", "suite_sha256", "source_diff_sha256")):
            raise ValueError("source commit or suite changed; use a new output directory")
    write_json(previous_runtime, runtime)
    with (output / "runtime-attempts.ndjson").open("a") as stream:
        stream.write(json.dumps(dict(runtime, unix_time_s=time.time()), allow_nan=False) + "\n")
    stopped = [False]
    def stop(signum, frame):
        stopped[0] = True
    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    started = time.monotonic()
    timing_path = output / "timing.json"
    timings = read_json(timing_path) if timing_path.exists() else []

    def execute(configuration, directory):
        aerosol = AerosolConfig(**configuration["intervention"]["aerosol"])
        planet = PlanetConfig(**configuration["planet"])
        if _already_finished(directory, aerosol.duration_s):
            print(f"Already complete: {directory.name}", flush=True)
            return aerosol, planet
        restart = directory / "restart.npz"
        before = time.monotonic()
        initial_elapsed = read_json(directory / "summary.json")["elapsed_seconds"] \
            if restart.exists() and (directory / "summary.json").exists() else None
        print(f"Running {directory.name}; resume={restart.exists()}", flush=True)
        try:
            report = run_aerosols(aerosol, planet, directory,
                initial_ls_deg=planet.initial_ls_deg, resume=restart if restart.exists() else None,
                should_stop=lambda: stopped[0],
                callback=lambda report, fields: print(json.dumps(dict(case=directory.name,
                    elapsed_s=report["elapsed_seconds"], status=report["status"],
                    surface_temperature_k=report["surface_temperature_k"])), flush=True))
        except Exception:
            timings.append(dict(case=directory.name, wall_seconds=time.monotonic()-before,
                                status="failed; inspect last-valid summary/history"))
            write_json(timing_path, timings)
            raise
        timings.append(dict(case=directory.name, wall_seconds=time.monotonic()-before,
                            initial_elapsed_s=initial_elapsed, final_elapsed_s=report["elapsed_seconds"],
                            status=report["status"], includes_compilation_and_io=True))
        write_json(timing_path, timings)
        if report["status"] != "completed":
            raise SystemExit("Stopped safely; restart is saved. Run the same command to continue.")
        return aerosol, planet

    first = manifest["cases"][0]["configuration"]
    spinup_input = json.loads(json.dumps(first))
    spinup_options = spinup_input["intervention"]["aerosol"]
    spinup_options.update(paired=False, mode="off", initial_burden_kg_m2=0., sources=[],
                          duration_s=manifest["spinup_s"])
    spinup_dir = output / "spinup"
    spinup_config, planet = execute(spinup_input, spinup_dir)
    physical_start = spinup_dir / "physical-start.npz"
    if not physical_start.exists():
        options = spinup_config.model_dump(exclude={"duration_s", "paired"})
        template, _, context = prepare(**options, surface_temperature_k=planet.surface_temperature,
            surface_pressure_pa=planet.surface_pressure, albedo=planet.albedo,
            initial_ls_deg=planet.initial_ls_deg)
        state = checkpoint.load(spinup_dir / "restart.npz", template,
                                checkpoint.read_configuration(spinup_dir / "restart.npz"))
        climate = state.climate._replace(dynamics=dataclasses.replace(state.climate.dynamics, tracers={}))
        save_physical_start(physical_start, climate, context["configuration"]["start_contract"])
    for case in manifest["cases"]:
        if case["name"] not in selected:
            continue
        cfg = json.loads(json.dumps(case["configuration"]))
        cfg["intervention"]["aerosol"]["initial_climate"] = str(physical_start)
        execute(cfg, output / case["name"])
    print(f"Selected experiments complete in {time.monotonic()-started:.1f} wall seconds", flush=True)


def reference_inventory(times, initial_kg, rate_kg_s, stop_s, lifetime_s):
    """One-box reference ONLY: dM/dt = release - M/tau, with finite shutdown."""
    import numpy as np
    times = np.asarray(times, dtype=float)
    before_stop = np.minimum(times, stop_s)
    return (initial_kg * np.exp(-times / lifetime_s) +
            rate_kg_s * lifetime_s * (-np.expm1(-before_stop / lifetime_s)) *
            np.exp(-np.maximum(times - stop_s, 0.) / lifetime_s))


def plot(args):
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    output = args.output.resolve()
    manifest = read_json(output / "suite.json")
    fig, axes = plt.subplots(2, 2, figsize=(13, 9), layout="constrained")
    records = []
    for case in manifest["cases"]:
        directory = output / case["name"]
        if not (directory / "configuration.json").exists():
            continue
        config = read_json(directory / "configuration.json")["cases"]["active"]
        area = config["body"]["radius_m"] ** 2 * 4 * math.pi
        density = config["particle"]["density_kg_m3"]
        history = directory / "history.ndjson"
        rows = []
        if history.exists():
            lines = history.read_text().splitlines()
            for index, line in enumerate(lines):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    if index == len(lines)-1:
                        break  # plotting a live run may encounter its unfinished last line
                    raise
                if "cases" in row:
                    rows.append(row)
        summary_path = directory / "summary.json"
        if summary_path.exists():
            summary = read_json(summary_path)
            if "cases" in summary:
                rows.append(summary)
        # A resume can repeat a terminal sample; retain its latest version.
        rows = sorted({row["elapsed_seconds"]: row for row in rows}.values(), key=lambda r:r["elapsed_seconds"])
        if not rows:
            continue
        times = np.array([r["elapsed_seconds"] for r in rows])
        active = [r["cases"]["active"] for r in rows]
        warming = np.array([a["surface_temperature_k"]-r["cases"]["off"]["surface_temperature_k"]
                            for a, r in zip(active, rows)])
        passive = [r["cases"]["passive"]["surface_temperature_k"]-r["cases"]["off"]["surface_temperature_k"] for r in rows]
        initial = config["initial_burden_kg_m2"] * area
        expected = np.array([initial+a["injected_mass_kg"] for a in active])
        relative_budget = np.array([a["material_budget_error_kg"] for a in active]) / np.maximum(expected, 1e-30)
        label = case["name"]
        line, = axes[0, 0].plot(times/SOL_S, warming, label=label)
        color = line.get_color()
        axes[0, 1].plot(times/SOL_S, [a["suspended_mass_kg"]*1e6/area for a in active], color=color, label=label)
        source = case["configuration"]["intervention"]["aerosol"]["sources"]
        rate = source[0]["rate"] * .001 * density if source else 0.
        stop_s = source[0]["stop_s"] if source else 0.
        assumed = reference_inventory(times, initial, rate, stop_s, case["reference_lifetime_s"])
        axes[0, 1].plot(times/SOL_S, assumed*1e6/area, color=color, linestyle=":", alpha=.8)
        axes[1, 0].plot(times/SOL_S, passive, color=color, label=label)
        axes[1, 1].plot(times/SOL_S, relative_budget, color=color, label=label)
        last = rows[-1]
        records.append(dict(case=label, status=last["status"], simulated_sols=times[-1]/SOL_S,
            simulated_mars_years=times[-1]/config["forcing"]["orbital_period_s"], final_warming_k=float(warming[-1]),
            paper_reference_warming_k=case["expected_warming_k"], reference_kind=case["expectation_kind"],
            final_loading_mg_m2=active[-1]["suspended_mass_kg"]*1e6/area,
            assumed_lifetime_years=1., reference_year=manifest["assumptions"]["lifetime_year_convention"],
            final_relative_material_residual=float(relative_budget[-1]),
            maximum_raw_relative_mass_error=max(a["maximum_raw_relative_mass_error"] for a in active),
            maximum_absolute_passive_control_k=float(np.max(np.abs(passive))),
            paired_time_mean_warming_k=last.get("warming", {}).get("paired_time_mean_warming_k"),
            completed_mars_years=last.get("warming", {}).get("completed_mars_years", 0),
            last_complete_annual_warming_k=last.get("warming", {}).get("consecutive_annual_warming_k", []),
            interpretation="transient, not an equilibrium/source reproduction verdict"))
    if not records:
        plt.close(fig)
        raise ValueError("no matched histories available yet")
    axes[0, 0].axhline(4, color="grey", linestyle="--", label="paper ~4 K reference")
    axes[0, 0].axhline(35, color="black", linestyle="--", label="paper >35 K reference")
    axes[0, 0].set(ylabel="Area-mean active − control temperature (K)", title="Transient response versus climate references")
    axes[0, 1].axhline(2, color="grey", linestyle="--", label="paper 2 mg/m² loading")
    convention = manifest["assumptions"]["lifetime_year_convention"]
    axes[0, 1].set(ylabel="Suspended column mass (mg/m²)", title=f"Dotted curves assume 1 {convention}-year removal lifetime")
    axes[1, 0].set(ylabel="Passive − control temperature (K)", title="Transport-only climate null check")
    axes[1, 1].set(ylabel="Material residual / total supplied mass", title="Post-repair accounting; inspect raw corrections too")
    for ax in axes.flat:
        ax.set_xlabel("Elapsed intervention sols (after common spinup)")
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.suptitle("Engineered aerosol experiments — flat-terrain baseline; paper assumptions are unverified")
    fig.savefig(output / "paper_comparison.png", dpi=160)
    plt.close(fig)
    annual = [record for record in records if record["last_complete_annual_warming_k"]]
    if annual:
        fig, ax = plt.subplots(figsize=(8, 5), layout="constrained")
        for record in annual:
            values = record["last_complete_annual_warming_k"]
            last_year = record["completed_mars_years"]
            ax.plot(range(last_year-len(values)+1, last_year+1), values, "o-", label=record["case"])
        ax.axhline(4, color="grey", linestyle="--", label="paper ~4 K reference")
        ax.axhline(35, color="black", linestyle="--", label="paper >35 K reference")
        ax.set(xlabel="Complete intervention Mars year", ylabel="Annual area-mean active − control (K)",
               title="Latest complete annual means; baseline/source equivalence remains unverified")
        ax.legend()
        ax.grid(alpha=.2)
        fig.savefig(output / "annual_warming.png", dpi=160)
        plt.close(fig)
    with (output / "paper_comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    write_json(output / "paper_comparison.json", dict(paper=PAPER_URL, cases=records,
        warning="No automatic claim of matching equilibrium warming, particle lifetime or validated Mars climatology."))
    for case in manifest["cases"]:
        directory = output / case["name"]
        if (directory / "results.nc").exists() and (directory / "history.ndjson").exists():
            subprocess.run([sys.executable, str(ROOT / "scripts/plot_aerosol_results.py"), str(directory)], check=True)
    print(f"Saved {output / 'paper_comparison.png'} and CSV/JSON comparisons")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    planner = commands.add_parser("plan", help="Write a suite without importing JAX or running a model")
    planner.add_argument("--output", type=Path, required=True)
    planner.add_argument("--sols", type=float, default=10.)
    planner.add_argument("--spinup-sols", type=float, default=10.)
    planner.add_argument("--dt", type=float, default=60.)
    planner.add_argument("--layers", type=int, default=12)
    planner.add_argument("--truncation", choices=["T21", "T42", "T85"], default="T21")
    planner.add_argument("--material", choices=["aluminium", "iron"], default="aluminium")
    planner.add_argument("--lifetime-year", choices=["earth", "mars"], default="earth",
                         help="Explicit year convention for the assumed-lifetime comparison curve")
    runner = commands.add_parser("run", help="Run or resume the planned spinup and selected cases")
    runner.add_argument("--output", type=Path, required=True)
    runner.add_argument("--case", action="append", help="Run only this named case; repeat to select several")
    runner.add_argument("--allow-cpu", action="store_true")
    plotter = commands.add_parser("plot", help="Plot saved histories, including incomplete experiments")
    plotter.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        dict(plan=plan, run=run, plot=plot)[args.command](args)
    except (ValueError, RuntimeError, FileNotFoundError) as exc:
        parser.exit(1, f"{exc}\n")


if __name__ == "__main__":
    main()
