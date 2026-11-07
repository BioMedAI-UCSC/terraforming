"""Frozen ICLR paper experiment commands; no Kubernetes or image dependencies."""
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

