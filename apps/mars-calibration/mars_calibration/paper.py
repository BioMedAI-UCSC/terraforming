"""Auditable paper experiments on the coupled Mars GCM, without neural physics.

Synthetic recovery uses multiple training times and an excluded continuation.
Ablations reset the identical restart and match physical observation times.
Runtime reports separate compilation, execution, and unavailable external data.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import resource
import subprocess
import sys
import time

import numpy as np
import scipy.optimize

from . import driver as d
from .launch import FILES, sha256
from src.framework.gcm.benchmarks import dry_conserved_quantities
from src.framework.gcm.restart import save_restart
from src.celestials.planets.mars.maps import state_to_comparison_dataset


def validate_config(config):
    required = {
        "schema_version", "dt_seconds", "train_seconds", "validation_seconds",
        "truth", "starts", "active_sets", "max_evaluations", "benchmark_repeats",
        "parameter_relative_tolerance", "gradient_relative_tolerance",
        "gradient_absolute_tolerance", "profile",
    }
    if set(config) != required or config["schema_version"] != 1:
        raise ValueError("Expected exactly the version-1 paper experiment fields")
    dt = config["dt_seconds"]
    if not np.isfinite(dt) or dt <= 0 or dt > 300:
        raise ValueError("T21/L12 experiments require 0 < dt_seconds <= 300")
    for key in ("train_seconds", "validation_seconds"):
        times = np.asarray(config[key], dtype=float)
        if (times.ndim != 1 or len(times) < 2 or not np.all(np.isfinite(times))
                or times[0] <= 0 or np.any(np.diff(times) <= 0)
                or not np.allclose(times / dt, np.round(times / dt), rtol=0, atol=1e-9)):
            raise ValueError(f"{key}: need >=2 increasing positive exact timestep multiples")
    if config["validation_seconds"][0] <= config["train_seconds"][-1]:
        raise ValueError("Validation must strictly follow all training timestamps")
    for values in [config["truth"], *config["starts"]]:
        values = np.asarray(values)
        if (values.shape != (3,) or not np.isfinite(values).all()
                or np.any(values <= d.LOWER_BOUNDS) or np.any(values >= d.UPPER_BOUNDS)):
            raise ValueError("truth/starts must contain three finite values inside bounds")
    if not config["starts"] or not config["active_sets"]:
        raise ValueError("At least one start and active parameter set required")
    for active in config["active_sets"]:
        if not active or len(set(active)) != len(active) or any(type(i) is not int or i not in range(3) for i in active):
            raise ValueError("active_sets must be unique parameter indices 0, 1, 2")
    for key in ("max_evaluations", "benchmark_repeats"):
        if type(config[key]) is not int or config[key] < 2:
            raise ValueError(f"{key} must be an integer >=2")
    for key in ("parameter_relative_tolerance", "gradient_relative_tolerance", "gradient_absolute_tolerance"):
        if not np.isfinite(config[key]) or config[key] <= 0:
            raise ValueError(f"{key} must be finite and positive")
    if not isinstance(config["profile"], str) or not config["profile"].strip():
        raise ValueError("profile must be a nonempty label")
    return config


def validate_inputs(inputs):
    manifest = json.loads((inputs / "manifest.json").read_text())
    for name in FILES:
        if sha256(inputs / name) != manifest["files"][name]["sha256"]:
            raise ValueError(f"Input hash mismatch: {name}")
    return manifest


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def block(value):
    return d.jax.block_until_ready(value)


def compile_function(function, argument):
    started = time.perf_counter()
    executable = d.jax.jit(function).lower(argument).compile()
    compilation = time.perf_counter() - started
    started = time.perf_counter()
    block(executable(argument))
    return executable, {"compile_seconds": compilation,
                        "warmup_seconds": time.perf_counter() - started}


def gradient_checks(forward, gradient, x, rtol, atol):
    checks = []
    for epsilon in (1e-3, 1e-4):
        numerical = []
        for i in range(len(x)):
            delta = np.zeros_like(x)
            delta[i] = epsilon
            numerical.append(float((forward(x + delta) - forward(x - delta)) / (2 * epsilon)))
        numerical = np.asarray(numerical)
        error = np.abs(gradient - numerical)
        checks.append({"epsilon": epsilon, "finite_difference": numerical.tolist(),
                       "absolute_error": error.tolist(),
                       "relative_error": (error / np.maximum(np.abs(numerical), 1e-12)).tolist(),
                       "pass": bool(np.all(error <= atol + rtol * np.abs(numerical)))})
    return checks


def optimize(forward, value_gradient, start, method, budget, emit):
    """Both methods operate in the same bounded normalized coordinates.

    Budgets are ceilings, not padded by repeated calls after convergence.
    Every trace row is a real, synchronized oracle evaluation.
    """
    trace = []
    started = time.perf_counter()

    def oracle(x):
        if len(trace) >= budget:
            raise d.EvaluationBudgetReached
        call_started = time.perf_counter()
        result = block(value_gradient(x) if method == "L-BFGS-B" else forward(x))
        value = float(result[0] if method == "L-BFGS-B" else result)
        if not np.isfinite(value) or (method == "L-BFGS-B" and not np.isfinite(result[1]).all()):
            raise FloatingPointError("Non-finite optimizer loss or gradient")
        row = {"method": method, "evaluation": len(trace) + 1,
               "loss": value, "normalized_parameters": np.asarray(x).tolist(),
               "oracle_seconds": time.perf_counter() - call_started,
               "cumulative_wall_seconds": time.perf_counter() - started}
        trace.append(row)
        emit(row)
        return (value, np.asarray(result[1])) if method == "L-BFGS-B" else value

    try:
        options = ({"maxfun": budget, "maxiter": budget, "ftol": 1e-12,
                    "gtol": 1e-10, "maxls": 30} if method == "L-BFGS-B" else
                   {"maxfev": budget, "maxiter": budget, "xtol": 1e-6, "ftol": 1e-12})
        result = scipy.optimize.minimize(oracle, np.asarray(start), method=method,
                                        jac=method == "L-BFGS-B", bounds=[(0.0, 1.0)] * len(start),
                                        options=options)
        termination = str(result.message)
    except d.EvaluationBudgetReached:
        termination = "evaluation budget exhausted"
    best = min(trace, key=lambda row: row["loss"])
    return {"best": best, "trace": trace, "termination": termination,
            "evaluations": len(trace), "wall_seconds": time.perf_counter() - started}


class Model:
    def __init__(self, inputs):
        self.coords = d.coordinate_system("T21", 12)
        self.specs = d.physics_specs(d.MARS_BODY_3D)
        self.initial = d.load_restart(inputs / "restart.npz")
        for leaf in d.jax.tree_util.tree_leaves(self.initial):
            if not np.isfinite(leaf).all():
                raise ValueError("Non-finite restart")
        if self.initial.surface_temperature.shape != (1, 64, 32):
            raise ValueError("Expected T21 restart")
        if self.initial.dynamics.temperature_variation.shape[0] != 12:
            raise ValueError("Expected L12 restart")
        self.forcing = d._forcing(inputs / "surface.nc", inputs / "dust.nc",
                                  self.coords.horizontal, d.jnp.ones(3))
        elevation = d.regrid_to_nodal(self.coords)
        self.orography = d.mola_modal_orography(self.coords, self.specs, elevation_nodal_m=elevation)
        self.weights = d.jnp.asarray(self.coords.horizontal.quadrature_weights)
        self.weights /= d.jnp.sum(self.weights)

    def observe(self, state):
        grid = self.coords.horizontal
        ref = d.reference_temperature(self.coords, d.MARS_BODY_3D).reshape(-1, 1, 1)
        temperature = grid.to_nodal(state.dynamics.temperature_variation) + ref
        u, v = d.spherical_harmonic.vor_div_to_uv_nodal(
            grid, state.dynamics.vorticity, state.dynamics.divergence)
        scale = float(self.specs.dimensionalize(1.0, d.scales.units.meter / d.scales.units.second).magnitude)
        return d.jnp.stack([temperature, u * scale, v * scale])

    def trajectory(self, seconds, dt, *, overrides=None, co2_exchange=True, diffusion_sols=0.1, states=False):
        offsets = np.asarray(seconds) / dt
        if not np.allclose(offsets, np.round(offsets), rtol=0, atol=1e-9):
            raise ValueError("Trajectory observations must align exactly to timestep")
        counts = d.jnp.asarray(np.diff(np.r_[0, np.round(offsets)]).astype(np.int32))

        def run(parameters):
            forcing = dataclasses.replace(self.forcing,
                ames_co2_longwave_opacity_scale=parameters[0],
                ames_dust_longwave_opacity_scale=parameters[1],
                surface_exchange_multiplier=parameters[2], **(overrides or {}))
            step = d._build_step(self.coords, self.specs, forcing, dt,
                                 orography=self.orography, co2_exchange=co2_exchange,
                                 diffusion_sols=diffusion_sols)

            def sample(state, count):
                next_state = d.jax.lax.fori_loop(0, count, lambda _, carry: step(carry), state)
                return next_state, next_state if states else self.observe(next_state)

            return d.jax.lax.scan(sample, self.initial, counts)[1]
        return run

    def loss(self, predicted, target, normalization):
        error = (predicted - target) / normalization[None, :, None, None, None]
        return d.jnp.mean(d.jnp.sum(error**2 * self.weights, axis=(-2, -1)))

    def diagnostics(self, state):
        if not all(np.isfinite(leaf).all() for leaf in d.jax.tree_util.tree_leaves(state)):
            raise FloatingPointError("Non-finite sampled state")
        fields = np.asarray(self.observe(state))
        ps = np.exp(np.asarray(self.coords.horizontal.to_nodal(state.dynamics.log_surface_pressure)))[0]
        scale = float(self.specs.dimensionalize(1.0, d.scales.units.pascal).magnitude)
        ps *= scale
        ice = np.asarray(state.co2_ice)[0] * scale
        weights = np.asarray(self.weights)
        dry = dry_conserved_quantities(state.dynamics, self.coords, self.specs,
            d.MARS_BODY_3D, np.full(12, d.MARS_BODY_3D.reference_temperature_k))
        return {"co2_mass_kg": float(np.sum((ps + ice) * weights) * d.MARS_BODY_3D.surface_area_m2 / d.MARS_BODY_3D.gravity_m_s2),
                "atmospheric_energy_excluding_surface_geopotential_j_m2_sr": float(dry.total_energy_j_m2_sr),
                "atmospheric_aam_kg_m_s_sr": float(dry.axial_angular_momentum_kg_m_s_sr),
                "min_air_temperature_k": float(fields[0].min()), "max_air_temperature_k": float(fields[0].max()),
                "max_wind_ms": float(np.hypot(fields[1], fields[2]).max()),
                "min_surface_temperature_k": float(np.min(state.surface_temperature)),
                "max_surface_temperature_k": float(np.max(state.surface_temperature)),
                "mean_deep_soil_temperature_k": float(np.sum(np.asarray(state.ground_temperature)[-1] * weights)),
                "min_frost_pa": float(ice.min())}


def recovery(model, config, output, emit):
    truth = d.jnp.asarray(config["truth"])
    times = config["train_seconds"] + config["validation_seconds"]
    ntrain = len(config["train_seconds"])
    run_all = model.trajectory(times, config["dt_seconds"])
    teacher, teacher_cost = compile_function(run_all, truth)
    targets = block(teacher(truth))
    if not np.isfinite(targets).all():
        raise FloatingPointError("Non-finite synthetic teacher")
    normalization = d.jnp.maximum(d.jnp.std(targets[:ntrain], axis=(0, 2, 3, 4)), 1.0)
    np.savez_compressed(output / "synthetic-targets.npz", seconds=times,
                        fields=np.asarray(targets), normalization=np.asarray(normalization), truth=np.asarray(truth))
    # Only training times enter the optimized function; continuation is evaluated afterward.
    train = model.trajectory(config["train_seconds"], config["dt_seconds"])
    results, flat_rows, trajectory_rows, recovery_rows = [], [], [], []
    for active in config["active_sets"]:
        index = d.jnp.asarray(active)
        lower, span = d.jnp.asarray(d.LOWER_BOUNDS)[index], d.jnp.asarray(d.UPPER_BOUNDS - d.LOWER_BOUNDS)[index]

        def expand(x):
            return truth.at[index].set(lower + span * x)

        def objective(x):
            return model.loss(train(expand(x)), targets[:ntrain], normalization)

        def objective_aux(x):
            loss = objective(x)
            return loss, loss

        tangent = d.jax.jacfwd(objective_aux, has_aux=True)

        def value_gradient(x):
            gradient, loss = tangent(x)
            return loss, gradient

        first = (d.jnp.asarray(config["starts"][0])[index] - lower) / span
        forward, forward_cost = compile_function(objective, first)
        vg, gradient_cost = compile_function(value_gradient, first)
        for start_id, physical_start in enumerate(config["starts"]):
            start = np.asarray((d.jnp.asarray(physical_start)[index] - lower) / span)
            initial_loss, initial_gradient = block(vg(start))
            checks = gradient_checks(forward, np.asarray(initial_gradient), start,
                                     config["gradient_relative_tolerance"], config["gradient_absolute_tolerance"])
            initial_prediction = block(teacher(expand(start)))
            initial_validation = float(model.loss(initial_prediction[ntrain:], targets[ntrain:], normalization))
            for method in ("L-BFGS-B", "Powell"):
                label = {"active": active, "start_id": start_id}
                result = optimize(forward, vg, start, method, config["max_evaluations"],
                                  lambda row: emit({"phase": "recovery", **label, **row}))
                best = result["best"]
                parameters = expand(d.jnp.asarray(best["normalized_parameters"]))
                predicted = block(teacher(parameters))
                prediction_name = f"fit-{'-'.join(map(str, active))}-start{start_id}-{method}.npz"
                np.savez_compressed(output / prediction_name, seconds=times,
                                    fields=np.asarray(predicted), parameters=np.asarray(parameters))
                per_time_rmse = np.sqrt(np.mean(np.sum(
                    (np.asarray(predicted) - np.asarray(targets))**2 * np.asarray(model.weights),
                    axis=(-2, -1)), axis=-1))
                for time_index, seconds in enumerate(times):
                    trajectory_rows.append({"active": str(active), "start_id": start_id,
                        "method": method, "seconds": seconds,
                        "split": "train" if time_index < ntrain else "validation",
                        "temperature_rmse_k": float(per_time_rmse[time_index, 0]),
                        "u_rmse_ms": float(per_time_rmse[time_index, 1]),
                        "v_rmse_ms": float(per_time_rmse[time_index, 2])})
                validation_loss = float(model.loss(predicted[ntrain:], targets[ntrain:], normalization))
                relative = np.abs(np.asarray(parameters)[active] / np.asarray(truth)[active] - 1)
                passed = bool(all(c["pass"] for c in checks) and
                              np.all(relative <= config["parameter_relative_tolerance"]) and
                              best["loss"] < float(initial_loss) and validation_loss < initial_validation)
                item = {**label, **result, "method": method, "parameters": np.asarray(parameters).tolist(),
                        "initial_parameters": np.asarray(expand(start)).tolist(),
                        "initial_loss": float(initial_loss), "initial_validation_loss": initial_validation,
                        "validation_loss": validation_loss, "parameter_relative_error": relative.tolist(),
                        "gradient_checks": checks, "acceptance_pass": passed,
                        "forward_preparation": forward_cost, "gradient_preparation": gradient_cost,
                        "finite_difference_forward_calls": 4 * len(active),
                        "validation_uses_training_normalization": True}
                results.append(item)
                recovery_rows.append({"active": str(active), "start_id": start_id,
                    "method": method, "evaluations": result["evaluations"],
                    "initial_loss": float(initial_loss), "final_loss": best["loss"],
                    "initial_validation_loss": initial_validation, "validation_loss": validation_loss,
                    "max_parameter_relative_error": float(relative.max()),
                    "parameters": np.asarray(parameters).tolist(), "acceptance_pass": passed,
                    "optimization_wall_seconds": result["wall_seconds"]})
                for row in result["trace"]:
                    flat_rows.append({"active": str(active), "start_id": start_id, **row})
                dump(output / "recovery.json", {"status": "in_progress", "results": results})
    report = {"status": "complete", "scientific_acceptance_pass": all(r["acceptance_pass"] for r in results),
              "teacher_preparation": teacher_cost, "results": results,
              "limitations": ["Noise-free identical-model recovery; not external validation.",
                              "Validation is later in the same trajectory, not an independent Mars year.",
                              "Inactive parameters are fixed to known truth in single-parameter experiments.",
                              "Parameter recovery can fail despite low trajectory error (weak identifiability)."]}
    dump(output / "recovery.json", report)
    write_csv(output / "optimization-traces.csv", flat_rows)
    write_csv(output / "trajectory-errors.csv", trajectory_rows)
    write_csv(output / "parameter-recovery.csv", recovery_rows)
    plot_recovery(output, results)
    return report


def plot_recovery(output, results):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for item in results:
        trace = item["trace"]
        label = f"{item['method']} {item['active']} start {item['start_id']}"
        loss = np.maximum(np.minimum.accumulate([r["loss"] for r in trace]), 1e-20)
        axes[0].semilogy([r["evaluation"] for r in trace], loss, label=label)
        axes[1].semilogy([r["cumulative_wall_seconds"] for r in trace], loss, label=label)
    axes[0].set_xlabel("Objective calls (different cost per method)")
    axes[1].set_xlabel("Optimization wall seconds, excluding preparation")
    for ax in axes:
        ax.set_ylabel("Best normalized training loss")
    axes[1].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(output / "recovery.png", dpi=160)
    plt.close(fig)


def ablation_cases(truth, dt):
    cases = [{"name": "full", "parameters": list(truth), "dt": dt}]
    for name, setting in (("no_pbl", "pbl_diffusion_enabled"),
                          ("no_convection", "convective_adjustment_enabled"),
                          ("no_regolith", "regolith_enabled")):
        cases.append({"name": name, "parameters": list(truth), "dt": dt, "overrides": {setting: False}})
    cases.extend([
        {"name": "no_co2_exchange", "parameters": list(truth), "dt": dt, "co2_exchange": False},
        {"name": "weaker_diffusion", "parameters": list(truth), "dt": dt, "diffusion_sols": 0.2},
        {"name": "half_timestep", "parameters": list(truth), "dt": dt / 2},
    ])
    for i, name in enumerate(d.PARAMETER_NAMES):
        for factor in (0.8, 1.2):
            parameters = list(truth)
            parameters[i] *= factor
            cases.append({"name": f"{name}_{factor:g}", "parameters": parameters, "dt": dt})
    return cases


def ablations(model, config, output, emit, selected=None):
    times = config["train_seconds"] + config["validation_seconds"]
    initial_diagnostics = model.diagnostics(model.initial)
    rows, records = [], []
    baseline = None
    cases = ablation_cases(config["truth"], config["dt_seconds"])
    if selected:
        unknown = set(selected) - {c["name"] for c in cases}
        if unknown:
            raise ValueError(f"Unknown ablation cases: {sorted(unknown)}")
        cases = [c for c in cases if c["name"] == "full" or c["name"] in selected]
    for case in cases:
        name = case["name"]
        kwargs = {k: case[k] for k in ("overrides", "co2_exchange", "diffusion_sols") if k in case}
        emit({"phase": "ablation_start", "case": case})
        run = model.trajectory(times, case["dt"], states=True, **kwargs)
        executable, cost = compile_function(run, d.jnp.asarray(case["parameters"]))
        started = time.perf_counter()
        history = block(executable(d.jnp.asarray(case["parameters"])))
        elapsed = time.perf_counter() - started
        try:
            states = [d.jax.tree_util.tree_map(lambda x: x[i], history) for i in range(len(times))]
            diagnostics = [model.diagnostics(state) for state in states]
            observations = np.stack([np.asarray(model.observe(state)) for state in states])
            if baseline is None:
                baseline = observations
            error = observations - baseline
            rmse = np.sqrt(np.mean(np.sum(error**2 * np.asarray(model.weights), axis=(-2, -1)), axis=(0, 2)))
            drift = max(abs(r["co2_mass_kg"] / initial_diagnostics["co2_mass_kg"] - 1) for r in diagnostics)
            final = states[-1]
            save_restart(final, output / f"{name}-restart.npz")
            datasets = [state_to_comparison_dataset(s, model.coords, model.specs, d.MARS_BODY_3D, model.forcing) for s in states]
            d.xr.concat(datasets, dim="time").assign_attrs(
                experiment=name, comparison="paired sensitivity, not independent accuracy",
                parameters=json.dumps(case["parameters"])).to_netcdf(output / f"{name}.nc")
            row = {"case": name, "status": "finite", "temperature_rmse_vs_full_k": float(rmse[0]),
                   "u_rmse_vs_full_ms": float(rmse[1]), "v_rmse_vs_full_ms": float(rmse[2]),
                   "max_relative_co2_drift": drift,
                   "max_wind_ms": max(r["max_wind_ms"] for r in diagnostics),
                   "energy_change_j_m2_sr": diagnostics[-1]["atmospheric_energy_excluding_surface_geopotential_j_m2_sr"] - initial_diagnostics["atmospheric_energy_excluding_surface_geopotential_j_m2_sr"],
                   "aam_change_kg_m_s_sr": diagnostics[-1]["atmospheric_aam_kg_m_s_sr"] - initial_diagnostics["atmospheric_aam_kg_m_s_sr"],
                   "wall_seconds": elapsed, "compile_seconds": cost["compile_seconds"]}
            records.append({"case": case, "status": "finite", "seconds": times,
                            "diagnostics": diagnostics, "preparation": cost})
        except FloatingPointError as exc:
            if name == "full":
                raise
            row = {k: None for k in rows[0]}
            row.update(case=name, status="nonfinite", wall_seconds=elapsed, compile_seconds=cost["compile_seconds"])
            records.append({"case": case, "status": "nonfinite", "error": str(exc)})
        rows.append(row)
        emit({"phase": "ablation_complete", **row})
        write_csv(output / "ablations.csv", rows)
        dump(output / "ablations.json", {"status": "in_progress", "records": records})
    report = {"status": "complete", "initial_diagnostics": initial_diagnostics, "records": records,
              "limitations": ["Paired transient sensitivities from identical initial state, not equilibrated accuracy.",
                              "Energy and angular momentum changes are diagnostics, not closed forced budgets.",
                              "Only IMEX-RK-SIL3 is exposed; half-timestep is not an alternative-integrator test.",
                              "The no-CO2 case removes paired mass/latent tendencies and freezes frost; global mass projection remains.",
                              "The no-regolith case removes both surface and soil conductive tendencies."]}
    dump(output / "ablations.json", report)
    return report


def external_timings(path):
    if path is None:
        return [{"simulator": "NASA Ames", "status": "not_measured"},
                {"simulator": "LMD/MCD parent simulator", "status": "not_measured"}]
    rows = json.loads(path.read_text())
    if not isinstance(rows, list) or not rows:
        raise ValueError("External timings must be a nonempty JSON list")
    for row in rows:
        for key in ("simulator", "source", "hardware", "resolution", "physics", "precision", "timed_region"):
            if not isinstance(row.get(key), str) or not row[key].strip():
                raise ValueError(f"External timing requires {key}")
        for key in ("simulated_seconds", "wall_seconds", "dt_seconds"):
            if not isinstance(row.get(key), (int, float)) or not math.isfinite(row[key]) or row[key] <= 0:
                raise ValueError(f"External timing requires positive {key}")
        row["simulated_seconds_per_wall_second"] = row["simulated_seconds"] / row["wall_seconds"]
        row["status"] = "user_supplied_measurement_or_citation"
    return rows


def benchmark(model, config, output, external):
    parameters = d.jnp.asarray(config["truth"])
    run = model.trajectory(config["train_seconds"], config["dt_seconds"])
    forward, forward_cost = compile_function(run, parameters)
    target = block(forward(parameters))
    normalization = d.jnp.maximum(d.jnp.std(target, axis=(0, 2, 3, 4)), 1.0)

    def objective_aux(p):
        value = model.loss(run(p), target, normalization)
        return value, value

    tangent = d.jax.jacfwd(objective_aux, has_aux=True)
    gradient, gradient_cost = compile_function(tangent, parameters)
    records = []
    for name, function, cost in (("forward_trajectory", forward, forward_cost),
                                  ("loss_and_three_forward_tangents", gradient, gradient_cost)):
        samples = []
        for _ in range(config["benchmark_repeats"]):
            started = time.perf_counter()
            block(function(parameters))
            samples.append(time.perf_counter() - started)
        median = float(np.median(samples))
        memory = function.memory_analysis()
        records.append({"workload": name, **cost, "execution_seconds": samples,
                        "median_seconds": median,
                        "simulated_sols_per_wall_hour": config["train_seconds"][-1] / d.MARS_BODY_3D.rotation_period_s / median * 3600,
                        "compiled_temporary_bytes": getattr(memory, "temp_size_in_bytes", None),
                        "peak_device_memory_bytes": None})
    report = {"status": "complete", "measurements": records, "external": external,
              "limitations": ["No cross-simulator speedup is inferred from unmatched external rows.",
                              "Compilation time may benefit from a persistent JAX compilation cache.",
                              "Compiled temporary memory is not measured peak accelerator memory.",
                              "Execution includes sampled field decoding; excludes input IO, compilation and warmup."]}
    dump(output / "timing.json", report)
    write_csv(output / "timing.csv", records)
    return report


def environment():
    def git(*args):
        try:
            return subprocess.check_output(["git", *args], text=True, stderr=subprocess.DEVNULL).strip()
        except (OSError, subprocess.CalledProcessError):
            return "unavailable"
    return {"python": sys.version, "platform": platform.platform(), "processor": platform.processor(),
            "cpu_count": os.cpu_count(), "backend": d.jax.default_backend(),
            "devices": [str(device) for device in d.jax.devices()],
            "packages": {name: importlib.metadata.version(name) for name in ("jax", "jaxlib", "dinosaur", "numpy", "scipy")},
            "git_commit": git("rev-parse", "HEAD"), "git_status": git("status", "--short"),
            "xla_flags": os.environ.get("XLA_FLAGS"), "jax_enable_x64": bool(d.jax.config.jax_enable_x64)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=("all", "recovery", "ablations", "benchmark"), default="all")
    parser.add_argument("--cases", nargs="+", help="Explicit subset of ablations (full is always included)")
    parser.add_argument("--reference-config", type=Path, help="Existing tform comparison config; cached inputs only")
    parser.add_argument("--external-timings", type=Path)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    args = parser.parse_args(argv)
    config = validate_config(json.loads(args.config.read_text()))
    if args.cases:
        unknown = set(args.cases) - {c["name"] for c in ablation_cases(config["truth"], config["dt_seconds"])}
        if unknown:
            parser.error(f"Unknown ablation cases: {sorted(unknown)}")
    manifest = validate_inputs(args.inputs)
    external = external_timings(args.external_timings)
    if args.validate_only:
        print(json.dumps({"status": "inputs_valid", "config": config, "external": external}))
        return 0
    os.environ["MOLA_PATH"] = str((args.inputs / "mola.img").resolve())
    d.jax.config.update("jax_enable_x64", True)
    if args.require_gpu and d.jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    args.output.mkdir(parents=True, exist_ok=False)
    os.environ.setdefault("MPLCONFIGDIR", str((args.output / "matplotlib-cache").resolve()))
    dump(args.output / "config.json", config)
    dump(args.output / "input-manifest.json", manifest)
    dump(args.output / "environment.json", environment())
    # Source hashes capture uncommitted implementation, which commit SHA alone cannot.
    from src.framework.physics import gcm as physics_module
    from src.framework.gcm import dynamics as dynamics_module
    dump(args.output / "source-hashes.json", {
        str(Path(module.__file__).resolve()): sha256(module.__file__)
        for module in (sys.modules[__name__], d, physics_module, dynamics_module)
    })
    dump(args.output / "invocation.json", vars(args) | {
        k: str(v) for k, v in vars(args).items() if isinstance(v, Path)})

    def emit(row):
        line = json.dumps(row, allow_nan=False)
        print(line, flush=True)
        with (args.output / "progress.jsonl").open("a") as stream:
            stream.write(line + "\n")

    results = {}
    try:
        emit({"phase": "initialize", "profile": config["profile"]})
        model = Model(args.inputs)
        for task in ("recovery", "ablations", "benchmark"):
            if args.task not in ("all", task):
                continue
            emit({"phase": task})
            if task == "recovery":
                results[task] = recovery(model, config, args.output, emit)
            elif task == "ablations":
                results[task] = ablations(model, config, args.output, emit, args.cases)
            else:
                results[task] = benchmark(model, config, args.output, external)
            # Separate tasks need not retain all compiled optimization closures.
            d.jax.clear_caches()
        if args.reference_config:
            from cli.comparison import build_report
            build_report(args.reference_config, args.output / "reference-comparison")
            results["references"] = {"status": "diagnostic_comparison_complete",
                                     "config": str(args.reference_config), "sha256": sha256(args.reference_config),
                                     "scope": "Sources explicitly named in comparison config; not automatically calibrated outputs"}
        peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        report = {"execution_status": "complete", "profile": config["profile"],
                  "model": {"truncation": "T21", "layers": 12, "integrator": "IMEX-RK-SIL3",
                            "pbl_closure_interval_seconds": model.forcing.pbl_implicit_timestep_s,
                            "co2_supply_limiter_interval_seconds": d.co2_forcing(energy_limited=True).exchange_timestep_s},
                  "tasks": {name: result.get("status") for name, result in results.items()},
                  "recovery_acceptance_pass": results.get("recovery", {}).get("scientific_acceptance_pass"),
                  "external_timings_supplied": bool(args.external_timings),
                  "process_peak_rss_bytes": peak_rss if sys.platform == "darwin" else peak_rss * 1024,
                  "process_peak_rss_scope": "whole process including compilation; not accelerator peak",
                  "artifacts": {str(p.relative_to(args.output)): sha256(p) for p in args.output.rglob("*") if p.is_file()}}
        dump(args.output / "report.json", report)
        (args.output / "summary.md").write_text(
            "# Mars paper experiment\n\n"
            f"Profile: `{config['profile']}`. Execution completed.\n\n"
            f"Recovery acceptance: `{report['recovery_acceptance_pass']}` (null means not run).\n\n"
            "See recovery.json/recovery.png, ablations.csv and timing.json for requested tasks.\n\n"
            "Synthetic continuation is not independent-year or observational validation. "
            "Ablation energy changes are not budget closure. No external speedup is claimed.\n")
        print(json.dumps(report, indent=2))
        return 10 if report["recovery_acceptance_pass"] is False else 0
    except Exception as exc:
        dump(args.output / "failure.json", {"execution_status": "error", "error": str(exc),
                                            "completed_tasks": list(results)})
        raise


if __name__ == "__main__":
    raise SystemExit(main())
