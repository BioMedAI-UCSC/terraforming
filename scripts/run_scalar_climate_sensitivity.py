#!/usr/bin/env python3
"""Differentiate global Mars climate diagnostics with respect to one parameter.

This is a short-horizon tangent experiment, separate from the annual stability
run. It writes the diagnostic values, exact JAX forward-mode derivatives, a
centered finite-difference check, and a publication-ready figure.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "package"))
sys.path.insert(0, str(ROOT / "apps" / "mars-calibration"))
from mars_calibration import driver as d  # noqa: E402
from mars_calibration import paper as p  # noqa: E402

PARAMETERS = {
    "co2_longwave_opacity_scale": 0,
    "dust_longwave_opacity_scale": 1,
    "surface_exchange_multiplier": 2,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True,
                        help="prepared native inputs containing restart.npz, surface.nc, dust.nc, mola.img")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parameter", choices=PARAMETERS, default="co2_longwave_opacity_scale")
    parser.add_argument("--value", type=float, default=1.0)
    parser.add_argument("--sols", type=float, nargs="+", default=[0.25, 1.0, 5.0])
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if (args.sols != sorted(set(args.sols)) or min(args.sols) <= 0 or args.dt <= 0
            or args.dt > 300 or args.epsilon <= 0):
        parser.error("sols must be positive, sorted and unique; require 0 < dt <= 300 and epsilon > 0")
    index = PARAMETERS[args.parameter]
    if not d.LOWER_BOUNDS[index] < args.value < d.UPPER_BOUNDS[index]:
        parser.error("parameter value must lie strictly inside its calibration bounds")

    d.jax.config.update("jax_enable_x64", True)
    if args.require_gpu and d.jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    os.environ["MOLA_PATH"] = str((args.inputs / "mola.img").resolve())
    inputs_manifest = p.validate_inputs(args.inputs)
    model = p.Model(args.inputs)
    seconds = np.asarray(args.sols) * d.MARS_BODY_3D.rotation_period_s
    # Sampling must land on exact integration steps.
    steps = np.rint(seconds / args.dt).astype(int)
    seconds = steps * args.dt
    actual_sols = seconds / d.MARS_BODY_3D.rotation_period_s
    run = model.trajectory(seconds, args.dt, states=True)
    weights = model.weights
    pressure_scale = float(model.specs.dimensionalize(1.0, d.scales.units.pascal).magnitude)
    temperature_scale = float(model.specs.dimensionalize(1.0, d.scales.units.kelvin).magnitude)

    def diagnostics(theta):
        parameters = d.jnp.ones(3).at[index].set(theta)
        states = run(parameters)
        pressure = d.jnp.exp(model.coords.horizontal.to_nodal(
            states.dynamics.log_surface_pressure
        )[:, 0]) * pressure_scale
        surface_temperature = states.surface_temperature[:, 0] * temperature_scale
        co2_ice = states.co2_ice[:, 0] * pressure_scale
        mean = lambda values: d.jnp.sum(values * weights, axis=(-2, -1))
        return d.jnp.stack([
            mean(surface_temperature), mean(pressure), mean(co2_ice)
        ], axis=1)

    started = time.perf_counter()
    value_and_tangent = d.jax.jit(lambda x: (diagnostics(x), d.jax.jacfwd(diagnostics)(x)))
    values, derivatives = p.block(value_and_tangent(args.value))
    compile_and_first_seconds = time.perf_counter() - started
    started = time.perf_counter()
    values, derivatives = p.block(value_and_tangent(args.value))
    warm_seconds = time.perf_counter() - started
    forward = d.jax.jit(diagnostics)
    plus = np.asarray(p.block(forward(args.value + args.epsilon)))
    minus = np.asarray(p.block(forward(args.value - args.epsilon)))
    finite_difference = (plus - minus) / (2 * args.epsilon)
    values, derivatives = np.asarray(values), np.asarray(derivatives)
    error = np.abs(derivatives - finite_difference)
    relative = error / np.maximum(np.abs(finite_difference), 1e-12)

    names = ["mean_surface_temperature_k", "mean_surface_pressure_pa", "mean_co2_ice_pa"]
    labels = ["Mean surface temperature", "Mean surface pressure", "Mean surface CO₂ frost"]
    units = ["K", "Pa", "Pa-equivalent"]
    rows = []
    for i, sol in enumerate(actual_sols):
        for j, name in enumerate(names):
            rows.append({"requested_sols": args.sols[i], "actual_sols": float(sol),
                         "steps": int(steps[i]), "parameter": args.parameter,
                         "parameter_value": args.value, "diagnostic": name,
                         "units": units[j], "value": float(values[i, j]),
                         "autodiff_derivative": float(derivatives[i, j]),
                         "finite_difference_derivative": float(finite_difference[i, j]),
                         "relative_error": float(relative[i, j]),
                         "gradient_check_pass": bool(error[i, j] <= 1e-7 + 1e-3 * abs(finite_difference[i, j]))})
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with (args.output_dir / "sensitivity.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.5), constrained_layout=True)
    colors = ["#c54e20", "#2962a3", "#5c8d3b"]
    for j, (label, unit, color) in enumerate(zip(labels, units, colors, strict=True)):
        axes[0, j].plot(actual_sols, values[:, j], marker="o", color=color, lw=2)
        axes[0, j].set(title=label, xlabel="Rollout horizon (sols)", ylabel=unit)
        axes[1, j].plot(actual_sols, derivatives[:, j], marker="o", color=color, lw=2,
                        label="JAX autodiff")
        axes[1, j].plot(actual_sols, finite_difference[:, j], marker="x", color="black",
                        ls="none", label="Centered finite difference")
        axes[1, j].axhline(0, color="0.5", lw=.7)
        axes[1, j].set(title=f"Sensitivity of {label.lower()}", xlabel="Rollout horizon (sols)",
                       ylabel=f"d({unit}) / d(parameter unit)")
        axes[1, j].legend(frameon=False, fontsize=8)
        for ax in axes[:, j]: ax.grid(alpha=.2)
    fig.suptitle(f"AEGIS differentiable response to {args.parameter} at θ={args.value:g}")
    fig.savefig(args.output_dir / "sensitivity.png", dpi=220)
    fig.savefig(args.output_dir / "sensitivity.pdf")
    plt.close(fig)
    checks_pass = bool(all(row["gradient_check_pass"] for row in rows))
    report = {
        "status": "pass" if np.isfinite(values).all() and np.isfinite(derivatives).all() and checks_pass else "fail",
        "parameter": args.parameter, "parameter_value": args.value,
        "method": "jax.jacfwd through the complete coupled rollout",
        "finite_difference_epsilon": args.epsilon,
        "maximum_relative_error": float(np.max(relative)),
        "all_finite_difference_checks_pass": checks_pass,
        "compile_and_first_execution_seconds": compile_and_first_seconds,
        "warm_value_and_jacobian_seconds": warm_seconds,
        "backend": d.jax.default_backend(), "devices": [str(x) for x in d.jax.devices()],
        "inputs": inputs_manifest,
        "limitations": [
            "This is a local tangent sensitivity around one parameter value.",
            "Short rollouts test differentiability; they are not equilibrium climate responses.",
            "Piecewise CO2 condensation and convection can make derivatives horizon and state dependent.",
        ],
    }
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": report["status"], "output": str(args.output_dir)}))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
