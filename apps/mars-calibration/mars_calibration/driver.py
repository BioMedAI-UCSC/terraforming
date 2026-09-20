#!/usr/bin/env python3
"""Calibrate three interpretable Mars-GCM parameters through a coupled rollout.

This script contains no learned component.  It fits CO2 longwave opacity, dust
longwave opacity, and bulk surface exchange against a staged ARCO-MACDA state,
then runs an equal-budget bounded Powell baseline.  The initial state, target,
parameter bounds, optimizer budget, and every gradient check are recorded.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import functools
import importlib.metadata
import sys
import json
import os
import time
from pathlib import Path

os.environ.setdefault(
    "XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2"
)
os.environ.setdefault(
    "JAX_COMPILATION_CACHE_DIR", str(Path("outputs/.jax_compilation_cache").resolve())
)

import numpy as np
import scipy.optimize
import xarray as xr

from src.celestials.planets.mars import MARS_BODY_3D
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import forcing_with_surface_properties
from src.celestials.planets.mars.topography import mola_modal_orography, regrid_to_nodal
from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic, time_integration
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import integrate, reference_temperature, stepper
from src.framework.gcm.restart import load_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics.gcm import (
    forced_co2_primitive_equations,
    mean_anomaly_for_ls,
    positivity_preserving_co2_step,
)
@functools.lru_cache(maxsize=2)
def _load_ames_dust(path: Path):
    import xarray as xr

    with xr.open_dataset(path) as opened:
        return opened[["dust_visible_optical_depth", "dust_longwave_optical_depth"]].load()


def _ames_dust_climatology(path: Path, grid):
    """Regrid the complete Ames seasonal dust table to Dinosaur nodal points."""
    import xarray as xr

    dust = _load_ames_dust(path)
    periodic = xr.concat(
        [dust.isel(lon=-1).assign_coords(lon=float(dust.lon[-1]) - 360.0),
         dust,
         dust.isel(lon=0).assign_coords(lon=float(dust.lon[0]) + 360.0)],
        dim="lon",
    )
    target = periodic.interp(
        lon=np.mod(np.degrees(np.asarray(grid.longitudes)), 360.0),
        lat=np.degrees(np.asarray(grid.latitudes)),
    )
    return (
        np.asarray(target.ls),
        np.asarray(target.dust_visible_optical_depth.transpose("ls", "lon", "lat")),
        np.asarray(target.dust_longwave_optical_depth.transpose("ls", "lon", "lat")),
    )




PARAMETER_NAMES = (
    "co2_longwave_opacity_scale",
    "dust_longwave_opacity_scale",
    "surface_exchange_multiplier",
)
LOWER_BOUNDS = np.asarray([0.05, 0.05, 0.25], dtype=np.float64)
UPPER_BOUNDS = np.asarray([2.0, 2.0, 4.0], dtype=np.float64)


class EvaluationBudgetReached(RuntimeError):
    """Internal optimizer stop used to enforce an exact forward-call budget."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def raw_to_physical(raw):
    """Map unconstrained optimizer variables into fixed positive bounds."""
    lower = jnp.asarray(LOWER_BOUNDS)
    upper = jnp.asarray(UPPER_BOUNDS)
    return lower + (upper - lower) * jax.nn.sigmoid(jnp.asarray(raw))


def physical_to_raw(values):
    """Numerically stable inverse of :func:`raw_to_physical`."""
    values = np.asarray(values, dtype=np.float64)
    fraction = (values - LOWER_BOUNDS) / (UPPER_BOUNDS - LOWER_BOUNDS)
    fraction = np.clip(fraction, 1e-12, 1.0 - 1e-12)
    return np.log(fraction / (1.0 - fraction))


def _forcing(surface: Path, dust: Path, grid, parameters):
    base = radiative_forcing(
        diurnal=True,
        co2_radiation_enabled=True,
        dust_visible_optical_depth=0.3,
        dust_longwave_optical_depth=0.1,
    )
    base = dataclasses.replace(
        base, init_orbital_angle_rad=mean_anomaly_for_ls(0.0, base)
    )
    base = forcing_with_surface_properties(base, grid, surface)
    dust_ls, visible, longwave = _ames_dust_climatology(dust, grid)
    return dataclasses.replace(
        base,
        regolith_enabled=True,
        stability_exchange_enabled=True,
        pbl_diffusion_enabled=True,
        convective_adjustment_enabled=True,
        ames_co2_longwave_opacity_scale=parameters[0],
        ames_dust_longwave_opacity_scale=parameters[1],
        surface_exchange_multiplier=parameters[2],
        dust_climatology_ls_deg=jnp.asarray(dust_ls),
        dust_visible_climatology=jnp.asarray(visible),
        dust_longwave_climatology=jnp.asarray(longwave),
    )


def _advance(initial, coords, specs, forcing, steps: int, dt_seconds: float):
    elevation = regrid_to_nodal(coords)
    orography = mola_modal_orography(coords, specs, elevation_nodal_m=elevation)
    equation = forced_co2_primitive_equations(
        coords,
        MARS_BODY_3D,
        forcing,
        co2_forcing(energy_limited=True),
        specs=specs,
        orography=orography,
    )
    advance = stepper(equation, dt_seconds, specs)
    dt_nd = float(specs.nondimensionalize(dt_seconds * scales.units.second))
    tau_nd = float(specs.nondimensionalize(
        0.1 * MARS_BODY_3D.rotation_period_s * scales.units.second
    ))
    dynamics_filter = time_integration.horizontal_diffusion_step_filter(
        coords.horizontal, dt_nd, tau_nd, order=4
    )

    def diffusion(previous, following):
        return following._replace(
            dynamics=dynamics_filter(previous.dynamics, following.dynamics)
        )

    advance = time_integration.step_with_filters(advance, [diffusion])
    advance = positivity_preserving_co2_step(
        advance,
        coords,
        specs,
        body=MARS_BODY_3D,
        co2_forcing=co2_forcing(energy_limited=True),
        dt_seconds=dt_seconds,
    )
    # Reverse-mode differentiation through a multi-sol scan otherwise retains
    # every timestep's intermediates.  Rematerializing the step keeps memory
    # bounded if this driver is used with reverse mode.
    advance = jax.checkpoint(advance)
    return integrate(advance, initial, steps)


def _target(path: Path, index: int):
    with xr.open_dataset(path, decode_times=False) as opened:
        data = opened.isel(time=index).load()
    return {
        "temperature": jnp.asarray(np.asarray(data.temp).transpose(0, 2, 1)),
        "eastward_wind": jnp.asarray(np.asarray(data.uwind).transpose(0, 2, 1)),
        "northward_wind": jnp.asarray(np.asarray(data.vwind).transpose(0, 2, 1)),
        "ls_deg": float(data.Ls),
        "mars_year": float(data.MY_Ls),
        "time_sol": float(data.time),
    }


def _objective_factory(initial, coords, specs, surface, dust, target, steps, dt_seconds):
    weights = jnp.asarray(coords.horizontal.quadrature_weights)
    weights = weights / jnp.sum(weights)
    reference = jnp.asarray(reference_temperature(coords, MARS_BODY_3D)).reshape(-1, 1, 1)
    temperature_scale = float(specs.dimensionalize(1.0, scales.units.kelvin).magnitude)
    velocity_scale = float(
        specs.dimensionalize(1.0, scales.units.meter / scales.units.second).magnitude
    )
    target_t = target["temperature"]
    target_u = target["eastward_wind"]
    target_v = target["northward_wind"]
    t_scale = jnp.maximum(jnp.std(target_t), 1.0)
    u_scale = jnp.maximum(jnp.std(target_u), 1.0)
    v_scale = jnp.maximum(jnp.std(target_v), 1.0)

    def weighted_mse(error):
        return jnp.sum(error**2 * weights[None, ...]) / error.shape[0]

    def objective(raw):
        parameters = raw_to_physical(raw)
        forcing = _forcing(surface, dust, coords.horizontal, parameters)
        final = _advance(initial, coords, specs, forcing, steps, dt_seconds)
        temperature = (
            coords.horizontal.to_nodal(final.dynamics.temperature_variation) + reference
        ) * temperature_scale
        u_nd, v_nd = spherical_harmonic.vor_div_to_uv_nodal(
            coords.horizontal,
            final.dynamics.vorticity,
            final.dynamics.divergence,
        )
        return (
            weighted_mse((temperature - target_t) / t_scale)
            + weighted_mse((u_nd * velocity_scale - target_u) / u_scale)
            + weighted_mse((v_nd * velocity_scale - target_v) / v_scale)
        ) / 3.0

    return objective


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("restart", type=Path)
    parser.add_argument("target", type=Path)
    parser.add_argument("surface_properties", type=Path)
    parser.add_argument("ames_dust_reference", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-index", type=int, default=-1)
    parser.add_argument("--rollout-sols", type=float, default=0.25)
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--max-evaluations", type=int, default=20)
    parser.add_argument("--initial", default="0.25,0.25,1.0")
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if args.rollout_sols <= 0 or args.dt <= 0 or args.max_evaluations < 1:
        parser.error("rollout, timestep, and evaluation budget must be positive")
    initial_values = np.asarray([float(item) for item in args.initial.split(",")])
    if initial_values.shape != (3,):
        parser.error("--initial requires exactly three comma-separated values")
    if not np.all(np.isfinite(initial_values)):
        parser.error("--initial values must be finite")
    if np.any((initial_values <= LOWER_BOUNDS) | (initial_values >= UPPER_BOUNDS)):
        parser.error("--initial values must lie strictly inside the fixed bounds")

    jax.config.update("jax_enable_x64", True)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing a CPU fallback")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    progress_path = args.output.with_suffix(".jsonl")
    def log_record(record):
        line = json.dumps(record, allow_nan=False)
        with progress_path.open("a") as stream:
            stream.write(line + "\n")
        print(line, flush=True)

    coords = coordinate_system("T21", 12)
    specs = physics_specs(MARS_BODY_3D)
    initial = load_restart(args.restart)
    target = _target(args.target, args.target_index)
    steps = round(args.rollout_sols * MARS_BODY_3D.rotation_period_s / args.dt)
    if steps < 1:
        parser.error("rollout must contain at least one physical step")
    for leaf in jax.tree_util.tree_leaves(initial):
        if not np.all(np.isfinite(leaf)):
            raise ValueError("Restart contains non-finite values")
    for name in ("temperature", "eastward_wind", "northward_wind"):
        if target[name].shape != (12, 64, 32) or not np.all(np.isfinite(target[name])):
            raise ValueError(f"Invalid T21/L12 target: {name}")
    objective = _objective_factory(
        initial, coords, specs, args.surface_properties,
        args.ames_dust_reference, target, steps, args.dt,
    )
    # There are only three calibrated scalars. Forward mode propagates those
    # three tangents with bounded memory and is substantially cheaper here than
    # storing or rematerializing a multi-sol reverse-mode trajectory.
    def objective_with_aux(raw):
        value = objective(raw)
        return value, value

    tangent = jax.jacfwd(objective_with_aux, has_aux=True)
    @jax.jit
    def value_and_grad(raw):
        gradient, value = tangent(raw)
        return value, gradient
    forward = jax.jit(objective)
    raw_initial = physical_to_raw(initial_values)
    trace = []

    def gradient_call(raw):
        if len(trace) >= args.max_evaluations:
            raise EvaluationBudgetReached
        started = time.perf_counter()
        value, gradient = value_and_grad(jnp.asarray(raw))
        record = {
            "method": "L-BFGS-B",
            "evaluation": len([x for x in trace if x["method"] == "L-BFGS-B"]) + 1,
            "loss": float(value),
            "parameters": dict(zip(PARAMETER_NAMES, map(float, raw_to_physical(raw)), strict=True)),
            "gradient_raw": list(map(float, gradient)),
            "wall_seconds": time.perf_counter() - started,
        }
        log_record(record)
        trace.append(record)
        return float(value), np.asarray(gradient, dtype=np.float64)

    lbfgs_start = raw_initial
    lbfgs_restarts = 0
    while len(trace) < args.max_evaluations:
        try:
            lbfgs_result = scipy.optimize.minimize(
                gradient_call,
                lbfgs_start,
                jac=True,
                method="L-BFGS-B",
                options={
                    "maxfun": args.max_evaluations,
                    "maxiter": args.max_evaluations,
                    "ftol": 0.0,
                    "gtol": 0.0,
                    "maxls": args.max_evaluations,
                },
            )
        except EvaluationBudgetReached:
            lbfgs_result = None
        if len(trace) < args.max_evaluations:
            lbfgs_restarts += 1
            best = min(trace, key=lambda record: record["loss"])
            lbfgs_start = physical_to_raw([best["parameters"][name] for name in PARAMETER_NAMES])

    powell_trace = []

    def powell_call(values):
        if len(powell_trace) >= args.max_evaluations:
            raise EvaluationBudgetReached
        started = time.perf_counter()
        raw = physical_to_raw(values)
        value = float(forward(jnp.asarray(raw)))
        record = {
            "method": "Powell",
            "evaluation": len(powell_trace) + 1,
            "loss": value,
            "parameters": dict(zip(PARAMETER_NAMES, map(float, values), strict=True)),
            "wall_seconds": time.perf_counter() - started,
        }
        log_record(record)
        powell_trace.append(record)
        return value

    powell_start = initial_values
    powell_restarts = 0
    while len(powell_trace) < args.max_evaluations:
        try:
            powell_result = scipy.optimize.minimize(
                powell_call,
                powell_start,
                method="Powell",
                bounds=list(zip(LOWER_BOUNDS, UPPER_BOUNDS, strict=True)),
                options={
                    "maxfev": args.max_evaluations,
                    "maxiter": args.max_evaluations,
                    "ftol": 0.0,
                    "xtol": 0.0,
                },
            )
        except EvaluationBudgetReached:
            powell_result = None
        if len(powell_trace) < args.max_evaluations:
            powell_restarts += 1
            best = min(powell_trace, key=lambda record: record["loss"])
            powell_start = np.asarray([best["parameters"][name] for name in PARAMETER_NAMES])

    gradient_checks = []
    autodiff = np.asarray(trace[0]["gradient_raw"])
    for epsilon in (1e-3, 1e-4):
        finite_difference = []
        for index in range(3):
            delta = np.zeros(3)
            delta[index] = epsilon
            finite_difference.append(float(
                (forward(raw_initial + delta) - forward(raw_initial - delta))
                / (2.0 * epsilon)
            ))
        finite_difference = np.asarray(finite_difference)
        relative = np.abs(np.asarray(autodiff) - finite_difference) / np.maximum(
            np.abs(finite_difference), 1e-12
        )
        gradient_checks.append({
            "epsilon": epsilon,
            "finite_difference_raw": finite_difference.tolist(),
            "relative_error": relative.tolist(),
            "pass": bool(np.all(relative < 1e-4)),
        })

    best_lbfgs = min(trace, key=lambda item: item["loss"])
    best_powell = min(powell_trace, key=lambda item: item["loss"])
    calibration_passed = (
        best_lbfgs["loss"] < trace[0]["loss"]
        and best_lbfgs["loss"] <= best_powell["loss"]
    )
    budget_passed = len(trace) == len(powell_trace) == args.max_evaluations
    report = {
        "claim_scope": "short-horizon coupled calibration; not equilibrium tuning",
        "budget": {"per_method": args.max_evaluations, "exact": budget_passed,
                   "finite_difference_forward_calls": 12,
                   "lbfgs_restarts": lbfgs_restarts, "powell_restarts": powell_restarts,
                   "unit": "objective oracle calls, not equal FLOPs or wall time"},
        "runtime": {"python": sys.version, "jax": jax.__version__,
                    "backend": jax.default_backend(), "devices": [str(d) for d in jax.devices()],
                    "packages": {name: importlib.metadata.version(name) for name in
                                 ("terraforming", "dinosaur", "jaxlib", "numpy", "scipy")}},
        "surface_properties": {"path": str(args.surface_properties), "sha256": _sha256(args.surface_properties)},
        "dust_reference": {"path": str(args.ames_dust_reference), "sha256": _sha256(args.ames_dust_reference)},
        "actual_rollout_sols": steps * args.dt / MARS_BODY_3D.rotation_period_s,
        "status": "pass" if (
            all(item["pass"] for item in gradient_checks) and calibration_passed and budget_passed
        ) else "fail",
        "model": "deterministic differentiable Mars GCM; no neural component",
        "rollout_sols": args.rollout_sols,
        "steps": steps,
        "dt_seconds": args.dt,
        "target": {
            "path": str(args.target), "sha256": _sha256(args.target),
            "index": args.target_index, "mars_year": target["mars_year"],
            "ls_deg": target["ls_deg"], "time_sol": target["time_sol"],
        },
        "restart": {"path": str(args.restart), "sha256": _sha256(args.restart)},
        "parameter_names": PARAMETER_NAMES,
        "bounds": dict(zip(PARAMETER_NAMES, zip(LOWER_BOUNDS.tolist(), UPPER_BOUNDS.tolist(), strict=True), strict=True)),
        "initial_parameters": dict(zip(PARAMETER_NAMES, initial_values.tolist(), strict=True)),
        "autodiff_initial_gradient_raw": list(map(float, autodiff)),
        "gradient_checks": gradient_checks,
        "lbfgs": {
            "success": calibration_passed,
            "message": (
                str(lbfgs_result.message) if lbfgs_result is not None
                else "exact evaluation budget reached"
            ),
            "evaluations": len(trace), "loss": best_lbfgs["loss"],
            "parameters": best_lbfgs["parameters"],
        },
        "powell": {
            "success": bool(powell_result.success) if powell_result is not None else False,
            "message": (
                str(powell_result.message) if powell_result is not None
                else "exact evaluation budget reached"
            ),
            "evaluations": len(powell_trace), "loss": best_powell["loss"],
            "parameters": best_powell["parameters"],
        },
        "trace": trace + powell_trace,
        "limitations": [
            "This single-window fit does not establish held-out or four-season calibration performance.",
            "The instantaneous endpoint is compared to a staged daily mean; temporal averaging differs.",
            "The target is a reanalysis state, not direct observational truth.",
            "A short initialized rollout tests parameter calibration, not equilibrium climate sensitivity.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps(report, indent=2, allow_nan=False))
    # Distinguish a completed scientific rejection from an execution failure.
    # The Kubernetes Job treats 10 as final and may retry other failures.
    return 0 if report["status"] == "pass" else 10


if __name__ == "__main__":
    raise SystemExit(main())
