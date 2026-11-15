#!/usr/bin/env python3
"""Frozen rebuttal protocols: annual projection test and spatial inverse design."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/mars-calibration"))
import numpy as np
from mars_calibration import driver as d
from mars_calibration.provenance import hardware, snapshot_sources
from src.framework.physics import gcm as physics
from src.framework.gcm.dynamics import stepper, reference_temperature
from src.framework.gcm.restart import save_restart, load_restart
from src.celestials.planets.mars.maps import initial_rest_state
from run_gcm3d_ablation import _true_anomaly


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False,
        default=lambda x: np.asarray(x).tolist()) + "\n")


def setup(inputs, output, layers=12):
    os.environ["MOLA_PATH"] = str((inputs / "mola.img").resolve())
    coords = d.coordinate_system("T21", layers)
    specs = d.physics_specs(d.MARS_BODY_3D)
    forcing = d._forcing(inputs / "surface.nc", inputs / "dust.nc", coords.horizontal, d.jnp.ones(3))
    elevation = d.regrid_to_nodal(coords)
    orography = d.mola_modal_orography(coords, specs, elevation_nodal_m=elevation)
    initial = physics.initial_column_state(dataclasses.replace(initial_rest_state(
        coords, specs, d.MARS_BODY_3D, elevation), sim_time=0.), coords,
        d.MARS_BODY_3D.reference_temperature_k, specs, forcing=forcing)
    output.mkdir(parents=True, exist_ok=False)
    snapshot_sources(ROOT, output)
    write(output / "environment.json", dict(hardware=hardware(d.jax.devices()), jax=d.jax.__version__,
        inputs={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs.iterdir() if p.is_file()},
        forcing=dataclasses.asdict(forcing), layers=layers, truncation="T21"))
    save_restart(initial, output / "initial.npz")
    return coords, specs, forcing, orography, initial, elevation


def climate(args):
    coords, specs, forcing, orography, state, elevation = setup(args.inputs, args.output)
    jax, jnp = d.jax, d.jnp
    grid, body = coords.horizontal, d.MARS_BODY_3D
    # An integer number of steps per solar day makes every daily mean complete.
    steps_per_sol = 300
    dt = body.solar_day_seconds / steps_per_sol
    cf = d.co2_forcing(energy_limited=True)
    raw = stepper(physics.forced_co2_primitive_equations(coords, body, forcing, cf,
        specs=specs, orography=orography), dt, specs, coords=coords,
        hyperdiffusion_tau_seconds=.1 * body.solar_day_seconds)
    advance = physics.positivity_preserving_co2_step(raw, coords, specs, body=body,
        co2_forcing=cf, dt_seconds=dt, return_diagnostics=True,
        inventory_projection=args.projection)
    pressure_scale = float(specs.dimensionalize(1., d.scales.units.pascal).magnitude)
    speed_scale = float(specs.dimensionalize(1., d.scales.units.meter / d.scales.units.second).magnitude)
    weights = jnp.asarray(grid.quadrature_weights)
    weights = weights / jnp.sum(weights)
    ref = jnp.asarray(reference_temperature(coords, body))[:, None, None]
    dx = body.radius_m * jnp.cos(jnp.asarray(grid.latitudes))[None] * (2 * np.pi / grid.nodal_shape[0])
    dy = body.radius_m * jnp.asarray(np.gradient(np.asarray(grid.latitudes)))[None]

    def observe(s):
        pressure = jnp.exp(grid.to_nodal(s.dynamics.log_surface_pressure))[0] * pressure_scale
        u, v = d.spherical_harmonic.vor_div_to_uv_nodal(grid, s.dynamics.vorticity, s.dynamics.divergence)
        u, v = u * speed_scale, v * speed_scale
        fields = jnp.stack([s.surface_temperature[0], pressure, u[-1], v[-1],
                           s.co2_ice[0] * pressure_scale,
                           (grid.to_nodal(s.dynamics.temperature_variation) + ref)[-1],
                           jnp.hypot(u[-1], v[-1])])
        cfl = jnp.max(dt * (jnp.abs(u) / dx + jnp.abs(v) / dy))
        mean_pressure = jnp.sum(pressure * weights)
        return fields, cfl, mean_pressure

    def tick(s, _):
        final, budget = advance(s)
        _, cfl, pressure = observe(final)
        return final, jnp.stack([budget.raw_total_mass_kg, budget.target_total_mass_kg,
            budget.corrected_total_mass_kg, budget.atmospheric_rescale_kg,
            budget.negative_frost_repair_kg, budget.pressure_factor, cfl, pressure])

    def sample(s, _):
        final, budgets = jax.lax.scan(tick, s, None, length=12)
        fields, _, _ = observe(final)
        return final, (fields, budgets)

    day = jax.jit(lambda s: jax.lax.scan(sample, s, None, length=25))
    sample_dir = args.output / "samples"
    sample_dir.mkdir()
    np.savez_compressed(args.output / "grid.npz", lat=np.degrees(grid.latitudes),
        lon=np.degrees(grid.longitudes), elevation_m=elevation, weights=weights)
    write(args.output / "protocol.json", dict(projection=args.projection, dt_seconds=dt,
        steps_per_sol=steps_per_sol, sample_interval_seconds=12 * dt,
        steps_per_sample=12, years=args.years, controls=[1., 1., 1.],
        co2_forcing=dataclasses.asdict(cf), diffusion_laplacian_power=4, diffusion_tau_sols=.1,
        field_order=["temperature", "surface_pressure", "u", "v", "co2_ice", "lowest_air_temperature", "wind_speed"],
        limitations="Cold-start transient; evaluation year is not certified equilibrated. Clouds are included in total mass. Disabling numerical inventory projection retains physical settling and saturation adjustment."))
    initial_mass = None
    max_drift, max_cfl, correction_sum, correction_abs_sum, correction_max = 0., 0., 0., 0., 0.
    count = 0
    days = int(np.ceil(args.years * forcing.orbital_period_s / body.solar_day_seconds))
    started = time.perf_counter()
    for sol in range(days):
        first, _, _ = observe(state)
        state, (fields, budgets) = day(state)
        fields, budgets = np.asarray(fields), np.asarray(budgets).reshape(-1, 8)
        if not np.isfinite(fields).all() or not np.isfinite(budgets).all():
            save_restart(state, args.output / "failed-state.npz")
            write(args.output / "report.json", dict(status="nonfinite", completed_sols=sol,
                requested_sols=days, projection=args.projection))
            raise FloatingPointError("Nonfinite climate state; saved outputs retained")
        if initial_mass is None:
            initial_mass = float(budgets[0, 1])
        correction = budgets[:, 3] / initial_mass
        correction_sum += float(correction.sum())
        correction_abs_sum += float(np.abs(correction).sum())
        correction_max = max(correction_max, float(np.abs(correction).max()))
        max_drift = max(max_drift, float(np.abs(budgets[:, 2] / initial_mass - 1).max()))
        max_cfl = max(max_cfl, float(budgets[:, 6].max()))
        count += len(budgets)
        seconds = (sol + np.arange(26) / 25) * body.solar_day_seconds
        ls = np.degrees(np.asarray(_true_anomaly(jnp.asarray(seconds), forcing)) + forcing.ls_perihelion_rad) % 360
        np.savez_compressed(sample_dir / f"sol-{sol:04d}.npz", seconds=seconds, ls=ls,
            fields=np.concatenate([np.asarray(first)[None], fields]), budgets=budgets)
        if (sol + 1) % 5 == 0 or sol == days - 1:
            save_restart(state, args.output / "latest.npz")
            report = dict(status="complete" if sol == days-1 else "running", completed_sols=sol+1,
                requested_sols=days, steps=count, initial_mass_kg=initial_mass,
                mean_absolute_correction_fraction=correction_abs_sum/count,
                max_absolute_correction_fraction=correction_max,
                cumulative_signed_correction_fraction=correction_sum,
                cumulative_absolute_correction_fraction=correction_abs_sum,
                max_relative_inventory_drift=max_drift, max_horizontal_courant=max_cfl,
                wall_seconds_including_compile=time.perf_counter()-started)
            write(args.output / "report.json", report)
            print(json.dumps(report), flush=True)


def spatial(args):
    import cma
    coords, specs, forcing, orography, initial, _ = setup(args.inputs, args.output)
    jax, jnp = d.jax, d.jnp
    grid, body = coords.horizontal, d.MARS_BODY_3D
    weights = jnp.asarray(grid.quadrature_weights)
    weights = weights / jnp.sum(weights)
    base = np.broadcast_to(np.asarray(forcing.albedo), grid.nodal_shape).copy().ravel()
    low, high = np.maximum(.02, base-.05), np.minimum(.8, base+.05)
    steps = args.steps
    dt = body.solar_day_seconds / 300

    def objective(albedo):
        f = dataclasses.replace(forcing, albedo=albedo.reshape(grid.nodal_shape))
        advance = d._build_step(coords, specs, f, dt, orography=orography)
        final = jax.lax.scan(lambda s, _: (advance(s), None), initial, None, length=steps)[0]
        return jnp.sum(final.surface_temperature[0] * weights)

    forward, vg = jax.jit(objective), jax.jit(jax.value_and_grad(objective))
    started = time.perf_counter()
    value, gradient = vg(jnp.asarray(base))
    value, gradient = float(value), np.asarray(gradient)
    first_seconds = time.perf_counter()-started
    checks = []
    indices = np.linspace(0, base.size-1, 12, dtype=int)
    for index in indices:
        eps = 1e-4
        delta = np.zeros_like(base)
        delta[index] = eps
        fd = float((forward(jnp.asarray(base+delta))-forward(jnp.asarray(base-delta)))/(2*eps))
        checks.append(dict(index=int(index), autodiff=float(gradient[index]), finite_difference=fd,
            passed=bool(np.isclose(gradient[index], fd, rtol=.02, atol=1e-6))))
    np.savez_compressed(args.output / "sensitivity.npz", gradient=gradient.reshape(grid.nodal_shape),
        albedo=base.reshape(grid.nodal_shape), lat=np.degrees(grid.latitudes), lon=np.degrees(grid.longitudes))
    write(args.output / "gradient-checks.json", dict(initial_objective_k=value,
        first_value_and_gradient_seconds=first_seconds, checks=checks))
    if not np.isfinite(gradient).all() or not all(x["passed"] for x in checks):
        raise ValueError("Spatial AD/FD check failed; optimization not launched")
    results = {}
    for method in ("gradient", "cma_es", "uniform"):
        trace = []
        best, best_x = value, base.copy()
        began = time.perf_counter()
        def evaluate(x, derivative=False):
            nonlocal best, best_x
            if derivative:
                y, g = vg(jnp.asarray(x))
                y, g = float(y), np.asarray(g)
            else:
                y, g = float(forward(jnp.asarray(x))), None
            if not np.isfinite(y) or (g is not None and not np.isfinite(g).all()):
                raise FloatingPointError("Nonfinite spatial objective/gradient")
            if y > best:
                best, best_x = y, np.asarray(x).copy()
            trace.append(dict(objective_k=y, best_k=best, gradient=derivative,
                seconds=time.perf_counter()-began))
            return y, g
        if method == "gradient":
            x = base.copy()
            for _ in range(args.evaluations):
                _, g = evaluate(x, True)
                # Scale the step in albedo units; projection enforces common bounds.
                x = np.clip(x + .005*g/max(np.max(np.abs(g)), 1e-30), low, high)
        elif method == "cma_es":
            es = cma.CMAEvolutionStrategy(base.tolist(), .01, dict(CMA_diagonal=True,
                bounds=[low.tolist(), high.tolist()], seed=args.seed+1, verbose=-9,
                popsize=8))
            while len(trace) < args.evaluations:
                candidates = es.ask()
                if len(trace)+len(candidates) > args.evaluations:
                    candidates = candidates[:args.evaluations-len(trace)]
                losses = [-evaluate(np.asarray(x))[0] for x in candidates]
                if len(candidates) == es.popsize:
                    es.tell(candidates, losses)
        else:
            for delta in np.linspace(-.05, .05, args.evaluations):
                evaluate(np.clip(base+delta, low, high))
        np.savez_compressed(args.output / f"{method}.npz", albedo=best_x.reshape(grid.nodal_shape))
        results[method] = dict(best_objective_k=best, forward_calls=len(trace),
            backward_calls=sum(t["gradient"] for t in trace), trace=trace)
        write(args.output / "optimization.json", results)
    write(args.output / "report.json", dict(status="complete", results=results, steps=steps,
        dt_seconds=dt, controls=base.size, seed=args.seed, budget_per_method=args.evaluations,
        objective="maximize area-mean final surface temperature", initial_objective_k=value,
        note="Equal forward calls; backward calls and synchronized wall time are additional costs. Diagonal CMA-ES. Gradient audit/warmup costs reported separately. Cold-start inverse-design pilot, not a climate intervention result."))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("task", choices=("climate", "spatial"))
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--projection", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--years", type=float, default=2.)
    p.add_argument("--steps", type=int, default=300)
    p.add_argument("--evaluations", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--require-gpu", action="store_true")
    a = p.parse_args()
    if not np.isfinite(a.years) or a.years <= 0 or min(a.steps, a.evaluations) < 1:
        p.error("years, steps and evaluations must be positive")
    d.jax.config.update("jax_enable_x64", True)
    if a.require_gpu and d.jax.default_backend() != "gpu":
        raise RuntimeError("GPU required")
    (climate if a.task == "climate" else spatial)(a)


if __name__ == "__main__":
    main()
