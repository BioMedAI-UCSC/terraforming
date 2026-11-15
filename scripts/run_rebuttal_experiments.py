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


def control_statistics(advance, observe, initial, steps, block_steps=300):
    """Integrate with block rematerialization; return final state and step mean."""
    if type(steps) is not int or steps < 1 or type(block_steps) is not int or block_steps < 1:
        raise ValueError("steps and block_steps must be positive static integers")
    jax, jnp = d.jax, d.jnp

    def tick(carry, _):
        state, total = carry
        state = advance(state)
        return (state, total + observe(state)), None

    def block(carry, _):
        return jax.lax.scan(tick, carry, None, length=block_steps)[0], None

    full, remainder = divmod(steps, block_steps)
    carry = (initial, jnp.asarray(0., dtype=jnp.float64))
    if full:
        # Keep block boundaries, and recompute each block during the backward pass.
        carry, _ = jax.lax.scan(jax.checkpoint(block), carry, None, length=full)
    if remainder:
        carry, _ = jax.lax.scan(tick, carry, None, length=remainder)
    return carry[0], carry[1] / steps


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

    observe = lambda state: jnp.sum(state.surface_temperature[0] * weights)

    def diagnostics(albedo):
        f = dataclasses.replace(forcing, albedo=albedo.reshape(grid.nodal_shape))
        advance = d._build_step(coords, specs, f, dt, orography=orography)
        final, mean = control_statistics(advance, observe, initial, steps)
        return jnp.stack([observe(final), mean])

    objective_index = 1 if args.objective == "mean" else 0
    def objective_aux(albedo):
        stats = diagnostics(albedo)
        return stats[objective_index], stats

    statistics_forward = jax.jit(diagnostics)
    forward = lambda albedo: statistics_forward(albedo)[objective_index]
    vg = jax.jit(jax.value_and_grad(objective_aux, has_aux=True))
    protocol = dict(steps=steps, dt_seconds=dt, elapsed_sols=steps/300,
        elapsed_mars_years=steps*dt/forcing.orbital_period_s, objective=args.objective,
        controls=base.size, seed=args.seed, evaluations=args.evaluations,
        audit_directions=args.audit_directions, checkpoint_block_steps=300,
        bounds=dict(lower=low.tolist(), upper=high.tolist()), methods=args.methods,
        note="Mean objective is a right-endpoint time mean over all integration steps; no spinup is omitted. Cold-start seasonal control test, not equilibrium or observational calibration.")
    write(args.output / "protocol.json", protocol)
    print(json.dumps(dict(phase="baseline_forward", steps=steps, objective=args.objective)), flush=True)
    baseline_start = time.perf_counter()
    baseline = np.asarray(statistics_forward(jnp.asarray(base)))
    if not np.isfinite(baseline).all():
        write(args.output / "report.json", dict(status="nonfinite_baseline", **protocol))
        raise FloatingPointError("Nonfinite baseline objective")
    write(args.output / "baseline.json", dict(final_surface_temperature_k=float(baseline[0]),
        mean_surface_temperature_k=float(baseline[1]), first_forward_seconds=time.perf_counter()-baseline_start))
    print(json.dumps(dict(phase="baseline_complete", values_k=baseline.tolist())), flush=True)
    started = time.perf_counter()
    value, gradient, audit_pass = float(baseline[objective_index]), None, True
    checks = []
    if "gradient" in args.methods:
        print(json.dumps(dict(phase="baseline_reverse", steps=steps)), flush=True)
        (ad_value, _), gradient = vg(jnp.asarray(base))
        ad_value, gradient = float(ad_value), np.asarray(gradient)
        finite = bool(np.isfinite(ad_value) and np.isfinite(gradient).all())
        audit_pass = finite and bool(np.isclose(ad_value, value, rtol=1e-10, atol=1e-8))
        np.savez_compressed(args.output / "sensitivity.npz", gradient=gradient.reshape(grid.nodal_shape),
            albedo=base.reshape(grid.nodal_shape), lat=np.degrees(grid.latitudes), lon=np.degrees(grid.longitudes))
        if finite:
            if args.audit_directions:
                rng = np.random.default_rng(0)
                directions = [rng.normal(size=base.shape) for _ in range(args.audit_directions)]
                directions = [v/np.linalg.norm(v) for v in directions]
                epsilons = (1e-3, 1e-4, 1e-5)
            else:
                indices = np.linspace(0, base.size-1, 12, dtype=int)
                directions = np.zeros((len(indices), base.size))
                directions[np.arange(len(indices)), indices] = 1.
                epsilons = (1e-4,)
            for index, direction in enumerate(directions):
                ad = float(np.dot(gradient, direction))
                if not np.isfinite(ad):
                    checks.append(dict(direction=index, autodiff=None, finite_difference=None,
                        passed=False, reason="nonfinite_directional_derivative"))
                    audit_pass = False
                    continue
                passed = 0
                for eps in epsilons:
                    fd = float((forward(jnp.asarray(base+eps*direction))-
                                forward(jnp.asarray(base-eps*direction)))/(2*eps))
                    agrees = bool(np.isfinite(fd) and np.isclose(ad, fd, rtol=.02, atol=1e-6))
                    checks.append(dict(direction=index, epsilon=eps, autodiff=ad,
                        finite_difference=fd if np.isfinite(fd) else None, passed=agrees))
                    passed += agrees
                    print(json.dumps(dict(phase="gradient_check", **checks[-1])), flush=True)
                audit_pass = audit_pass and passed >= (2 if args.audit_directions else 1)
        write(args.output / "gradient-checks.json", dict(initial_objective_k=value,
            first_value_gradient_and_checks_seconds=time.perf_counter()-started,
            finite_gradient=finite, gradient_max_abs=float(np.abs(gradient).max()) if finite else None,
            scientific_acceptance_pass=bool(audit_pass), checks=checks))
        if args.audit_only:
            write(args.output / "report.json", dict(status="audit_pass" if audit_pass else "audit_fail", **protocol))
            if not audit_pass:
                raise SystemExit(2)
            return
    results = {}
    for method in args.methods:
        if method == "gradient" and not audit_pass:
            results[method] = dict(status="skipped_failed_gradient_audit", forward_calls=0, backward_calls=0)
            write(args.output / "optimization.json", results)
            continue
        print(json.dumps(dict(phase="optimization_start", method=method)), flush=True)
        trace = []
        best, best_x, best_statistics = value, base.copy(), baseline.copy()
        began = time.perf_counter()
        def evaluate(x, derivative=False):
            nonlocal best, best_x, best_statistics
            if derivative:
                (y, stats), g = vg(jnp.asarray(x))
                y, stats, g = float(y), np.asarray(stats), np.asarray(g)
            else:
                stats = np.asarray(statistics_forward(jnp.asarray(x)))
                y, g = float(stats[objective_index]), None
            valid = bool(np.isfinite(y) and np.isfinite(stats).all() and
                (g is None or np.isfinite(g).all()))
            if valid and y > best:
                best, best_x, best_statistics = y, np.asarray(x).copy(), stats.copy()
            if not valid:
                np.savez_compressed(args.output / f"{method}-failed-{len(trace):03d}.npz", albedo=x,
                    statistics=stats, **({"gradient": g} if g is not None else {}))
            trace.append(dict(objective_k=y if valid else None, best_k=best, gradient=derivative,
                finite=valid, final_surface_temperature_k=float(stats[0]) if valid else None,
                mean_surface_temperature_k=float(stats[1]) if valid else None,
                seconds=time.perf_counter()-began))
            write(args.output / f"{method}-progress.json", dict(trace=trace, best_objective_k=best))
            np.savez_compressed(args.output / f"{method}-best.npz", albedo=best_x.reshape(grid.nodal_shape))
            print(json.dumps(dict(phase="evaluation", method=method, evaluation=len(trace), **trace[-1])), flush=True)
            return (y if valid else None), g
        if method == "gradient":
            x = base.copy()
            for _ in range(args.evaluations):
                y, g = evaluate(x, True)
                if y is None:
                    break
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
                values = [evaluate(np.asarray(x))[0] for x in candidates]
                losses = [-y if y is not None else 1e30 for y in values]
                if len(candidates) == es.popsize:
                    es.tell(candidates, losses)
        else:
            for delta in np.linspace(-.05, .05, args.evaluations):
                evaluate(np.clip(base+delta, low, high))
        np.savez_compressed(args.output / f"{method}.npz", albedo=best_x.reshape(grid.nodal_shape))
        results[method] = dict(best_objective_k=best, forward_calls=len(trace),
            failed_evaluations=sum(not t["finite"] for t in trace),
            best_final_surface_temperature_k=float(best_statistics[0]),
            best_mean_surface_temperature_k=float(best_statistics[1]),
            backward_calls=sum(t["gradient"] for t in trace), trace=trace)
        write(args.output / "optimization.json", results)
    failed_evaluations = any(r.get("failed_evaluations", 0) for r in results.values())
    status = ("gradient_audit_failed" if not audit_pass else
              "nonfinite_optimization" if failed_evaluations else "complete")
    write(args.output / "report.json", dict(status=status, results=results, steps=steps,
        dt_seconds=dt, controls=base.size, seed=args.seed, budget_per_method=args.evaluations,
        objective=("maximize surface temperature averaged over area and time" if args.objective == "mean"
                   else "maximize area-mean final surface temperature"), initial_objective_k=value,
        elapsed_sols=steps/300, elapsed_mars_years=steps*dt/forcing.orbital_period_s,
        scientific_gradient_acceptance_pass=bool(audit_pass) if "gradient" in args.methods else None,
        note="Equal forward calls; backward calls and synchronized wall time are additional costs. Diagonal CMA-ES. Gradient audit/warmup costs reported separately. Cold-start inverse-design pilot, not a climate intervention result."))
    if not audit_pass or failed_evaluations:
        raise SystemExit(2)


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
    p.add_argument("--objective", choices=("final", "mean"), default="final")
    p.add_argument("--methods", nargs="+", choices=("gradient", "cma_es", "uniform"),
                   default=["gradient", "cma_es", "uniform"])
    p.add_argument("--audit-directions", type=int, default=0)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--require-gpu", action="store_true")
    a = p.parse_args()
    if not np.isfinite(a.years) or a.years <= 0 or min(a.steps, a.evaluations) < 1 or a.audit_directions < 0:
        p.error("years, steps and evaluations must be positive")
    if len(set(a.methods)) != len(a.methods) or (a.audit_only and "gradient" not in a.methods):
        p.error("methods must be unique; audit-only requires the gradient method")
    d.jax.config.update("jax_enable_x64", True)
    if a.require_gpu and d.jax.default_backend() != "gpu":
        raise RuntimeError("GPU required")
    (climate if a.task == "climate" else spatial)(a)


if __name__ == "__main__":
    main()
