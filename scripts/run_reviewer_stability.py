#!/usr/bin/env python3
"""Archive default-ultra stability or a production-physics timestep trajectory."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/mars-calibration"))

import numpy as np
from mars_calibration.provenance import hardware, snapshot_sources
from src.framework.gcm._dinosaur import jax, jnp, scales, spherical_harmonic
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import reference_temperature, stepper
from src.framework.gcm.restart import save_restart, load_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import initial_rest_state, resolve_scale
from src.celestials.planets.mars.topography import regrid_to_nodal, mola_modal_orography
from run_gcm3d_ablation import _checkpoint_diagnostics, _ames_dust_climatology
from neural_temp_data import write_json


def configuration(case, dt, years):
    if case == "ultra":
        return resolve_scale("ultra")
    if dt not in (150., 300., 450.) or not np.isfinite(years) or years <= 0:
        raise ValueError("timestep study requires 150, 300 or 450 s and positive orbital years")
    duration = round(years * radiative_forcing().orbital_period_s / 900.) * 900.
    if duration <= 0:
        raise ValueError("duration must include a common 900-second interval")
    return dict(truncation="T21", n_layers=12, dt_seconds=dt,
                n_steps=int(round(duration / dt)), hyperdiffusion_tau_seconds=.1 * BODY.solar_day_seconds)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", choices=("ultra", "timestep", "smoke"))
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dt", type=float, default=300.)
    parser.add_argument("--years", type=float, default=1.)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    jax.config.update("jax_enable_x64", True)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    cfg = (dict(truncation="T21", n_layers=4, dt_seconds=300., n_steps=2,
                hyperdiffusion_tau_seconds="auto") if args.case == "smoke" else
           configuration(args.case, args.dt, args.years))
    if args.output.exists() and not args.resume:
        raise ValueError("output exists; use --resume or a new directory")
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["MOLA_PATH"] = str((args.inputs / "mola.img").resolve())
    coords = coordinate_system(cfg["truncation"], cfg["n_layers"])
    grid, specs = coords.horizontal, physics_specs(BODY)
    tau = cfg["hyperdiffusion_tau_seconds"]
    if tau == "auto":
        tau = .1 * BODY.solar_day_seconds * 42 / int(cfg["truncation"][1:])
    forcing = radiative_forcing(diurnal=args.case == "timestep", co2_radiation_enabled=args.case == "timestep")
    forcing = dataclasses.replace(forcing, init_orbital_angle_rad=physics.mean_anomaly_for_ls(0., forcing))
    if args.case == "timestep":
        from src.celestials.planets.mars.maps import forcing_with_surface_properties
        forcing = forcing_with_surface_properties(forcing, grid, args.inputs / "surface.nc")
        ls, visible, infrared = _ames_dust_climatology(args.inputs / "dust.nc", grid)
        forcing = dataclasses.replace(forcing, regolith_enabled=True, stability_exchange_enabled=True,
            pbl_diffusion_enabled=True, convective_adjustment_enabled=True,
            dust_climatology_ls_deg=jnp.asarray(ls), dust_visible_climatology=jnp.asarray(visible),
            dust_longwave_climatology=jnp.asarray(infrared))
    elevation = regrid_to_nodal(coords)
    orography = mola_modal_orography(coords, specs, elevation_nodal_m=elevation)
    initial = physics.initial_column_state(dataclasses.replace(
        initial_rest_state(coords, specs, BODY, elevation), sim_time=0.),
        coords, BODY.reference_temperature_k, specs, forcing=forcing)
    cf = co2_forcing()
    equation = physics.forced_co2_primitive_equations(coords, BODY, forcing, cf, specs=specs, orography=orography)
    raw = stepper(equation, cfg["dt_seconds"], specs, coords=coords, hyperdiffusion_tau_seconds=tau)
    advance = physics.positivity_preserving_co2_step(raw, coords, specs, body=BODY,
        co2_forcing=cf, dt_seconds=cfg["dt_seconds"], return_diagnostics=True)
    # Check identity before copying source bytes over a previous attempt.
    source_hashes = {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for folder in ("scripts", "package/src", "apps/mars-calibration") for p in sorted((ROOT/folder).rglob("*.py"))}
    identity = dict(case=args.case, grid=cfg, jax=jax.__version__, source_sha256=source_hashes,
        forcing=dataclasses.asdict(forcing), co2_forcing=dataclasses.asdict(cf),
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.iterdir() if p.is_file()})
    identity = json.loads(json.dumps(identity, default=lambda x: np.asarray(x).tolist()))
    configuration_path = args.output / "configuration.json"
    if configuration_path.exists() and json.loads(configuration_path.read_text())["identity"] != identity:
        raise ValueError("source, inputs or numerical configuration changed; use a new output directory")
    source = snapshot_sources(ROOT, args.output)
    configuration_path.write_text(json.dumps(dict(case=args.case, grid=cfg, identity=identity,
        forcing=dataclasses.asdict(forcing), co2_forcing=dataclasses.asdict(cf), hardware=hardware(jax.devices()),
        jax=jax.__version__, source_snapshot=source,
        input_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in args.inputs.iterdir() if p.is_file()},
        note="450 s is exercised by this isolated evaluation; calibration's 300 s guard is unchanged"),
        indent=2, default=lambda x: np.asarray(x).tolist()) + "\n")
    write_json(args.output / f"hardware-attempt-{len(list(args.output.glob('hardware-attempt-*.json'))):03d}.json",
               hardware(jax.devices()))
    save_restart(initial, args.output / "initial.npz")
    scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
    velocity = float(specs.dimensionalize(1., scales.units.meter / scales.units.second).magnitude)
    ref = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    degree = jnp.asarray(grid.modal_mesh[1])
    mask = degree >= .8 * (grid.total_wavenumbers - 2)
    sigma = jnp.asarray(coords.vertical.centers)[:, None, None]

    def tick(state, _):
        final, budget = advance(state)
        # Retain every raw mass correction; large atmospheric arrays stay in the carry.
        pressure = jnp.exp(grid.to_nodal(final.dynamics.log_surface_pressure))[0] * scale
        correction = pressure * (1. - 1. / budget.pressure_factor)
        weights = jnp.asarray(grid.quadrature_weights)
        regions = [jnp.abs(grid.latitudes) < np.pi/3, grid.latitudes >= np.pi/3, grid.latitudes <= -np.pi/3]
        regional = [jnp.sum(correction * weights * region[None]) / jnp.sum(weights * region[None]) for region in regions]
        return final, jnp.stack([budget.raw_total_mass_kg, budget.target_total_mass_kg,
            budget.corrected_total_mass_kg, budget.pressure_factor,
            budget.raw_exchange_energy_j, budget.frost_repaired_exchange_energy_j,
            budget.settled_exchange_energy_j, budget.corrected_exchange_energy_j,
            budget.raw_axial_angular_momentum_kg_m2_s, budget.corrected_axial_angular_momentum_kg_m2_s,
            jnp.max(jnp.abs(correction)), *regional])

    def diagnostic(state):
        record = _checkpoint_diagnostics(state, coords, specs, cfg["dt_seconds"])
        wind = jnp.stack(spherical_harmonic.vor_div_to_uv_nodal(grid,
            state.dynamics.vorticity, state.dynamics.divergence)) * velocity
        air = grid.to_nodal(state.dynamics.temperature_variation) + ref
        speed = jnp.sqrt(jnp.sum(wind**2, axis=0))
        gamma = BODY.cp_j_kg_k / (BODY.cp_j_kg_k - BODY.gas_constant_j_kg_k)
        pressure = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure)) * scale
        record.update(max_mach=float(jnp.max(speed / jnp.sqrt(gamma * BODY.gas_constant_j_kg_k * air))),
            max_frost_deficit_k=float(jnp.max(jnp.maximum(physics.co2_frost_point_k(sigma * pressure)-air, 0.))))
        for name in ("vorticity", "divergence", "temperature_variation"):
            values = getattr(state.dynamics, name)**2
            record[name + "_high_degree_fraction"] = float(jnp.sum(values * mask) / jnp.maximum(jnp.sum(values), 1.e-300))
        # For constant density, inverse Laplacian weighting converts vorticity/divergence power to a kinetic-energy indicator.
        kinetic = (state.dynamics.vorticity**2 + state.dynamics.divergence**2) / jnp.maximum(degree*(degree+1), 1.)
        record["wind_modal_kinetic_high_degree_fraction"] = float(jnp.sum(kinetic * mask) / jnp.maximum(jnp.sum(kinetic), 1.e-300))
        record["wind_modal_kinetic_indicator_spectrum"] = np.asarray(jnp.sum(kinetic, axis=(0, 1))).tolist()
        record["max_absolute_vorticity_nd"] = float(jnp.max(jnp.abs(grid.to_nodal(state.dynamics.vorticity))))
        return record

    state, completed, rows = initial, 0, [diagnostic(initial)]
    budget_maxima = np.zeros(4)
    progress_path = args.output / "progress.json"
    if progress_path.exists():
        progress = json.loads(progress_path.read_text())
        completed, rows = progress["completed_steps"], progress["diagnostics"]
        budget_maxima = np.asarray(progress["budget_maxima"])
        state = load_restart(args.output / progress["checkpoint"])
    chunk = min(100, cfg["n_steps"])
    full_chunk = jax.jit(lambda state: jax.lax.scan(tick, state, None, length=chunk))
    initial_mass = rows[0]["total_co2_mass_kg"]
    while completed < cfg["n_steps"]:
        count = min(chunk, cfg["n_steps"] - completed)
        state, history = (full_chunk(state) if count == chunk else
            jax.jit(lambda s: jax.lax.scan(tick, s, None, length=count))(state))
        history = np.asarray(history)
        if not all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(state)) or not np.isfinite(history).all():
            (args.output / "failure.json").write_text(json.dumps(dict(completed_steps=completed, error="nonfinite state/budget")))
            raise FloatingPointError("nonfinite trajectory; previous checkpoint preserved")
        completed += count
        budget_maxima = np.maximum(budget_maxima, [np.max(np.abs(history[:,0]-history[:,1])),
            np.max(np.abs(history[:,2]/initial_mass-1.)), np.max(np.abs(history[:,7]-history[:,6])),
            np.max(np.abs(history[:,9]-history[:,8]))])
        columns = ("raw_total_mass_kg", "target_total_mass_kg", "corrected_total_mass_kg", "pressure_factor",
            "raw_exchange_energy_j", "frost_repaired_exchange_energy_j", "settled_exchange_energy_j",
            "corrected_exchange_energy_j", "raw_axial_angular_momentum_kg_m2_s",
            "corrected_axial_angular_momentum_kg_m2_s", "max_local_pressure_correction_pa",
            "mean_nonpolar_pressure_correction_pa", "mean_north_polar_pressure_correction_pa",
            "mean_south_polar_pressure_correction_pa")
        np.savez_compressed(args.output / f"budgets-{completed:09d}.npz", **dict(zip(columns, history.T, strict=True)))
        checkpoint = f"state-{completed:09d}.npz"
        save_restart(state, args.output / checkpoint)
        rows.append(diagnostic(state))
        write_json(args.output / "diagnostics.json", rows)
        write_json(progress_path, dict(completed_steps=completed, checkpoint=checkpoint,
            budget_maxima=budget_maxima.tolist(), diagnostics=rows))
        print(json.dumps(dict(case=args.case, completed=completed, total=cfg["n_steps"],
            min_air_k=rows[-1]["min_air_temperature_k"], max_wind_ms=rows[-1]["max_wind_speed_ms"],
            vorticity_tail=rows[-1]["vorticity_high_degree_fraction"])), flush=True)
    (args.output / "report.json").write_text(json.dumps(dict(status="finite_complete", steps=completed,
        seconds=completed*cfg["dt_seconds"], max_raw_step_mass_error_kg=budget_maxima[0],
        max_post_projection_relative_co2_drift=budget_maxima[1], max_projection_exchange_energy_change_j=budget_maxima[2],
        max_raw_to_corrected_aam_change_kg_m2_s=budget_maxima[3],
        limitations="Finite trajectory, spectra and diagnosed budgets; no equilibrium or observational skill claim. Exchange energy is not the complete dry-core energy theorem."), indent=2)+"\n")


if __name__ == "__main__":
    main()
