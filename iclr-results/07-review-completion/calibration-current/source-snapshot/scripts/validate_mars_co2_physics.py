#!/usr/bin/env python3
"""Paired, synthetic polar-night experiment; not an observed climate benchmark."""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import zipfile

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")
os.environ.setdefault("JAX_COMPILATION_CACHE_DIR", str(Path("outputs/.jax_compilation_cache").resolve()))

import numpy as np
from src.framework.gcm._dinosaur import jax, jnp, scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import reference_temperature, stepper
from src.framework.gcm.restart import save_restart
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import initial_rest_state
from run_gcm3d_ablation import _checkpoint_diagnostics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sols", type=float, default=1.)
    parser.add_argument("--dt", type=float, default=300.)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not np.isfinite(args.sols) or args.sols <= 0 or not np.isfinite(args.dt) or not 0 < args.dt <= 300:
        parser.error("sols must be positive and finite; dt must be in (0, 300] seconds")
    jax.config.update("jax_enable_x64", True)
    args.output.mkdir(parents=True, exist_ok=True)
    root = Path(__file__).resolve().parents[1]
    sources = sorted((root / "package/src").rglob("*.py")) + sorted((root / "package/src/framework/physics").glob("*.npz"))
    sources += [Path(__file__).resolve(), root / "scripts/run_gcm3d_ablation.py"]
    sources += [root / path for path in (
        "AmesGCM/atmos_param_mars/mars_surface.F90",
        "AmesGCM/atmos_param_mars/update_mars_atmos.F90",
        "AmesGCM/build_run/fms_mars_default_v3.2",
    )]
    hashes = {}
    with zipfile.ZipFile(args.output / "executed_source.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sources:
            name = str(path.relative_to(root))
            content = path.read_bytes()
            hashes[name] = hashlib.sha256(content).hexdigest()
            archive.writestr(name, content)
    coords = coordinate_system("T21", args.layers)
    specs = physics_specs(BODY)
    grid = coords.horizontal
    scale = float(specs.dimensionalize(1., scales.units.pascal).magnitude)
    reference = jnp.asarray(reference_temperature(coords, BODY))[:, None, None]
    sigma = jnp.asarray(coords.vertical.centers)[:, None, None]
    forcing = dataclasses.replace(radiative_forcing(diurnal=False, co2_radiation_enabled=True,
        dust_visible_optical_depth=.3, dust_longwave_optical_depth=.1),
        tsi_1au_w_m2=0., regolith_enabled=True, stability_exchange_enabled=True,
        pbl_diffusion_enabled=True, convective_adjustment_enabled=True)
    dynamics = dataclasses.replace(initial_rest_state(coords, specs, BODY, np.zeros(grid.nodal_shape)), sim_time=0.)
    initial = physics.initial_column_state(dynamics, coords, 148., specs, forcing=forcing)
    frost = physics.co2_frost_point_k(sigma * 610.)
    initial = initial._replace(dynamics=dataclasses.replace(initial.dynamics,
        temperature_variation=grid.to_modal(jnp.broadcast_to(frost - 10. - reference,
            (args.layers,) + grid.nodal_shape))))
    save_restart(initial, args.output / "initial.npz")
    steps = int(np.ceil(args.sols * BODY.solar_day_seconds / args.dt))
    cases = {}
    for enabled in (False, True):
        name = "surface_only" if not enabled else "atmospheric_condensation"
        cf = dataclasses.replace(co2_forcing(), atmospheric_condensation=enabled)
        equation = physics.forced_co2_primitive_equations(coords, BODY, forcing, cf, specs=specs)
        advance = physics.positivity_preserving_co2_step(
            stepper(equation, args.dt, specs, coords=coords,
                hyperdiffusion_tau_seconds=.1 * BODY.solar_day_seconds),
            coords, specs, body=BODY, co2_forcing=cf, dt_seconds=args.dt,
            return_diagnostics=True)

        def sample(state):
            pressure = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure)) * scale
            air = grid.to_nodal(state.dynamics.temperature_variation) + reference
            deficit = jnp.maximum(physics.co2_frost_point_k(sigma * pressure) - air, 0.)
            return jnp.max(deficit)

        def tick(state, _):
            final, budget = advance(state)
            return final, (sample(final), budget)

        final, (deficits, budgets) = jax.jit(lambda s: jax.lax.scan(tick, s, None, length=steps))(initial)
        final.dynamics.temperature_variation.block_until_ready()
        save_restart(final, args.output / f"{name}.npz")
        np.savez_compressed(args.output / f"{name}_history.npz", frost_deficit_k=np.asarray(deficits),
            **{key: np.asarray(value) for key, value in budgets._asdict().items() if value is not None})
        before = _checkpoint_diagnostics(initial, coords, specs, args.dt)
        after = _checkpoint_diagnostics(final, coords, specs, args.dt)
        corrected = np.asarray(budgets.corrected_total_mass_kg)
        mass_drift = float(np.max(np.abs(corrected / before["total_co2_mass_kg"] - 1.)))
        cases[name] = dict(co2_forcing=dataclasses.asdict(cf), initial=before, final=after, initial_frost_deficit_k=float(sample(initial)),
            final_frost_deficit_k=float(deficits[-1]), peak_frost_deficit_k=float(jnp.max(deficits)),
            max_relative_total_co2_mass_drift=mass_drift,
            max_raw_step_mass_error_kg=float(jnp.max(jnp.abs(budgets.raw_total_mass_kg - budgets.target_total_mass_kg))),
            max_projection_energy_change_j=float(jnp.max(jnp.abs(
                budgets.corrected_exchange_energy_j - budgets.settled_exchange_energy_j))),
            valid_inventory=bool(jnp.all(budgets.valid_inventory)),
            finite=all(np.isfinite(np.asarray(x)).all() for x in jax.tree.leaves(final)))
        print(json.dumps({name: cases[name]}), flush=True)
    source_changed = any(hashlib.sha256(path.read_bytes()).hexdigest() != hashes[str(path.relative_to(root))]
                         for path in sources)
    passed = (not source_changed
              and all(c["finite"] and c["valid_inventory"] and c["max_relative_total_co2_mass_drift"] < 1.e-10 for c in cases.values())
              and cases["atmospheric_condensation"]["final_frost_deficit_k"] < cases["surface_only"]["final_frost_deficit_k"])
    report = dict(status="pass" if passed else "fail", cases=cases,
        configuration=dict(truncation="T21", layers=args.layers, dt_seconds=args.dt, steps=steps,
            elapsed_sols=steps * args.dt / BODY.solar_day_seconds, precision="float64",
            terrain="flat", sunlight="zero everywhere, synthetic polar night", initial_pressure_pa=610.,
            initial_air_frost_deficit_k=10., forcing=dataclasses.asdict(forcing)),
        environment=dict(jax=jax.__version__, devices=[str(device) for device in jax.devices()],
            system=platform.platform(), machine=platform.machine()),
        git_revision=subprocess.check_output(["rtk", "proxy", "git", "rev-parse", "HEAD"], cwd=root, text=True).strip(),
        source_sha256=hashes, source_changed_during_run=source_changed,
        artifact_sha256={path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.output.iterdir() if path.suffix in (".npz", ".zip")},
        limitations="Synthetic paired mechanism test. No observational skill, seasonal equilibrium, terrain stability or full dry-core energy/momentum closure is established. Raw errors and projection energy changes remain reported.")
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
