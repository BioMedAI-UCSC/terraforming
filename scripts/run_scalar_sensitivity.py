#!/usr/bin/env python3
"""Check surface-temperature/pressure/frost derivatives through a coupled rollout."""
import argparse
import dataclasses
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps/mars-calibration"))
from mars_calibration import paper as p
from mars_calibration import driver as d
from mars_calibration.provenance import snapshot_sources


def diagnostics(model, states):
    """Differentiable area means in K, Pa, and frost pressure-equivalent Pa."""
    scale = float(model.specs.dimensionalize(1., d.scales.units.pascal).magnitude)
    grid = model.coords.horizontal
    pressure = d.jnp.exp(grid.to_nodal(states.dynamics.log_surface_pressure))[:, 0] * scale
    fields = (states.surface_temperature[:, 0], pressure, states.co2_ice[:, 0] * scale)
    return d.jnp.stack([d.jnp.sum(field * model.weights, axis=(-2, -1)) for field in fields], axis=-1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", nargs="+", type=int, default=[74, 296, 1480])
    parser.add_argument("--dt", type=float, default=300.)
    parser.add_argument("--epsilon", type=float, default=1e-3)
    parser.add_argument("--require-gpu", action="store_true")
    args = parser.parse_args()
    if (not np.isfinite([args.dt, args.epsilon]).all() or not 0 < args.dt <= 300 or
            not 0 < args.epsilon < 1 or min(args.steps) < 1 or args.steps != sorted(set(args.steps))):
        parser.error("positive ordered horizons, 0 < dt <= 300, and 0 < epsilon < 1 required")
    inputs = p.validate_inputs(args.inputs)
    os.environ["MOLA_PATH"] = str((args.inputs / "mola.img").resolve())
    d.jax.config.update("jax_enable_x64", True)
    if args.require_gpu and d.jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    model = p.Model(args.inputs)
    args.output.mkdir(parents=True, exist_ok=False)
    p.dump(args.output / "provenance.json", dict(inputs=inputs, environment=p.environment(),
        arguments={k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        body=dataclasses.asdict(d.MARS_BODY_3D), executed_source=snapshot_sources(ROOT, args.output)))
    trajectory = model.trajectory(np.asarray(args.steps) * args.dt, args.dt, states=True)
    def objective(beta):
        return diagnostics(model, trajectory(d.jnp.asarray([beta, 1., 1.])))
    forward, timing = p.compile_function(objective, d.jnp.asarray(1.))
    derivative, derivative_timing = p.compile_function(d.jax.jacfwd(objective), d.jnp.asarray(1.))
    value, ad = np.asarray(forward(1.)), np.asarray(derivative(1.))
    fd = np.asarray((forward(1. + args.epsilon) - forward(1. - args.epsilon)) / (2 * args.epsilon))
    error = np.abs(ad - fd)
    passed = np.isfinite([value, ad, fd]).all() and np.all(error <= 1e-8 + 1e-4 * np.abs(fd))
    rows = [dict(steps=steps, seconds=steps * args.dt, diagnostic=name,
                 value=float(value[i, j]), autodiff=float(ad[i, j]),
                 finite_difference=float(fd[i, j]), absolute_error=float(error[i, j]))
            for i, steps in enumerate(args.steps)
            for j, name in enumerate(("surface_temperature_k", "surface_pressure_pa", "surface_frost_pa"))]
    p.write_csv(args.output / "sensitivity.csv", rows)
    p.dump(args.output / "acceptance.json", dict(all_checks_pass=bool(passed),
        forward_timing=timing, derivative_timing=derivative_timing,
        scope="local tangent/finite-difference check, not equilibrium climate or raw budget closure"))
    if not passed:
        raise SystemExit("scalar derivative check failed; see saved CSV")
    print(args.output / "sensitivity.csv")


if __name__ == "__main__":
    main()
