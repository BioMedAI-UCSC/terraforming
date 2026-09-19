#!/usr/bin/env python3
"""Reproducible short coupled-rollout derivative check; not a climate benchmark."""
from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path

os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=2")

import numpy as np
from src.framework.gcm._dinosaur import jax, jnp, scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.dynamics import integrate, stepper
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics
from src.celestials.planets.mars import MARS_BODY_3D
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.celestials.planets.mars.maps import initial_rest_state


def main():
    jax.config.update("jax_enable_x64", True)
    coords = coordinate_system("T21", 4)
    specs = physics_specs(MARS_BODY_3D)
    base = dataclasses.replace(
        radiative_forcing(diurnal=False, co2_radiation_enabled=True,
                          dust_visible_optical_depth=0.3, dust_longwave_optical_depth=0.1),
        regolith_enabled=True, stability_exchange_enabled=True,
        pbl_diffusion_enabled=True, convective_adjustment_enabled=True,
    )
    base = dataclasses.replace(base, init_orbital_angle_rad=physics.mean_anomaly_for_ls(0.0, base))
    dynamics = dataclasses.replace(initial_rest_state(
        coords, specs, MARS_BODY_3D, np.zeros(coords.horizontal.nodal_shape)
    ), sim_time=0.0)
    initial = physics.initial_column_state(dynamics, coords, 200.0, specs, forcing=base)
    weights = jnp.asarray(coords.horizontal.quadrature_weights)
    weights = weights / jnp.sum(weights)
    temperature_scale = float(specs.dimensionalize(1.0, scales.units.kelvin).magnitude)

    def objective(albedo):
        forcing = dataclasses.replace(base, albedo=albedo)
        equation = physics.forced_co2_primitive_equations(
            coords, MARS_BODY_3D, forcing, co2_forcing(), specs=specs
        )
        advance = physics.positivity_preserving_co2_step(stepper(equation, 60.0, specs), coords, specs)
        final = integrate(advance, initial, 8)
        return jnp.sum(final.surface_temperature[0] * weights) * temperature_scale

    value_grad = jax.jit(jax.value_and_grad(objective))
    forward = jax.jit(objective)
    value, gradient = map(float, value_grad(0.25))
    checks = []
    for epsilon in (1e-3, 1e-4):
        fd = float((forward(0.25 + epsilon) - forward(0.25 - epsilon)) / (2 * epsilon))
        relative_error = abs(gradient - fd) / max(abs(fd), 1e-12)
        checks.append(dict(epsilon=epsilon, finite_difference=fd, relative_error=relative_error))
    passed = bool(np.isfinite(gradient) and gradient < 0 and all(c["relative_error"] < 1e-3 for c in checks))
    report = dict(status="pass" if passed else "fail", truncation="T21", layers=4,
                  steps=8, dt_seconds=60, initial_ls_deg=0, terrain="flat",
                  parameter="albedo", objective="area_mean_surface_temperature_k",
                  value=value, autodiff=gradient, finite_difference_checks=checks,
                  limitations="Short smooth trajectory; does not establish gradients at frost/convection switching boundaries or climate skill.")
    output = Path("outputs/iclr_validation/gradient.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
