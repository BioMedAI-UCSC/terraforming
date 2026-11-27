#!/usr/bin/env python3
"""Reproduce one prescribed column budget defect and its conservative correction."""
import argparse
import dataclasses
import hashlib
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from src.celestials.planets.mars import MARS_BODY_3D as body
from src.celestials.planets.mars.gcm import co2_forcing
from src.framework.physics.co2_clouds import column_budgets, surface_exchange


def validate():
    jax.config.update("jax_enable_x64", True)
    sigma = (np.arange(12) + .5) / 12
    dsigma = np.full(12, 1. / 12)
    temperature = jnp.linspace(140., 200., 12)[:, None, None]
    pressure = jnp.full((1, 1), 610.)
    cloud = jnp.full_like(temperature, .01)
    wind = jnp.stack([jnp.linspace(-30., 100., 12)[:, None, None], jnp.full_like(temperature, 5.)])
    surface, frost = jnp.full_like(pressure, 150.), jnp.full_like(pressure, 20.)
    rate = jnp.full_like(pressure, 1.e-4)
    cf = co2_forcing()
    corrected = surface_exchange(temperature, pressure, cloud, wind, surface, frost,
        rate, sigma, dsigma, body, cf)
    # Reconstruct the archived pressure/frost/latent-only surface operator.
    legacy = (jnp.zeros_like(temperature), cloud * body.gravity_m_s2 * rate[None] / pressure[None],
        jnp.zeros_like(wind), cf.latent_heat_j_kg * rate / cf.thermal_inertia,
        jnp.zeros_like(wind[:, 0]))
    def budget(epsilon, rates):
        heat, ice, acceleration, ground_heat, impulse = rates
        mass, energy, momentum = column_budgets(temperature + epsilon * heat,
            pressure - epsilon * body.gravity_m_s2 * rate, cloud + epsilon * ice,
            wind + epsilon * acceleration, sigma, dsigma, body, cf)
        ts, solid = surface + epsilon * ground_heat, frost + epsilon * rate
        return (mass + solid, energy + cf.thermal_inertia * ts + solid * (
            cf.cloud_ice_heat_capacity_j_kg_k * ts - cf.latent_heat_j_kg), momentum + epsilon * impulse)
    results = {}
    for name, rates in (("legacy", legacy), ("corrected", corrected)):
        tangent = jax.jvp(lambda e: budget(e, rates), (jnp.asarray(0.),), (jnp.asarray(1.),))[1]
        results[name] = {key: np.asarray(value).squeeze().tolist() for key, value in zip(
            ("mass_residual_kg_m2_s", "energy_residual_w_m2", "horizontal_momentum_residual_pa"), tangent)}
    assert abs(results["legacy"]["energy_residual_w_m2"]) > 10.
    assert abs(results["corrected"]["energy_residual_w_m2"]) < 2.e-10
    assert abs(results["corrected"]["mass_residual_kg_m2_s"]) < 1.e-14
    assert np.max(np.abs(results["corrected"]["horizontal_momentum_residual_pa"])) < 1.e-14
    root = Path(__file__).resolve().parents[1]
    paths = ["scripts/validate_surface_mass_transport.py", "package/src/framework/physics/co2_clouds.py",
             "package/src/framework/physics/gcm.py", "package/tests/gcm3d/test_co2_clouds.py"]
    return dict(scope="Prescribed exchange-only column, not observed Mars accuracy or full-model energy closure",
        legacy_source_commit="313e43532e5addff0ca6b2dbbafae56b01b68fe2",
        source_sha256={p: hashlib.sha256((root / p).read_bytes()).hexdigest() for p in paths},
        forcing=dataclasses.asdict(cf), pressure_pa=610., surface_temperature_k=150., frost_mass_kg_m2=20.,
        temperature_k=np.asarray(temperature).ravel().tolist(),
        wind_ms=np.asarray(wind).reshape(2, 12).tolist(), cloud_ice_kg_per_kg_gas=.01,
        surface_deposition_kg_m2_s=1.e-4, results=results)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(validate(), indent=2, allow_nan=False) + "\n")
