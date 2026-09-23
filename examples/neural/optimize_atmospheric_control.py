#!/usr/bin/env python3
"""Optimize a bounded neural heating policy through a low-resolution Mars rollout.

This is a capability demonstration: the target is synthetic and no observational
or terraforming claim is intended. A physical GCM rollout supplies the target
trajectory; the policy is optimized only from the final-state objective.
"""
import argparse
from pathlib import Path

import numpy as np

from src.framework.gcm._dinosaur import jax, jnp, primitive_equations, scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.learning import make_parameterized_step, rollout
from src.framework.gcm.specs import physics_specs
from src.framework.neural import NeuralTendency, fit_normalization, state_features, state_feature_schema
from src.framework.neural.training import adam_init, adam_update, tree_norm
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import radiative_forcing
from src.framework.physics.gcm import forced_primitive_equations, initial_column_state
from common import provenance, write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="outputs/neural_framework/atmospheric_control")
    parser.add_argument("--iterations", type=int, default=40)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--target-shift-k", type=float, default=-0.2)
    args = parser.parse_args()
    if min(args.iterations, args.steps, args.layers) < 1:
        parser.error("iterations, steps and layers must be positive")
    jax.config.update("jax_enable_x64", True)

    coords = coordinate_system("T21", args.layers)
    specs = physics_specs(BODY)
    forcing = radiative_forcing(co2_radiation_enabled=False)
    grid = coords.horizontal
    zeros = jnp.zeros((args.layers,) + grid.modal_shape)
    ps_nd = float(specs.nondimensionalize(610.0 * scales.units.pascal))
    dynamics = primitive_equations.State(
        vorticity=zeros, divergence=zeros, temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(ps_nd))),
        sim_time=0.0,
    )
    initial = initial_column_state(dynamics, coords, 220.0, specs, forcing=forcing)
    policy = NeuralTendency(args.layers, maximum_heating_k_s=2e-3, hidden_sizes=(16, 16))
    normalization = fit_normalization(
        np.asarray(state_features(initial, coords, specs, BODY, forcing)), split="train")
    params = policy.init(jax.random.PRNGKey(args.seed))
    # Start from the physical model (zero neural intervention), then let the
    # trajectory objective discover a bounded correction.
    last_weight, last_bias = params[-1]
    params = params[:-1] + ((jnp.zeros_like(last_weight), jnp.zeros_like(last_bias)),)

    physical_step = make_parameterized_step(
        lambda _: forced_primitive_equations(coords, BODY, forcing, specs=specs),
        30.0, specs)
    physical = jax.jit(lambda: rollout(physical_step, (), initial, args.steps).final_state)
    teacher = physical()
    target_air = grid.to_nodal(teacher.dynamics.temperature_variation) + BODY.reference_temperature_k + args.target_shift_k

    def control_step(parameters):
        equation = forced_primitive_equations(
            coords, BODY, forcing, specs=specs,
            neural_tendency=policy.bind(parameters, normalization, coords=coords,
                                         specs=specs, body=BODY, forcing=forcing),
        )
        return make_parameterized_step(lambda _: equation, 30.0, specs)

    def objective(parameters):
        result = rollout(control_step(parameters), parameters, initial, args.steps, remat=True)
        final_air = grid.to_nodal(result.final_state.dynamics.temperature_variation) + BODY.reference_temperature_k
        state_error = jnp.mean((final_air - target_air)**2)
        # Penalize policy magnitude at every current state only through the
        # parameter norm. The hard tanh bound remains the safety constraint.
        regularization = 1e-6 * tree_norm(parameters)**2
        return state_error + regularization

    update = jax.jit(lambda p, opt: (
        lambda value, grad: (*adam_update(p, grad, opt, learning_rate=0.03, clip_norm=5.), value)
    )(*jax.value_and_grad(objective)(p)))
    initial_loss = float(objective(params))
    optimizer = adam_init(params)
    history = []
    for iteration in range(args.iterations):
        params, optimizer, value = update(params, optimizer)
        history.append({"iteration": iteration + 1, "loss_before_update": float(value),
                        "gradient_norm": float(tree_norm(jax.grad(objective)(params)))})

    result = rollout(control_step(params), params, initial, args.steps, save_every=max(1, args.steps // 4))
    final_air = grid.to_nodal(result.final_state.dynamics.temperature_variation) + BODY.reference_temperature_k
    initial_air = grid.to_nodal(initial.dynamics.temperature_variation) + BODY.reference_temperature_k
    policy_features = state_features(initial, coords, specs, BODY, forcing)
    raw = policy.model.apply(params, normalization.apply(policy_features))[..., 0]
    heating = policy.maximum_heating_k_s * jnp.tanh(raw)
    report = {
        "provenance": provenance(args),
        "purpose": "synthetic neural inverse-control capability demonstration",
        "configuration": {"truncation": "T21", "layers": args.layers, "dt_seconds": 30.0,
                          "steps": args.steps, "target_shift_k": args.target_shift_k,
                          "feature_schema": state_feature_schema(args.layers),
                          "maximum_heating_k_s": policy.maximum_heating_k_s},
        "initial_objective": initial_loss, "final_objective": float(objective(params)),
        "initial_rmse_k": float(jnp.sqrt(jnp.mean((initial_air - target_air)**2))),
        "final_rmse_k": float(jnp.sqrt(jnp.mean((final_air - target_air)**2))),
        "max_abs_policy_heating_k_s": float(jnp.max(jnp.abs(heating))),
        "finite_rollout": bool(all(np.isfinite(x).all() for x in jax.tree.leaves(result.final_state))),
        "history": history,
    }
    write_report(args.output, report, "Differentiable neural atmospheric control")
    np.savez_compressed(Path(args.output) / "control_trajectory.npz",
                        steps=np.asarray(result.steps),
                        initial_air=np.asarray(initial_air), target_air=np.asarray(target_air),
                        final_air=np.asarray(final_air), heating=np.asarray(heating))
    print(f"Wrote {args.output}/report.md")


if __name__ == "__main__":
    main()
