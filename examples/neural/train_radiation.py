#!/usr/bin/env python3
"""Train radiation locally and through columns, then deploy it in a T21 GCM."""
import argparse
import dataclasses
from pathlib import Path
import time

import numpy as np

from src.framework.gcm._dinosaur import jax, jnp, primitive_equations, scales
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm.learning import make_parameterized_step, rollout
from src.framework.physics.gcm import forced_primitive_equations, initial_column_state, two_stream_radiative_fluxes
from src.framework.neural import (
    NeuralRadiation, column_features, fit_normalization, reference_fluxes, heating_rates,
    radiation_budget_residual, feature_schema, save_checkpoint, load_checkpoint, weighted_mse,
)
from src.framework.neural.training import adam_init, adam_update, tree_norm
from src.framework.neural.columns import initial_column, make_column_step, column_energy
from common import BODY, configuration, generate_columns, provenance, synchronized_call, write_report


def fit(params, loss, validation_loss, updates, learning_rate, *, time_budget=None):
    """Validation selection includes the input checkpoint; never inspect test data."""
    @jax.jit
    def update(p, opt):
        value, grad = jax.value_and_grad(loss)(p)
        next_p, next_opt = adam_update(p, grad, opt, learning_rate=learning_rate)
        return next_p, next_opt, value, tree_norm(grad)

    validate = jax.jit(validation_loss)
    best, best_loss = params, float(validate(params))
    opt = adam_init(params)
    _, first_call = synchronized_call(update, params, opt)
    start = time.perf_counter()
    history = []
    # A continuation control may use the fine-tuning wall-time budget. Cap the
    # loop explicitly and disclose if it could not spend the matching budget.
    cap = updates if time_budget is None else max(updates, 10000)
    failed = False
    for epoch in range(cap):
        params, opt, value, norm = update(params, opt)
        val = float(validate(params))
        history.append({"update": epoch+1, "train_before_update": float(value),
                        "validation_after_update": val, "gradient_norm": float(norm)})
        if not np.isfinite([float(value), val, float(norm)]).all():
            failed = True
            break
        if val < best_loss:
            best, best_loss = params, val
        if epoch+1 >= updates and (time_budget is None or time.perf_counter()-start >= time_budget):
            break
    return best, {"history": history, "best_validation_loss": best_loss,
                  "first_update_compile_and_execution_seconds": first_call,
                  "training_seconds": time.perf_counter()-start, "failed": failed,
                  "time_budget_seconds": time_budget, "updates": len(history)}


def coupled_evaluation(adapter, norm, models, forcing, layers, steps):
    """Use the same trained checkpoint in the actual Dinosaur GCM, with no terrain download."""
    coords, specs = coordinate_system("T21", layers), physics_specs(BODY)
    grid = coords.horizontal
    zeros = jnp.zeros((layers,) + grid.modal_shape)
    ps_nd = float(specs.nondimensionalize(610 * scales.units.pascal))
    dynamics = primitive_equations.State(
        vorticity=zeros, divergence=zeros, temperature_variation=zeros,
        log_surface_pressure=grid.to_modal(jnp.full((1,) + grid.nodal_shape, np.log(ps_nd))), sim_time=0.)
    initial = initial_column_state(dynamics, coords, 220., specs, forcing=forcing)

    conventional_step = make_parameterized_step(
        lambda p: forced_primitive_equations(coords, BODY, forcing, specs=specs), 30., specs)
    learned_step = make_parameterized_step(
        lambda p: forced_primitive_equations(coords, BODY, forcing, specs=specs,
                                            radiation_component=adapter.bind(p, norm)), 30., specs)
    physical = jax.jit(lambda: rollout(conventional_step, (), initial, steps).final_state)
    neural = jax.jit(lambda params: rollout(learned_step, params, initial, steps).final_state)
    target, compile_time = synchronized_call(physical)
    _, physical_time = synchronized_call(physical)
    report = {"truncation": "T21", "layers": layers, "steps": steps, "dt_seconds": 30.,
              "purpose": "short coupled execution check; not a climate-stability result",
              "physical_first_call_seconds": compile_time, "physical_warm_seconds": physical_time,
              "models": {}}
    weights = jnp.asarray(grid.quadrature_weights)
    initial_mass = jnp.sum(jnp.exp(grid.to_nodal(initial.dynamics.log_surface_pressure)) * weights)
    for name, params in models.items():
        final, first_time = synchronized_call(neural, params)
        _, warm_time = synchronized_call(neural, params)
        air = grid.to_nodal(final.dynamics.temperature_variation) + BODY.reference_temperature_k
        flux = adapter.bind(params, norm)(final, coords, specs, BODY, forcing)
        mass = jnp.sum(jnp.exp(grid.to_nodal(final.dynamics.log_surface_pressure)) * weights)
        report["models"][name] = {
            "first_call_seconds": first_time, "warm_seconds": warm_time,
            "air_rmse_k": float(jnp.sqrt(weighted_mse(
                grid.to_nodal(final.dynamics.temperature_variation),
                grid.to_nodal(target.dynamics.temperature_variation), weights))),
            "surface_rmse_k": float(jnp.sqrt(weighted_mse(final.surface_temperature, target.surface_temperature, weights))),
            "relative_atmospheric_mass_drift": float((mass-initial_mass)/initial_mass),
            "instantaneous_radiation_budget_max_abs_w_m2": float(jnp.max(jnp.abs(radiation_budget_residual(flux)))),
            "failed": not bool(jnp.all(jnp.isfinite(air)) & jnp.all((air > 50) & (air < 400))
                               & jnp.all((final.surface_temperature > 50) & (final.surface_temperature < 400)))
                      or not all(np.isfinite(a).all() for a in jax.tree.leaves(final)),
        }
    # Reference substitution is tested in the unit suite; report the same flux
    # accounting diagnostic for the conventional simulation here.
    reference_flux = two_stream_radiative_fluxes(target, coords, specs, BODY, forcing)
    report["physical_radiation_budget_max_abs_w_m2"] = float(jnp.max(jnp.abs(radiation_budget_residual(reference_flux))))
    return report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/neural_framework/radiation")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--columns", type=int, default=96)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--fine-tune-epochs", type=int, default=20)
    p.add_argument("--steps", type=int, default=12)
    p.add_argument("--long-steps", type=int, default=48)
    p.add_argument("--coupled-steps", type=int, default=2)
    p.add_argument("--ames", action="store_true", help="use Ames correlated-k rather than compact reference radiation")
    args = p.parse_args()
    if min(args.layers, args.columns, args.epochs, args.fine_tune_epochs, args.steps,
           args.long_steps, args.coupled_steps) < 1 or args.long_steps <= args.steps:
        p.error("counts must be positive and long-steps must exceed steps")
    jax.config.update("jax_enable_x64", True)
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    boundaries, forcing = configuration(args.layers, args.ames)
    adapter = NeuralRadiation(boundaries, forcing.stefan_boltzmann, compact_shortwave=not args.ames)
    splits = {name: generate_columns(args.seed+i, count, args.layers, extrapolation=name == "extrapolation")
              for i, (name, count) in enumerate([
                  ("train", args.columns), ("validation", max(8, args.columns//4)),
                  ("test", max(8, args.columns//4)), ("extrapolation", max(8, args.columns//4))])}
    norm = fit_normalization(column_features(splits["train"], boundaries), split="train")
    targets = {name: reference_fluxes(x, boundaries, BODY, forcing) for name, x in splits.items()}
    flux_array = lambda f: jnp.stack(f[:4])

    def local_loss(split):
        x, target = splits[split], targets[split]
        reference_heating = heating_rates(target, x.surface_pressure_pa, boundaries, BODY, forcing.thermal_inertia)[0]
        def loss(params):
            prediction = adapter.predict(params, x, norm)
            heating = heating_rates(prediction, x.surface_pressure_pa, boundaries, BODY, forcing.thermal_inertia)[0]
            return (weighted_mse(flux_array(prediction)/100, flux_array(target)/100)
                    + .1*weighted_mse(heating/.01, reference_heating/.01))
        return loss

    params = adapter.init(jax.random.PRNGKey(args.seed))
    local, local_history = fit(params, local_loss("train"), local_loss("validation"), args.epochs, .003)
    print("Local fitting complete", flush=True)

    def predict_trajectory(x, params, steps, learned):
        flux_fn = (lambda p, inputs: adapter.predict(p, inputs, norm)) if learned else (
            lambda p, inputs: reference_fluxes(inputs, boundaries, BODY, forcing))
        step = make_column_step(x, boundaries, BODY, forcing.thermal_inertia, flux_fn)
        return rollout(step, params, initial_column(x), steps, save_every=max(1, steps//4), remat=True)

    reference_trajectories = {name: jax.jit(lambda x: predict_trajectory(x, (), args.steps, False))(splits[name])
                              for name in ("train", "validation")}

    def trajectory_loss(split):
        reference = reference_trajectories[split].trajectory
        def loss(params):
            prediction = predict_trajectory(splits[split], params, args.steps, True).trajectory
            return (weighted_mse(prediction.air_temperature_k, reference.air_temperature_k)
                    + weighted_mse(prediction.surface_temperature_k, reference.surface_temperature_k))
        return loss

    fine, fine_history = fit(local, trajectory_loss("train"), trajectory_loss("validation"),
                             args.fine_tune_epochs, .0003)
    control, control_history = fit(local, local_loss("train"), local_loss("validation"),
                                   args.fine_tune_epochs, .0003, time_budget=fine_history["training_seconds"])
    print("Trajectory fitting and local continuation complete", flush=True)
    models = {"local": local, "trajectory": fine, "continued_local": control}
    metadata = {"feature_schema": feature_schema(args.layers), "physics": {
        "sigma_boundaries": boundaries, "body": dataclasses.asdict(BODY),
        "forcing": dataclasses.asdict(forcing), "compact_shortwave": adapter.compact_shortwave,
        "column_dt_seconds": 30., "fixed_pressure_dust_illumination": True}}
    for name, values in models.items():
        save_checkpoint(out / f"{name}.npz", values, norm, adapter.model, metadata=metadata)
        restored, _ = load_checkpoint(out / f"{name}.npz", adapter.model, expected_metadata=metadata)
        assert all(np.array_equal(a, b) for a, b in zip(jax.tree.leaves(values), jax.tree.leaves(restored)))

    report = {"provenance": provenance(args), "metadata": metadata,
              "training": {"local": local_history, "trajectory": fine_history, "continued_local": control_history},
              "split_policy": "independent whole columns; held-out dust [1,1.5] versus training [0,.8]",
              "selection": "validation only, includes input checkpoint; no test-driven tuning",
              "column_evaluation": {}}
    artifacts = {}
    for split in ("test", "extrapolation"):
        x = splits[split]
        result = {}
        for steps in (args.steps, args.long_steps):
            physical_fn = jax.jit(lambda: predict_trajectory(x, (), steps, False))
            reference, physical_compile = synchronized_call(physical_fn)
            _, physical_seconds = synchronized_call(physical_fn)
            result[str(steps)] = {"physical_first_call_seconds": physical_compile,
                                  "physical_warm_seconds": physical_seconds, "models": {}}
            predicted_fn = jax.jit(lambda values: predict_trajectory(x, values, steps, True))
            for name, values in models.items():
                prediction, compile_seconds = synchronized_call(predicted_fn, values)
                _, seconds = synchronized_call(predicted_fn, values)
                final = prediction.final_state
                residual = (column_energy(final, x, boundaries, BODY, forcing.thermal_inertia)
                            - column_energy(initial_column(x), x, boundaries, BODY, forcing.thermal_inertia)
                            - final.external_energy_j_m2)
                finite = (jnp.all(jnp.isfinite(prediction.trajectory.air_temperature_k), axis=(0, 1, 3))
                          & jnp.all(jnp.isfinite(prediction.trajectory.surface_temperature_k), axis=(0, 2)))
                bounded = (jnp.all((prediction.trajectory.air_temperature_k > 50)
                                   & (prediction.trajectory.air_temperature_k < 400), axis=(0, 1, 3))
                           & jnp.all((prediction.trajectory.surface_temperature_k > 50)
                                     & (prediction.trajectory.surface_temperature_k < 400), axis=(0, 2)))
                result[str(steps)]["models"][name] = {
                    "air_rmse_k": float(jnp.sqrt(weighted_mse(final.air_temperature_k, reference.final_state.air_temperature_k))),
                    "surface_rmse_k": float(jnp.sqrt(weighted_mse(final.surface_temperature_k, reference.final_state.surface_temperature_k))),
                    "budget_max_abs_j_m2": float(jnp.max(jnp.abs(residual))),
                    "failed_columns": int(jnp.sum(~(finite & bounded))), "total_columns": int(finite.size),
                    "first_call_seconds": compile_seconds, "warm_seconds": seconds,
                }
                artifacts[f"{split}_{steps}_{name}_air"] = np.asarray(prediction.trajectory.air_temperature_k)
                artifacts[f"{split}_{steps}_{name}_surface"] = np.asarray(prediction.trajectory.surface_temperature_k)
            artifacts[f"{split}_{steps}_reference_air"] = np.asarray(reference.trajectory.air_temperature_k)
            artifacts[f"{split}_{steps}_reference_surface"] = np.asarray(reference.trajectory.surface_temperature_k)
            artifacts[f"{split}_{steps}_sample_steps"] = np.asarray(reference.steps)
        ref_call = jax.jit(lambda: reference_fluxes(x, boundaries, BODY, forcing))
        _, reference_first = synchronized_call(ref_call)
        _, reference_warm = synchronized_call(ref_call)
        local_metrics = {"reference_first_call_seconds": reference_first, "reference_warm_seconds": reference_warm}
        pred_call = jax.jit(lambda values: adapter.predict(values, x, norm))
        for name, values in models.items():
            flux, first = synchronized_call(pred_call, values)
            _, warm = synchronized_call(pred_call, values)
            local_metrics[name] = {"flux_rmse_w_m2": float(jnp.sqrt(weighted_mse(flux_array(flux), flux_array(targets[split])))),
                                   "first_call_seconds": first, "warm_seconds": warm,
                                   "budget_max_abs_w_m2": float(jnp.max(jnp.abs(radiation_budget_residual(flux))))}
        result["local_flux"] = local_metrics
        report["column_evaluation"][split] = result
    np.savez_compressed(out / "predictions.npz", **artifacts)
    # Write column results before the more expensive coupled execution check.
    report["coupled_status"] = "pending"
    write_report(out, report, "Neural radiation experiment")
    print("Column evaluation complete; checking T21 coupled execution", flush=True)
    report["coupled"] = coupled_evaluation(adapter, norm, models, forcing, args.layers, args.coupled_steps)
    report["coupled_status"] = "complete"
    write_report(out, report, "Neural radiation experiment")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, history in report["training"].items():
        axes[0].plot([x["validation_after_update"] for x in history["history"]], label=name)
    axes[0].set(xlabel="Updates", ylabel="Validation objective (different losses)", yscale="log")
    axes[0].legend()
    for name in models:
        axes[1].plot([args.steps*30, args.long_steps*30],
                     [report["column_evaluation"]["test"][str(n)]["models"][name]["air_rmse_k"]
                      for n in (args.steps, args.long_steps)], marker="o", label=name)
    axes[1].set(xlabel="Lead (seconds)", ylabel="Held-out atmospheric RMSE (K)")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(out / "training_and_rollouts.png", dpi=160)
    plt.close(fig)
    print(f"Wrote {out}/report.md")


if __name__ == "__main__":
    main()
