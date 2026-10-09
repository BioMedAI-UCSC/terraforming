#!/usr/bin/env python3
"""Recover a bounded bulk sensible-exchange multiplier through column trajectories."""
import argparse
import time

import numpy as np
from scipy.optimize import minimize_scalar

from src.framework.gcm._dinosaur import jax, jnp
from src.framework.gcm.learning import rollout
from src.framework.neural import reference_fluxes, weighted_mse, directional_gradient_check
from src.framework.neural.training import adam_init, adam_update
from src.framework.neural.columns import initial_column, make_column_step, column_energy
from common import BODY, configuration, generate_columns, provenance, synchronized_call, write_report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", default="outputs/neural_framework/recovery")
    p.add_argument("--iterations", type=int, default=120)
    p.add_argument("--steps", type=int, default=24)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    if min(args.iterations, args.steps) < 1:
        p.error("iterations and steps must be positive")
    jax.config.update("jax_enable_x64", True)
    boundaries, forcing = configuration()
    train, test = [generate_columns(args.seed+i, 8, 4) for i in (0, 1)]
    truth = 1.6
    dt, capacity = 30., forcing.thermal_inertia

    def make_predict(inputs):
        step = make_column_step(inputs, boundaries, BODY, capacity,
                                lambda params, x: reference_fluxes(x, boundaries, BODY, forcing),
                                dt_seconds=dt, exchange_fn=lambda multiplier: 2.0*multiplier)
        return jax.jit(lambda value: rollout(step, value, initial_column(inputs), args.steps,
                                            save_every=max(1, args.steps//4)))

    predict, heldout = make_predict(train), make_predict(test)
    reference, reference_seconds = synchronized_call(predict, truth)
    test_reference, _ = synchronized_call(heldout, truth)

    def error(result, target):
        return (weighted_mse(result.trajectory.air_temperature_k, target.trajectory.air_temperature_k)
                + weighted_mse(result.trajectory.surface_temperature_k, target.trajectory.surface_temperature_k))

    loss = jax.jit(lambda value: error(predict(value), reference))
    checks = {str(x): directional_gradient_check(loss, jnp.asarray(x), jnp.asarray(1.0))
              for x in (.5, 1., 2.5)}

    @jax.jit
    def update(value, opt):
        loss_value, gradient = jax.value_and_grad(loss)(value)
        candidate, opt = adam_update(value, gradient, opt, learning_rate=.05, clip_norm=10.)
        return jnp.clip(candidate, .2, 4.), opt, loss_value, gradient

    seed_value = jnp.asarray(.5, dtype=jnp.float64)
    _, compile_seconds = synchronized_call(update, seed_value, adam_init(seed_value))
    runs = []
    for start_value in (.5, 1., 3.):
        value = jnp.asarray(start_value, dtype=jnp.float64)
        opt = adam_init(value)
        history = []
        start = time.perf_counter()
        for iteration in range(args.iterations):
            value, opt, objective, gradient = update(value, opt)
            history.append({"iteration": iteration, "loss_before_update": float(objective),
                            "multiplier_after_update": float(value), "gradient": float(gradient)})
            if not np.isfinite([float(value), float(objective), float(gradient)]).all():
                break
        seconds = time.perf_counter()-start
        test_result = heldout(value)
        final = test_result.final_state
        residual = (column_energy(final, test, boundaries, BODY, capacity)
                    - column_energy(initial_column(test), test, boundaries, BODY, capacity)
                    - final.external_energy_j_m2)
        runs.append({"initial_multiplier": start_value, "estimated_multiplier": float(value),
                     "absolute_parameter_error": abs(float(value)-truth), "training_loss": float(loss(value)),
                     "heldout_loss": float(error(test_result, test_reference)), "seconds": seconds,
                     "gradient_objective_evaluations": len(history), "history": history,
                     "budget_max_abs_j_m2": float(jnp.max(jnp.abs(residual))),
                     "failed": not bool(jnp.all(jnp.isfinite(final.air_temperature_k)))})
    baseline_start = time.perf_counter()
    baseline = minimize_scalar(lambda x: float(loss(x)), bounds=(.2, 4.), method="bounded",
                               options={"maxiter": args.iterations, "xatol": 1e-6})
    baseline_seconds = time.perf_counter()-baseline_start
    report = {"provenance": provenance(args), "true_multiplier": truth,
              "definition": "multiplier of the 2 W/m²/K bulk sensible conductance; not the aerodynamic GCM closure",
              "forcing": "fixed illumination, dust and pressure per column", "gradient_checks": checks,
              "reference_first_call_seconds": reference_seconds, "update_first_call_seconds": compile_seconds,
              "gradient_runs": runs, "derivative_free": {
                  "estimated_multiplier": float(baseline.x), "training_loss": float(baseline.fun),
                  "heldout_loss": float(error(heldout(baseline.x), test_reference)),
                  "objective_evaluations": int(baseline.nfev), "seconds": baseline_seconds,
                  "success": bool(baseline.success)}}
    write_report(args.output, report, "Physical parameter recovery")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    for run in runs:
        plt.plot([x["multiplier_after_update"] for x in run["history"]], label=str(run["initial_multiplier"]))
    plt.axhline(truth, color="black", linestyle="--", label="reference")
    plt.xlabel("Gradient updates")
    plt.ylabel("Surface-exchange multiplier")
    plt.legend()
    plt.savefig(f"{args.output}/recovery.png", dpi=160)
    plt.close()
    print(f"Wrote {args.output}/report.md")


if __name__ == "__main__":
    main()
