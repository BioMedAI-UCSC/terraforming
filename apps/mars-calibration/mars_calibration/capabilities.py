#!/usr/bin/env python3
"""JAX-MD-style capability experiments for the coupled AEGIS-Mars GCM.

Three experiments reuse the verified coupled step and the MY32 Ls45 calibration
inputs (the same restart used by the scalar CO2-opacity sensitivity run):

  sensitivity-map   One reverse-mode pass gives the gradient of a scalar climate
                    diagnostic with respect to an entire surface map (albedo and
                    thermal inertia, 64x32 values each), checked against centered
                    finite differences along random directions.
  optimize-albedo   Gradient-based design of a bounded, smooth surface-albedo
                    intervention that warms a target region, compared with
                    matched-norm hand-designed interventions.
  ensemble          jax.vmap runs many coupled trajectories in one compiled call;
                    reports throughput versus batch size and ensemble spread.

Every run writes report.json (metrics, gradient checks, timings, environment,
input hashes), arrays (.npz), and figures (.png/.pdf).  Use --smoke for a fast
CPU check of the full pipeline at tiny horizons.

Run as ``mars-capabilities`` (installed entry point) or via
``scripts/differentiable_capabilities.py`` from the repository root.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
import time

import numpy as np

from jax.flatten_util import ravel_pytree

from . import driver as d
from . import paper as p
from .launch import FILES, sha256

jax, jnp = d.jax, d.jnp
SOL_SECONDS = d.MARS_BODY_3D.rotation_period_s
RTOL, ATOL = 1e-4, 1e-8  # same criterion as the paper's derivative-check table


# --------------------------------------------------------------------------- #
# shared setup
# --------------------------------------------------------------------------- #
class Setup:
    """Model, grid geometry, masks, and helpers shared by all experiments."""

    def __init__(self, inputs: Path, dt: float):
        self.inputs = inputs
        self.dt = dt
        self.model = p.Model(inputs)
        grid = self.model.coords.horizontal
        self.lon = np.degrees(np.asarray(grid.longitudes))  # (64,)
        self.lat = np.degrees(np.asarray(grid.latitudes))   # (32,)
        self.weights = np.asarray(self.model.weights)        # (64, 32), sums to 1
        self.elevation = np.asarray(d.regrid_to_nodal(self.model.coords))
        self.pa = float(self.model.specs.dimensionalize(1.0, d.scales.units.pascal).magnitude)
        self.lat2d = np.broadcast_to(self.lat[None, :], self.weights.shape)
        self.lon2d = np.broadcast_to(self.lon[:, None], self.weights.shape)

    def mask(self, lat_range, lon_range=None):
        m = (self.lat2d >= lat_range[0]) & (self.lat2d <= lat_range[1])
        if lon_range is not None:
            m &= (self.lon2d >= lon_range[0]) & (self.lon2d <= lon_range[1])
        if not m.any():
            raise ValueError("empty region mask")
        return m

    def regional_mean(self, field, mask):
        w = jnp.asarray(self.weights * mask)
        return jnp.sum(field * w) / jnp.sum(w)

    def step(self, forcing):
        return d._build_step(self.model.coords, self.model.specs, forcing, self.dt,
                             orography=self.model.orography)

    def forcing(self, **replacements):
        return dataclasses.replace(self.model.forcing, **replacements)

    def input_hashes(self):
        return {name: sha256(self.inputs / name) for name in FILES if (self.inputs / name).exists()}


def steps_for(sols: float, dt: float) -> int:
    n = int(round(sols * SOL_SECONDS / dt))
    if n < 1:
        raise ValueError("horizon must contain at least one step")
    return n


def timed(fn, *args):
    started = time.perf_counter()
    out = jax.block_until_ready(fn(*args))
    return out, time.perf_counter() - started


def check_direction(f, grad_dot, x, direction, epsilons):
    """Compare an AD directional derivative with centered finite differences."""
    rows = []
    for eps in epsilons:
        fd = (float(f(x + eps * direction)) - float(f(x - eps * direction))) / (2 * eps)
        err = abs(grad_dot - fd)
        rows.append({"epsilon": eps, "autodiff": grad_dot, "finite_difference": fd,
                     "absolute_error": err,
                     "relative_error": err / max(abs(fd), abs(grad_dot), 1e-30),
                     "pass": bool(np.isfinite([grad_dot, fd]).all() and err <= ATOL + RTOL * abs(fd))})
    return rows


def write_report(out: Path, setup: Setup, args, payload: dict):
    out.mkdir(parents=True, exist_ok=True)
    report = {"experiment": args.command, "arguments": {k: (str(v) if isinstance(v, Path) else v)
                                                       for k, v in vars(args).items()},
              "dt_seconds": setup.dt, "inputs": {"directory": str(setup.inputs),
                                                  "sha256": setup.input_hashes()},
              "environment": p.environment(), **payload}
    (out / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "figure.dpi": 130,
                         "savefig.bbox": "tight", "savefig.dpi": 300})
    return plt


def save_fig(fig, out: Path, stem: str):
    for ext in ("png", "pdf"):
        fig.savefig(out / f"{stem}.{ext}")


def map_panel(ax, setup, field, title, cbar_label, *, diverging=True, fig=None):
    plt = _plt()
    data = np.asarray(field).T
    if diverging:
        lim = float(np.nanmax(np.abs(data))) or 1.0
        mesh = ax.pcolormesh(setup.lon, setup.lat, data, cmap="RdBu_r", vmin=-lim, vmax=lim,
                             shading="nearest")
    else:
        mesh = ax.pcolormesh(setup.lon, setup.lat, data, cmap="viridis", shading="nearest")
    ax.contour(setup.lon, setup.lat, setup.elevation.T, levels=[0.0], colors="k",
               linewidths=0.5, alpha=0.6)
    ax.set_title(title)
    ax.set_xlabel("Longitude (°E)")
    ax.set_ylabel("Latitude (°N)")
    (fig or plt.gcf()).colorbar(mesh, ax=ax, label=cbar_label, shrink=0.85)


# --------------------------------------------------------------------------- #
# 1. adjoint sensitivity maps
# --------------------------------------------------------------------------- #
def sensitivity_map(args):
    setup = Setup(args.inputs, args.dt)
    n = steps_for(args.sols, args.dt)
    polar = setup.mask((-90.0, args.polar_lat))

    def final_state(fields):
        f = setup.forcing(albedo=fields["albedo"],
                          surface_thermal_inertia_tiu=fields["thermal_inertia"])
        step = setup.step(f)
        return jax.lax.fori_loop(0, n, lambda _, s: step(s), setup.model.initial)

    objectives = {
        "global_mean_surface_temperature_k":
            lambda s: jnp.sum(s.surface_temperature[0] * jnp.asarray(setup.weights)),
        "south_polar_co2_frost_pa":
            lambda s: setup.regional_mean(s.co2_ice[0] * setup.pa, polar),
    }
    x0 = {"albedo": setup.model.forcing.albedo,
          "thermal_inertia": setup.model.forcing.surface_thermal_inertia_tiu}
    # Perturbations are reported per 0.01 albedo and per 10 TIU, typical local contrasts.
    unit = {"albedo": 0.01, "thermal_inertia": 10.0}
    rng = np.random.default_rng(args.seed)

    results, arrays, checks, timings = {}, {}, [], {}
    for name, obj in objectives.items():
        J = lambda fields, obj=obj: obj(final_state(fields))  # noqa: E731
        vg = jax.jit(jax.value_and_grad(J))
        (value, grad), first = timed(vg, x0)
        (_, _), warm = timed(vg, x0)
        f_jit = jax.jit(J)
        (_, fwd_warm) = timed(f_jit, x0)
        (_, fwd_warm) = timed(f_jit, x0)
        timings[name] = {"reverse_first_call_seconds": first, "reverse_warm_seconds": warm,
                         "forward_warm_seconds": fwd_warm,
                         "reverse_to_forward_cost_ratio": warm / fwd_warm,
                         "finite_difference_equivalent_forward_calls": 2 * sum(
                             int(np.asarray(v).size) for v in x0.values())}
        results[name] = {"value": float(value)}
        for field in x0:
            g = np.asarray(grad[field])
            arrays[f"{name}__d_{field}"] = g
            area_weighted = g / setup.weights  # sensitivity density per unit field change per unit area fraction
            results[name][field] = {
                "sum_of_gradient_uniform_unit_change": float(g.sum()),
                "response_to_uniform_change_of_" + str(unit[field]): float(g.sum() * unit[field]),
                "max_abs_gradient": float(np.abs(g).max()),
                "argmax_lat_lon_deg": [float(setup.lat2d.flat[np.abs(g).argmax()]),
                                       float(setup.lon2d.flat[np.abs(g).argmax()])],
                "max_abs_density": float(np.abs(area_weighted).max()),
            }
            arrays[f"{name}__density_{field}"] = area_weighted
            for k in range(args.directions):
                direction = rng.normal(size=g.shape) * float(unit[field])
                grad_dot = float(np.sum(g * direction))

                def f_dir(xdir, field=field, name=name):
                    return f_jit({**x0, field: xdir})
                for row in check_direction(f_dir, grad_dot, np.asarray(x0[field]), direction,
                                           args.epsilons):
                    checks.append({"objective": name, "input": field, "direction": k, **row})

    Path(args.output).mkdir(parents=True, exist_ok=True)
    np.savez_compressed(Path(args.output) / "sensitivity.npz", lon=setup.lon, lat=setup.lat, **arrays)

    # Figure: sensitivity density (response per unit field change per unit area fraction).
    plt = _plt()
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.2))
    labels = {"global_mean_surface_temperature_k": ("Global-mean surface temperature", "K"),
              "south_polar_co2_frost_pa": ("South-polar CO$_2$ frost", "Pa")}
    for i, name in enumerate(objectives):
        for j, field in enumerate(x0):
            title, u = labels[name]
            scale = unit[field]
            dens = arrays[f"{name}__density_{field}"] * scale
            fu = "0.01 albedo" if field == "albedo" else "10 TIU"
            map_panel(axes[i, j], setup, dens,
                      f"∂({title})/∂({field.replace('_', ' ')})",
                      f"{u} per {fu} (area-normalized)", fig=fig)
    fig.suptitle(f"Adjoint sensitivity maps after {n * args.dt / SOL_SECONDS:.3g} sol ({n} steps); "
                 "one reverse-mode pass per objective", fontsize=10)
    fig.tight_layout()
    save_fig(fig, Path(args.output), "sensitivity_maps")
    plt.close(fig)

    payload = {"steps": n, "sols": n * args.dt / SOL_SECONDS,
               "polar_region": {"lat_max_deg": args.polar_lat},
               "results": results, "timings": timings, "gradient_checks": checks,
               "all_gradient_checks_pass": bool(all(c["pass"] for c in checks)),
               "max_relative_gradient_error": float(max(c["relative_error"] for c in checks)),
               "status": "pass" if all(c["pass"] for c in checks) else "fail",
               "interpretation": ("Each map is the full gradient of one scalar diagnostic with "
                                  "respect to a 64x32 surface field, obtained from a single "
                                  "reverse-mode pass through the coupled rollout.")}
    write_report(Path(args.output), setup, args, payload)
    return payload


# --------------------------------------------------------------------------- #
# 2. gradient-based surface-albedo intervention design
# --------------------------------------------------------------------------- #
def optimize_albedo(args):
    """Per-pixel albedo design under an explicit budget, started from a hand design.

    The design variable is the full 64x32 albedo-change field.  The feasible set is
    a per-pixel bound |da| <= max_change together with an area-weighted L2 budget
    ||da||_w <= r.  Optimization starts from the natural hand design (uniform
    darkening of the target region, scaled to the budget) and performs projected
    gradient ascent with backtracking, accepting only improvements.  The result is
    therefore at least as good as the hand design at an identical budget; the gain
    measures what the adjoint gradient adds.
    """
    setup = Setup(args.inputs, args.dt)
    n = steps_for(args.sols, args.dt)
    region = setup.mask(tuple(args.region_lat), tuple(args.region_lon))
    w_np = setup.weights
    area = float(np.sum(w_np * region))
    # Budget: the region-box design at half the per-pixel bound.
    radius = args.budget_fraction * args.max_change * np.sqrt(area)
    a0 = setup.model.forcing.albedo

    def time_mean_ts(delta):
        step = setup.step(setup.forcing(albedo=a0 + delta))

        def body(_, carry):
            s, acc = carry
            s = step(s)
            return s, acc + s.surface_temperature[0]
        zero = jnp.zeros_like(setup.model.initial.surface_temperature[0])
        return jax.lax.fori_loop(0, n, body, (setup.model.initial, zero))[1] / n

    def regional(delta):
        return setup.regional_mean(time_mean_ts(delta), region)

    f_fields = jax.jit(time_mean_ts)
    f_region = jax.jit(regional)
    vg = jax.jit(jax.value_and_grad(regional))

    def norm(delta):
        return float(np.sqrt(np.sum(w_np * delta * delta)))

    def project(delta):
        delta = np.clip(delta, -args.max_change, args.max_change)
        size = norm(delta)
        return delta * (radius / size) if size > radius else delta

    def matched(pattern):
        pattern = np.asarray(pattern, dtype=float)
        return -pattern * radius / norm(pattern)  # darkening scaled to the budget

    box = matched(region.astype(float))
    uniform = matched(np.ones_like(w_np))

    # Gradient checks at the hand design, along random per-pixel directions.
    (value0, grad0), first = timed(vg, jnp.asarray(box))
    rng = np.random.default_rng(args.seed)
    checks = []
    for k in range(args.directions):
        direction = rng.normal(size=box.shape) * 0.01
        checks += [{"direction": k, **row} for row in check_direction(
            f_region, float(np.sum(np.asarray(grad0) * direction)), box, direction, args.epsilons)]

    delta, value, grad = box, float(value0), np.asarray(grad0)
    history = [{"iteration": 0, "regional_mean_ts_k": value, "step": None,
                "accepted": True, "seconds": first}]
    step_size = 1.0
    for it in range(1, args.iterations + 1):
        started = time.perf_counter()
        # Riesz representer of the gradient in the area-weighted L2 geometry.
        direction = grad / w_np
        direction /= norm(direction)
        accepted = False
        for _ in range(args.backtracks):
            candidate = project(delta + step_size * radius * direction)
            cand_value = float(f_region(jnp.asarray(candidate)))
            if cand_value > value:
                accepted = True
                break
            step_size *= 0.5
        if accepted:
            delta = candidate
            value, grad = vg(jnp.asarray(delta))
            value, grad = float(value), np.asarray(grad)
            step_size = min(1.0, step_size * 2.0)
        history.append({"iteration": it, "regional_mean_ts_k": value, "step": step_size,
                        "accepted": accepted, "seconds": time.perf_counter() - started})
        print(json.dumps(history[-1]), flush=True)
        if not accepted:
            break

    candidates = {"baseline": np.zeros_like(box), "optimized": delta,
                  "regional_box": box, "global_uniform": uniform}
    base = np.asarray(f_fields(jnp.zeros_like(a0)))
    fields, evaluation = {}, {}
    for key, da in candidates.items():
        tm = np.asarray(f_fields(jnp.asarray(da)))
        fields[key] = tm
        evaluation[key] = {
            "regional_warming_k": float(np.sum(w_np * region * (tm - base)) / area),
            "global_warming_k": float(np.sum(w_np * (tm - base))),
            "outside_region_warming_k": float(np.sum(w_np * ~region * (tm - base))
                                              / np.sum(w_np * ~region)),
            "albedo_change_norm": norm(da),
            "albedo_change_max_abs": float(np.abs(da).max()),
            "albedo_range": [float(np.min(np.asarray(a0) + da)), float(np.max(np.asarray(a0) + da))],
        }
    gain = (evaluation["optimized"]["regional_warming_k"]
            / evaluation["regional_box"]["regional_warming_k"] - 1.0)

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "optimization.npz", lon=setup.lon, lat=setup.lat, region=region,
                        **{f"delta_albedo_{k}": v for k, v in candidates.items()},
                        **{f"time_mean_ts_{k}": v for k, v in fields.items()})

    plt = _plt()
    fig = plt.figure(figsize=(12, 6.5))
    gs = fig.add_gridspec(2, 3)
    ax = fig.add_subplot(gs[0, 0])
    base_regional = float(np.sum(w_np * region * base) / area)
    ax.plot([h["iteration"] for h in history],
            [h["regional_mean_ts_k"] - base_regional for h in history], marker="o", ms=3,
            label="gradient design")
    ax.axhline(evaluation["regional_box"]["regional_warming_k"], color="#E69F00", ls="--",
               label="hand design (region box)")
    ax.axhline(evaluation["global_uniform"]["regional_warming_k"], color="#999999", ls=":",
               label="uniform darkening")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Regional warming (K)")
    ax.set_title("Equal albedo-change budget")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    map_panel(fig.add_subplot(gs[0, 1]), setup, delta, "Gradient-designed Δα", "Δα", fig=fig)
    map_panel(fig.add_subplot(gs[0, 2]), setup, fields["optimized"] - base,
              "Time-mean ΔT$_s$ (gradient design)", "K", fig=fig)
    map_panel(fig.add_subplot(gs[1, 0]), setup, box, "Hand design Δα (region box)", "Δα", fig=fig)
    map_panel(fig.add_subplot(gs[1, 1]), setup, fields["regional_box"] - base,
              "Time-mean ΔT$_s$ (hand design)", "K", fig=fig)
    ax = fig.add_subplot(gs[1, 2])
    keys = ["optimized", "regional_box", "global_uniform"]
    ax.bar(["gradient\ndesign", "hand\ndesign", "uniform"],
           [evaluation[k]["regional_warming_k"] for k in keys],
           color=["#0072B2", "#E69F00", "#999999"])
    ax.set_ylabel("Regional warming (K)")
    ax.set_title("Same budget, same bound")
    ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"Per-pixel albedo design (2048 parameters) over {n * args.dt / SOL_SECONDS:.2g} sol "
                 f"({n} steps); target {args.region_lat[0]:g}–{args.region_lat[1]:g}°N, "
                 f"{args.region_lon[0]:g}–{args.region_lon[1]:g}°E", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out, "albedo_intervention")
    plt.close(fig)

    payload = {"steps": n, "sols": n * args.dt / SOL_SECONDS,
               "region": {"lat_deg": args.region_lat, "lon_deg": args.region_lon,
                          "area_fraction": area},
               "design_parameters": int(box.size), "max_change": args.max_change,
               "budget_area_weighted_l2": radius, "history": history, "evaluation": evaluation,
               "relative_gain_over_hand_design": gain,
               "gradient_checks": checks,
               "all_gradient_checks_pass": bool(all(r["pass"] for r in checks)),
               "status": "pass" if all(r["pass"] for r in checks) else "fail"}
    write_report(out, setup, args, payload)
    return payload


# --------------------------------------------------------------------------- #
# 3. vmapped ensembles
# --------------------------------------------------------------------------- #
def ensemble(args):
    setup = Setup(args.inputs, args.dt)
    n = steps_for(args.sols, args.dt)
    samples = max(1, n // args.samples)
    w = jnp.asarray(setup.weights)

    def run(state, co2):
        step = setup.step(setup.forcing(ames_co2_longwave_opacity_scale=co2))

        def sample(s, _):
            s = jax.lax.fori_loop(0, samples, lambda __, x: step(x), s)
            return s, jnp.sum(s.surface_temperature[0] * w)
        return jax.lax.scan(sample, state, None, length=args.samples)[1]

    def members(batch, key):
        rng = np.random.default_rng(key)
        states = jax.tree_util.tree_map(lambda x: jnp.broadcast_to(x, (batch,) + x.shape),
                                        setup.model.initial)
        noise = rng.normal(size=(batch,) + setup.model.initial.surface_temperature.shape)
        states = states._replace(surface_temperature=states.surface_temperature
                                 + jnp.asarray(noise * args.noise_k))
        co2 = jnp.asarray(np.linspace(args.co2_range[0], args.co2_range[1], batch)
                          if batch > 1 else [1.0])
        return states, co2

    batched = jax.jit(jax.vmap(run))
    single = jax.jit(run)
    scaling = []
    for batch in args.batches:
        states, co2 = members(batch, args.seed)
        try:
            _, first = timed(batched, states, co2)
            _, warm = timed(batched, states, co2)
        except Exception as error:  # out-of-memory or similar: record and stop scaling
            scaling.append({"batch": batch, "status": "failed", "error": str(error)[:300]})
            break
        simulated_sols = batch * n * args.dt / SOL_SECONDS
        scaling.append({"batch": batch, "status": "ok", "first_call_seconds": first,
                        "warm_seconds": warm,
                        "aggregate_simulated_sols_per_wall_hour": simulated_sols / warm * 3600,
                        "memory": _memory_stats()})
        print(json.dumps(scaling[-1]), flush=True)

    one_state, one_co2 = members(1, args.seed)
    single_args = (jax.tree_util.tree_map(lambda x: x[0], one_state), one_co2[0])
    _, _ = timed(single, *single_args)
    _, single_warm = timed(single, *single_args)
    for row in scaling:
        if row["status"] == "ok":
            row["speedup_vs_serial"] = row["batch"] * single_warm / row["warm_seconds"]

    # Ensemble spread for the largest successful batch.
    ok = [r for r in scaling if r["status"] == "ok"]
    spread = None
    if ok:
        batch = ok[-1]["batch"]
        states, co2 = members(batch, args.seed)
        traj = np.asarray(batched(states, co2))  # (batch, samples)
        times = (np.arange(1, args.samples + 1) * samples * args.dt / SOL_SECONDS)
        spread = {"batch": batch, "sols": times.tolist(), "co2_scale": np.asarray(co2).tolist(),
                  "global_mean_ts_k": traj.tolist(),
                  "member_consistency_check": _consistency(single, states, co2, traj, args.check_members)}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    plt = _plt()
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4))
    b = [r["batch"] for r in ok]
    axes[0].plot(b, [r["aggregate_simulated_sols_per_wall_hour"] for r in ok], marker="o",
                 label="vmapped ensemble")
    axes[0].plot(b, [bb * n * args.dt / SOL_SECONDS / (bb * single_warm) * 3600 for bb in b],
                 ls="--", color="gray", label="serial runs")
    axes[0].set_xscale("log", base=2)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Ensemble size")
    axes[0].set_ylabel("Aggregate simulated sols / wall hour")
    axes[0].set_title("Batched throughput")
    axes[0].legend()
    axes[0].grid(alpha=0.3, which="both")
    if spread:
        traj = np.asarray(spread["global_mean_ts_k"])
        cmap = plt.get_cmap("viridis")
        c = np.asarray(spread["co2_scale"])
        for k in range(traj.shape[0]):
            frac = 0.5 if len(c) == 1 else (c[k] - c.min()) / (c.max() - c.min())
            axes[1].plot(spread["sols"], traj[k], color=cmap(frac), lw=1)
        axes[1].set_xlabel("Sols")
        axes[1].set_ylabel("Global-mean T$_s$ (K)")
        axes[1].set_title(f"{spread['batch']}-member ensemble "
                          f"(CO$_2$ opacity {args.co2_range[0]}–{args.co2_range[1]})")
        axes[1].grid(alpha=0.3)
    fig.tight_layout()
    save_fig(fig, out, "ensemble")
    plt.close(fig)

    payload = {"steps": n, "sols": n * args.dt / SOL_SECONDS,
               "single_trajectory_warm_seconds": single_warm, "scaling": scaling,
               "ensemble": spread,
               "status": "pass" if ok and spread and spread["member_consistency_check"]["pass"]
               else "fail"}
    write_report(out, setup, args, payload)
    return payload


def _consistency(single, states, co2, traj, count):
    """Batched members must match independent single runs."""
    worst = 0.0
    for k in range(min(count, len(co2))):
        member = np.asarray(single(jax.tree_util.tree_map(lambda x: x[k], states), co2[k]))
        worst = max(worst, float(np.max(np.abs(member - traj[k]))))
    return {"members_checked": int(min(count, len(co2))), "max_abs_difference_k": worst,
            "pass": bool(worst <= 1e-8)}


def _memory_stats():
    try:
        stats = jax.devices()[0].memory_stats() or {}
        return {k: int(v) for k, v in stats.items() if k in ("peak_bytes_in_use", "bytes_in_use",
                                                            "bytes_limit")}
    except Exception:  # CPU backends typically do not report
        return {}


# --------------------------------------------------------------------------- #
# 4. neural surface field recovered through the coupled model
# --------------------------------------------------------------------------- #
ALBEDO_BOUNDS = (0.05, 0.5)


def _albedo_from_logit(z):
    lo, hi = ALBEDO_BOUNDS
    return lo + (hi - lo) * jax.nn.sigmoid(z)


def _logit_of_albedo(a):
    lo, hi = ALBEDO_BOUNDS
    f = (np.asarray(a) - lo) / (hi - lo)
    return np.log(f / (1.0 - f))


def _field_features(setup: Setup, frequencies: int):
    """Coordinate features for the neural field: Fourier terms plus MOLA elevation."""
    lam = np.radians(setup.lon2d)
    phi = np.radians(setup.lat2d)
    feats = []
    for k in range(1, frequencies + 1):
        feats += [np.sin(k * lam), np.cos(k * lam), np.sin(k * phi), np.cos(k * phi)]
    elev = setup.elevation
    feats.append((elev - elev.mean()) / elev.std())
    return np.stack(feats, axis=-1)  # (64, 32, F)


def _mlp_init(key, sizes, out_bias):
    params = []
    keys = jax.random.split(key, len(sizes) - 1)
    for i, (m, n) in enumerate(zip(sizes[:-1], sizes[1:])):
        last = i == len(sizes) - 2
        w = (jnp.zeros((m, n)) if last
             else jax.random.normal(keys[i], (m, n)) * jnp.sqrt(2.0 / m))
        b = jnp.full((n,), out_bias) if last else jnp.zeros((n,))
        params.append((w, b))
    return params


def _mlp_apply(params, x):
    for w, b in params[:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = params[-1]
    return (x @ w + b)[..., 0]


def neural_field(args):
    """Recover the TES surface-albedo map from surface temperatures (twin experiment).

    Truth uses the TES albedo map.  Both learners start from the uniform
    area-weighted mean albedo and see only surface-temperature maps sampled along
    a coupled rollout.  The neural field maps (Fourier coordinates, MOLA elevation)
    to albedo; the per-pixel baseline fits 2048 independent values.  Recovered maps
    are then evaluated on a later, unseen time window.
    """
    setup = Setup(args.inputs, args.dt)
    n_train = steps_for(args.sols, args.dt)
    n_test = steps_for(args.test_sols, args.dt)
    every = max(1, args.observe_every)
    n_train -= n_train % every
    n_test -= n_test % every
    if n_train < every or n_test < every:
        raise ValueError("horizons must contain at least one observation interval")
    w = jnp.asarray(setup.weights)
    a_true = setup.model.forcing.albedo
    a_uniform = float(np.sum(setup.weights * np.asarray(a_true)))

    def observe(albedo, total):
        step = setup.step(setup.forcing(albedo=albedo))

        def sample(s, _):
            s = jax.lax.fori_loop(0, every, lambda __, x: step(x), s)
            return s, (s.surface_temperature[0], s.co2_ice[0])
        _, (ts, ice) = jax.lax.scan(sample, setup.model.initial, None, length=total // every)
        return ts, ice

    observe_all = jax.jit(lambda a: observe(a, n_train + n_test))
    truth_ts, truth_ice = (np.asarray(x) for x in observe_all(a_true))
    k_train = n_train // every
    target = jnp.asarray(truth_ts[:k_train])
    observable = np.asarray(truth_ice[:k_train].max(axis=0) <= 0.0)  # frost-free throughout

    def ts_loss(albedo):
        step = setup.step(setup.forcing(albedo=albedo))

        def sample(carry, target_k):
            s, acc = carry
            s = jax.lax.fori_loop(0, every, lambda __, x: step(x), s)
            err = s.surface_temperature[0] - target_k
            return (s, acc + jnp.sum(w * err * err)), None
        (_, total), _ = jax.lax.scan(sample, (setup.model.initial, 0.0), target)
        return total / k_train

    feats = jnp.asarray(_field_features(setup, args.frequencies))
    learners = {
        "neural_field": {
            "init": _mlp_init(jax.random.PRNGKey(args.seed),
                              [feats.shape[-1], *args.hidden, 1],
                              float(_logit_of_albedo(a_uniform))),
            "albedo": lambda p: _albedo_from_logit(_mlp_apply(p, feats)),
            "lr": args.lr_neural},
        "per_pixel": {
            "init": jnp.full(setup.weights.shape, float(_logit_of_albedo(a_uniform))),
            "albedo": _albedo_from_logit,
            "lr": args.lr_pixel},
    }

    def albedo_metrics(albedo):
        a = np.asarray(albedo)
        t = np.asarray(a_true)
        def stats(mask):
            ww = setup.weights * mask
            ww = ww / ww.sum()
            err = a - t
            am, tm = np.sum(ww * a), np.sum(ww * t)
            cov = np.sum(ww * (a - am) * (t - tm))
            corr = cov / np.sqrt(np.sum(ww * (a - am) ** 2) * np.sum(ww * (t - tm) ** 2) + 1e-30)
            return {"rmse": float(np.sqrt(np.sum(ww * err * err))), "correlation": float(corr)}
        return {"all": stats(np.ones_like(t)), "observable": stats(observable),
                "unobservable": stats(~observable) if (~observable).any() else None}

    rng = np.random.default_rng(args.seed)
    results, checks, recovered = {}, [], {}
    b1, b2, eps = 0.9, 0.999, 1e-8
    for name, spec in learners.items():
        loss_fn = lambda p, spec=spec: ts_loss(spec["albedo"](p))  # noqa: E731
        vg = jax.jit(jax.value_and_grad(loss_fn))
        f_loss = jax.jit(loss_fn)
        params = spec["init"]
        flat0, unravel = ravel_pytree(params)

        # Gradient check along one random direction in parameter space.
        (_, g0), first = timed(vg, params)
        g_flat = np.asarray(ravel_pytree(g0)[0])
        direction = rng.normal(size=g_flat.shape)
        direction /= np.linalg.norm(direction)
        checks += [{"learner": name, **row} for row in check_direction(
            lambda x: f_loss(unravel(jnp.asarray(x))), float(g_flat @ direction),
            np.asarray(flat0), direction, args.epsilons)]

        m_t = jax.tree_util.tree_map(jnp.zeros_like, params)
        v_t = jax.tree_util.tree_map(jnp.zeros_like, params)
        history = []
        best = (np.inf, params)
        for it in range(1, args.updates + 1):
            (value, g), seconds = timed(vg, params)
            if float(value) < best[0]:
                best = (float(value), params)
            m_t = jax.tree_util.tree_map(lambda m, gg: b1 * m + (1 - b1) * gg, m_t, g)
            v_t = jax.tree_util.tree_map(lambda v, gg: b2 * v + (1 - b2) * gg * gg, v_t, g)
            params = jax.tree_util.tree_map(
                lambda p, m, v: p - spec["lr"] * (m / (1 - b1 ** it))
                / (jnp.sqrt(v / (1 - b2 ** it)) + eps), params, m_t, v_t)
            if it == 1 or it % args.log_every == 0 or it == args.updates:
                metrics = albedo_metrics(spec["albedo"](params))
                history.append({"update": it, "train_ts_mse_k2": float(value),
                                "albedo_rmse": metrics["all"]["rmse"],
                                "albedo_rmse_observable": metrics["observable"]["rmse"],
                                "seconds": seconds})
                print(json.dumps({"learner": name, **history[-1]}), flush=True)
        final = float(f_loss(params))
        if final < best[0]:
            best = (final, params)
        albedo = np.asarray(spec["albedo"](best[1]))
        recovered[name] = albedo
        ts, _ = observe_all(jnp.asarray(albedo))
        ts = np.asarray(ts)
        err = ts - truth_ts
        mse = np.sum(setup.weights * err * err, axis=(-2, -1))
        results[name] = {"selected_train_ts_mse_k2": best[0],
                         "train_ts_rmse_k": float(np.sqrt(mse[:k_train].mean())),
                         "test_ts_rmse_k": float(np.sqrt(mse[k_train:].mean())),
                         "albedo": albedo_metrics(albedo), "history": history,
                         "parameters": int(flat0.size), "first_gradient_seconds": first}

    # Uniform starting map as the reference.
    ts_u, _ = observe_all(jnp.full_like(a_true, a_uniform))
    err_u = np.asarray(ts_u) - truth_ts
    mse_u = np.sum(setup.weights * err_u * err_u, axis=(-2, -1))
    results["uniform_start"] = {"train_ts_rmse_k": float(np.sqrt(mse_u[:k_train].mean())),
                                "test_ts_rmse_k": float(np.sqrt(mse_u[k_train:].mean())),
                                "albedo": albedo_metrics(np.full(setup.weights.shape, a_uniform))}

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "neural_field.npz", lon=setup.lon, lat=setup.lat,
                        albedo_true=np.asarray(a_true), observable=observable,
                        **{f"albedo_{k}": v for k, v in recovered.items()})

    plt = _plt()
    fig = plt.figure(figsize=(12.5, 7))
    gs = fig.add_gridspec(2, 3)
    lim = (float(np.asarray(a_true).min()), float(np.asarray(a_true).max()))
    for i, (key, title) in enumerate([("true", "True TES albedo"),
                                      ("neural_field", "Neural field (recovered)"),
                                      ("per_pixel", "Per-pixel fit (recovered)")]):
        ax = fig.add_subplot(gs[0, i])
        data = np.asarray(a_true) if key == "true" else recovered[key]
        mesh = ax.pcolormesh(setup.lon, setup.lat, data.T, cmap="viridis", vmin=lim[0], vmax=lim[1],
                             shading="nearest")
        ax.contour(setup.lon, setup.lat, observable.T.astype(float), levels=[0.5], colors="w",
                   linewidths=0.8)
        ax.set_title(title)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        fig.colorbar(mesh, ax=ax, label="albedo", shrink=0.85)
    map_panel(fig.add_subplot(gs[1, 0]), setup, recovered["neural_field"] - np.asarray(a_true),
              "Neural field error", "Δ albedo", fig=fig)
    map_panel(fig.add_subplot(gs[1, 1]), setup, recovered["per_pixel"] - np.asarray(a_true),
              "Per-pixel error", "Δ albedo", fig=fig)
    ax = fig.add_subplot(gs[1, 2])
    for name, color in (("neural_field", "#0072B2"), ("per_pixel", "#E69F00")):
        h = results[name]["history"]
        ax.plot([x["update"] for x in h], [x["albedo_rmse_observable"] for x in h], color=color,
                label=name.replace("_", " "))
    ax.axhline(results["uniform_start"]["albedo"]["observable"]["rmse"], color="gray", ls=":",
               label="uniform start")
    ax.set_xlabel("Update")
    ax.set_ylabel("Albedo RMSE (frost-free pixels)")
    ax.set_title("Recovery progress")
    ax.legend(fontsize=7)
    ax.grid(alpha=0.3)
    fig.suptitle(f"Recovering surface albedo from surface temperature through the coupled model "
                 f"({n_train} training steps; white contour: frost-free region)", fontsize=10)
    fig.tight_layout()
    save_fig(fig, out, "neural_field")
    plt.close(fig)

    payload = {"train_steps": n_train, "test_steps": n_test, "observe_every_steps": every,
               "observable_area_fraction": float(np.sum(setup.weights * observable)),
               "uniform_albedo": a_uniform, "albedo_bounds": ALBEDO_BOUNDS,
               "neural_field": {"hidden": args.hidden, "fourier_frequencies": args.frequencies},
               "results": results, "gradient_checks": checks,
               "all_gradient_checks_pass": bool(all(r["pass"] for r in checks)),
               "status": "pass" if all(r["pass"] for r in checks) else "fail"}
    write_report(out, setup, args, payload)
    return payload


# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--inputs", type=Path, default=Path("outputs/nautilus-calibration-inputs"))
    parser.add_argument("--dt", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="tiny horizons for a CPU pipeline check")
    sub = parser.add_subparsers(dest="command", required=True)

    s = sub.add_parser("sensitivity-map")
    s.add_argument("--output", type=Path, default=Path("outputs/capabilities/sensitivity-map"))
    s.add_argument("--sols", type=float, default=1.0)
    s.add_argument("--polar-lat", type=float, default=-60.0)
    s.add_argument("--directions", type=int, default=2)
    s.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-4])

    o = sub.add_parser("optimize-albedo")
    o.add_argument("--output", type=Path, default=Path("outputs/capabilities/optimize-albedo"))
    o.add_argument("--sols", type=float, default=1.0)
    o.add_argument("--region-lat", type=float, nargs=2, default=[15.0, 45.0])  # frost-free northern mid-latitudes at Ls 45
    o.add_argument("--region-lon", type=float, nargs=2, default=[240.0, 300.0])
    o.add_argument("--max-change", type=float, default=0.08)
    o.add_argument("--budget-fraction", type=float, default=0.5,
                   help="budget = region-box design at this fraction of --max-change")
    o.add_argument("--iterations", type=int, default=15)
    o.add_argument("--backtracks", type=int, default=6)
    o.add_argument("--directions", type=int, default=2)
    o.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-4])

    e = sub.add_parser("ensemble")
    e.add_argument("--output", type=Path, default=Path("outputs/capabilities/ensemble"))
    e.add_argument("--sols", type=float, default=1.0)
    e.add_argument("--samples", type=int, default=24)
    e.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    e.add_argument("--noise-k", type=float, default=0.1)
    e.add_argument("--co2-range", type=float, nargs=2, default=[0.8, 1.2])
    e.add_argument("--check-members", type=int, default=2)

    n = sub.add_parser("neural-field")
    n.add_argument("--output", type=Path, default=Path("outputs/capabilities/neural-field"))
    n.add_argument("--sols", type=float, default=1.0, help="training window")
    n.add_argument("--test-sols", type=float, default=1.0, help="later, unseen evaluation window")
    n.add_argument("--observe-every", type=int, default=12, help="steps between observed Ts maps")
    n.add_argument("--hidden", type=int, nargs="+", default=[64, 64])
    n.add_argument("--frequencies", type=int, default=6)
    n.add_argument("--updates", type=int, default=200)
    n.add_argument("--lr-neural", type=float, default=3e-3)
    n.add_argument("--lr-pixel", type=float, default=5e-2)
    n.add_argument("--log-every", type=int, default=10)
    n.add_argument("--epsilons", type=float, nargs="+", default=[1e-3, 1e-4])

    args = parser.parse_args(argv)
    jax.config.update("jax_enable_x64", True)
    if args.require_gpu and jax.default_backend() != "gpu":
        raise RuntimeError("GPU required; refusing CPU fallback")
    if args.smoke:
        args.sols = 4 * args.dt / SOL_SECONDS
        if args.command == "optimize-albedo":
            args.iterations = 2
        if args.command == "neural-field":
            args.test_sols, args.observe_every, args.updates, args.log_every = args.sols, 2, 3, 1
        if args.command == "ensemble":
            args.samples, args.batches = 2, [1, 2]
        args.output = Path(str(args.output) + "-smoke")
    runner = {"sensitivity-map": sensitivity_map, "optimize-albedo": optimize_albedo,
              "neural-field": neural_field,
              "ensemble": ensemble}[args.command]
    payload = runner(args)
    print(json.dumps({"status": payload["status"], "output": str(args.output)}, indent=2))
    return 0 if payload["status"] == "pass" else 10


if __name__ == "__main__":
    raise SystemExit(main())
