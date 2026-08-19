"""Dry-dynamics Mars maps over real MOLA terrain — lat/lon fields for each variable.

Runs the ``gcm3d`` primitive-equations core on Mars with the MOLA orography as
its lower boundary and returns 2-D (lat/lon) maps of surface pressure,
near-surface temperature, and the horizontal wind components — the fields used
to eyeball a Mars GCM against the Ames MGCM and LMD PCM.

Scope / honesty about fidelity: this is the **dry dynamical core** only. There is
no radiative transfer, CO2 condensation cycle, or dust, so the *magnitudes* of
the climatology are not yet quantitatively comparable to Ames/LMD. What *is*
comparable is the structure the terrain imposes — most directly the surface
pressure, which sits in Mars hydrostatic balance with the topography (low over
Tharsis/Olympus, high in Hellas), the canonical first validation map. Adding the
physics that closes the gap is a separate, staged effort.

Requires the optional ``gcm3d`` extra.
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path
from typing import Callable

import numpy as np

from src.gcm3d._dinosaur import jax, jnp, primitive_equations, scales, spherical_harmonic
from src.gcm3d.coordinates import coordinate_system
from src.gcm3d.dynamics import integrate as _integrate
from src.gcm3d.dynamics import primitive_equations as build_primitive_equations
from src.gcm3d.dynamics import reference_temperature, stepper
from src.gcm3d.specs import physics_specs
from src.gcm3d.topography import mola_modal_orography, regrid_to_nodal

_u = scales.units


# ── Runtime scale presets ─────────────────────────────────────────────────────
# Named resolution/step configurations, selectable per run (CLI --scale, server
# `scale` field). Each maps to run_maps' truncation / n_layers / dt / n_steps.
# dt is kept below the daily-mean stability limit at each resolution; the diurnal
# terminator CFL (dt <= rotation/(2*n_lon)) is enforced separately in run_maps.
MAP_SCALES: dict[str, dict] = {
    "fast":     dict(truncation="T42",  n_layers=12, dt_seconds=450.0, n_steps=700),
    "balanced": dict(truncation="T85",  n_layers=20, dt_seconds=300.0, n_steps=1000),
    "high":     dict(truncation="T106", n_layers=30, dt_seconds=225.0, n_steps=1400),
    "ultra":    dict(truncation="T170", n_layers=40, dt_seconds=150.0, n_steps=2000),
}
DEFAULT_SCALE = "fast"


def resolve_scale(scale: str | None) -> dict:
    """Return the run_maps kwargs for a named scale preset (see :data:`MAP_SCALES`)."""
    key = (scale or DEFAULT_SCALE).lower()
    if key not in MAP_SCALES:
        raise ValueError(
            f"unknown scale {scale!r}; choose one of {list(MAP_SCALES)}"
        )
    return dict(MAP_SCALES[key])


def forcing_with_surface_properties(forcing, grid, path):
    """Return forcing with explicitly supplied nodal albedo/TES inertia fields."""
    from src.gcm3d.surface import surface_boundary_fields_on_grid

    fields = surface_boundary_fields_on_grid(grid, path)
    updates = dict(
        albedo=jnp.asarray(fields["albedo"]),
        surface_thermal_inertia_tiu=jnp.asarray(fields["thermal_inertia"]),
        regolith_enabled=True,
        stability_exchange_enabled=True,
        pbl_diffusion_enabled=True,
        convective_adjustment_enabled=True,
        co2_radiation_enabled=True,
    )
    if "emissivity" in fields:
        updates["emissivity"] = jnp.asarray(fields["emissivity"])
    if "roughness" in fields:
        updates["surface_roughness_m"] = jnp.asarray(fields["roughness"])
    return dataclasses.replace(forcing, **updates)


def forcing_with_ames_dust(forcing, grid, path, ls_deg: float):
    """Attach a seasonally and spatially matched Ames prescribed-dust scenario."""
    from src.gcm3d.dust import seasonal_dust_on_grid

    tau, zmax = seasonal_dust_on_grid(grid, path, ls_deg)
    return dataclasses.replace(
        forcing,
        dust_visible_optical_depth=jnp.asarray(tau),
        # Ames input opacity is referenced in the visible; retain the established
        # compact IR/visible ratio until the spectral aerosol solver is enabled.
        dust_longwave_optical_depth=jnp.asarray(tau) * 0.33,
        dust_top_height_km=jnp.asarray(zmax),
    )


def hydrostatic_surface_pressure_pa(elevation_m, body, t_ref_k=None, p0_pa=None):
    """Mars-hydrostatic surface pressure over terrain (Pa).

    ``p_s(z) = p0 · exp(−g·z / (R·T))`` with Mars's own gravity, CO2 gas
    constant and reference temperature — *not* the Earth barometric constants
    dinosaur's ``isothermal_rest_atmosphere`` hardcodes. Mars's ~10 km scale
    height gives ~85 Pa over Olympus Mons and ~1200 Pa in Hellas from a 610 Pa
    global reference.
    """
    p0 = body.reference_surface_pressure_pa if p0_pa is None else p0_pa
    t_ref = body.reference_temperature_k if t_ref_k is None else t_ref_k
    scale_height = body.gas_constant_j_kg_k * t_ref / body.gravity_m_s2
    return p0 * np.exp(-np.asarray(elevation_m) / scale_height)


def initial_rest_state(coords, specs, body, elevation_nodal_m, t_ref_k=None, p0_pa=None):
    """A rest, isothermal Mars state with surface pressure balanced to the terrain.

    Winds are zero and the temperature is the reference profile (temperature
    variation zero); only the surface pressure carries the topography, via
    :func:`hydrostatic_surface_pressure_pa`. Running the dycore from here lets
    the terrain-driven pressure gradients spin up the circulation.
    """
    grid = coords.horizontal
    # Surface pressure on the (1, n_lon, n_lat) surface nodal grid.
    ps_pa = hydrostatic_surface_pressure_pa(
        np.asarray(elevation_nodal_m), body, t_ref_k, p0_pa
    ).reshape(coords.surface_nodal_shape)
    ps_nd = np.asarray(specs.nondimensionalize(ps_pa * _u.pascal))
    log_sp = grid.to_modal(jnp.asarray(np.log(ps_nd)))

    zeros_modal = jnp.zeros((coords.vertical.layers,) + grid.modal_shape)
    return primitive_equations.State(
        vorticity=zeros_modal,
        divergence=zeros_modal,
        temperature_variation=zeros_modal,
        log_surface_pressure=log_sp,
    )


@dataclasses.dataclass(frozen=True)
class MarsMapFields:
    """Lat/lon map fields from a dry-dynamics Mars run.

    All 2-D fields are shaped ``(n_lat, n_lon)`` with ``lat_deg`` ascending
    (−90→90) and ``lon_deg`` ascending (0→360), so they drop straight into
    ``pcolormesh(lon_deg, lat_deg, field)``. Winds/temperature are taken at the
    near-surface sigma level.
    """

    lon_deg: np.ndarray            # (n_lon,)
    lat_deg: np.ndarray            # (n_lat,)
    elevation_m: np.ndarray        # (n_lat, n_lon)   MOLA, regridded
    surface_pressure_pa: np.ndarray
    temperature_k: np.ndarray      # near-surface level
    u_ms: np.ndarray               # near-surface zonal wind
    v_ms: np.ndarray               # near-surface meridional wind
    truncation: str
    n_layers: int
    n_steps: int
    dt_seconds: float
    physics: str = "dry dynamics; no radiation/CO2/dust"
    co2_ice_pa: np.ndarray | None = None  # (n_lat, n_lon) surface CO2 frost, Pa-equiv
    rotation_period_s: float = 88775.244  # for the duration diagnostic

    @property
    def wind_speed_ms(self) -> np.ndarray:
        return np.hypot(self.u_ms, self.v_ms)

    @property
    def duration_sols(self) -> float:
        """Physical duration of the run in sols (n_steps * dt / rotation period)."""
        return self.n_steps * self.dt_seconds / self.rotation_period_s

    @property
    def is_transient(self) -> bool:
        """True if the run is too short (< 1 Mars year ~ 668 sols) to be a
        seasonally-equilibrated climatology — it is a spin-up transient snapshot."""
        return self.duration_sols < 668.0

    @property
    def wind_level_sigma(self) -> float:
        """Sigma midpoint represented by the exported lowest-layer wind."""
        return 1.0 - 0.5 / self.n_layers

    @property
    def approximate_wind_height_m(self) -> float:
        """Hydrostatic reference height of the exported lowest-layer midpoint."""
        from src.celestials.planets.mars import MARS_BODY_3D

        scale_height = (
            MARS_BODY_3D.gas_constant_j_kg_k
            * MARS_BODY_3D.reference_temperature_k
            / MARS_BODY_3D.gravity_m_s2
        )
        return -scale_height * math.log(self.wind_level_sigma)


def run_maps(
    body=None,
    truncation: str = "T42",
    n_layers: int = 25,
    dt_seconds: float = 600.0,
    n_steps: int = 200,
    t_ref_k: float | None = None,
    p0_pa: float | None = None,
    mola_path=None,
    forcing=None,
    co2_forcing=None,
    surface_properties_path=None,
    initial_state=None,
    return_final_state: bool = False,
    progress_callback: Callable[[int, int], None] | None = None,
    diagnostic_callback: Callable[[int, int, object, object, object], None] | None = None,
    stop_requested: Callable[[], bool] | None = None,
    progress_chunk_steps: int = 32,
) -> MarsMapFields | tuple[MarsMapFields, object]:
    """Run the Mars dycore over MOLA terrain and return lat/lon map fields.

    Builds the coordinate system, the MOLA modal orography, a terrain-balanced
    rest state, integrates ``n_steps`` of the primitive equations, then converts
    the final modal state to dimensional lat/lon fields (surface pressure,
    near-surface temperature, and near-surface ``u``/``v``).

    Pass ``forcing`` (a :class:`src.gcm3d.physics.RadiativeForcing`, e.g. from
    :func:`src.gcm3d.physics.mars_radiative_forcing`) to add the per-column
    radiative energy balance on top of the dry dynamics; the fluid then develops
    its own temperature structure instead of holding the isothermal rest profile.
    With ``forcing=None`` this is the dry dynamical core (the previous behaviour).
    """
    if body is None:
        from src.celestials.planets.mars import MARS_BODY_3D

        body = MARS_BODY_3D
    if n_steps < 1:
        raise ValueError(f"n_steps must be >= 1, got {n_steps}")
    if not np.isfinite(dt_seconds) or dt_seconds <= 0:
        raise ValueError(f"dt_seconds must be finite and > 0, got {dt_seconds}")
    if co2_forcing is not None and forcing is None:
        raise ValueError(
            "co2_forcing requires a radiative `forcing` (the CO2 cycle is driven "
            "by the radiatively-forced surface temperature)."
        )

    coords = coordinate_system(truncation=truncation, n_layers=n_layers)
    specs = physics_specs(body)
    grid = coords.horizontal

    spatial_surface = False
    if forcing is not None and surface_properties_path is not None:
        forcing = forcing_with_surface_properties(
            forcing, grid, surface_properties_path
        )
        spatial_surface = True

    elevation_nodal_m = regrid_to_nodal(coords, mola_path=mola_path)  # (n_lon, n_lat)
    orography = mola_modal_orography(coords, specs, elevation_nodal_m=elevation_nodal_m)

    state0 = (initial_rest_state(coords, specs, body, elevation_nodal_m, t_ref_k, p0_pa)
              if initial_state is None else initial_state)
    if forcing is None:
        equation = build_primitive_equations(coords, body, specs=specs, orography=orography)
        physics_label = "dry dynamics; no radiation/CO2/dust"
    else:
        from src.gcm3d.physics import forced_primitive_equations, initial_column_state

        equation = forced_primitive_equations(
            coords, body, forcing, specs=specs, orography=orography
        )
        # sim_time must be present (0.0) for the diurnal/seasonal forcing to advance.
        if initial_state is None:
            state0 = dataclasses.replace(state0, sim_time=0.0)
            state0 = initial_column_state(
                state0, coords, t_ref_k or body.reference_temperature_k, specs,
                forcing=forcing,
            )
        radiation_name = (
            "two-stream CO2-band radiation" if forcing.co2_radiation_enabled
            else "grey radiative energy balance"
        )
        dust_name = (
            " + prescribed radiatively active dust"
            if (np.any(np.asarray(forcing.dust_visible_optical_depth) != 0.0)
                or np.any(np.asarray(forcing.dust_longwave_optical_depth) != 0.0))
            else "; no dust"
        )
        physics_label = f"dry dynamics + {radiation_name}{dust_name}"
        if co2_forcing is not None:
            from src.gcm3d.physics import (
                forced_co2_primitive_equations,
            )

            equation = forced_co2_primitive_equations(
                coords, body, forcing, co2_forcing, specs=specs, orography=orography
            )
            # The radiation-only state already contains the surface reservoir;
            # enabling CO2 simply uses its existing zero frost field.
            physics_label = (
                f"dry dynamics + {radiation_name} + CO2 condensation cycle{dust_name}"
            )
        if spatial_surface:
            physics_label += (
                " + explicit spatial albedo/TI + multilayer regolith"
                " + Richardson/PBL diffusion + dry convective adjustment"
            )

        # Diurnal-terminator CFL: the subsolar point sweeps 360° per rotation, so
        # the day/night terminator crosses one longitude cell in
        # rotation_period/n_lon seconds. The sharp terminator is not band-limited,
        # and with the nonlinear T^4 feedback a step that lets it jump more than
        # ~half a cell aliases and diverges to NaN (verified: T42 needs
        # dt<=~350s, T21 <=~700s). Require >=2 steps per cell and refuse a step
        # too coarse rather than emit silently-NaN maps.
        if getattr(forcing, "diurnal", False):
            n_lon = len(grid.longitudes)
            max_stable_dt = forcing.rotation_period_s / (2.0 * n_lon)
            if dt_seconds > max_stable_dt:
                raise ValueError(
                    f"dt_seconds={dt_seconds:g} is too coarse for the diurnal "
                    f"forcing at {truncation} (n_lon={n_lon}); the moving "
                    f"terminator aliases and the run diverges. Use dt_seconds "
                    f"<= {max_stable_dt:g}, coarsen the truncation, or pass a "
                    f"forcing with diurnal=False (daily-mean insolation)."
                )

    step = stepper(equation, dt_seconds, specs)
    if co2_forcing is not None and co2_forcing.energy_limited:
        from src.gcm3d.physics import positivity_preserving_co2_step
        step = positivity_preserving_co2_step(step, coords, specs)
    if (progress_callback is None and diagnostic_callback is None
            and stop_requested is None):
        # Keep the single scan for training/gradient callers. Debugger runs opt
        # into bounded chunks so progress and cancellation are observable.
        final = _integrate(step, state0, n_steps)
    else:
        if progress_chunk_steps < 1:
            raise ValueError("progress_chunk_steps must be >= 1")
        final = state0
        completed = 0
        advance_chunk = jax.jit(
            lambda state: _integrate(step, state, progress_chunk_steps)
        )
        while completed < n_steps:
            if stop_requested is not None and stop_requested():
                raise InterruptedError("simulation stopped by user")
            count = min(progress_chunk_steps, n_steps - completed)
            final = (advance_chunk(final) if count == progress_chunk_steps
                     else _integrate(step, final, count))
            # Synchronise here: without it JAX dispatch would make UI progress
            # describe queued work instead of completed integration steps.
            jax.block_until_ready(final)
            completed += count
            if progress_callback is not None:
                progress_callback(completed, n_steps)
            if diagnostic_callback is not None:
                diagnostic_callback(completed, n_steps, final, coords, specs)
    if not all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(final)
    ):
        raise FloatingPointError(
            "GCM state became non-finite; reduce the timestep or inspect the "
            "last progress chunk instead of exporting invalid maps"
        )
    final_state = final

    # Unwrap the JCM-style column state used by all forced integrations.
    co2_ice_map = None
    surface_temperature_map = None
    if forcing is not None:
        column_final = final
        final = column_final.dynamics
        surface_temperature_map = np.asarray(
            specs.dimensionalize(column_final.surface_temperature, _u.kelvin).magnitude
        )[0].T
    if co2_forcing is not None:
        co2_ice = column_final.co2_ice
        ice_pa = np.asarray(
            specs.dimensionalize(jnp.asarray(co2_ice), _u.pascal).magnitude
        )  # (1, n_lon, n_lat)
        co2_ice_map = np.asarray(ice_pa[0]).T  # (n_lat, n_lon)

    # ── modal → nodal, dimensionalize ─────────────────────────────────────────
    # Surface pressure: exp(log_sp) is nondimensional p_s.
    ps_nd = jnp.exp(grid.to_nodal(final.log_surface_pressure))
    ps_pa = np.asarray(specs.dimensionalize(ps_nd, _u.pascal).magnitude)  # (1,n_lon,n_lat)

    # Temperature = reference profile + variation, at the near-surface level.
    ref_t = np.asarray(reference_temperature(coords, body)).reshape(n_layers, 1, 1)
    t_var_nd = np.asarray(grid.to_nodal(final.temperature_variation))     # (L,n_lon,n_lat)
    t_nd = t_var_nd + ref_t
    t_k = np.asarray(specs.dimensionalize(t_nd, _u.kelvin).magnitude)

    # Winds from vorticity/divergence.
    u_nd, v_nd = spherical_harmonic.vor_div_to_uv_nodal(
        grid, final.vorticity, final.divergence
    )
    u_ms = np.asarray(specs.dimensionalize(u_nd, _u.meter / _u.second).magnitude)
    v_ms = np.asarray(specs.dimensionalize(v_nd, _u.meter / _u.second).magnitude)

    surf = n_layers - 1  # near-surface sigma level (sigma ≈ 1)

    # Honesty about duration: use prognostic time, including resumed chunks.
    if getattr(final, "sim_time", None) is not None:
        elapsed_seconds = float(final.sim_time) / float(
            specs.nondimensionalize(1.0 * _u.second)
        )
        total_steps = round(elapsed_seconds / dt_seconds)
    else:
        elapsed_seconds = n_steps * dt_seconds
        total_steps = n_steps
    duration_sols = elapsed_seconds / body.rotation_period_s
    if duration_sols < 668.0:
        physics_label += (
            f" | TRANSIENT: {duration_sols:.1f}-sol spin-up snapshot, "
            f"NOT a seasonally-equilibrated climatology (needs >= 1 Mars year)"
        )

    def to_map(field_lonlat):
        """(n_lon, n_lat) → (n_lat, n_lon)."""
        return np.asarray(field_lonlat).T

    fields = MarsMapFields(
        lon_deg=np.degrees(np.asarray(grid.longitudes)),
        lat_deg=np.degrees(np.asarray(grid.latitudes)),
        elevation_m=to_map(elevation_nodal_m),
        surface_pressure_pa=to_map(ps_pa[0]),
        temperature_k=(surface_temperature_map if surface_temperature_map is not None
                       else to_map(t_k[surf])),
        u_ms=to_map(u_ms[surf]),
        v_ms=to_map(v_ms[surf]),
        truncation=truncation,
        n_layers=n_layers,
        n_steps=total_steps,
        dt_seconds=dt_seconds,
        physics=physics_label,
        co2_ice_pa=co2_ice_map,
        rotation_period_s=body.rotation_period_s,
    )
    return (fields, final_state) if return_final_state else fields


# ── Output: NetCDF export + lat/lon map plots ─────────────────────────────────

def save_netcdf(fields: MarsMapFields, path) -> Path:
    """Write all map fields to a NetCDF file (lat/lon coords) via xarray.

    Gives you the raw gridded data for quantitative comparison against Ames/LMD
    output later (same coordinate convention: lat −90→90, lon 0→360).
    """
    import xarray as xr

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = ("lat", "lon")
    coords = {"lat": fields.lat_deg, "lon": fields.lon_deg}
    ds = xr.Dataset(
        {
            "elevation": (dims, fields.elevation_m, {"units": "m", "long_name": "MOLA elevation"}),
            "surface_pressure": (dims, fields.surface_pressure_pa, {"units": "Pa"}),
            "temperature": (dims, fields.temperature_k, {"units": "K", "long_name": "near-surface temperature"}),
            "u": (dims, fields.u_ms, {"units": "m/s", "long_name": "zonal wind"}),
            "v": (dims, fields.v_ms, {"units": "m/s", "long_name": "meridional wind"}),
            "wind_speed": (dims, fields.wind_speed_ms, {"units": "m/s"}),
            **(
                {"co2_ice": (dims, fields.co2_ice_pa,
                             {"units": "Pa", "long_name": "CO2 surface frost (pressure-equiv)"})}
                if fields.co2_ice_pa is not None else {}
            ),
        },
        coords=coords,
        attrs={
            "title": "gcm3d dry-dynamics Mars maps over MOLA terrain",
            "truncation": fields.truncation,
            "n_layers": fields.n_layers,
            "n_steps": fields.n_steps,
            "dt_seconds": fields.dt_seconds,
            "fidelity": fields.physics,
            "wind_level_sigma": fields.wind_level_sigma,
            "approximate_wind_height_m": fields.approximate_wind_height_m,
        },
    )
    ds.lat.attrs.update(units="degrees_north")
    ds.lon.attrs.update(units="degrees_east")
    ds.to_netcdf(path)
    return path


def plot_maps(fields: MarsMapFields, outdir, prefix: str = "mars") -> list[Path]:
    """Render one lat/lon PNG per variable, MOLA terrain contoured on each.

    Uses the non-interactive Agg backend (no display needed). Each map is a
    simple-cylindrical ``pcolormesh`` with the MOLA elevation overlaid as thin
    contours so the terrain control on every field is visible.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    lon, lat, elev = fields.lon_deg, fields.lat_deg, fields.elevation_m

    panels = [
        ("topography", fields.elevation_m, "MOLA elevation (m)", "terrain", None),
        ("surface_pressure", fields.surface_pressure_pa, "Surface pressure (Pa)", "viridis", None),
        ("temperature", fields.temperature_k, "Near-surface temperature (K)", "inferno", None),
        ("wind_speed", fields.wind_speed_ms, "Near-surface wind speed (m/s)", "cividis", "quiver"),
    ]
    if fields.co2_ice_pa is not None:
        panels.append(
            ("co2_ice", fields.co2_ice_pa, "CO2 surface frost (Pa-equiv)", "bone", None)
        )
    paths: list[Path] = []
    for name, field, label, cmap, overlay in panels:
        fig, ax = plt.subplots(figsize=(9, 4.5))
        mesh = ax.pcolormesh(lon, lat, field, cmap=cmap, shading="auto")
        # MOLA terrain contours on every panel for geographic reference.
        ax.contour(lon, lat, elev, levels=8, colors="k", linewidths=0.3, alpha=0.4)
        if overlay == "quiver":
            # Subsample lon/lat independently (grids are not square) and scale
            # arrows to the field's own peak speed so they stay short and legible.
            slon = max(1, field.shape[1] // 20)
            slat = max(1, field.shape[0] // 12)
            speed_max = float(np.nanmax(fields.wind_speed_ms)) or 1.0
            ax.quiver(
                lon[::slon], lat[::slat],
                fields.u_ms[::slat, ::slon], fields.v_ms[::slat, ::slon],
                scale=speed_max * 12.0, width=0.002, color="white", alpha=0.85,
            )
        fig.colorbar(mesh, ax=ax, label=label, shrink=0.85)
        ax.set_xlabel("Longitude (°E)")
        ax.set_ylabel("Latitude (°N)")
        model_tag = "gcm3d+radiation" if "radiat" in fields.physics else "dry gcm3d"
        ax.set_title(
            f"Mars {name.replace('_', ' ')} — {model_tag}, {fields.truncation}, "
            f"{fields.n_steps} steps × {fields.dt_seconds:g}s"
        )
        ax.set_xlim(0, 360)
        ax.set_ylim(-90, 90)
        fig.tight_layout()
        p = outdir / f"{prefix}_{name}.png"
        fig.savefig(p, dpi=130)
        plt.close(fig)
        paths.append(p)
    return paths


def save_maps(fields: MarsMapFields, outdir, prefix: str = "mars") -> list[Path]:
    """Write the NetCDF dataset and all per-variable PNGs; return the file paths."""
    outdir = Path(outdir)
    nc = save_netcdf(fields, outdir / f"{prefix}_maps.nc")
    pngs = plot_maps(fields, outdir, prefix=prefix)
    return [nc, *pngs]
