"""Column radiative physics as an explicit forcing on the 3-D primitive equations.

This is the bridge from the 0-D terraforming ODE to the 3-D dycore: it takes the
diurnal/seasonal radiative energy balance that :mod:`src.gcm3d.terraforming_ode`
integrates for a *single* global-mean column and evaluates it **per grid column**,
then adds the result as a temperature tendency onto dinosaur's dry
``PrimitiveEquationsSigma``. The dry maps path (:mod:`src.gcm3d.maps`) is
dynamics-only — dinosaur computes every tendency and nothing heats the fluid; here
we supply the missing radiative forcing so the model develops its *own* temperature
structure (hot subsolar point, cold poles/nightside) instead of relaxing a
prescribed rest state. Those horizontal temperature gradients are what drive the
circulation, so this is the step that turns the dry dynamical core into a (grey,
single-band) Mars climate model.

How the coupling works
----------------------
dinosaur's equations are an ``ImplicitExplicitODE`` with a
``explicit_terms(state) -> State`` method. We wrap it: the forced ODE calls the
base ``explicit_terms`` and adds a radiative temperature tendency, leaving the
implicit (gravity-wave) side untouched. The forcing needs the current time for the
diurnal + seasonal cycle — dinosaur's ``State`` already carries a ``sim_time``
field that the base equations advance at unit (nondimensional) rate, so we read it
with no change to the state layout. Initialise the state with ``sim_time=0.0`` (see
:func:`src.gcm3d.maps.run_maps` with ``forcing=...``) for time to advance.

The nonlinear surface energy balance is evaluated once per column on a prognostic
surface temperature. A conservative bulk sensible flux couples that reservoir to
the lowest atmospheric sigma layer using its mass-dependent heat capacity
``cp*dp/g``. Surface loss and atmospheric gain therefore cancel exactly and the
column budget is independent of vertical layer count.

The legacy grey forcing remains available for regression experiments. The
JCM-style path uses a compact pressure-scaled two-stream solver with CO2 near-IR
and thermal bands plus prescribed dust. It is not yet correlated-k and therefore
must not be described as full LMD/JCM radiative fidelity. Requires the optional
``gcm3d`` extra.
"""

from __future__ import annotations

import dataclasses
import math
from typing import NamedTuple

import numpy as np

from src.gcm3d._dinosaur import jnp, scales, spherical_harmonic, time_integration
from src.gcm3d.body import BodyConstants
from src.gcm3d.dynamics import primitive_equations as _build_primitive_equations
from src.gcm3d.dynamics import reference_temperature
from src.gcm3d.specs import physics_specs

_u = scales.units
_TWO_PI = 2.0 * math.pi
_REGOLITH_LAYER_FRACTIONS = (0.5, 1.0, 2.0, 4.0)

# Astronomical constants shared with the 0-D orbit model.
TSI_1AU_W_M2 = 1361.0
AU_M = 1.49597870700e11


@dataclasses.dataclass(frozen=True)
class RadiativeForcing:
    """Radiative + orbital constants for the per-column energy-balance forcing.

    Mirrors the subset of :class:`src.gcm3d.terraforming_ode.SeasonalForcing`
    needed for the temperature tendency (no polar-cap fields — the CO2 cycle is a
    separate increment). Every field is a plain float/bool, so this stays pure
    Python; the values that are *not* on :class:`~src.gcm3d.body.BodyConstants`
    (albedo, greenhouse, emissivity, thermal inertia, orbit) live here.
    """

    # Radiative / thermal
    albedo: object
    greenhouse_factor: float
    emissivity: float
    stefan_boltzmann: float
    thermal_inertia: float
    # Rotation (sets the diurnal hour angle) and obliquity/precession
    rotation_period_s: float
    axial_tilt_rad: float
    ls_perihelion_rad: float
    # Orbit (Keplerian ellipse; angle advances at the mean rate)
    orbital_period_s: float
    semi_major_axis_m: float
    eccentricity: float
    init_orbital_angle_rad: float = 0.0
    # Astronomical constants (overridable for other bodies)
    tsi_1au_w_m2: float = TSI_1AU_W_M2
    au_m: float = AU_M
    # Diurnal cycle on (per-longitude hour angle) or daily-mean insolation.
    diurnal: bool = True
    # Bulk surface-atmosphere sensible heat exchange coefficient (W m-2 K-1).
    # This is deliberately explicit and replaceable: unlike the old layer-uniform
    # heating it transfers energy between two reservoirs without creating energy.
    sensible_heat_transfer_w_m2_k: float = 2.0
    # Neutral bulk aerodynamic surface drag.  The coefficient is diagnosed from
    # the logarithmic surface-layer law at the centre of the lowest sigma layer.
    surface_roughness_m: float = 0.01
    von_karman_constant: float = 0.4
    minimum_wind_ms: float = 0.1
    # Regolith properties. Thermal inertia may be replaced by a nodal TES field.
    surface_thermal_inertia_tiu: object = 250.0
    regolith_volumetric_heat_capacity_j_m3_k: float = 1.0e6
    regolith_layer_skin_depth_fractions: tuple[float, ...] = _REGOLITH_LAYER_FRACTIONS
    regolith_enabled: bool = False
    stability_exchange_enabled: bool = False
    pbl_diffusion_enabled: bool = False
    pbl_height_m: float = 5000.0
    pbl_implicit_timestep_s: float = 1800.0
    convective_adjustment_enabled: bool = False
    convective_relaxation_s: float = 900.0
    # Vertically resolved multiband Mars radiation. The two scalar optical depths
    # remain useful experiment controls; the tuples split them into spectral
    # bands with distinct pressure/temperature responses. This is a compact
    # differentiable precursor to Ames-style correlated-k table interpolation,
    # not a replacement for those tables.
    co2_radiation_enabled: bool = False
    # Use the bundled Ames 12-band correlated-k coefficients. Disable only for
    # compact-scheme ablations or installations without the staged asset.
    ames_correlated_k_enabled: bool = True
    co2_longwave_optical_depth: float = 0.35
    co2_near_ir_optical_depth: float = 0.08
    co2_reference_temperature_k: float = 200.0
    co2_shortwave_band_weights: tuple[float, ...] = (0.72, 0.28)
    co2_shortwave_band_strengths: tuple[float, ...] = (0.35, 2.67)
    co2_shortwave_pressure_exponents: tuple[float, ...] = (1.0, 1.18)
    co2_shortwave_temperature_exponents: tuple[float, ...] = (0.15, 0.55)
    co2_longwave_band_weights: tuple[float, ...] = (0.18, 0.62, 0.20)
    co2_longwave_band_strengths: tuple[float, ...] = (0.12, 1.32, 0.48)
    co2_longwave_pressure_exponents: tuple[float, ...] = (1.0, 1.22, 1.08)
    co2_longwave_temperature_exponents: tuple[float, ...] = (0.10, 0.75, 0.35)
    # Prescribed visible/IR dust column opacity. May be a scalar or nodal field.
    dust_visible_optical_depth: object = 0.0
    dust_longwave_optical_depth: object = 0.0
    dust_single_scattering_albedo: float = 0.92


def mars_radiative_forcing(
    albedo: float = 0.25,
    greenhouse_factor: float = 1.02,
    diurnal: bool = True,
    init_orbital_angle_rad: float = 0.0,
    co2_radiation_enabled: bool = False,
    dust_visible_optical_depth: object = 0.0,
    dust_longwave_optical_depth: object = 0.0,
) -> RadiativeForcing:
    """A :class:`RadiativeForcing` built from the package's Mars constants.

    Pulls obliquity, precession, orbit, emissivity and thermal inertia from
    ``src.celestials.planets.mars`` so the 3-D forcing uses the same numbers as
    the 0-D/torch Mars model. ``albedo`` and ``greenhouse_factor`` default to the
    torch ``Mars`` defaults but are overridable (e.g. to sweep terraforming
    scenarios).
    """
    from src.celestials.planets import mars as _m

    return RadiativeForcing(
        albedo=albedo,
        greenhouse_factor=greenhouse_factor,
        emissivity=float(_m.MARS_SURFACE_EMISSIVITY),
        stefan_boltzmann=float(_m.STEFAN_BOLTZMANN),
        thermal_inertia=float(_m.MARS_THERMAL_INERTIA),
        rotation_period_s=float(_m.MARS_ROTATION_PERIOD),
        axial_tilt_rad=float(_m.MARS_AXIAL_TILT),
        ls_perihelion_rad=float(_m.MARS_LS_PERIHELION),
        orbital_period_s=float(_m.MARS_ORBITAL_PERIOD),
        semi_major_axis_m=float(_m.MARS_SEMI_MAJOR_AXIS),
        eccentricity=float(_m.MARS_ECCENTRICITY),
        init_orbital_angle_rad=init_orbital_angle_rad,
        diurnal=diurnal,
        co2_radiation_enabled=co2_radiation_enabled,
        dust_visible_optical_depth=dust_visible_optical_depth,
        dust_longwave_optical_depth=dust_longwave_optical_depth,
    )


# ── Keplerian orbit ───────────────────────────────────────────────────────────
# Epoch convention: ``init_orbital_angle_rad`` is the *mean anomaly* M0 at t=0,
# measured from perihelion (M=0 at perihelion). The mean anomaly advances
# uniformly; the eccentric anomaly E solves Kepler's equation M = E - e sin E;
# the true anomaly nu and heliocentric distance follow from E. Areocentric solar
# longitude is Ls = nu + Ls_perihelion (Ls_perihelion ~= 251 deg for Mars). Use
# :func:`mean_anomaly_for_ls` to set the epoch from a desired Ls consistently.
_KEPLER_ITERS = 6  # Newton iterations; e<0.1 converges to ~machine eps in ~4


def _mean_anomaly(t_s, f: RadiativeForcing):
    """Mean anomaly (rad) at elapsed seconds ``t_s`` — advances uniformly."""
    return f.init_orbital_angle_rad + _TWO_PI * t_s / f.orbital_period_s


def _eccentric_anomaly(mean_anomaly, e: float):
    """Solve Kepler's equation ``M = E - e sin E`` for E (Newton's method)."""
    E = mean_anomaly  # good initial guess for small e
    for _ in range(_KEPLER_ITERS):
        E = E - (E - e * jnp.sin(E) - mean_anomaly) / (1.0 - e * jnp.cos(E))
    return E


def _true_anomaly(t_s, f: RadiativeForcing):
    """True anomaly nu (rad) from the Kepler solution at elapsed seconds ``t_s``."""
    E = _eccentric_anomaly(_mean_anomaly(t_s, f), f.eccentricity)
    e = f.eccentricity
    return 2.0 * jnp.arctan2(
        jnp.sqrt(1.0 + e) * jnp.sin(E / 2.0),
        jnp.sqrt(1.0 - e) * jnp.cos(E / 2.0),
    )


def orbital_distance(t_s, f: RadiativeForcing):
    """Heliocentric distance (m): ``r = a(1 - e cos E)``."""
    E = _eccentric_anomaly(_mean_anomaly(t_s, f), f.eccentricity)
    return f.semi_major_axis_m * (1.0 - f.eccentricity * jnp.cos(E))


def mean_anomaly_for_ls(ls_rad: float, f: RadiativeForcing) -> float:
    """The epoch mean anomaly that places the orbit at solar longitude ``ls_rad``.

    Inverts the Kepler chain (Ls -> nu -> E -> M) so callers can specify a season
    (Ls) rather than a mean anomaly, keeping epoch/perihelion/Ls mutually
    consistent. Returned as a plain float (static epoch config).
    """
    nu = float(ls_rad) - f.ls_perihelion_rad
    e = f.eccentricity
    E = 2.0 * math.atan2(
        math.sqrt(1.0 - e) * math.sin(nu / 2.0),
        math.sqrt(1.0 + e) * math.cos(nu / 2.0),
    )
    return E - e * math.sin(E)


def solar_flux(t_s, f: RadiativeForcing):
    """Inverse-square solar flux (W m^-2) at elapsed seconds ``t_s``.

    Uses the true Keplerian distance ``r = a(1 - e cos E)`` then
    ``S = S_1AU (AU/r)^2``.
    """
    return f.tsi_1au_w_m2 * (f.au_m / orbital_distance(t_s, f)) ** 2


def _declination(t_s, f: RadiativeForcing):
    """Solar declination (rad): ``arcsin(sin(tilt) sin(Ls))``, Ls = nu + Ls_peri."""
    ls = _true_anomaly(t_s, f) + f.ls_perihelion_rad
    return jnp.arcsin(jnp.sin(f.axial_tilt_rad) * jnp.sin(ls))


def cos_zenith_nodal(t_s, lat_rad, lon_rad, f: RadiativeForcing):
    """Cosine of the solar zenith angle on the ``(n_lon, n_lat)`` nodal grid.

    With ``diurnal`` the hour angle rotates with the planet
    (``h = lon + 2 pi t / rotation_period``), giving a day/night terminator that
    sweeps westward each sol. Without it, the daily-mean insolation factor is used
    (integrated over the hour angle), which is smooth in time and lets the ODE be
    stepped more coarsely. Result is clipped at 0 (no negative insolation) and
    shaped ``(n_lon, n_lat)`` to match dinosaur's nodal fields.
    """
    delta = _declination(t_s, f)
    lat = jnp.asarray(lat_rad)  # (n_lat,)
    sin_lat, cos_lat = jnp.sin(lat), jnp.cos(lat)
    sin_d, cos_d = jnp.sin(delta), jnp.cos(delta)

    if f.diurnal:
        lon = jnp.asarray(lon_rad)  # (n_lon,)
        h = lon[:, None] + (_TWO_PI / f.rotation_period_s) * t_s  # (n_lon,1)
        cz = sin_lat[None, :] * sin_d + cos_lat[None, :] * cos_d * jnp.cos(h)
        return jnp.clip(cz, 0.0, None)

    # Daily-mean insolation factor (mean of clipped cos_zenith over a rotation):
    #   H0 = arccos(-tan(lat) tan(delta));  <cz> = (H0 sinφ sinδ + cosφ cosδ sinH0)/π
    x = jnp.clip(-jnp.tan(lat) * (sin_d / jnp.clip(cos_d, 1e-6, None)), -1.0, 1.0)
    h0 = jnp.arccos(x)  # (n_lat,)
    mean_cz = (h0 * sin_lat * sin_d + cos_lat * cos_d * jnp.sin(h0)) / math.pi
    mean_cz = jnp.clip(mean_cz, 0.0, None)  # (n_lat,)
    n_lon = jnp.asarray(lon_rad).shape[0]
    return jnp.broadcast_to(mean_cz[None, :], (n_lon, mean_cz.shape[0]))


class ColumnPhysicsState(NamedTuple):
    """JAX-pytree state for the dycore plus non-advected surface reservoirs.

    Keeping the Dinosaur state intact makes this the stable seam for conventional
    or learned JCM-style column parameterizations. Surface temperature and frost
    are nodal ``(1, n_lon, n_lat)`` fields; frost is pressure-equivalent and may be
    zero when the CO2 cycle is disabled.
    """

    dynamics: object
    surface_temperature: object
    co2_ice: object
    ground_temperature: object


class SurfaceEnergyDiagnostics(NamedTuple):
    absorbed_shortwave_w_m2: object
    outgoing_longwave_w_m2: object
    sensible_heat_w_m2: object
    net_external_w_m2: object


class RadiativeFluxDiagnostics(NamedTuple):
    """Interface fluxes, ordered from TOA (0) to the surface (L)."""

    shortwave_down_w_m2: object
    longwave_up_w_m2: object
    longwave_down_w_m2: object
    atmospheric_convergence_w_m2: object
    surface_net_w_m2: object
    toa_net_down_w_m2: object


def _validate_radiative_bands(f: RadiativeForcing) -> None:
    """Fail early when static spectral configuration has inconsistent lengths."""
    sw_lengths = {
        len(f.co2_shortwave_band_weights),
        len(f.co2_shortwave_band_strengths),
        len(f.co2_shortwave_pressure_exponents),
        len(f.co2_shortwave_temperature_exponents),
    }
    lw_lengths = {
        len(f.co2_longwave_band_weights),
        len(f.co2_longwave_band_strengths),
        len(f.co2_longwave_pressure_exponents),
        len(f.co2_longwave_temperature_exponents),
    }
    if len(sw_lengths) != 1 or 0 in sw_lengths:
        raise ValueError("shortwave CO2 band tuples must have one common non-zero length")
    if len(lw_lengths) != 1 or 0 in lw_lengths:
        raise ValueError("longwave CO2 band tuples must have one common non-zero length")
    if sum(f.co2_shortwave_band_weights) <= 0.0:
        raise ValueError("shortwave CO2 band weights must have a positive sum")
    if sum(f.co2_longwave_band_weights) <= 0.0:
        raise ValueError("longwave CO2 band weights must have a positive sum")


class ColumnPhysicsTendencies(NamedTuple):
    """Replaceable JCM-style physics contribution to the prognostic tendencies."""

    vorticity: object
    divergence: object
    temperature_variation: object
    log_surface_pressure: object
    surface_temperature: object
    co2_ice: object
    ground_temperature: object
    tracers: object


def column_primitive_equations(base, parameterization):
    """Compose Dinosaur dynamics with a conventional or learned column-physics callable.

    ``parameterization(state)`` returns :class:`ColumnPhysicsTendencies`. This is
    the stable integration seam intended for later JCM-inspired deterministic
    packages and NeuralGCM-style learned residual tendencies.
    """
    def explicit_terms(state):
        dry = base.explicit_terms(state.dynamics)
        phy = parameterization(state)
        dynamics = dataclasses.replace(
            dry,
            vorticity=dry.vorticity + phy.vorticity,
            divergence=dry.divergence + phy.divergence,
            temperature_variation=(dry.temperature_variation
                                   + phy.temperature_variation),
            log_surface_pressure=(dry.log_surface_pressure
                                  + phy.log_surface_pressure),
            tracers={
                name: value + phy.tracers.get(name, 0.0)
                for name, value in dry.tracers.items()
            },
        )
        return ColumnPhysicsState(
            dynamics, phy.surface_temperature, phy.co2_ice, phy.ground_temperature
        )

    def implicit_terms(state):
        return ColumnPhysicsState(
            base.implicit_terms(state.dynamics),
            jnp.zeros_like(state.surface_temperature),
            jnp.zeros_like(state.co2_ice),
            jnp.zeros_like(state.ground_temperature),
        )

    def implicit_inverse(state, step_size):
        return ColumnPhysicsState(
            base.implicit_inverse(state.dynamics, step_size),
            state.surface_temperature,
            state.co2_ice,
            state.ground_temperature,
        )

    return time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms, implicit_terms, implicit_inverse
    )


def initial_column_state(dyn_state, coords, surface_temperature_k: float, specs,
                         ice_pa: float = 0.0) -> ColumnPhysicsState:
    """Attach prognostic surface temperature and frost reservoirs to a dycore state."""
    ts_nd = float(specs.nondimensionalize(surface_temperature_k * _u.kelvin))
    ice_nd = float(specs.nondimensionalize(ice_pa * _u.pascal))
    shape = coords.surface_nodal_shape
    ground = jnp.full(
        (len(_REGOLITH_LAYER_FRACTIONS),) + shape[1:],
        ts_nd,
    )
    return ColumnPhysicsState(
        dyn_state, jnp.full(shape, ts_nd), jnp.full(shape, ice_nd), ground
    )


def two_stream_radiative_fluxes(
    state: ColumnPhysicsState, coords, specs, body: BodyConstants,
    f: RadiativeForcing,
) -> RadiativeFluxDiagnostics:
    """Multiband pressure/temperature-scaled CO2/dust radiative fluxes.

    The compact solver uses Beer--Lambert transmission in two solar bands and a
    hemispheric two-stream solve in three thermal bands. Each band responds
    differently to local pressure and temperature. Thermal emission is divided
    by normalized Planck-like band weights and emitted equally upward/downward
    according to Kirchhoff's law. Its principal contract is exact discrete energy
    closure; Ames-style correlated-k coefficients can later replace the optical
    depth closure without changing callers.
    """
    _validate_radiative_bands(f)
    grid, dyn = coords.horizontal, state.dynamics
    n = coords.vertical.layers
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
    t_s = dyn.sim_time * time_scale_s
    cz = cos_zenith_nodal(t_s, grid.latitudes, grid.longitudes, f)
    incoming = solar_flux(t_s, f) * cz
    ps_nd = jnp.exp(grid.to_nodal(dyn.log_surface_pressure))[0]
    ps_pa = ps_nd * float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    dsigma = jnp.asarray(np.diff(np.asarray(coords.vertical.boundaries)))[:, None, None]

    ref = np.asarray(reference_temperature(coords, body)).reshape(n, 1, 1)
    air_k = jnp.clip(grid.to_nodal(dyn.temperature_variation) + ref, 1.0, None)
    surface_k = jnp.clip(state.surface_temperature[0], 1.0, None)

    sigma_mid = jnp.asarray(
        0.5 * (np.asarray(coords.vertical.boundaries[:-1])
               + np.asarray(coords.vertical.boundaries[1:]))
    )[:, None, None]
    local_pressure_ratio = jnp.clip(
        sigma_mid * ps_pa[None] / body.reference_surface_pressure_pa, 1.0e-6, None
    )
    column_pressure_ratio = ps_pa / body.reference_surface_pressure_pa
    temperature_ratio = jnp.clip(
        f.co2_reference_temperature_k / air_k, 0.25, 4.0
    )

    def band_tau(base_tau, strengths, p_exponents, t_exponents, dust_tau=0.0):
        """Return ``(band, layer, lon, lat)`` optical depths."""
        values = []
        for strength, p_exp, t_exp in zip(strengths, p_exponents, t_exponents):
            gas = (
                base_tau * strength * column_pressure_ratio[None] * dsigma
                * local_pressure_ratio ** (p_exp - 1.0)
                * temperature_ratio ** t_exp
            )
            values.append(gas + jnp.asarray(dust_tau)[None] * dsigma)
        return jnp.stack(values)

    blackbody_air = f.stefan_boltzmann * air_k**4
    surface_emission = f.emissivity * f.stefan_boltzmann * surface_k**4
    if f.ames_correlated_k_enabled:
        from src.gcm3d import ames_radiation

        pressure_mid_pa = sigma_mid * ps_pa[None]
        delta_pressure_pa = dsigma * ps_pa[None]
        sw_tau, lw_tau = ames_radiation.correlated_k_optical_depths(
            air_k, pressure_mid_pa, delta_pressure_pa
        )
        data = ames_radiation.load_ames_co2_tables()
        sw_channel_weights = ames_radiation.channel_weights(
            data["clear_fraction_sw"]
        )
        lw_channel_weights = ames_radiation.channel_weights(
            data["clear_fraction_ir"]
        )
        # Dust is still grey within SW/LW here; spectral aerosol properties are
        # the next milestone. It is applied to every k channel consistently.
        dust_sw = (jnp.asarray(f.dust_visible_optical_depth)
                   * (1.0 - f.dust_single_scattering_albedo)
                   * column_pressure_ratio)[None, None, None] * dsigma[None, None]
        dust_lw = (jnp.asarray(f.dust_longwave_optical_depth)
                   * column_pressure_ratio)[None, None, None] * dsigma[None, None]
        sw_tau = sw_tau + dust_sw
        lw_tau = lw_tau + dust_lw
        # Direct solar path length scales as 1/cos(zenith), as in Ames dsolflux.
        mu = jnp.clip(cz, 0.05, 1.0)
        sw_trans = jnp.exp(-jnp.clip(sw_tau / mu[None, None, None], 0.0, 50.0))
        lw_trans = jnp.exp(-jnp.clip(lw_tau / 0.5, 0.0, 50.0))
        solar_weights = jnp.asarray(data["solar_weights"])
        flux = (incoming[None, None] * solar_weights[:, None, None, None]
                * sw_channel_weights[:, :, None, None])
        sw_levels = [flux]
        for k in range(n):
            flux = flux * sw_trans[:, :, k]
            sw_levels.append(flux)
        sw = jnp.sum(jnp.stack(sw_levels), axis=(1, 2))

        planck_air = ames_radiation.planck_band_fractions(air_k)
        planck_surface = ames_radiation.planck_band_fractions(surface_k)
        channel = lw_channel_weights[:, :, None, None]
        up = [surface_emission[None, None] * planck_surface[:, None] * channel]
        for k in range(n - 1, -1, -1):
            emission = blackbody_air[k][None, None] * planck_air[:, None, k] * channel
            up.append(up[-1] * lw_trans[:, :, k]
                      + (1.0 - lw_trans[:, :, k]) * emission)
        lw_up = jnp.sum(jnp.stack(up[::-1]), axis=(1, 2))
        down = [jnp.zeros_like(up[0])]
        for k in range(n):
            emission = blackbody_air[k][None, None] * planck_air[:, None, k] * channel
            down.append(down[-1] * lw_trans[:, :, k]
                        + (1.0 - lw_trans[:, :, k]) * emission)
        lw_down = jnp.sum(jnp.stack(down), axis=(1, 2))
    else:
        # Compact multiband fallback used for ablations.
        sw_tau = band_tau(
            f.co2_near_ir_optical_depth,
            f.co2_shortwave_band_strengths,
            f.co2_shortwave_pressure_exponents,
            f.co2_shortwave_temperature_exponents,
            jnp.asarray(f.dust_visible_optical_depth)
            * (1.0 - f.dust_single_scattering_albedo)
            * column_pressure_ratio,
        )
        lw_tau = band_tau(
            f.co2_longwave_optical_depth,
            f.co2_longwave_band_strengths,
            f.co2_longwave_pressure_exponents,
            f.co2_longwave_temperature_exponents,
            jnp.asarray(f.dust_longwave_optical_depth) * column_pressure_ratio,
        )
        sw_trans = jnp.exp(-jnp.clip(sw_tau, 0.0, 50.0))
        lw_trans = jnp.exp(-jnp.clip(lw_tau, 0.0, 50.0))
        sw_weights = jnp.asarray(f.co2_shortwave_band_weights)
        sw_weights = sw_weights / jnp.sum(sw_weights)
        sw_bands = []
        for band in range(len(f.co2_shortwave_band_weights)):
            flux = [incoming * sw_weights[band]]
            for k in range(n):
                flux.append(flux[-1] * sw_trans[band, k])
            sw_bands.append(jnp.stack(flux))
        sw = jnp.sum(jnp.stack(sw_bands), axis=0)
        lw_weights = jnp.asarray(f.co2_longwave_band_weights)
        lw_weights = lw_weights / jnp.sum(lw_weights)
        up_bands, down_bands = [], []
        for band in range(len(f.co2_longwave_band_weights)):
            up = [surface_emission * lw_weights[band]]
            for k in range(n - 1, -1, -1):
                emission = blackbody_air[k] * lw_weights[band]
                up.append(up[-1] * lw_trans[band, k]
                          + (1.0 - lw_trans[band, k]) * emission)
            up_bands.append(jnp.stack(up[::-1]))
            down = [jnp.zeros_like(surface_k)]
            for k in range(n):
                emission = blackbody_air[k] * lw_weights[band]
                down.append(down[-1] * lw_trans[band, k]
                            + (1.0 - lw_trans[band, k]) * emission)
            down_bands.append(jnp.stack(down))
        lw_up = jnp.sum(jnp.stack(up_bands), axis=0)
        lw_down = jnp.sum(jnp.stack(down_bands), axis=0)

    net_up = lw_up - lw_down - sw
    convergence = net_up[1:] - net_up[:-1]
    surface_net = (1.0 - f.albedo) * sw[-1] + lw_down[-1] - surface_emission
    reflected = f.albedo * sw[-1]
    toa_net_down = sw[0] - lw_up[0] - reflected
    return RadiativeFluxDiagnostics(
        sw, lw_up, lw_down, convergence, surface_net, toa_net_down
    )


def surface_energy_tendencies(state: ColumnPhysicsState, coords, specs,
                              body: BodyConstants, f: RadiativeForcing,
                              wind_nodal=None):
    """Conservative surface/atmosphere energy exchange.

    Solar and longwave fluxes act once on a prognostic surface reservoir. A bulk
    sensible flux transfers energy to the lowest atmospheric sigma layer, divided
    by its actual areal heat capacity ``cp * dp/g``. Thus the surface loss and
    atmospheric gain cancel exactly and do not depend on the number of layers.
    """
    grid = coords.horizontal
    n_layers = coords.vertical.layers
    dyn = state.dynamics

    # sim_time is nondimensional; convert to seconds (time scale = 1/(2 Omega)).
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
    t_s = dyn.sim_time * time_scale_s

    cz = cos_zenith_nodal(t_s, grid.latitudes, grid.longitudes, f)  # (n_lon,n_lat)
    # Lowest atmospheric-layer and surface temperatures in kelvin.
    ref_t = np.asarray(reference_temperature(coords, body)).reshape(n_layers, 1, 1)
    air_k = jnp.clip(grid.to_nodal(dyn.temperature_variation) + ref_t, 1.0, None)
    surface_k = jnp.clip(state.surface_temperature[0], 1.0, None)
    if f.co2_radiation_enabled:
        radiation = two_stream_radiative_fluxes(state, coords, specs, body, f)
        q_in = (1.0 - f.albedo) * radiation.shortwave_down_w_m2[-1]
        q_out = radiation.longwave_up_w_m2[-1] - radiation.longwave_down_w_m2[-1]
    else:
        q_in = (1.0 - f.albedo) * solar_flux(t_s, f) * cz
        q_out = f.emissivity * f.stefan_boltzmann * (
            surface_k / max(f.greenhouse_factor, 1.0)
        ) ** 4
    if f.stability_exchange_enabled:
        exchange, _, _ = _surface_exchange_properties(
            state, coords, specs, body, f, air_k[-1], surface_k, ps_pa=None,
            wind_nodal=wind_nodal,
        )
        sensible = exchange * (surface_k - air_k[-1])
    else:
        sensible = f.sensible_heat_transfer_w_m2_k * (surface_k - air_k[-1])

    # Surface reservoir: external radiation minus energy transferred to atmosphere.
    dts_si = (q_in - q_out - sensible) / f.thermal_inertia

    # Lowest-layer areal heat capacity cp*dp/g. In sigma coordinates
    # dp = p_s * delta_sigma, so thickening the atmosphere correctly increases its
    # thermal inertia. No flux is duplicated in the other layers.
    ps_nd = jnp.exp(grid.to_nodal(dyn.log_surface_pressure))[0]
    pa_per_nd = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    ps_pa = ps_nd * pa_per_nd
    dsigma = float(np.diff(np.asarray(coords.vertical.boundaries))[-1])
    cp = body.gas_constant_j_kg_k / body.kappa
    lowest_capacity = cp * jnp.clip(ps_pa * dsigma / body.gravity_m_s2, 1e-6, None)
    dtair_si = sensible / lowest_capacity
    atm_nodal = jnp.zeros((n_layers,) + sensible.shape).at[-1].set(
        dtair_si * time_scale_s
    )
    if f.co2_radiation_enabled:
        layer_capacity = (
            body.cp_j_kg_k * ps_pa[None] *
            jnp.asarray(np.diff(np.asarray(coords.vertical.boundaries)))[:, None, None]
            / body.gravity_m_s2
        )
        atm_nodal = atm_nodal + radiation.atmospheric_convergence_w_m2 / jnp.clip(
            layer_capacity, 1e-6, None
        ) * time_scale_s
    surface_nd = (dts_si * time_scale_s)[None, :, :]
    diagnostics = SurfaceEnergyDiagnostics(q_in, q_out, sensible, q_in - q_out)
    return grid.to_modal(atm_nodal), surface_nd, diagnostics


def surface_momentum_tendencies(
    state: ColumnPhysicsState, coords, specs, body: BodyConstants, f: RadiativeForcing,
    wind_nodal=None,
):
    """Neutral-log-law surface stress applied to the lowest sigma layer.

    The stress is ``tau = rho C_D |V| V`` and the layer acceleration is
    ``-tau / (dp/g)``.  ``C_D = (kappa/log(z/z0))**2`` uses the hydrostatic
    height of the lowest-layer pressure midpoint.  The tendency always removes
    resolved kinetic energy and approaches zero continuously with wind speed.
    """
    grid, dyn = coords.horizontal, state.dynamics
    if wind_nodal is None:
        wind_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            grid, dyn.vorticity, dyn.divergence
        )
    u_nd, v_nd = wind_nodal
    velocity_unit = _u.meter / _u.second
    u_ms = specs.dimensionalize(u_nd, velocity_unit).magnitude
    v_ms = specs.dimensionalize(v_nd, velocity_unit).magnitude

    ref_t = np.asarray(reference_temperature(coords, body)).reshape(coords.vertical.layers, 1, 1)
    air_k = jnp.clip(grid.to_nodal(dyn.temperature_variation)[-1] + ref_t[-1], 50.0, None)
    surface_k = jnp.clip(state.surface_temperature[0], 50.0, None)
    _, drag_coefficient, speed = _surface_exchange_properties(
        state, coords, specs, body, f, air_k, surface_k, ps_pa=None,
        wind_nodal=wind_nodal,
    )
    dsigma = float(np.diff(np.asarray(coords.vertical.boundaries))[-1])
    # rho/(dp/g) = g/(R*T*dsigma), using the actual lowest-layer temperature.
    rate_s = drag_coefficient * speed * body.gravity_m_s2 / (
        body.gas_constant_j_kg_k * air_k * dsigma
    )
    du_si = jnp.zeros_like(u_ms).at[-1].set(-rate_s * u_ms[-1])
    dv_si = jnp.zeros_like(v_ms).at[-1].set(-rate_s * v_ms[-1])
    acceleration_unit = _u.meter / (_u.second**2)
    du_nd = specs.nondimensionalize(du_si * acceleration_unit)
    dv_nd = specs.nondimensionalize(dv_si * acceleration_unit)
    return spherical_harmonic.uv_nodal_to_vor_div_modal(grid, du_nd, dv_nd)


def _surface_exchange_properties(
    state, coords, specs, body, f, air_k, surface_k, ps_pa=None, wind_nodal=None
):
    """Return sensible conductance, drag coefficient and resolved wind speed."""
    grid = coords.horizontal
    if wind_nodal is None:
        wind_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            grid, state.dynamics.vorticity, state.dynamics.divergence
        )
    u_nd, v_nd = wind_nodal
    unit = _u.meter / _u.second
    u = specs.dimensionalize(u_nd[-1], unit).magnitude
    v = specs.dimensionalize(v_nd[-1], unit).magnitude
    speed = jnp.sqrt(u**2 + v**2 + f.minimum_wind_ms**2)
    sigma = float(np.asarray(coords.vertical.centers)[-1])
    scale_height = body.gas_constant_j_kg_k * body.reference_temperature_k / body.gravity_m_s2
    height = max(-scale_height * math.log(sigma), 1.01 * f.surface_roughness_m)
    neutral_cd = (f.von_karman_constant / math.log(height / f.surface_roughness_m)) ** 2
    if f.stability_exchange_enabled:
        ri = body.gravity_m_s2 * height * (air_k - surface_k) / (
            jnp.clip(air_k, 50.0, None) * speed**2
        )
        stable = jnp.clip(1.0 - 5.0 * ri, 0.1, 1.0) ** 2
        unstable = jnp.sqrt(jnp.clip(1.0 - 16.0 * ri, 1.0, None))
        cd = neutral_cd * jnp.where(ri >= 0.0, stable, unstable)
    else:
        cd = neutral_cd
    if ps_pa is None:
        ps_nd = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]
        ps_pa = ps_nd * float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    rho = ps_pa / (body.gas_constant_j_kg_k * jnp.clip(air_k, 50.0, None))
    sensible_conductance = rho * body.cp_j_kg_k * cd * speed
    return sensible_conductance, cd, speed


def pbl_vertical_diffusion_tendencies(
    state, coords, specs, body, f, wind_nodal=None
):
    """Mass-conserving adjacent-layer diffusion of momentum and temperature."""
    dyn, grid = state.dynamics, coords.horizontal
    zeros = jnp.zeros_like(dyn.temperature_variation)
    if not f.pbl_diffusion_enabled:
        return (
            jnp.zeros_like(dyn.vorticity), jnp.zeros_like(dyn.divergence), zeros,
            {name: jnp.zeros_like(value) for name, value in dyn.tracers.items()},
        )
    if wind_nodal is None:
        wind_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            grid, dyn.vorticity, dyn.divergence
        )
    u, v = wind_nodal
    temperature = grid.to_nodal(dyn.temperature_variation)
    sigma = np.asarray(coords.vertical.centers)
    dsigma = np.diff(np.asarray(coords.vertical.boundaries))
    scale_height = body.gas_constant_j_kg_k * body.reference_temperature_k / body.gravity_m_s2
    z = -scale_height * np.log(np.clip(sigma, 1e-6, None))
    velocity_unit = _u.meter / _u.second
    u_ms = specs.dimensionalize(u, velocity_unit).magnitude
    v_ms = specs.dimensionalize(v, velocity_unit).magnitude
    ref = np.asarray(reference_temperature(coords, body)).reshape(coords.vertical.layers, 1, 1)
    actual_t = temperature + ref
    theta = actual_t / jnp.asarray(sigma[:, None, None]) ** body.kappa
    _, cd, speed = _surface_exchange_properties(
        state, coords, specs, body, f,
        jnp.ones(grid.nodal_shape) * body.reference_temperature_k,
        state.surface_temperature[0],
        wind_nodal=wind_nodal,
    )
    ustar = jnp.sqrt(cd) * speed
    time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
    interface_rates = []
    def diffuse(field):
        tendency = jnp.zeros_like(field)
        for k in range(coords.vertical.layers - 1):
            interface_z = min(z[k], z[k + 1])
            shape = max(0.0, 1.0 - interface_z / f.pbl_height_m) ** 2
            rate = f.von_karman_constant * ustar * max(interface_z, 1.0) * shape
            dz_local = max(z[k] - z[k + 1], 1.0)
            shear2 = (
                ((u_ms[k] - u_ms[k + 1]) / dz_local) ** 2
                + ((v_ms[k] - v_ms[k + 1]) / dz_local) ** 2
                + 1.0e-10
            )
            ri_gradient = (
                body.gravity_m_s2
                / jnp.clip(0.5 * (theta[k] + theta[k + 1]), 50.0, None)
                * ((theta[k] - theta[k + 1]) / dz_local)
                / shear2
            )
            stable_factor = jnp.clip(1.0 - 5.0 * ri_gradient, 0.0, 1.0) ** 2
            unstable_factor = jnp.clip(
                jnp.sqrt(jnp.clip(1.0 - 16.0 * ri_gradient, 1.0, None)),
                1.0,
                4.0,
            )
            rate = rate * jnp.where(
                ri_gradient >= 0.0, stable_factor, unstable_factor
            )
            rate = rate / max((z[k] - z[k + 1]) ** 2, 1.0)
            rate = jnp.minimum(rate, 1.0 / 1800.0) * time_scale
            if len(interface_rates) <= k:
                interface_rates.append(rate / time_scale)
            ratio = float(dsigma[k] / dsigma[k + 1])
            exchange = rate * (field[k + 1] - field[k])
            tendency = tendency.at[k].add(exchange)
            tendency = tendency.at[k + 1].add(-ratio * exchange)
        return tendency
    dt = diffuse(temperature)
    rates_si = jnp.stack(interface_rates)
    du, u_new = _implicit_vertical_diffusion_tendency(
        u_ms, rates_si, dsigma, f.pbl_implicit_timestep_s, specs, velocity=True
    )
    dv, v_new = _implicit_vertical_diffusion_tendency(
        v_ms, rates_si, dsigma, f.pbl_implicit_timestep_s, specs, velocity=True
    )
    # Backward diffusion dissipates resolved kinetic energy. Return that energy
    # uniformly to the mixed air column as heat, preserving the resolved budget.
    old_ke = jnp.sum(jnp.asarray(dsigma)[:, None, None] * (u_ms**2 + v_ms**2) / 2.0, axis=0)
    new_ke = jnp.sum(jnp.asarray(dsigma)[:, None, None] * (u_new**2 + v_new**2) / 2.0, axis=0)
    heat_si = jnp.clip(old_ke - new_ke, 0.0, None) / (
        body.cp_j_kg_k * jnp.sum(jnp.asarray(dsigma)) * f.pbl_implicit_timestep_s
    )
    dt = dt + heat_si[None] * time_scale
    vor, div = spherical_harmonic.uv_nodal_to_vor_div_modal(grid, du, dv)
    tracer_tendencies = {
        name: grid.to_modal(diffuse(grid.to_nodal(value)))
        for name, value in dyn.tracers.items()
    }
    return vor, div, grid.to_modal(dt), tracer_tendencies


def _implicit_vertical_diffusion_tendency(
    field_si, interface_rate_s, layer_weights, timestep_s, specs, *, velocity=False
):
    """Backward-Euler conservative column diffusion on arbitrary nodal columns."""
    n = field_si.shape[0]
    weights = jnp.asarray(layer_weights)
    operator = jnp.zeros(field_si.shape[1:] + (n, n))
    for k in range(n - 1):
        rate = interface_rate_s[k]
        ratio = weights[k] / weights[k + 1]
        operator = operator.at[..., k, k].add(-rate)
        operator = operator.at[..., k, k + 1].add(rate)
        operator = operator.at[..., k + 1, k].add(ratio * rate)
        operator = operator.at[..., k + 1, k + 1].add(-ratio * rate)
    matrix = jnp.eye(n) - timestep_s * operator
    old = jnp.moveaxis(field_si, 0, -1)
    new = jnp.linalg.solve(matrix, old[..., None])[..., 0]
    new = jnp.moveaxis(new, -1, 0)
    tendency_si = (new - field_si) / timestep_s
    if velocity:
        tendency = specs.nondimensionalize(
            tendency_si * (_u.meter / (_u.second**2))
        )
    else:
        time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
        tendency = tendency_si * time_scale
    return tendency, new


def dry_convective_adjusted_temperature(state, coords, body):
    """Return an enthalpy-conserving, statically neutral/stable temperature field."""
    grid = coords.horizontal
    n = coords.vertical.layers
    ref = np.asarray(reference_temperature(coords, body)).reshape(n, 1, 1)
    temperature = grid.to_nodal(state.dynamics.temperature_variation) + ref
    sigma = jnp.asarray(coords.vertical.centers).reshape(n, 1, 1)
    exner = sigma ** body.kappa
    theta = temperature / exner
    weights = np.diff(np.asarray(coords.vertical.boundaries))
    # First milestone: if a column contains any unstable pair, mix the dry column
    # to a single potential temperature. This is intentionally more diffusive than
    # a later PAVA/block adjustment, but is exact, deterministic and conservative.
    unstable = jnp.any(theta[:-1] < theta[1:], axis=0)
    w = jnp.asarray(weights).reshape(n, 1, 1)
    mixed = jnp.sum(w * theta * exner, axis=0) / jnp.sum(w * exner, axis=0)
    theta = jnp.where(unstable[None, :, :], mixed[None, :, :], theta)
    adjusted = theta * exner
    return grid.to_modal(adjusted - ref)


def dry_convective_adjustment_tendency(state, coords, specs, body, f):
    if not f.convective_adjustment_enabled:
        return jnp.zeros_like(state.dynamics.temperature_variation)
    target = dry_convective_adjusted_temperature(state, coords, body)
    time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
    return (
        target - state.dynamics.temperature_variation
    ) * time_scale / f.convective_relaxation_s


def regolith_conduction_tendencies(
    state: ColumnPhysicsState, specs, f: RadiativeForcing
):
    """Conservative finite-volume conduction through skin-depth-scaled soil layers.

    Positive interface flux points downward. The top flux is removed from the
    prognostic surface reservoir and added to the first soil layer; the bottom
    boundary has zero flux. Thus the area-integrated internal energy tendency is
    zero to roundoff in every column.
    """
    if not f.regolith_enabled:
        return (
            jnp.zeros_like(state.surface_temperature),
            jnp.zeros_like(state.ground_temperature),
        )
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
    ts = state.surface_temperature[0]
    tg = state.ground_temperature
    inertia = jnp.asarray(f.surface_thermal_inertia_tiu)
    cv = f.regolith_volumetric_heat_capacity_j_m3_k
    skin_depth = inertia / cv * math.sqrt(f.rotation_period_s / math.pi)
    fractions = jnp.asarray(f.regolith_layer_skin_depth_fractions).reshape((-1, 1, 1))
    dz = jnp.clip(fractions * skin_depth, 1.0e-4, None)
    conductivity = inertia**2 / cv

    top_distance = 0.5 * dz[0]
    top_flux = conductivity * (ts - tg[0]) / top_distance
    center_distance = 0.5 * (dz[:-1] + dz[1:])
    internal_flux = conductivity * (tg[:-1] - tg[1:]) / center_distance
    fluxes = jnp.concatenate(
        [top_flux[None], internal_flux, jnp.zeros_like(top_flux)[None]], axis=0
    )
    ground_si = (fluxes[:-1] - fluxes[1:]) / (cv * dz)
    surface_si = -top_flux / f.thermal_inertia
    return surface_si[None] * time_scale_s, ground_si * time_scale_s


def radiative_heating_tendency(state, coords, specs, body: BodyConstants,
                               f: RadiativeForcing):
    """Compatibility accessor for the conservative atmospheric tendency.

    ``state`` must now be :class:`ColumnPhysicsState`; radiation cannot be closed
    without the prognostic surface reservoir.
    """
    if not isinstance(state, ColumnPhysicsState):
        raise TypeError("radiative forcing requires a ColumnPhysicsState")
    return surface_energy_tendencies(state, coords, specs, body, f)[0]


def forced_primitive_equations(
    coords,
    body: BodyConstants,
    forcing: RadiativeForcing,
    specs=None,
    orography=None,
) -> "time_integration.ImplicitExplicitODE":
    """dinosaur dry dynamics with the radiative energy balance added as forcing.

    Returns an ``ImplicitExplicitODE`` whose implicit side is dinosaur's unchanged
    (the semi-implicit gravity-wave treatment) and whose explicit side is the dry
    dynamical tendency **plus** :func:`radiative_heating_tendency`. Integrate it
    with :func:`src.gcm3d.stepper` / :func:`src.gcm3d.integrate` exactly like the
    dry equations; the only requirement is that the state carries ``sim_time`` (set
    it to ``0.0``) so the diurnal/seasonal forcing advances.
    """
    if specs is None:
        specs = physics_specs(body)
    base = _build_primitive_equations(coords, body, specs=specs, orography=orography)

    def parameterization(state):
        wind_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            coords.horizontal,
            state.dynamics.vorticity,
            state.dynamics.divergence,
        )
        heat, surface_tendency, _ = surface_energy_tendencies(
            state, coords, specs, body, forcing, wind_nodal=wind_nodal
        )
        drag_vor, drag_div = surface_momentum_tendencies(
            state, coords, specs, body, forcing, wind_nodal=wind_nodal
        )
        mix_vor, mix_div, mix_heat, mix_tracers = pbl_vertical_diffusion_tendencies(
            state, coords, specs, body, forcing, wind_nodal=wind_nodal
        )
        convection = dry_convective_adjustment_tendency(
            state, coords, specs, body, forcing
        )
        ground_surface, ground = regolith_conduction_tendencies(state, specs, forcing)
        return ColumnPhysicsTendencies(
            drag_vor + mix_vor,
            drag_div + mix_div,
            heat + mix_heat + convection,
            jnp.zeros_like(state.dynamics.log_surface_pressure),
            surface_tendency + ground_surface,
            jnp.zeros_like(state.co2_ice),
            ground,
            mix_tracers,
        )
    return column_primitive_equations(base, parameterization)


# ==============================================================================
# CO2 condensation cycle (the second half of the 0-D physics, now spatial).
#
# The 0-D model carries two global polar-cap reservoirs. In 3-D the honest analogue
# is the Leighton-Murray cycle: CO2 condenses out of the atmosphere wherever the
# surface reaches the CO2 frost point (the winter poles), releasing latent heat and
# *removing atmospheric mass* (local surface pressure drops); the deposited frost
# sublimes back when insolation warms it, restoring the mass. This buffers polar
# temperatures near the frost point — the missing physics that let the dry+radiation
# run's winter pole cool to an unphysical ~66 K.
#
# Surface frost sits on the ground, so it is *not* advected by the 3-D wind and must
# not be a dinosaur tracer (those get transported). Instead the state becomes a
# :class:`ColumnPhysicsState` JAX pytree. ``co2_ice`` is stored in pressure-equivalent
# units (the surface pressure the frost would contribute if fully sublimed), so mass
# conservation is exact per cell: ``d(p_s) = -d(ice) - escape``.
# ==============================================================================


@dataclasses.dataclass(frozen=True)
class CO2Forcing:
    """Constants for the 3-D CO2 condensation/sublimation cycle.

    ``frost_point_k`` is treated as fixed (the CO2 saturation temperature varies
    only weakly over Mars's pressure range). ``exchange_rate_pa_s_per_k`` sets how
    fast the surface relaxes toward the frost point via condensation — the free
    parameter of this smooth (differentiable) scheme; the default buffers polar
    temperatures over a few sols without destabilising the explicit step.
    """

    frost_point_k: float               # constant fallback (used if use_pressure_frost=False)
    latent_heat_j_kg: float
    gravity_m_s2: float
    thermal_inertia: float
    # Use the pressure-dependent CO2 saturation temperature (Clausius-Clapeyron
    # curve) instead of the constant frost_point_k. See :func:`co2_frost_point_k`.
    use_pressure_frost: bool = True
    # Smooth condensation/sublimation rate (Pa of exchange per K of frost offset
    # per second) and the ice scale that gates sublimation to zero as frost runs out.
    exchange_rate_pa_s_per_k: float = 1.0e-4
    ice_ref_pa: float = 100.0
    # Condensation supply limit: the rate is gated by tanh(p_s / supply_scale_pa)
    # so it vanishes as the local atmosphere thins. Without this, runaway
    # condensation collapses the polar surface pressure to zero (d(ln p_s) blows
    # up as p_s -> 0) and the integration diverges — you cannot condense CO2 that
    # is no longer in the column.
    supply_scale_pa: float = 50.0
    # Non-thermal escape (uniform atmospheric mass sink), kg s^-1 over the globe.
    escape_rate_kg_s: float = 0.0
    # Diagnose phase change from the non-latent surface energy residual.  This
    # removes the tunable temperature-relaxation rate from the active scheme.
    energy_limited: bool = False
    # Four-stage IMEX evaluations can sample intermediate reservoirs, so use a
    # conservative four-step positivity horizon for the default 600 s step.
    exchange_timestep_s: float = 24000.0


def mars_co2_forcing(
    exchange_rate_pa_s_per_k: float = 1.0e-4,
    escape_rate_kg_s: float = 0.0,
    energy_limited: bool = True,
) -> CO2Forcing:
    """A :class:`CO2Forcing` built from the package's Mars CO2 constants."""
    from src.celestials.planets import mars as _m

    return CO2Forcing(
        frost_point_k=float(_m.MARS_CO2_FROST_POINT),
        latent_heat_j_kg=float(_m.MARS_CO2_LATENT_HEAT),
        gravity_m_s2=float(_m.MARS_BODY_3D.gravity_m_s2),
        thermal_inertia=float(_m.MARS_THERMAL_INERTIA),
        exchange_rate_pa_s_per_k=exchange_rate_pa_s_per_k,
        escape_rate_kg_s=escape_rate_kg_s,
        energy_limited=energy_limited,
    )


def co2_frost_point_k(pressure_pa):
    """CO2 condensation (frost-point) temperature (K) at partial pressure ``pressure_pa``.

    Uses the CO2 saturation-vapour-pressure (Clausius-Clapeyron) relation in the
    inverted, GCM-standard form

        T_sat = 3182.48 / (23.3494 - ln p[hPa])

    which is the CO2 condensation temperature used in Mars GCMs. It is well
    calibrated against the CO2 phase curve: it returns ~147.7 K at 6.1 hPa (Mars
    surface pressure) and 194.6 K at 1013 hPa (the CO2 sublimation point at 1 atm),
    and rises with pressure as Clausius-Clapeyron requires. ``pressure_pa`` is
    clipped to a small positive floor so the log stays finite in vacuum cells.
    """
    p_hpa = jnp.clip(jnp.asarray(pressure_pa) / 100.0, 1e-6, None)
    return 3182.48 / (23.3494 - jnp.log(p_hpa))


def _co2_surface_tendencies(
    state, coords, specs, body, cf: CO2Forcing,
    available_surface_flux_w_m2=None,
):
    """Per-cell CO2 exchange: returns pressure, frost, and surface-T tendencies.

    Works in SI (Pa s^-1, K s^-1) then nondimensionalises. ``ice_nd`` is the frost
    reservoir in nondimensional pressure-equivalent units (same scaling as p_s).
    Condensation (surface below the frost point) moves mass atmosphere -> ice and
    warms; sublimation (above frost, gated by available ice) moves it back and cools.
    """
    grid = coords.horizontal
    dyn, ice_nd = state.dynamics, state.co2_ice

    # Prognostic physical surface temperature (K) and surface pressure (Pa).
    t_surf_k = jnp.clip(state.surface_temperature[0], 1.0, None)
    ps_nd = jnp.exp(grid.to_nodal(dyn.log_surface_pressure))           # (1,n_lon,n_lat)
    ps_pa = np.asarray(specs.dimensionalize(1.0, _u.pascal).magnitude) * ps_nd

    # Ice reservoir in Pa (dimensional) for gating and rates.
    ice_pa = specs.dimensionalize(ice_nd, _u.pascal).magnitude        # (1,n_lon,n_lat)

    # Frost point: pressure-dependent CO2 saturation temperature per cell (or the
    # constant fallback). Using the local p_s makes the winter-pole cap form at the
    # right temperature and respond as the atmosphere thickens/thins.
    frost_k = (co2_frost_point_k(ps_pa[0]) if cf.use_pressure_frost
               else cf.frost_point_k)
    supply_gate = jnp.tanh(ps_pa[0] / cf.supply_scale_pa)
    ice_gate = jnp.tanh(jnp.clip(ice_pa[0], 0.0, None) / cf.ice_ref_pa)
    if cf.energy_limited:
        if available_surface_flux_w_m2 is None:
            raise ValueError("energy-limited CO2 exchange requires the surface-energy residual")
        residual = jnp.asarray(available_surface_flux_w_m2)
        pa_per_watt = cf.gravity_m_s2 / cf.latent_heat_j_kg
        cond_pa_s = jnp.where(
            t_surf_k <= frost_k, jnp.clip(-residual, 0.0, None) * pa_per_watt * supply_gate, 0.0
        )
        subl_pa_s = jnp.where(
            (t_surf_k >= frost_k) & (ice_pa[0] > 0.0),
            jnp.clip(residual, 0.0, None) * pa_per_watt * ice_gate,
            0.0,
        )
    else:
        below = jnp.clip(frost_k - t_surf_k, 0.0, None)
        above = jnp.clip(t_surf_k - frost_k, 0.0, None)
        cond_pa_s = cf.exchange_rate_pa_s_per_k * below * supply_gate
        subl_pa_s = cf.exchange_rate_pa_s_per_k * above * ice_gate
    # Positivity limiter for the explicit phase-change tendency. The configured
    # interval is the maximum physics step supported by this forcing.
    cond_pa_s = jnp.minimum(cond_pa_s, jnp.clip(ps_pa[0], 0.0, None) / cf.exchange_timestep_s)
    subl_pa_s = jnp.minimum(subl_pa_s, jnp.clip(ice_pa[0], 0.0, None) / cf.exchange_timestep_s)
    net_pa_s = cond_pa_s - subl_pa_s                                   # (n_lon,n_lat), >0 = condensing

    # Positivity repair for intermediate Runge--Kutta stages. It transfers any
    # tiny negative numerical reservoir back from the atmosphere, so total CO2
    # remains unchanged while the physical state is restored rapidly.
    repair_pa_s = jnp.clip(-ice_pa[0], 0.0, None) / 60.0

    # Escape: uniform atmospheric mass sink spread over the globe (Pa s^-1).
    escape_pa_s = cf.escape_rate_kg_s * cf.gravity_m_s2 / body.surface_area_m2

    # Mass budget (SI, per cell): ice gains net, atmosphere loses net + escape.
    dice_pa_s = (net_pa_s + repair_pa_s)[None, :, :]                   # (1,n_lon,n_lat)
    dps_pa_s = (-net_pa_s - repair_pa_s)[None, :, :] - escape_pa_s

    # Latent heating: condensing mass (net/g kg m^-2 s^-1) releases L (W m^-2).
    dT_latent_si = cf.latent_heat_j_kg * (net_pa_s / cf.gravity_m_s2) / cf.thermal_inertia

    # --- nondimensionalise ---
    to_nd_press_rate = lambda x: specs.nondimensionalize(x * (_u.pascal / _u.second))
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))

    dice_nd = to_nd_press_rate(dice_pa_s)                              # (1,n_lon,n_lat)
    dps_nd = to_nd_press_rate(dps_pa_s)                               # (1,n_lon,n_lat)
    d_logsp_nodal = dps_nd / ps_nd                                    # d(ln p_s)/dt
    d_logsp_modal = grid.to_modal(d_logsp_nodal)

    # Latent energy belongs to the surface reservoir where frost forms/sublimes.
    dT_latent_nd = (dT_latent_si * time_scale_s)[None, :, :]
    return d_logsp_modal, dice_nd, dT_latent_nd


def forced_co2_primitive_equations(
    coords,
    body: BodyConstants,
    forcing: RadiativeForcing,
    co2_forcing: CO2Forcing,
    specs=None,
    orography=None,
) -> "time_integration.ImplicitExplicitODE":
    """Dry dynamics + radiative forcing + CO2 condensation cycle on a tuple state.

    The ODE operates on :class:`ColumnPhysicsState`, carrying the Dinosaur state,
    prognostic surface temperature, and a nodal surface frost reservoir. Build it
    with :func:`initial_column_state` or :func:`initial_co2_state`; integrate with
    the ordinary :func:`src.gcm3d.stepper`/:func:`src.gcm3d.integrate`.
    """
    if specs is None:
        specs = physics_specs(body)
    base = _build_primitive_equations(coords, body, specs=specs, orography=orography)

    def parameterization(state):
        wind_nodal = spherical_harmonic.vor_div_to_uv_nodal(
            coords.horizontal,
            state.dynamics.vorticity,
            state.dynamics.divergence,
        )
        heat, dsurface, _ = surface_energy_tendencies(
            state, coords, specs, body, forcing, wind_nodal=wind_nodal
        )
        ground_surface, ground = regolith_conduction_tendencies(state, specs, forcing)
        time_scale = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
        surface_flux = (
            (dsurface + ground_surface)[0] / time_scale * forcing.thermal_inertia
        )
        d_logsp, dice, dlatent_surface = _co2_surface_tendencies(
            state, coords, specs, body, co2_forcing,
            available_surface_flux_w_m2=surface_flux,
        )
        drag_vor, drag_div = surface_momentum_tendencies(
            state, coords, specs, body, forcing, wind_nodal=wind_nodal
        )
        mix_vor, mix_div, mix_heat, mix_tracers = pbl_vertical_diffusion_tendencies(
            state, coords, specs, body, forcing, wind_nodal=wind_nodal
        )
        convection = dry_convective_adjustment_tendency(
            state, coords, specs, body, forcing
        )
        return ColumnPhysicsTendencies(
            drag_vor + mix_vor, drag_div + mix_div,
            heat + mix_heat + convection, d_logsp,
            dsurface + dlatent_surface + ground_surface,
            dice,
            ground,
            mix_tracers,
        )
    return column_primitive_equations(base, parameterization)


def project_co2_reservoirs(state: ColumnPhysicsState, coords, specs):
    """Project frost to non-negative values while conserving local CO2 mass.

    Explicit multistage schemes do not preserve positivity at a moving frost
    boundary. Any negative pressure-equivalent frost is set to zero and the same
    pressure deficit is removed from the atmospheric column. The correction is
    applied after a complete timestep, not within Runge--Kutta stages.
    """
    grid = coords.horizontal
    pressure_scale = float(specs.dimensionalize(1.0, _u.pascal).magnitude)
    ice_pa = state.co2_ice * pressure_scale
    ps_nd = jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))
    ps_pa = ps_nd * pressure_scale
    deficit_pa = jnp.clip(-ice_pa, 0.0, None)
    ice_pa = jnp.clip(ice_pa, 0.0, None)
    ps_pa = jnp.clip(ps_pa - deficit_pa, 1.0e-6, None)
    dynamics = dataclasses.replace(
        state.dynamics,
        log_surface_pressure=grid.to_modal(jnp.log(ps_pa / pressure_scale)),
    )
    return state._replace(
        dynamics=dynamics,
        co2_ice=ice_pa / pressure_scale,
    )


def positivity_preserving_co2_step(step_fn, coords, specs):
    """Wrap an IMEX step with the conservative CO2-reservoir projection."""
    def step(state):
        return project_co2_reservoirs(step_fn(state), coords, specs)
    return step


def initial_co2_state(dyn_state, coords, ice_pa: float = 0.0, specs=None, body=None,
                      surface_temperature_k: float | None = None):
    """Pair a dynamical ``State`` with an initial (uniform) CO2 frost field.

    ``ice_pa`` is the starting frost everywhere in Pa-equivalent (0 by default, i.e.
    frost forms from the atmosphere as poles cool). Returns the ``(dyn, ice)`` tuple
    the CO2 ODE integrates.
    """
    if body is None:
        from src.celestials.planets.mars import MARS_BODY_3D
        body = MARS_BODY_3D
    if specs is None:
        specs = physics_specs(body)
    if surface_temperature_k is None:
        surface_temperature_k = body.reference_temperature_k
    return initial_column_state(
        dyn_state, coords, surface_temperature_k, specs, ice_pa=ice_pa
    )
