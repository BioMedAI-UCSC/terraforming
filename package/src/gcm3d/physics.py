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

The heating law is the *same* nonlinear surface energy balance as the 0-D kernel
(``compute_derivatives``): ``dT/dt = (Q_in - eps*sigma*(T/greenhouse)**4) / C``,
with ``Q_in = (1-albedo)*S*cos_zenith`` and a per-column solar zenith angle from
latitude, solar declination and hour angle. It is applied uniformly through the
column (the dry core has no convection/vertical diffusion to redistribute a
surface-only flux, so a bottom-only heating would be statically unstable). Because
it is the explicit 0-D balance, it inherits the 0-D stability limit: the step must
resolve the diurnal cycle (``dt <= rotation_period/8``); use daily-mean insolation
(``diurnal=False``) if you must step coarser.

Honesty about fidelity: this is a single-band ("grey") radiative *forcing*, not a
radiative-transfer scheme — no spectral CO2 bands, no dust, and (yet) no CO2
condensation cycle. It fixes the largest gap in the dry maps (a fluid with no
thermal forcing); the CO2 cap mass/pressure exchange is the next increment and is
deliberately not included here. Requires the optional ``gcm3d`` extra.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np

from src.gcm3d._dinosaur import jnp, scales, time_integration
from src.gcm3d.body import BodyConstants
from src.gcm3d.dynamics import primitive_equations as _build_primitive_equations
from src.gcm3d.dynamics import reference_temperature
from src.gcm3d.specs import physics_specs

_u = scales.units
_TWO_PI = 2.0 * math.pi

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
    albedo: float
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


def mars_radiative_forcing(
    albedo: float = 0.25,
    greenhouse_factor: float = 1.02,
    diurnal: bool = True,
    init_orbital_angle_rad: float = 0.0,
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
    )


def _orbital_angle(t_s, f: RadiativeForcing):
    """Mean-rate orbital angle (0 = perihelion) at elapsed seconds ``t_s``."""
    return f.init_orbital_angle_rad + _TWO_PI * t_s / f.orbital_period_s


def solar_flux(t_s, f: RadiativeForcing):
    """Inverse-square solar flux (W m^-2) at elapsed seconds ``t_s``.

    Kepler distance ``r = a(1-e^2)/(1+e cos theta)`` then ``S = S_1AU (AU/r)^2``,
    matching :func:`src.gcm3d.terraforming_ode.solar_flux`.
    """
    theta = _orbital_angle(t_s, f)
    distance = (
        f.semi_major_axis_m * (1.0 - f.eccentricity**2)
        / (1.0 + f.eccentricity * jnp.cos(theta))
    )
    return f.tsi_1au_w_m2 * (f.au_m / distance) ** 2


def _declination(t_s, f: RadiativeForcing):
    """Solar declination (rad): ``arcsin(sin(tilt) sin(Ls))``."""
    ls = _orbital_angle(t_s, f) + f.ls_perihelion_rad
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


def radiative_heating_tendency(state, coords, specs, body: BodyConstants, f: RadiativeForcing):
    """Modal temperature-variation tendency (nondimensional) from the energy balance.

    Evaluates the 0-D surface energy balance per grid column:
    ``dT/dt = (Q_in - eps*sigma*(T/greenhouse)**4) / thermal_inertia`` in SI
    (K s^-1), applied uniformly through the vertical, then converts to dinosaur's
    nondimensional time units and returns it in the modal basis so it can be added
    directly to ``base.explicit_terms(state).temperature_variation``.
    """
    grid = coords.horizontal
    n_layers = coords.vertical.layers

    # sim_time is nondimensional; convert to seconds (time scale = 1/(2 Omega)).
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))
    t_s = state.sim_time * time_scale_s

    cz = cos_zenith_nodal(t_s, grid.latitudes, grid.longitudes, f)  # (n_lon,n_lat)
    q_in = (1.0 - f.albedo) * solar_flux(t_s, f) * cz                # (n_lon,n_lat), W m^-2

    # Absolute temperature per layer (K): reference profile + variation.
    ref_t = np.asarray(reference_temperature(coords, body)).reshape(n_layers, 1, 1)
    t_var = grid.to_nodal(state.temperature_variation)              # (L,n_lon,n_lat)
    temp_k = jnp.clip(t_var + ref_t, 1.0, None)

    q_out = f.emissivity * f.stefan_boltzmann * (temp_k / max(f.greenhouse_factor, 1.0)) ** 4
    dtdt_si = (q_in[None, :, :] - q_out) / f.thermal_inertia        # K s^-1, (L,n_lon,n_lat)

    # K s^-1 -> nondimensional (T scale = 1 K, so only the time scale enters).
    dtdt_nd = dtdt_si * time_scale_s
    return grid.to_modal(dtdt_nd)


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

    def explicit_terms(state):
        tend = base.explicit_terms(state)
        heat = radiative_heating_tendency(state, coords, specs, body, forcing)
        return dataclasses.replace(
            tend, temperature_variation=tend.temperature_variation + heat
        )

    return time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms, base.implicit_terms, base.implicit_inverse
    )


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
# tuple ``(dyn_state, co2_ice)`` — JAX treats tuples as pytrees, so dinosaur's
# stepper/scan operate on it unchanged. ``co2_ice`` is stored in pressure-equivalent
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

    frost_point_k: float
    latent_heat_j_kg: float
    gravity_m_s2: float
    thermal_inertia: float
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


def mars_co2_forcing(
    exchange_rate_pa_s_per_k: float = 1.0e-4,
    escape_rate_kg_s: float = 0.0,
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
    )


def _co2_surface_tendencies(dyn, ice_nd, coords, specs, body, cf: CO2Forcing):
    """Per-cell CO2 exchange: returns (d_logsp_modal, d_ice_nd_nodal, dT_latent_nd).

    Works in SI (Pa s^-1, K s^-1) then nondimensionalises. ``ice_nd`` is the frost
    reservoir in nondimensional pressure-equivalent units (same scaling as p_s).
    Condensation (surface below the frost point) moves mass atmosphere -> ice and
    warms; sublimation (above frost, gated by available ice) moves it back and cools.
    """
    grid = coords.horizontal
    n_layers = coords.vertical.layers

    # Surface-layer absolute temperature (K) and surface pressure (Pa).
    ref_t = np.asarray(reference_temperature(coords, body)).reshape(n_layers, 1, 1)
    t_surf_k = (grid.to_nodal(dyn.temperature_variation) + ref_t)[-1]  # (n_lon,n_lat)
    ps_nd = jnp.exp(grid.to_nodal(dyn.log_surface_pressure))           # (1,n_lon,n_lat)
    ps_pa = np.asarray(specs.dimensionalize(1.0, _u.pascal).magnitude) * ps_nd

    # Ice reservoir in Pa (dimensional) for gating and rates.
    ice_pa = specs.dimensionalize(ice_nd, _u.pascal).magnitude        # (1,n_lon,n_lat)

    below = jnp.clip(cf.frost_point_k - t_surf_k, 0.0, None)           # K, cooling below frost
    above = jnp.clip(t_surf_k - cf.frost_point_k, 0.0, None)          # K, warm enough to sublime
    # Supply-limited condensation: vanishes as the local column thins (p_s -> 0),
    # which keeps p_s strictly positive and the log-pressure tendency bounded.
    supply_gate = jnp.tanh(ps_pa[0] / cf.supply_scale_pa)
    cond_pa_s = cf.exchange_rate_pa_s_per_k * below * supply_gate      # atmosphere -> ice
    # Gate sublimation on available frost (clip >= 0 so spectral noise near the
    # cap edge can never drive sublimation of non-existent ice).
    subl_pa_s = (
        cf.exchange_rate_pa_s_per_k
        * above
        * jnp.tanh(jnp.clip(ice_pa[0], 0.0, None) / cf.ice_ref_pa)
    )
    net_pa_s = cond_pa_s - subl_pa_s                                   # (n_lon,n_lat), >0 = condensing

    # Escape: uniform atmospheric mass sink spread over the globe (Pa s^-1).
    escape_pa_s = cf.escape_rate_kg_s * cf.gravity_m_s2 / body.surface_area_m2

    # Mass budget (SI, per cell): ice gains net, atmosphere loses net + escape.
    dice_pa_s = net_pa_s[None, :, :]                                  # (1,n_lon,n_lat)
    dps_pa_s = -net_pa_s[None, :, :] - escape_pa_s

    # Latent heating: condensing mass (net/g kg m^-2 s^-1) releases L (W m^-2).
    dT_latent_si = cf.latent_heat_j_kg * (net_pa_s / cf.gravity_m_s2) / cf.thermal_inertia

    # --- nondimensionalise ---
    to_nd_press_rate = lambda x: specs.nondimensionalize(x * (_u.pascal / _u.second))
    time_scale_s = 1.0 / float(specs.nondimensionalize(1.0 * _u.second))

    dice_nd = to_nd_press_rate(dice_pa_s)                              # (1,n_lon,n_lat)
    dps_nd = to_nd_press_rate(dps_pa_s)                               # (1,n_lon,n_lat)
    d_logsp_nodal = dps_nd / ps_nd                                    # d(ln p_s)/dt
    d_logsp_modal = grid.to_modal(d_logsp_nodal)

    dT_latent_nd = dT_latent_si * time_scale_s                        # (n_lon,n_lat)
    # Apply latent heating column-uniform (matching the radiative treatment).
    dT_latent_modal = grid.to_modal(
        jnp.broadcast_to(dT_latent_nd[None, :, :], (n_layers,) + dT_latent_nd.shape)
    )
    return d_logsp_modal, dice_nd, dT_latent_modal


def forced_co2_primitive_equations(
    coords,
    body: BodyConstants,
    forcing: RadiativeForcing,
    co2_forcing: CO2Forcing,
    specs=None,
    orography=None,
) -> "time_integration.ImplicitExplicitODE":
    """Dry dynamics + radiative forcing + CO2 condensation cycle on a tuple state.

    The ODE operates on ``(dyn_state, co2_ice)`` where ``co2_ice`` is a nodal
    surface field ``(1, n_lon, n_lat)`` of frost in nondimensional pressure-equiv
    units. Build the initial tuple with :func:`initial_co2_state`; integrate with
    the ordinary :func:`src.gcm3d.stepper`/:func:`src.gcm3d.integrate`.
    """
    if specs is None:
        specs = physics_specs(body)
    base = _build_primitive_equations(coords, body, specs=specs, orography=orography)

    def explicit_terms(state):
        dyn, ice = state
        tend = base.explicit_terms(dyn)
        heat = radiative_heating_tendency(dyn, coords, specs, body, forcing)
        d_logsp, dice, dlatent = _co2_surface_tendencies(
            dyn, ice, coords, specs, body, co2_forcing
        )
        dyn_tend = dataclasses.replace(
            tend,
            temperature_variation=tend.temperature_variation + heat + dlatent,
            log_surface_pressure=tend.log_surface_pressure + d_logsp,
        )
        return (dyn_tend, dice)

    def implicit_terms(state):
        dyn, ice = state
        return (base.implicit_terms(dyn), jnp.zeros_like(ice))

    def implicit_inverse(state, step_size):
        dyn, ice = state
        return (base.implicit_inverse(dyn, step_size), ice)

    return time_integration.ImplicitExplicitODE.from_functions(
        explicit_terms, implicit_terms, implicit_inverse
    )


def initial_co2_state(dyn_state, coords, ice_pa: float = 0.0, specs=None, body=None):
    """Pair a dynamical ``State`` with an initial (uniform) CO2 frost field.

    ``ice_pa`` is the starting frost everywhere in Pa-equivalent (0 by default, i.e.
    frost forms from the atmosphere as poles cool). Returns the ``(dyn, ice)`` tuple
    the CO2 ODE integrates.
    """
    if specs is None:
        specs = physics_specs(body)
    ice_nd = float(specs.nondimensionalize(ice_pa * _u.pascal))
    ice = jnp.full(coords.surface_nodal_shape, ice_nd)
    return (dyn_state, ice)
