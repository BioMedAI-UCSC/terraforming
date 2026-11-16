"""Explicit NASA Ames v3.2.1 benchmark constants.

Values follow patches/srcFMS.patch and the GNU build without SECDAY86400.
These describe the reference model configuration, including its model-day
clock. They do not replace the framework's separate present-Mars defaults.
"""
import math
from src.framework.gcm.body import BodyConstants


AMES_BODY_3D=BodyConstants(
    name="Mars",radius_m=3386.e3,gravity_m_s2=3.72,
    rotation_period_s=2.*math.pi/7.0882e-5,solar_day_s=88704.,
    gas_constant_j_kg_k=189.02,cp_j_kg_k=735.94,
    reference_temperature_k=200.,reference_surface_pressure_pa=610.)

AMES_CO2_LATENT_HEAT_J_KG=5.9e5


def reference_radiative_forcing(*,solar_constant_w_m2=None):
    """Reference orbit; optional source solar constant is at mean distance."""
    import dataclasses
    from src.celestials.planets.mars.gcm import radiative_forcing
    from src.framework.physics.gcm import AU_M

    eccentricity=.093
    mean_equinox=2.*math.pi*.269461
    eccentric=mean_equinox
    for _ in range(6):
        eccentric-=(eccentric-eccentricity*math.sin(eccentric)-mean_equinox)/(
            1.-eccentricity*math.cos(eccentric))
    true_equinox=2.*math.atan(math.sqrt((1.+eccentricity)/(1.-eccentricity))*math.tan(eccentric*.5))
    if solar_constant_w_m2 is not None and (not math.isfinite(solar_constant_w_m2) or solar_constant_w_m2<=0.):
        raise ValueError('solar constant must be finite and positive')
    tsi=1356.1 if solar_constant_w_m2 is None else solar_constant_w_m2*1.524**2
    return dataclasses.replace(radiative_forcing(co2_radiation_enabled=True),
        greenhouse_factor=1.,emissivity=1.,stefan_boltzmann=5.6734e-8,
        rotation_period_s=88704.,orbital_period_s=668.*88704.,
        semi_major_axis_m=1.524*AU_M,eccentricity=eccentricity,
        axial_tilt_rad=math.radians(25.19),ls_perihelion_rad=-true_equinox,
        init_orbital_angle_rad=-2.*math.pi*488./668.,tsi_1au_w_m2=tsi)
