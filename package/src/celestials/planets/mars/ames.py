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
