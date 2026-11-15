import math
from src.celestials.planets.mars.ames import AMES_BODY_3D,AMES_CO2_LATENT_HEAT_J_KG


def test_reference_constants_match_the_pinned_gnu_build():
    body=AMES_BODY_3D
    assert body.radius_m==3386000.
    assert body.gravity_m_s2==3.72
    assert body.gas_constant_j_kg_k==189.02
    assert body.cp_j_kg_k==735.94
    assert math.isclose(body.angular_velocity_s,7.0882e-5,rel_tol=2e-15)
    assert body.solar_day_seconds==88704.
    assert AMES_CO2_LATENT_HEAT_J_KG==590000.
