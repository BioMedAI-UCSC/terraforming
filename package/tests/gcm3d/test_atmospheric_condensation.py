"""Reject invalid atmospheric phase-change timescales before integration."""
import dataclasses

import numpy as np
import pytest

pytest.importorskip("dinosaur")
from src.celestials.planets.mars import MARS_BODY_3D as BODY
from src.celestials.planets.mars.gcm import co2_forcing, radiative_forcing
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.physics import gcm as physics


@pytest.mark.parametrize("value", [0., -1., np.nan, np.inf])
def test_invalid_condensation_timescale_is_rejected(value):
    coords, specs, cf = coordinate_system("T21", 4), physics_specs(BODY), co2_forcing()
    with pytest.raises(ValueError, match="atmospheric_condensation_timescale_s"):
        physics.forced_co2_primitive_equations(coords, BODY, radiative_forcing(),
            dataclasses.replace(cf, atmospheric_condensation_timescale_s=value), specs=specs)
