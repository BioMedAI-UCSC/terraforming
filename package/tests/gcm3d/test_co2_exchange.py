"""Initialization checks for the released surface CO2 exchange depth."""
import numpy as np
import pytest
from src.framework.physics.co2_condensation import surface_exchange_start


@pytest.mark.parametrize('latitude,qmass,expected',[
    ([0.],True,5),([np.pi/4.],True,5),([-np.pi/4.],True,5),
    ([0.,np.nextafter(np.pi/4.,np.inf)],True,2),
    ([-np.nextafter(np.pi/4.,np.inf)],True,2),([1.],False,5)])
def test_exchange_depth_uses_strict_domain_latitude_and_pressure_boundaries(latitude,qmass,expected):
    pressure=np.arange(1.,11.)
    assert surface_exchange_start(pressure,latitude,do_qmass=qmass,polar_ratio=.7,tropical_ratio=1.)==expected
    assert surface_exchange_start(pressure*610.,latitude,do_qmass=qmass,polar_ratio=.7,tropical_ratio=1.)==expected
    if expected==2:
        assert surface_exchange_start(pressure,latitude,do_qmass=qmass,
            polar_ratio=np.nextafter(.7,np.inf),tropical_ratio=1.)==3


@pytest.mark.parametrize('kwargs',[
    dict(pressure_half=[]),dict(pressure_half=[1.,0.]),dict(pressure_half=[1.,1.]),
    dict(pressure_half=[1.,np.nan]),dict(latitude_rad=[]),dict(latitude_rad=[np.inf]),
    dict(latitude_rad=[np.pi]),dict(polar_ratio=0.),dict(tropical_ratio=np.nan),
    dict(tropical_ratio=.01)])
def test_exchange_depth_rejects_invalid_reference_inputs(kwargs):
    options=dict(pressure_half=np.arange(1.,11.),latitude_rad=[0.])
    options.update(kwargs)
    with pytest.raises(ValueError): surface_exchange_start(**options)


def test_coarse_grid_bounds_extra_exchange_layers_at_first_layer():
    assert surface_exchange_start([1.,2.,3.],[0.])==0


def test_released_interfaces_fix_custom_ratio_without_changing_default_depth():
    from src.framework.physics.vertical_grid import initialization_reference_interfaces,sigma_boundaries
    half=initialization_reference_interfaces(sigma_boundaries(56),705.)
    assert half[0]>0.
    assert surface_exchange_start(half,[0.])==49
    assert surface_exchange_start(half,[0.],tropical_ratio=.995)==49
    centers=.5*(half[1:]+half[:-1])
    old_start=int(np.flatnonzero(centers/centers[-1]<.995)[-1])-2
    assert old_start==50
