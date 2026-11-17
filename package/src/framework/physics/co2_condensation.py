"""Ames v3.2.1 simple atmospheric CO2 condensation, in JAX.

NASA-derived algorithm: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
import numpy as np
from src.framework.gcm._dinosaur import jnp


def surface_exchange_start(pressure_half,latitude_rad,*,do_qmass=False,polar_ratio=.92,tropical_ratio=.991):
    """Released local-domain depth choice, converted to a zero-based start.

    Dinosaur uses one global domain. The first layer bounds the source's two
    extra exchange layers on coarse grids that cannot contain both of them.
    """
    pressure=np.asarray(pressure_half,dtype=float)
    latitude=np.asarray(latitude_rad,dtype=float)
    if pressure.ndim!=1 or len(pressure)<2 or not np.isfinite(pressure).all() or \
            pressure[0]<0. or np.any(pressure[1:]<=0.) or np.any(np.diff(pressure)<=0.):
        raise ValueError('CO2 exchange reference interfaces must be nonnegative and increase')
    if not latitude.size or not np.isfinite(latitude).all() or np.any(abs(latitude)>np.pi/2.):
        raise ValueError('CO2 exchange latitudes must be finite radians within the poles')
    if not np.isfinite([polar_ratio,tropical_ratio]).all() or min(polar_ratio,tropical_ratio)<=0.:
        raise ValueError('CO2 exchange pressure ratios must be finite and positive')
    ratio=polar_ratio if do_qmass and np.any(abs(latitude)>np.pi/4.) else tropical_ratio
    below=np.flatnonzero(pressure/pressure[-1]<ratio)
    if not len(below): raise ValueError('CO2 exchange ratio does not select a reference layer')
    # Initialization has one more interface than the runtime has model layers.
    return max(0,int(below[-1])-3)


def simple_condensation(predicted_temperature,pressure_half,pressure_full,cp,gravity,latent_heat=5.9e5):
    """Reproduce co2_condense's temperature, immediate snow and mass-loss guard.

    Levels come first. Return adjusted temperature, layer mass loss (kg/m2),
    surface precipitation (kg/m2) and the guard-active mask. The reference
    caps dmass at 0.1*dp but still applies the full frost-point heating. This
    guard is copied literally; its activation must be reported, since mass
    and latent heating then cease to agree. Do not treat it as extra energy.
    """
    predicted_temperature,pressure_half,pressure_full=map(jnp.asarray,
        (predicted_temperature,pressure_half,pressure_full))
    if pressure_full.shape!=predicted_temperature.shape or pressure_half.shape!=(
            predicted_temperature.shape[0]+1,*predicted_temperature.shape[1:]):
        raise ValueError("CO2 temperature and pressure grids do not match")
    frost=3182.48/(23.3494-jnp.log(pressure_full*.01))
    deficit=jnp.maximum(0.,frost-predicted_temperature)
    dp=jnp.diff(pressure_half,axis=0)
    requested=(cp/latent_heat)/gravity*deficit*dp
    mass=jnp.minimum(requested,.1*dp)
    return predicted_temperature+deficit,mass,jnp.sum(mass,axis=0),requested>.1*dp
