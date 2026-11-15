"""Ames v3.2.1 simple atmospheric CO2 condensation, in JAX.

NASA-derived algorithm: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
from src.framework.gcm._dinosaur import jnp


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
