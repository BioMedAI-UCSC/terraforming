"""Released Mars hydrostatic cold-start equations.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. Apache License 2.0; see licenses/VERTICAL-GRID-NOTICE.txt.
"""
from src.framework.gcm._dinosaur import jnp


def surface_pressure(geopotential,area,*,dry_mass_pa=705.,temperature_k=170.,
                     gas_constant=189.02,top_pressure_pa=.001319):
    """Original hydro_eq pressure and log-weighted global mass correction."""
    phi=jnp.asarray(geopotential);weights=jnp.broadcast_to(jnp.asarray(area),phi.shape)
    rt=gas_constant*temperature_k
    top=jnp.asarray(top_pressure_pa)
    height=rt*jnp.log(.8*dry_mass_pa/top)
    pressure=top*jnp.exp((height-phi)/rt)
    mean=jnp.sum(weights*pressure)/jnp.sum(weights)
    return pressure+(dry_mass_pa-mean)*jnp.log(pressure)/jnp.log(jnp.asarray(dry_mass_pa))


def air_temperature(pressure_half,pressure_surface,*,temperature_k=170.,isothermal=True,
                    gravity=3.72,cp=735.94):
    """Original optional frost-limited profile, on supplied pressure interfaces."""
    half=jnp.asarray(pressure_half)
    if half.shape[0]<2: raise ValueError('initial atmosphere needs pressure interfaces')
    pressure=jnp.cumsum(jnp.diff(half,axis=0),axis=0)
    if isothermal: return jnp.full_like(pressure,temperature_k)
    frost=3182.48/(23.3494-jnp.log(pressure*.01))
    return jnp.maximum(frost,temperature_k+(gravity/cp)*11100.*jnp.log(pressure/pressure_surface))
