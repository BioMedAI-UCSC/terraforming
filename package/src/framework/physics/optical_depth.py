"""Ames optcv15/optci15 optical-depth aggregation on the radiation subgrid.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.correlated_k import correlated_k_optical_depths


class OpticalColumn(NamedTuple):
    optical_depth: object
    scattering_albedo: object
    asymmetry: object
    cumulative: object
    gas_column: object


def gas_subgrid(grid,*,mode='12band',water_mass_fraction=None,linear_k_interpolation=False):
    """Dry-CO2 optical depths at every original fillpt subinterval, in Pa."""
    sw,ir=correlated_k_optical_depths(grid.lookup_temperature[1:],
        grid.lookup_pressure[1:],jnp.diff(grid.pressure,axis=0),mode=mode,
        water_mass_fraction=water_mass_fraction,linear_k_interpolation=linear_k_interpolation)
    return tuple(jnp.moveaxis(x,(0,1),(1,2)) for x in (sw,ir))


def aggregate_optics(gas,particle_extinction,particle_scattering,
                     particle_scattering_moment,rayleigh=None,*,infrared=False):
    """Combine gas and supplied particle/Rayleigh depths as in optcv15/optci15.

    Gas shape is (subinterval, band, Gaussian channel, ...). Particle arrays
    omit the Gaussian channel. Their moment is the sum of asymmetry times
    scattering depth over species. Include the top ghost subinterval. This
    routine does not infer particle size, loading or vertical distribution.
    """
    gas,ext,scat,moment=map(jnp.asarray,
        (gas,particle_extinction,particle_scattering,particle_scattering_moment))
    expected=gas.shape[:2]+gas.shape[3:]
    if ext.shape!=expected or scat.shape!=expected or moment.shape!=expected:
        raise ValueError("particle optical depths must match subintervals and spectral bands")
    if gas.shape[0]%2 or gas.shape[2] not in (17,33):
        raise ValueError("reference optics requires paired subintervals and 17 or 33 Gaussian channels")
    ray=jnp.zeros_like(ext) if rayleigh is None else jnp.broadcast_to(rayleigh,ext.shape)
    if infrared and rayleigh is not None:
        raise ValueError("Ames infrared optics does not include Rayleigh scattering")
    sub=gas+(ext+ray)[:,:,None]

    def pair(values):
        padded=jnp.concatenate([jnp.zeros_like(values[:1]),values,jnp.zeros_like(values[:1])],axis=0)
        return padded[2::2]+padded[3::2]

    tau=pair(sub)
    scattering=pair(scat)
    ray_sum=pair(ray)
    numerator=pair(moment)
    denominator=scattering+ray_sum
    safe=jnp.where(denominator>0.,denominator,1.)
    asym=jnp.where(denominator>0.,numerator/safe,0.)
    if infrared:
        tau=tau+1.e-50
        resolved=tau>1.e-9
        tau=jnp.where(resolved,tau,1.e-9)
        albedo=jnp.where(resolved,scattering[:,:,None]/tau,0.)
    else:
        safe_tau=jnp.where(tau>0.,tau,1.)
        albedo=jnp.where(tau>0.,(scattering+.9999*ray_sum)[:,:,None]/safe_tau,0.)
    asym=jnp.broadcast_to(asym[:,:,None],tau.shape)
    cumulative=jnp.concatenate([jnp.zeros_like(sub[:1]),jnp.cumsum(sub,axis=0)],axis=0)
    return OpticalColumn(tau,albedo,asym,cumulative,jnp.sum(gas[:,:,:-1],axis=0))
