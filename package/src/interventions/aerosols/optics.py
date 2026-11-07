"""Planck-weight source spectra on the *Mars* Ames radiation intervals."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from src.framework.gcm._dinosaur import jnp

# NASA Ames rtmod_mgcm.F90 setrad: seven solar and five thermal intervals.
# Arrays run long-to-short wavelength, matching the staged gas channels.
SW_EDGES_UM=np.array([4.5,3.24,2.48,1.86,1.31,.8,.4,.24])
LW_EDGES_UM=np.array([1000.,60.,24.,12.,8.,4.5])
# IAU 2015 Resolution B3 nominal solar effective temperature; blackbody is an approximation.
SOLAR_TEMPERATURE_K=5772.


def planck_lambda(wavelength_m,temperature_k):
    h,c,k=6.62607015e-34,299792458.,1.380649e-23
    exponent=h*c/(wavelength_m*k*temperature_k)
    return 2*h*c*c/wavelength_m**5/np.expm1(np.minimum(exponent,700.))


def integrate_band(table,lo_m,hi_m,temperature_k,*,tail,points_per_interval=12):
    """Resolve every source-table knot, rather than smoothing narrow resonances."""
    if lo_m<table.wavelength_m[0]*(1-1e-8):
        raise ValueError('band extends below source spectral range')
    if hi_m>table.wavelength_m[-1] and tail!='source_constant':
        raise ValueError('longwave tail requires explicit source_constant selection')
    knots=np.unique(np.r_[lo_m,table.wavelength_m[(table.wavelength_m>lo_m)&(table.wavelength_m<hi_m)],hi_m])
    nodes,weights=np.polynomial.legendre.leggauss(points_per_interval)
    half=np.diff(knots)/2
    x=(knots[:-1,None]+knots[1:,None])/2+half[:,None]*nodes
    w=half[:,None]*weights*planck_lambda(x,temperature_k)
    norm=w.sum()
    a=np.interp(x,table.wavelength_m,table.absorption_m2)
    s=np.interp(x,table.wavelength_m,table.scattering_m2)
    sg=s*np.interp(x,table.wavelength_m,table.asymmetry)
    return np.array([(w*a).sum(),(w*s).sum(),(w*sg).sum()])/norm


@dataclass(frozen=True)
class BandTable:
    temperatures_k: np.ndarray
    sw: np.ndarray
    lw: np.ndarray
    metadata: dict

    def optical_depths(self,burden_kg_m2,temperature_k):
        sw=jnp.asarray(self.sw)[:,:,None,None,None]*burden_kg_m2[None,None]
        values=[]
        for component in range(3):
            values.append(jnp.stack([jnp.interp(temperature_k,jnp.asarray(self.temperatures_k),
                jnp.asarray(self.lw[:,component,band]),left=jnp.nan,right=jnp.nan) for band in range(self.lw.shape[-1])]))
        lw=jnp.stack(values)*burden_kg_m2[None,None]
        return dict(sw_extinction=sw[0]+sw[1],sw_scattering=sw[1],sw_scattering_g=sw[2],
                    lw_extinction=lw[0]+lw[1],lw_scattering=lw[1],lw_scattering_g=lw[2])


def make_band_table(table,particle,*,tail='source_constant',temperature_min_k=None,temperature_max_k=None,temperature_step_k=5.):
    from .validity import radiation_limits
    domain=radiation_limits()['air_temperature_k']
    temperature_min_k=domain[0] if temperature_min_k is None else temperature_min_k
    temperature_max_k=domain[1] if temperature_max_k is None else temperature_max_k
    if not all(np.isfinite(x) for x in (temperature_min_k,temperature_max_k,temperature_step_k)) or not 0<temperature_min_k<temperature_max_k or temperature_step_k<=0:
        raise ValueError('finite positive temperature grid with increasing bounds required')
    temperatures=np.linspace(temperature_min_k,temperature_max_k,
        int(np.ceil((temperature_max_k-temperature_min_k)/temperature_step_k))+1)
    sw=np.stack([integrate_band(table,b*1e-6,a*1e-6,SOLAR_TEMPERATURE_K,tail=tail) for a,b in zip(SW_EDGES_UM[:-1],SW_EDGES_UM[1:])],axis=1)/particle.mass_kg
    lw=np.array([np.stack([integrate_band(table,b*1e-6,a*1e-6,t,tail=tail)
                   for a,b in zip(LW_EDGES_UM[:-1],LW_EDGES_UM[1:])],axis=1) for t in temperatures])/particle.mass_kg
    return BandTable(temperatures,sw,lw,dict(source=table.provenance,tail=tail,
        band_order='Ames: seven SW and five LW; no Earth RRTMGP band defaults',
        units='absorption/scattering/scattering*g in m2/kg',temperature_k=[temperature_min_k,temperature_max_k],
        temperature_domain_basis='Ames gas-opacity table coverage; not a validated material temperature range',
        solar_weighting=dict(temperature_k=SOLAR_TEMPERATURE_K,source='IAU 2015 Resolution B3',
            approximation='blackbody within each Ames solar band, not a measured solar spectrum'),
        optical_constants='temperature-independent source simulation; Planck weights vary with temperature'))
