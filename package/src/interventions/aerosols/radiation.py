"""Bind prognostic particle burden to the existing GCM radiation seam."""
import numpy as np
from src.framework.gcm._dinosaur import jnp,scales
from src.framework.gcm.dynamics import reference_temperature
from src.framework.physics.gcm import (
    column_radiative_fluxes,cos_zenith_nodal,solar_flux,solar_path_factor_nodal,dust_optical_depths)
from .state import MASS
from .transport import gas_layer_mass


def radiation_component(bands,forcing):
    if not forcing.ames_correlated_k_enabled or not forcing.co2_radiation_enabled:
        raise ValueError('active aerosols require Ames correlated-k radiation')
    if not np.allclose(np.asarray(forcing.emissivity),1.,rtol=0,atol=1e-12):
        raise ValueError('active prototype requires surface emissivity 1 for Kirchhoff consistency and legacy zero-loading parity')
    def flux(state,coords,specs,body,f,*,air_temperature_k=None,surface_pressure_pa=None):
        grid=coords.horizontal
        ps=surface_pressure_pa
        if ps is None:
            ps=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
        air=air_temperature_k
        if air is None:
            ref=np.asarray(reference_temperature(coords,body))[:,None,None]
            air=grid.to_nodal(state.dynamics.temperature_variation)+ref
        gas=gas_layer_mass(ps,coords.vertical.boundaries,body.gravity_m_s2)
        # Optical lookup must be nonnegative at intermediate spectral stages.
        # State material is NOT clipped here; the outer step diagnoses/repairs
        # undershoots conservatively and records correction magnitudes.
        burden=jnp.maximum(grid.to_nodal(state.dynamics.tracers[MASS]),0.)*gas
        valid=jnp.all((air>=bands.temperatures_k[0])&(air<=bands.temperatures_k[-1]))
        optics=bands.optical_depths(burden,air)
        optics={k:jnp.where(valid,v,jnp.nan) for k,v in optics.items()}
        time=state.dynamics.sim_time/float(specs.nondimensionalize(1.*scales.units.second))
        cz=cos_zenith_nodal(time,grid.latitudes,grid.longitudes,f)
        path=solar_path_factor_nodal(time,grid.latitudes,grid.longitudes,f)
        visible,longwave=dust_optical_depths(time,f)
        return column_radiative_fluxes(air,state.surface_temperature[0],ps,solar_flux(time,f)*cz,path,
            visible,longwave,coords.vertical.boundaries,body,f,aerosol_optics=optics)
    return flux
