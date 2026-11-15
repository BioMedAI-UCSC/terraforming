"""JAX adaptation of Ames v3.2.1 soil and surface CO2 phase adjustment.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
"""
import numpy as np
from src.framework.gcm._dinosaur import jnp
from src.framework.physics.vertical_solve import solve_tridiagonal


def soil_layer_thickness(nlayers=16):
    """Ames cold-start soil grid, including its extra lower boundary thickness."""
    if not isinstance(nlayers,int) or isinstance(nlayers,bool) or nlayers<2:
        raise ValueError("soil grid requires at least two layers")
    initial=np.array([.003,.005,.0075,.011,.015,.02,.04,.08,.16,.32,.64,1.5,3.5])
    if nlayers<=12:
        return initial[:nlayers+1].copy()
    if nlayers<=20:
        return np.concatenate([initial[:12],1.5*2.**np.arange(1,nlayers-10)])
    return .015*1.2**np.arange(nlayers+1)


def ice_depth(latitude_rad,minimum_depth_m,grs_latitude_deg):
    """Ames icedepth transition; north/south sign and ten-metre cap retained."""
    latitude_rad=jnp.asarray(latitude_rad)
    sign=jnp.where(latitude_rad>=0.,1.,-1.)
    alpha=sign*25.*(jnp.pi/180.)/jnp.pi
    beta=sign*(jnp.abs(grs_latitude_deg)-25./1.25)*(jnp.pi/180.)
    transition=jnp.tanh((latitude_rad-beta)/alpha)-1.
    return jnp.minimum(-(40.*transition/2.-minimum_depth_m),10.)


def subsurface_properties(latitude_rad,thermal_inertia,grs_latitude_deg,soil_layers=16,*,
                          legacy=False,subsurface_ice=True,ice_case=7,
                          southern_minimum_layers=7,north_ice_inertia=1000.,south_ice_inertia=1000.):
    """Original soil diffusivity and heat capacity, including legacy materials.

    Only case 7 changes the profile in the released source; other integer cases
    retain a uniform column. The deepest node is fixed by the surface solver.
    """
    dz=soil_layer_thickness(soil_layers)
    if not isinstance(ice_case,int) or isinstance(ice_case,bool):
        raise ValueError('soil ice case must be an integer')
    if not isinstance(southern_minimum_layers,int) or southern_minimum_layers<1:
        raise ValueError('southern minimum must select valid soil layers')
    if subsurface_ice and ice_case==7 and soil_layers<max(4,southern_minimum_layers):
        raise ValueError('case-7 subsurface ice requires enough soil nodes for its minimum depths')
    latitude,inertia,grs=jnp.broadcast_arrays(latitude_rad,thermal_inertia,grs_latitude_deg)
    capacity=1481.39*840. if legacy else 1500.*627.9
    icy_capacity=1781.99*1404.09 if legacy else capacity
    levels=jnp.arange(1,soil_layers+1).reshape((soil_layers,)+(1,)*latitude.ndim)
    cumulative=jnp.asarray(np.cumsum(dz[:soil_layers])).reshape((soil_layers,)+(1,)*latitude.ndim)
    minimum=jnp.where(latitude>=0.,jnp.sum(dz[:4]),jnp.sum(dz[:southern_minimum_layers]))
    depth=ice_depth(latitude,minimum,grs)
    icy=(levels>=2)&(levels<soil_layers)&(depth[None]<cumulative)
    icy=icy&subsurface_ice&(ice_case==7)
    ice_inertia=jnp.where(latitude>=0.,1100. if legacy else north_ice_inertia,
        2236.995 if legacy else south_ice_inertia)
    background=jnp.where(inertia>0.,(inertia/capacity)**2,jnp.nan)
    diffusion=jnp.where(icy[:-1],(ice_inertia/icy_capacity)**2,background[None])
    heat_capacity=jnp.where(icy,icy_capacity,capacity)
    return diffusion,heat_capacity


def subsurface_diffusivity(latitude_rad,thermal_inertia,grs_latitude_deg,soil_layers=16):
    """Default case-7 profile; use subsurface_properties for legacy material."""
    return subsurface_properties(latitude_rad,thermal_inertia,grs_latitude_deg,soil_layers)[0]


def surface_step(soil,thickness,diffusivity,volumetric_heat_capacity,pressure_pa,
                 co2_ice_kg_m2,downward_flux_w_m2,sensible_flux_w_m2,
                 sensible_surface_derivative,emissivity,dt_seconds,*,
                 condensation=True,alpha=.5,latent_heat=5.9e5,stefan=5.6734e-8,
                 bulk_latent_flux=0.,bulk_latent_derivative=0.):
    """Ames progts on supplied soil properties and surface fluxes.

    Soil has levels first and a fixed deepest node. diffusivity covers the
    interfaces between nodes. Positive external heat flux points downwards.
    The original first solve and one frost-triggered retry are retained.
    Return updated soil, surface CO2 mass and the phase-change diagnostics.
    """
    soil=jnp.asarray(soil)
    n=soil.shape[0]-1
    if n<2:
        raise ValueError("surface solve requires two evolving soil nodes and a fixed deep node")
    thickness=jnp.asarray(thickness)[:soil.shape[0]]
    if thickness.shape!=(soil.shape[0],):
        raise ValueError("soil thickness must cover every node")
    dz=thickness.reshape((soil.shape[0],)+(1,)*(soil.ndim-1))
    capacity=jnp.asarray(volumetric_heat_capacity)
    diffusion=jnp.asarray(diffusivity)
    if capacity.ndim==1 and soil.ndim>1:
        capacity=capacity.reshape((capacity.shape[0],)+(1,)*(soil.ndim-1))
    if diffusion.ndim==1 and soil.ndim>1:
        diffusion=diffusion.reshape((diffusion.shape[0],)+(1,)*(soil.ndim-1))
    capacity=jnp.broadcast_to(capacity,soil.shape)
    diffusion=jnp.broadcast_to(diffusion,(n,)+soil.shape[1:])
    interface=2./(dz[:-1]+dz[1:])
    conductance=diffusion*interface
    coefficient=alpha*dt_seconds/dz[:-1]
    lower=jnp.concatenate([jnp.zeros_like(conductance[:1]),-coefficient[1:]*conductance[:-1]],axis=0)
    upper=-coefficient*conductance
    diagonal=1.-lower-upper
    upper=upper.at[-1].set(-dt_seconds*conductance[-1]/dz[-2])
    crd=1./(capacity[:-1]*dz[:-1])
    flux=-diffusion*capacity[:-1]*interface*(soil[:-1]-soil[1:])
    frost=3182.48/(23.3494-jnp.log(jnp.asarray(pressure_pa)*.01))
    snow=jnp.asarray(co2_ice_kg_m2)
    maxlatent=snow*latent_heat/dt_seconds-capacity[0]*dz[0]*(soil[0]-frost)/dt_seconds

    def solve(seed):
        snowtmp=snow+seed
        ts=soil[0]
        ir=stefan*emissivity*ts**4
        nonlatent=downward_flux_w_m2+sensible_flux_w_m2-ir+bulk_latent_flux+bulk_latent_derivative
        rhs=soil[:-1]
        rhs=rhs.at[1:].add((1.-alpha)*dt_seconds*(flux[1:]-flux[:-1])*crd[1:])
        rhs0=ts+dt_seconds*crd[0]*(nonlatent+4.*ir+sensible_surface_derivative*ts+
            (1.-alpha)*flux[0])
        diag0=diagonal[0]+dt_seconds*crd[0]*(4.*stefan*emissivity*ts**3+sensible_surface_derivative)
        up0=upper[0]
        phase=jnp.zeros_like(ts)
        if condensation:
            frosted=snowtmp>0.
            persists=frosted & ((nonlatent+flux[0])<maxlatent)
            depleted=frosted & ~persists
            phase=jnp.where(persists,seed-(nonlatent+flux[0])*dt_seconds/latent_heat,
                jnp.where(depleted,-snowtmp,0.))
            depleted_ts=soil[0]-snowtmp*latent_heat*crd[0]
            depleted_ir=stefan*emissivity*depleted_ts**4
            depleted_flux=downward_flux_w_m2+sensible_flux_w_m2-depleted_ir+(1.-alpha)*flux[0]
            depleted_rhs=depleted_ts+dt_seconds*crd[0]*(depleted_flux+4.*depleted_ir+
                sensible_surface_derivative*depleted_ts)
            depleted_diag=diagonal[0]+dt_seconds*crd[0]*(4.*stefan*emissivity*depleted_ts**3+
                sensible_surface_derivative)
            rhs0=jnp.where(persists,frost,jnp.where(depleted,depleted_rhs,rhs0))
            diag0=jnp.where(persists,1.,jnp.where(depleted,depleted_diag,diag0))
            up0=jnp.where(persists,0.,up0)
        else:
            persists=jnp.zeros_like(ts,dtype=bool)
        diag=diagonal.at[0].set(diag0)
        up=upper.at[0].set(up0)
        rhs=rhs.at[0].set(rhs0)
        # Preserve the reference's special fixed-deep-boundary equation.
        rhs=rhs.at[-1].set(soil[-1]+(lower[-1]+diag[-1])*crd[-1]+
            up[-1]*crd[-1]*soil[-1])
        solved=solve_tridiagonal(lower,diag,up,rhs)
        surface=jnp.where(persists,frost,solved[0])
        solved=solved.at[0].set(surface)
        result=jnp.concatenate([solved,soil[-1:]],axis=0)
        return result,phase,snowtmp+phase

    first,phase,snowtmp=solve(jnp.zeros_like(soil[0]))
    retry=(first[0]<frost)&(snowtmp==0.) if condensation else jnp.zeros_like(soil[0],dtype=bool)
    second,second_phase,_=solve(jnp.where(retry,1.e-10,0.))
    result=jnp.where(retry[None],second,first)
    phase=jnp.where(retry,second_phase,phase)
    return result,snow+phase,dict(co2_phase_change_kg_m2=phase,
        frost_temperature_k=frost,frost_retry=retry)
