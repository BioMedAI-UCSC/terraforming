"""Bulk water-cloud saturation, precipitation threshold and sedimentation.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple,Any
from jax import Array
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.co2_microphysics import sedimentation

LATENT_HEAT=2.8e6
ICE_DENSITY=917.
WATER_VAPOR_TRACER='water_vapor'
WATER_CLOUD_TRACER='water_cloud_ice'


class WaterResult(NamedTuple):
    temperature: Array
    vapor_ratio: Array
    ice_ratio: Array
    deposition_kg_m2: Array
    radius_m: Array
    diagnostics: dict[str,Any]


def saturation_ratio(temperature,pressure):
    """Original bulk watsat, with pressure in Pa and mass ratio in kg/kg."""
    return 611.*jnp.exp(22.5*(1.-273.16/temperature))*18./(44.*pressure)


def _newton(qcond,qvap,p,t,latent,cp):
    new_t=jnp.maximum(t+qcond*latent/cp,60.)
    sat=jnp.maximum(saturation_ratio(new_t,p),1e-50)
    f=(qvap-qcond)/sat-1.
    # This is the source solver's derivative, including its unusual factor two.
    df=-2./sat-(f+1.)*(6146.1*latent/cp)/new_t**2
    return f,df


def _findq(qvap,p,t,guess,latent,cp):
    qvap,p,t,guess,latent,cp=jnp.broadcast_arrays(*map(jnp.asarray,(qvap,p,t,guess,latent,cp)))
    fl,_=_newton(jnp.zeros_like(guess),qvap,p,t,latent,cp)
    fh,_=_newton(guess,qvap,p,t,latent,cp)
    bracketed=fl*fh<0.
    low=jnp.where(fl<0.,0.,guess);high=jnp.where(fl<0.,guess,0.)
    root=.5*guess;dx=jnp.abs(guess)
    f,df=_newton(root,qvap,p,t,latent,cp)
    initial=(jnp.asarray(0),root,low,high,dx,dx,f,df,bracketed)
    def condition(state): return (state[0]<500)&jnp.any(state[-1])
    def iterate(state):
        iteration,root,low,high,dxold,dx,f,df,active=state
        bisect=(((root-high)*df-f)*((root-low)*df-f)>=0.)|(jnp.abs(2.*f)>jnp.abs(dxold*df))
        step=jnp.where(bisect,.5*(high-low),f/df)
        candidate=jnp.where(bisect,low+step,root-step)
        done=jnp.where(bisect,low==candidate,root==candidate)|(jnp.abs(step)<1e-8)
        updated=jnp.where(active,candidate,root)
        new_f,new_df=_newton(updated,qvap,p,t,latent,cp)
        remaining=active&~done
        next_low=jnp.where(remaining&(new_f<0.),updated,low)
        next_high=jnp.where(remaining&(new_f>=0.),updated,high)
        return (iteration+1,updated,next_low,next_high,jnp.where(active,dx,dxold),
            jnp.where(active,step,dx),jnp.where(remaining,new_f,f),
            jnp.where(remaining,new_df,df),remaining)
    final=jax.lax.while_loop(condition,iterate,initial)
    return jnp.where(bracketed,final[1],jnp.nan)


@jax.custom_jvp
def _saturation_root(qvap,p,t,guess,latent,cp):
    """Source findq initialization, polished saturation root and implicit derivative.

    The source stopping rule is |step|<1e-8, at most 500 iterations. Three
    Newton corrections remove its finite stopping noise, retaining the same
    saturation equation. This is a numerical adaptation for smooth objectives.
    Reverse
    differentiation through a data-dependent while loop is unavailable in JAX.
    Differentiate its saturation equation instead. The original iteration is
    checked separately against Fortran; the polished root is checked against
    an independent high-accuracy solve and the measured native residual.
    """
    initial=_findq(qvap,p,t,guess,latent,cp)
    def polish(_,value):
        raw_t=t+value*latent/cp
        adjusted=jnp.maximum(raw_t,60.)
        raw_sat=saturation_ratio(adjusted,p)
        sat=jnp.maximum(raw_sat,1e-50)
        slope=jnp.where((raw_t>60.)&(raw_sat>1e-50),sat*6146.1/adjusted**2,0.)
        return value+(qvap-value-sat)/(1.+slope*latent/cp)
    return jax.lax.fori_loop(0,3,polish,initial)


@_saturation_root.defjvp
def _condensed_mass_jvp(primals,tangents):
    qvap,p,t,guess,latent,cp=primals
    dqvap,dp,dt,_,dlatent,dcp=tangents
    condensed=_saturation_root(*primals)
    raw_t=t+condensed*latent/cp
    new_t=jnp.maximum(raw_t,60.)
    raw_sat=saturation_ratio(new_t,p)
    sat=jnp.maximum(raw_sat,1e-50)
    dsat_t=jnp.where((raw_t>60.)&(raw_sat>1e-50),sat*6146.1/new_t**2,0.)
    pressure_term=jnp.where(raw_sat>1e-50,sat/p*dp,0.)
    derivative=(dqvap+pressure_term-dsat_t*(dt+condensed*dlatent/cp-
        condensed*latent*dcp/cp**2))/(1.+dsat_t*latent/cp)
    return condensed,derivative


def find_condensed_mass(qvap,p,t,guess,latent=LATENT_HEAT,cp=735.94):
    """Polished reference saturation root; accept integer or floating SI inputs."""
    values=(qvap,p,t,guess,latent,cp)
    dtype=jnp.result_type(*values,float)
    return _saturation_root(*(jnp.asarray(value,dtype=dtype) for value in values))


def phase_change(temperature,pressure,vapor,ice,*,latent_heat=LATENT_HEAT,cp=735.94):
    """Original bulk driver evaporation/condensation order and saturation switches."""
    t,p,vapor,ice=map(jnp.asarray,(temperature,pressure,vapor,ice))
    if t.shape!=p.shape or t.shape!=vapor.shape or t.shape!=ice.shape:
        raise ValueError('water temperature, pressure, vapor and cloud grids must match')
    vapor,ice=jnp.maximum(vapor,0.),jnp.maximum(ice,0.)
    saturation=saturation_ratio(t,p)
    evaporate=(ice>0.)&(vapor<saturation*.999999)
    guess=-2.*jnp.maximum(ice,saturation)
    def solve(mask,guess,vapor,temperature):
        # Inactive columns get a well-bracketed surrogate, keeping AD finite.
        return jax.lax.cond(jnp.any(mask),lambda _:find_condensed_mass(
            jnp.where(mask,vapor,1.),jnp.where(mask,p,610.),jnp.where(mask,temperature,200.),
            jnp.where(mask,guess,1.),latent_heat,cp),lambda _:jnp.zeros_like(vapor),None)
    evaporation=solve(evaporate,guess,vapor,t)
    evaporation=jnp.where(evaporate,jnp.maximum(evaporation,-ice),0.)
    t=t+evaporation*latent_heat/cp;ice=ice+evaporation;vapor=vapor-evaporation
    condense=vapor>1.000001*saturation
    condensed=solve(condense,vapor,vapor,t)
    condensed=jnp.where(condense,condensed,0.)
    return t+condensed*latent_heat/cp,vapor-condensed,ice+condensed


def cloud_step(temperature,pressure_half,pressure_full,vapor_ratio,ice_ratio,
               temperature_surface,heat_diffusivity,dt_seconds,*,nuclei_per_kg=1.e5,
               precipitation_threshold=.001,gas_constant=189.02,gravity=3.72,cp=735.94):
    """Original bulk driver: phase changes, threshold precipitation, then settling."""
    t,p,full,vapor,ice=map(jnp.asarray,(temperature,pressure_half,pressure_full,vapor_ratio,ice_ratio))
    adjusted,vapor,formed=phase_change(t,full,vapor,ice,cp=cp)
    threshold=jnp.maximum(formed-precipitation_threshold,0.)
    formed=jnp.minimum(formed,precipitation_threshold)
    volume=formed/nuclei_per_kg*.75/(jnp.pi*ICE_DENSITY)
    radius=jnp.maximum(volume,1e-21)**(1./3.)
    settled,deposit,transport=sedimentation(formed,p,full,t,temperature_surface,radius,
        heat_diffusivity,dt_seconds,gas_constant=gas_constant,gravity=gravity,
        particle_density=ICE_DENSITY,density_temperature=adjusted)
    threshold_mass=jnp.sum(jnp.diff(p,axis=0)/gravity*threshold,axis=0)
    total=deposit+threshold_mass
    diagnostics=dict(sedimentation=transport,threshold_precipitation_kg_m2=threshold_mass,
        settling_precipitation_kg_m2=deposit,rain_kg_m2=jnp.where(t[-1]>=273.,total,0.),
        snow_kg_m2=jnp.where(t[-1]<273.,total,0.))
    return WaterResult(adjusted,vapor,settled,total,radius,diagnostics)
