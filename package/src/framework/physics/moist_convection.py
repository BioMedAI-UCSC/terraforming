"""Released AM4 moist convective adjustment with Mars constants in JAX.

SPDX-License-Identifier: LGPL-3.0-only
Derived from NOAA-GFDL atmos_phys revision 46817a1, moist_conv.F90.
See licenses/MOIST-CONVECTION-NOTICE.txt.
"""
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.saturation_pressure import vapor_pressure,specific_humidity


class MoistResult(NamedTuple):
    temperature_tendency: object
    vapor_tendency: object
    rain_kg_m2_s: object
    snow_kg_m2_s: object
    diagnostics: object


def _column(t,q,half,full,cold,*,humidity,tolerance_min,tolerance_max,iterations,
            simple_adjustment,saturation_model,gas_constant,cp,vapor_gas_constant):
    epsilon=gas_constant/vapor_gas_constant;latent=2.5e6 if simple_adjustment else jnp.where(cold,2.834e6,2.5e6)
    n=len(t);level=jnp.arange(n);interface=jnp.arange(n-1)
    difference=(full[1:]-full[:-1])/half[1:-1]
    def saturation(temp):
        return specific_humidity(temp,full,relative_humidity=humidity,model=saturation_model,
            gas_constant=gas_constant,vapor_gas_constant=vapor_gas_constant)
    def lapse(temp,scale):
        mid=.5*(temp[1:]+temp[:-1]);vapor=vapor_pressure(mid,model=saturation_model)
        es=scale*vapor.pressure_pa;des=scale*vapor.derivative_pa_k
        return gas_constant/cp*difference*mid*(half[1:-1]+epsilon*latent*es/mid/gas_constant)/(
            half[1:-1]+epsilon*latent*des/cp)
    def flags(temp,vapor,sat,adiabat,tolerance):
        excess=jnp.maximum(0.,vapor-sat)*sat>0.
        return jnp.concatenate([excess[:-1]&excess[1:]&
            (jnp.diff(temp)>adiabat+tolerance),jnp.array([False])])
    sat=saturation(t);adiabat=lapse(t,1.)  # Original initial lapse omits HC.
    unstable=flags(t,q,sat.specific_humidity,adiabat,tolerance_min)
    initial=(t,q,sat.specific_humidity,sat.temperature_derivative,sat.vapor_pressure_pa,
        adiabat,unstable,~jnp.any(unstable),jnp.asarray(0))
    tolerances=[];tolerance=tolerance_min
    while tolerance<=tolerance_max:
        tolerances.append(tolerance);tolerance*=2.
    tolerances=jnp.asarray(tolerances)
    def iteration(index,state):
        tolerance=tolerances[index//iterations]
        def calculate(state):
            temp,vapor,qs,dqs,es,adiabat,unstable,done,count=state
            def block(_,state):
                temp,vapor,qs,dqs,es,adiabat,start,done=state
                top=jnp.min(jnp.where((level>=start)&unstable,level,n))
                found=(top<n-1)&~done
                bottom=jnp.min(jnp.where((level>top)&~unstable,level,n-1))
                selected=(level>=top)&(level<=bottom)&found
                if simple_adjustment:
                    coefficient=epsilon*humidity*vapor_pressure(temp,model=saturation_model).derivative_pa_k/full
                else:
                    coefficient=jnp.where(full-(1.-epsilon)*es>0.,dqs,0.)
                # The source sums ALRM from each layer down to the block bottom.
                offsets=jnp.sum(jnp.where((interface[None,:]>=level[:,None])&
                    (interface[None,:]<bottom),adiabat[None,:],0.),axis=-1)
                carrier=jnp.diff(half)
                numerator=jnp.sum(jnp.where(selected,carrier*((cp+latent*coefficient)*
                    (temp+offsets)+latent*(vapor-qs)),0.))
                denominator=jnp.sum(jnp.where(selected,carrier*(cp+latent*coefficient),0.))
                target=numerator/jnp.where(denominator!=0.,denominator,1.)-offsets
                changed_q=qs+coefficient*(target-temp)
                temp=jnp.where(selected,target,temp);vapor=jnp.where(selected,changed_q,vapor)
                updated=saturation(temp)
                qs=jnp.where(selected,updated.specific_humidity,qs)
                dqs=jnp.where(selected,updated.temperature_derivative,dqs)
                es=jnp.where(selected,updated.vapor_pressure_pa,es)
                new_lapse=lapse(temp,humidity)
                adiabat=jnp.where((interface>=top)&(interface<bottom)&found,new_lapse,adiabat)
                next_start=bottom+1
                # A failed subsequent block search exits the whole source column.
                done=done|~found
                return temp,vapor,qs,dqs,es,adiabat,next_start,done
            block_state=(temp,vapor,qs,dqs,es,adiabat,jnp.asarray(0),done)
            def scan_block(_,state):
                return jax.lax.cond((state[6]<n-1)&~state[7],lambda s:block(None,s),lambda s:s,state)
            temp,vapor,qs,dqs,es,adiabat,_,done=jax.lax.fori_loop(0,n,scan_block,block_state)
            def refresh(_):
                new_lapse=lapse(temp,humidity)
                new_flags=flags(temp,vapor,qs,new_lapse,tolerance)
                return new_lapse,new_flags,~jnp.any(new_flags)
            # ITER==ITSMOD skips refresh, even before increasing tolerance.
            adiabat,unstable,done=jax.lax.cond(~done&((index+1)%iterations!=0),refresh,
                lambda _:(adiabat,unstable,done),None)
            return temp,vapor,qs,dqs,es,adiabat,unstable,done,count+1
        return jax.lax.cond(state[7],lambda s:s,calculate,state)
    final=jax.lax.fori_loop(0,iterations*len(tolerances),iteration,initial)
    return final[0],final[1],final[8],~final[7]


def adjust(temperature,vapor_ratio,pressure_half,pressure_full,dt_seconds,*,
           humidity=1.,tolerance_min=.02,tolerance_max=.10,iterations=30,
           simple_adjustment=False,saturation_model='smithsonian',gas_constant=189.02,
           cp=735.94,vapor_gas_constant=461.50,gravity=3.72):
    """Original Mars-call path without optional AM4 stratiform cloud inputs.

    Source tolerance passes bound the iteration count; this is not a job time
    limit. Native maximum-iteration output is retained and diagnosed. Cold
    columns use snow when their pre-adjustment bottom temperature is below
    273.15 K; the simple option always produces rain.
    """
    if not np.isfinite(dt_seconds) or dt_seconds<=0.:
        raise ValueError('moist convection requires a finite positive time step')
    if (not np.isfinite(humidity) or humidity<=0. or not np.isfinite(tolerance_min) or
        not np.isfinite(tolerance_max) or tolerance_min<=0. or tolerance_max<tolerance_min or
        not isinstance(iterations,int) or isinstance(iterations,bool) or iterations<1):
        raise ValueError('moist convection requires positive humidity, ordered tolerances and integer iterations')
    if any(not np.isfinite(value) or value<=0. for value in (gas_constant,cp,vapor_gas_constant,gravity)):
        raise ValueError('moist convection gas constants, heat capacity and gravity must be finite and positive')
    t,q,half,full=map(jnp.asarray,(temperature,vapor_ratio,pressure_half,pressure_full))
    if len(t)<2 or t.shape!=q.shape or t.shape!=full.shape or half.shape!=(len(t)+1,*t.shape[1:]):
        raise ValueError('moist convection requires matching temperature, vapor and pressure grids')
    flat=lambda value:value.reshape((len(value),-1))
    columns=jax.vmap(lambda tt,qq,hh,pp,cold:_column(tt,qq,hh,pp,cold,humidity=humidity,
        tolerance_min=tolerance_min,tolerance_max=tolerance_max,iterations=iterations,
        simple_adjustment=simple_adjustment,saturation_model=saturation_model,gas_constant=gas_constant,
        cp=cp,vapor_gas_constant=vapor_gas_constant),in_axes=(1,1,1,1,0),out_axes=(1,1,0,0))
    cold=t[-1]<273.15
    tn,qn,count,unconverged=columns(flat(t),flat(q),flat(half),flat(full),cold.reshape(-1))
    tn=tn.reshape(t.shape);qn=qn.reshape(q.shape)
    valid=jnp.all(specific_humidity(t,full,model=saturation_model).valid,axis=0)&jnp.all(
        jnp.isfinite(q)&(q>=0.)&jnp.isfinite(qn)&(qn>=0.)&
        vapor_pressure(tn,model=saturation_model).valid,axis=0)&jnp.all(jnp.diff(half,axis=0)>0.,axis=0)
    tn=jnp.where(valid[None],tn,jnp.nan);qn=jnp.where(valid[None],qn,jnp.nan)
    precipitation=jnp.maximum(jnp.sum((q-qn)*jnp.diff(half,axis=0)/gravity,axis=0),0.)/dt_seconds
    snow=(jnp.zeros_like(precipitation) if simple_adjustment else jnp.where(cold,precipitation,0.))
    rain=precipitation-snow
    latent=2.5e6 if simple_adjustment else jnp.where(cold,2.834e6,2.5e6)
    energy=jnp.sum((cp*(tn-t)+latent*(qn-q))*jnp.diff(half,axis=0)/gravity,axis=0)
    return MoistResult((tn-t)/dt_seconds,(qn-q)/dt_seconds,rain,snow,
        dict(iterations=count.reshape(t.shape[1:]),unconverged=unconverged.reshape(t.shape[1:]),
            valid=valid,enthalpy_water_change_j_m2=energy))
