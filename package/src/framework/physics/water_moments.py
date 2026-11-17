"""Moment-water heterogeneous nucleation and ice growth from the reference GCM.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
import numpy as np
from typing import NamedTuple
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.water_bulk import saturation_ratio,find_condensed_mass

BOLTZMANN=1.38064852e-23
WATER_MOLAR_MASS=.01801
GAS_CONSTANT=8.3143
ICE_DENSITY=917.
AVOGADRO=6.02e23
LATENT_HEAT=2.8343e6
MOMENT_PARTICLE_TRACERS=('dust_mass','dust_number','water_ice_number','water_core_mass')


def moment_particle_tracers(dust_mode_count=1):
    """Physical populations only; extra modes keep source mass/number/core order."""
    if not isinstance(dust_mode_count,int) or isinstance(dust_mode_count,bool) or dust_mode_count<1:
        raise ValueError('dust mode count must be a positive integer')
    return MOMENT_PARTICLE_TRACERS+tuple(f'dust_{kind}_{mode}'
        for mode in range(2,dust_mode_count+1) for kind in ('mass','number','core_mass'))


def dust_particle_tracers(dust_mode_count=1):
    """Dust mass/number pairs without the inactive water-cloud/core entries."""
    names=moment_particle_tracers(dust_mode_count)
    return names[:2]+tuple(name for start in range(4,len(names),3) for name in names[start:start+2])


def size_grid():
    """Original initmicro four-bin radii and volume-derived boundaries."""
    ratio=np.exp(np.log(1.e-5/1.e-7)/3.*3.)
    radii=[1.e-7]
    for _ in range(3): radii.append(radii[-1]*ratio**(1./3.))
    radii=np.asarray(radii)
    edges=np.r_[1.e-10,((2.*ratio)/(ratio+1.))**(1./3.)*radii]
    edges[-1]=1.e-2
    return radii,edges


def critical_saturation(temperature,*,contact_case=0,contact=.965):
    """Original scrit_tdep contact models; retain their published extrapolations."""
    t=jnp.asarray(temperature)
    if contact_case not in (0,1,2,3):
        raise ValueError('contact case must be 0, 1, 2 or 3')
    if contact_case==1: return -.0718*t+14.4,.0046*t+.1085
    if contact_case==2: return jnp.ones_like(t),.94-6005.*jnp.exp(-.065*t)
    if contact_case==3: return jnp.ones_like(t),-.698+1.635*jnp.tanh((t/144.21)**3.239)
    return jnp.ones_like(t),jnp.full_like(t,contact)


def shape_factor(contact,radius_ratio):
    """Original fshape2 geometry with factored terms to avoid large-ratio cancellation."""
    m,x=jnp.broadcast_arrays(jnp.asarray(contact),jnp.asarray(radius_ratio))
    planar=x>3000.
    x=jnp.where(planar,0.,x)
    phi=jnp.sqrt(1.-2.*m*x+x*x)
    y=(x-m)/phi
    one_minus=(1.-m*m)/(phi*(phi+x-m))
    a=1.+((1.-m*x)/phi)**3
    b=x**3*one_minus**2*(2.+y)
    c=-3.*m*x*x*one_minus
    return jnp.where(planar,(2.+m)*(1.-m)**2/4.,.5*(a+b+c))


def nucleation_rates(partial_pressure_pa,temperature,saturation_ratio,number_bins,*,
                     contact_case=0,contact=.965,radii_m=None):
    """Original nuclea per-bin rates in 1/s and its runtime contact parameter."""
    ph,t,sat=map(jnp.asarray,(partial_pressure_pa,temperature,saturation_ratio))
    count=jnp.asarray(number_bins)
    radii=jnp.asarray(size_grid()[0] if radii_m is None else radii_m)
    if count.shape[-1]!=radii.shape[0]: raise ValueError('nucleation counts must match dust radius bins')
    threshold,m=critical_saturation(t,contact_case=contact_case,contact=contact)
    active=sat>threshold
    safe_t=jnp.where(active,t,180.);safe_sat=jnp.where(active,sat,2.)
    safe_m=jnp.where(active,m,.965)
    sigma=(141.-.15*safe_t)*1.e-3
    rstar=2.*sigma*(WATER_MOLAR_MASS/ICE_DENSITY)/(GAS_CONSTANT*safe_t*jnp.log(safe_sat))
    gstar=4.*AVOGADRO*jnp.pi*rstar**3/(3.*WATER_MOLAR_MASS/ICE_DENSITY)
    shape=shape_factor(safe_m[...,None],radii/rstar[...,None])
    barrier=(4./3.*jnp.pi)*sigma[...,None]*rstar[...,None]**2*shape
    desorption=.288e-19
    delta=jnp.clip((2.*desorption-desorption/10.-barrier)/(BOLTZMANN*safe_t[...,None]),-100.,100.)
    density=ph/(BOLTZMANN*safe_t)
    zeldov=jnp.sqrt(barrier/(3.*jnp.pi*BOLTZMANN*safe_t[...,None]*gstar[...,None]**2))
    rate=(zeldov*BOLTZMANN*safe_t[...,None]*rstar[...,None]**2*4.*jnp.pi*
        (density[...,None]*radii)**2/(shape*1.e13*(WATER_MOLAR_MASS/AVOGADRO))*jnp.exp(delta))
    return jnp.where(active[...,None]&(count!=0.)&(delta!=-100.),rate,0.),m


def growth_coefficient(temperature,pressure_pa,saturation_pressure_pa,equilibrium_ratio,radius_m):
    """Original growthrate2 resistance coefficient; all arguments use SI units."""
    t,p,psat,seq,r=map(jnp.asarray,(temperature,pressure_pa,saturation_pressure_pa,equilibrium_ratio,radius_m))
    conductivity=(.17913*t-13.9789)*4.184e-4
    latent=(2834.3-.28*(t-273.15)-.004*(t-273.15)**2)*1.e3
    nav=6.023e23;rgp=8.3143
    mean_path_factor=.707*rgp/(4.*jnp.pi*(2.2e-10)**2*nav)
    diffusion=(1./3.*jnp.sqrt(8.*BOLTZMANN*t/(jnp.pi*.018/nav))*BOLTZMANN*t/
        (jnp.pi*p*(2.2e-10+1.2e-10)**2*jnp.sqrt(1.+.018/.044)))
    knudsen=t/p*mean_path_factor/r
    correction=(1.333+.71/knudsen)/(1.+1./knudsen)
    diffusion=diffusion/(1.+correction*knudsen)
    thermal=latent**2*917.*.018/(conductivity*rgp*t*t)
    molecular=rgp*t*917./(diffusion*psat*.018)
    return 1./(seq*thermal+molecular)


class MomentResult(NamedTuple):
    temperature_tendency: object
    tracer_tendency: object
    dust_mass_tendency: object
    dust_number_tendency: object
    saturation_ratio: object
    contact_parameter: object


def _distribution(mass,number):
    """Expand source dust moments to four lognormal mass and number bins."""
    radii,edges=size_grid()
    positive=(mass>0.)&(number>0.)
    volume=jnp.where(positive,mass,1.)/jnp.where(positive,number,1.)*.75/(jnp.pi*2500.)
    rn=jnp.where(positive,volume**(1./3.)*jnp.exp(-1.5*.63676**2),0.)
    rn=jnp.clip(rn,radii[0],radii[-1]);rm=rn*jnp.exp(3.*.63676**2)
    def bins(total,radius):
        arguments=jnp.log(jnp.asarray(edges)/radius[...,None])/(jnp.sqrt(2.)*.63676)
        return .5*total[...,None]*jnp.diff(jax.scipy.special.erf(arguments),axis=-1)
    return bins(mass,rm),bins(number,rn)


def _mode_weights(bins):
    total=jnp.sum(bins,axis=-2,keepdims=True)
    weights=jnp.where(total>0.,bins/jnp.where(total>0.,total,1.),0.)
    # The source normalizes each mode against the partly updated table in order.
    for mode in range(bins.shape[-2]):
        total=jnp.sum(weights,axis=-2)
        value=weights[...,mode,:]/jnp.where(total>0.,total,1.)
        weights=weights.at[...,mode,:].set(value)
    return weights


def nucleation_condensation(temperature,pressure_pa,tracers,dt_seconds,*,
                           dust_mass=None,dust_number=None,latent_heat=False,
                           contact_case=0,contact=.965,cp=735.94):
    """Original nucleacond at layer centers, with arbitrary leading grid axes.

    Last tracer axis: dust mass, dust number, ice mass, ice number, core mass,
    vapor mass. Optional dust-mode arrays have the same leading axes. They
    contain physical modes only; tagged species are handled by the outer driver.
    Empty bin weights and dry saturation pressure use continuous finite limits.
    The source's small-core branch retains the last dust mode's number.
    Tendencies remain relative to clipped inputs, as in the source routine.
    """
    t,p,q=map(jnp.asarray,(temperature,pressure_pa,tracers))
    if t.shape!=p.shape or q.shape!=(*t.shape,6):
        raise ValueError('moment tracers must have six entries per temperature/pressure cell')
    if (dust_mass is None)!=(dust_number is None):
        raise ValueError('dust mass and number modes must be supplied together')
    old=jnp.maximum(q,0.)
    dm=old[...,0,None] if dust_mass is None else jnp.maximum(jnp.asarray(dust_mass),0.)
    dn=old[...,1,None] if dust_number is None else jnp.maximum(jnp.asarray(dust_number),0.)
    if dm.shape!=dn.shape or dm.shape[:-1]!=t.shape or dm.shape[-1]<1:
        raise ValueError('dust mass and number modes must match the atmospheric grid')
    multi=dm.shape[-1]>1
    bm,bn=_distribution(dm,dn)
    wm,wn=(_mode_weights(bm),_mode_weights(bn)) if multi else (jnp.ones_like(bm),jnp.ones_like(bn))
    qs=saturation_ratio(t,p);sat=old[...,5]/qs
    rates,m=nucleation_rates(old[...,5]*(44./18.)*p,t,sat,jnp.sum(bn,axis=-2),
        contact_case=contact_case,contact=contact)
    factor=rates[...,None,:]*dt_seconds
    removed_m=jnp.sum(wm*bm/(1.+factor)*factor,axis=-1)
    removed_n=jnp.sum(wn*bn/(1.+factor)*factor,axis=-1)
    updated=old.at[...,0].add(-removed_m[...,0]).at[...,1].add(-removed_n[...,0])
    updated=updated.at[...,4].add(jnp.sum(removed_m,axis=-1))
    updated=updated.at[...,3].add(jnp.sum(removed_n,axis=-1))
    new_dm,new_dn=dm-removed_m,dn-removed_n
    active=(updated[...,3]>=1e-32)&(sat!=1.)
    total=old[...,2]+updated[...,4]
    tiny=total<1e-15
    safe_total=jnp.where(total>0.,total,1.)
    density=jnp.maximum((old[...,2]*917.+updated[...,4]*2500.)/safe_total,917.)
    number=jnp.where(tiny,dn[...,-1],updated[...,3])
    volume=jnp.where(active&~tiny,total,1.)/jnp.where(active&~tiny,number,1.)*.75/(jnp.pi*density)
    radius=jnp.where(tiny,1e-8,volume**(1./3.))
    # Inactive cells use a finite surrogate for AD through both select branches.
    work_t=jnp.where(active,t,180.);work_p=jnp.where(active,p,610.)
    radius=jnp.where(active,radius,1e-6)
    seq=jnp.exp(2.*(141.-.15*work_t)*1e-3*WATER_MOLAR_MASS/
        (917.*GAS_CONSTANT*work_t*radius))
    psat=saturation_ratio(work_t,work_p)*(44./18.)*work_p
    gr=growth_coefficient(work_t,work_p,psat,seq,radius)
    coefficient=dt_seconds*4.*jnp.pi*917.*gr*radius*number
    vapor=jnp.minimum((coefficient*seq+old[...,5])/(coefficient/qs+1.),old[...,2]+q[...,5])
    condensed=jnp.where(active,jnp.maximum(coefficient*(vapor/qs-seq),-old[...,2]),0.)
    if latent_heat:
        new_sat=(old[...,5]-condensed)/saturation_ratio(t+condensed*2.8343e6/cp,p)
        solve=active&(((sat<1.)&(new_sat>1.01))|((sat>1.)&(new_sat<.99)))
        root=find_condensed_mass(jnp.where(solve,old[...,5],1.),jnp.where(solve,p,610.),
            jnp.where(solve,t,200.),jnp.where(solve,condensed,1.),cp=cp)
        condensed=jnp.where(solve,root,condensed)
    ice=old[...,2]+condensed
    cleared=active&(ice<=0.)
    if multi:
        core_bins,cloud_bins=_distribution(updated[...,4],updated[...,3])
        new_dm=jnp.where(cleared[...,None],new_dm+jnp.sum(core_bins[...,None,:]*wm,axis=-1),new_dm)
        new_dn=jnp.where(cleared[...,None],new_dn+jnp.sum(cloud_bins[...,None,:]*wn,axis=-1),new_dn)
        updated=updated.at[...,0].set(jnp.where(cleared,new_dm[...,0],updated[...,0]))
        updated=updated.at[...,1].set(jnp.where(cleared,new_dn[...,0],updated[...,1]))
    else:
        updated=updated.at[...,0].add(jnp.where(cleared,updated[...,4],0.))
        updated=updated.at[...,1].add(jnp.where(cleared,updated[...,3],0.))
    updated=updated.at[...,2].set(jnp.where(cleared,0.,ice))
    updated=updated.at[...,4].set(jnp.where(cleared,0.,updated[...,4]))
    updated=updated.at[...,3].set(jnp.where(cleared,0.,updated[...,3]))
    updated=updated.at[...,5].set(old[...,5]-(updated[...,2]-old[...,2]))
    return MomentResult(condensed*LATENT_HEAT/cp/dt_seconds if latent_heat else jnp.zeros_like(t),
        jnp.maximum(-old,updated-old)/dt_seconds,jnp.maximum(-dm,new_dm-dm)/dt_seconds,
        jnp.maximum(-dn,new_dn-dn)/dt_seconds,sat,m)


class EffectiveState(NamedTuple):
    temperature: object
    tracers: object
    dust_mass: object
    dust_number: object
    initialized: object = True


def subcycled_cloud_step(initial_temperature,temperature,pressure_pa,initial_tracers,tracers,
                         dt_seconds,*,microtimestep=-1.,initial_dust_mass=None,dust_mass=None,
                         initial_dust_number=None,dust_number=None,effective_input_mode=0,
                         previous_state=None,**options):
    """Source micro_driver effective-input ramp and substeps.

    Inputs before physics and before clouds are separate. Positive tracer
    tendencies enter over the substeps. Negative tendencies act immediately.
    Return cloud-only tendencies relative to the supplied pre-cloud state.
    """
    if not np.isfinite(dt_seconds) or dt_seconds<=0.:
        raise ValueError('moment cloud time step must be finite and positive')
    if not np.isfinite(microtimestep) or microtimestep==0.:
        raise ValueError('microtimestep must be finite and nonzero')
    if effective_input_mode not in (0,1):
        raise ValueError('effective cloud input mode must be 0 or 1')
    count=max(int(dt_seconds/microtimestep),1);step=dt_seconds/count
    t0,t,p,q0,q=map(jnp.asarray,(initial_temperature,temperature,pressure_pa,initial_tracers,tracers))
    q0,q=jnp.maximum(q0,0.),jnp.maximum(q,0.)
    arrays=(initial_dust_mass,dust_mass,initial_dust_number,dust_number)
    if any(x is None for x in arrays) and not all(x is None for x in arrays):
        raise ValueError('initial and current dust mass and number modes must all be supplied')
    if dust_mass is None:
        dm0,dm=q0[...,0,None],q[...,0,None];dn0,dn=q0[...,1,None],q[...,1,None]
    else:
        dm0,dm,dn0,dn=map(lambda x:jnp.maximum(jnp.asarray(x),0.),arrays)
    if effective_input_mode==1 and previous_state is not None:
        old=previous_state
        if (np.shape(old.temperature)!=t0.shape or np.shape(old.tracers)!=q0.shape or
            np.shape(old.dust_mass)!=dm0.shape or np.shape(old.dust_number)!=dn0.shape):
            raise ValueError('previous effective cloud fields must match the current column and modes')
        t0=jnp.where(old.initialized,old.temperature,t0)
        q0=jnp.where(old.initialized,jnp.maximum(old.tracers,0.),q0)
        dm0=jnp.where(old.initialized,jnp.maximum(old.dust_mass,0.),dm0)
        dn0=jnp.where(old.initialized,jnp.maximum(old.dust_number,0.),dn0)
    effective=jnp.minimum(q0,q);dm_eff=jnp.minimum(dm0,dm);dn_eff=jnp.minimum(dn0,dn)
    heat_ramp=(t-t0)/count
    ramps=((q-effective)/count,(dm-dm_eff)/count,(dn-dn_eff)/count)
    def advance(state,_):
        ts,qs,ms,ns=state
        ts=ts+heat_ramp;qs=jnp.maximum(qs+ramps[0],0.)
        ms=jnp.maximum(ms+ramps[1],0.);ns=jnp.maximum(ns+ramps[2],0.)
        change=nucleation_condensation(ts,p,qs,step,dust_mass=ms,dust_number=ns,**options)
        return (ts+step*change.temperature_tendency,jnp.maximum(qs+step*change.tracer_tendency,0.),
            jnp.maximum(ms+step*change.dust_mass_tendency,0.),
            jnp.maximum(ns+step*change.dust_number_tendency,0.)),(change.saturation_ratio,change.contact_parameter)
    (new_t,new_q,new_dm,new_dn),(sat,contact)=jax.lax.scan(advance,(t0,effective,dm_eff,dn_eff),None,length=count)
    return MomentResult((new_t-t)/dt_seconds,(new_q-q)/dt_seconds,
        (new_dm-dm)/dt_seconds,(new_dn-dn)/dt_seconds,sat[-1],contact[-1])
