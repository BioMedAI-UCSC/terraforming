"""Reference 25-bin dust coagulation and its physical-moment reconstruction.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from functools import lru_cache
from typing import NamedTuple
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp

BOLTZMANN=1.38064852e-23
MOLECULE_MASS=.04401/6.02e23


def size_grid():
    ratio=(40e-6/1e-8)**(3./24.)
    index=np.arange(25,dtype=float)
    radius=1e-8*ratio**(index/3.)
    volume=4./3.*np.pi*1e-24*ratio**index
    width=2.*radius*2.**(1./3.)*(ratio**(1./3.)-1.)/(1.+ratio)**(1./3.)
    return radius,volume,width


def _kernels(t,path,r1,r2,rho,efficiency,coalescence,gravity,xp):
    viscosity=1.37e-5*(273.15+222.)/(t+222.)*(t/273.15)**1.5
    def thermal(mass): return xp.sqrt(8.*BOLTZMANN*t/(xp.pi*mass))
    def properties(radius):
        slip=1.+path/radius*(1.246+.42*xp.exp(-.87/(path/radius)))
        diffusion=BOLTZMANN*t/(6.*xp.pi*radius*viscosity)*slip
        mass=2500.*4./3.*xp.pi*radius**3
        mean=2.*diffusion/(xp.pi*thermal(mass))
        delta=((2.*radius+mean)**3-(4.*radius**2+mean**2)**1.5)/(6.*radius*mean)-2.*radius
        fall=2./9.*radius**2*2500.*gravity/viscosity*slip
        return diffusion,mass,delta,fall
    d1,m1,delta1,w1=properties(r1);d2,m2,delta2,w2=properties(r2)
    brownian=4.*xp.pi*(r1+r2)*(d1+d2)/((r1+r2)/(r1+r2+xp.sqrt(delta1**2+delta2**2))+
        4./efficiency*(d1+d2)/(xp.sqrt(thermal(m1)**2+thermal(m2)**2)*(r1+r2)))
    sticking=1. if coalescence==0 else (1.5 if coalescence==1 else .25)*xp.minimum(r1,r2)**2/(r1+r2)**2
    # Equal sizes have exactly zero differential settling. Make that identity
    # explicit before reverse-pass fusion can change abs(0)'s branch.
    relative_fall=xp.where(r1==r2,0.,xp.abs(w1-w2))
    gravitational=sticking*xp.pi*(r1+r2)**2*relative_fall
    large=xp.maximum(r1,r2);small=xp.minimum(r1,r2)
    dl,_,_,wl=properties(large);ds,_,_,_=properties(small)
    reynolds=2.*large*wl/(viscosity/rho)
    schmidt=viscosity/rho/ds
    enhancement=brownian*.45*xp.where(reynolds<=1.,reynolds**(1./3.),xp.sqrt(reynolds))*schmidt**(1./3.)
    return brownian,gravitational,enhancement


def mean_free_path(temperature,density):
    t,rho=map(jnp.asarray,(temperature,density))
    viscosity=1.37e-5*(273.15+222.)/(t+222.)*(t/273.15)**1.5
    return 2.*viscosity/rho/jnp.sqrt(8.*BOLTZMANN*t/(jnp.pi*MOLECULE_MASS))


def pair_kernels(temperature,path,radius1,radius2,density,*,efficiency=1.,coalescence=1,gravity=3.72):
    """Original betab, betag and betade, in m3/s; turbulence has no source epsilon."""
    if coalescence not in (0,1,2): raise ValueError('coalescence must be 0, 1 or 2')
    return _kernels(temperature,path,radius1,radius2,density,efficiency,coalescence,gravity,jnp)


@lru_cache(maxsize=8)
def _tables(efficiency,coalescence,gravity,gas_constant):
    radius,_,_=size_grid()
    t=(50.+25.*np.arange(1,16))[:,None,None,None]
    axis=(1e-8*10.**(np.arange(1,26)/2.))[None,:,None,None]
    r1=radius[None,None,:,None];r2=radius[None,None,None,:]
    brownian,gravitational,_=_kernels(t,axis,r1,r2,1.,efficiency,coalescence,gravity,np)
    rho=axis/(gas_constant*t)
    viscosity=1.37e-5*(273.15+222.)/(t+222.)*(t/273.15)**1.5
    path=2.*viscosity/rho/np.sqrt(8.*BOLTZMANN*t/(np.pi*MOLECULE_MASS))
    enhancement=_kernels(t,path,r1,r2,rho,efficiency,coalescence,gravity,np)[2]
    return brownian,gravitational,enhancement


def _lookup(table,temperature,value,*,pressure=False):
    temperatures=jnp.arange(75.,426.,25.)
    axis=1e-8*10.**(jnp.arange(1,26)/2.)
    ti=jnp.clip(jnp.searchsorted(temperatures,temperature,side='right'),1,14)
    vi=jnp.searchsorted(axis,value,side='right')
    # The source pressure fallback uses table_numt (15), rather than 25.
    vi=jnp.where(vi==25,14 if pressure else 24,jnp.maximum(vi,1))
    tw=(temperature-temperatures[ti-1])/(temperatures[ti]-temperatures[ti-1])
    vw=(jnp.log10(value)-jnp.log10(axis[vi-1]))/(jnp.log10(axis[vi])-jnp.log10(axis[vi-1]))
    table=jnp.asarray(table)
    a,b,c,d=table[ti-1,vi-1],table[ti-1,vi],table[ti,vi-1],table[ti,vi]
    # Keep the original call's swapped middle corners in boxinterp.
    left=a+(b-a)*tw[...,None,None];right=c+(d-c)*tw[...,None,None]
    return left+(right-left)*vw[...,None,None]


@lru_cache(maxsize=1)
def _fractions():
    _,volume,_=size_grid()
    combined=volume[:,None]+volume[None,:]
    result=np.zeros((25,25,25))
    for k in range(25):
        if k<24:
            mask=(combined>=volume[k])&(combined<volume[k+1])
            result[:,:,k]=np.where(mask,(volume[k+1]-combined)/(volume[k+1]-volume[k])*volume[k]/combined,0.)
        else:
            result[:,:,k]=np.where(combined>=volume[k],1.,0.)
        if k>0:
            mask=(combined<volume[k])&(combined>volume[k-1])
            result[:,:,k]=np.where(mask,(combined-volume[k-1])/(volume[k]-volume[k-1])*volume[k]/combined,result[:,:,k])
    return result


class CoagulationResult(NamedTuple):
    mass_tendency: object
    number_tendency: object
    diagnostics: object


def coagulation_step(temperature,pressure,mass_ratios,number_ratios,dt_seconds,*,
                     brownian=True,gravitational=True,diffusion_enhancement=True,
                     efficiency=1.,coalescence=1,factor=1.,direct=False,
                     gravity=3.72,gas_constant=189.02):
    """Original coagul_main for physical modes, preserving source reconstructions.

    Last axis indexes physical dust modes. Tagged copies are excluded.
    Default table mode retains source corner ordering and pressure fallback.
    `direct=True` selects its FULLCOAG path. Turbulent inertia is undefined in
    the release because epsilon is uninitialized, and is not selectable here.
    """
    t,p,mass,number=map(jnp.asarray,(temperature,pressure,mass_ratios,number_ratios))
    if t.shape!=p.shape or mass.shape!=number.shape or mass.shape[:-1]!=t.shape or mass.shape[-1]<1:
        raise ValueError('coagulation requires matching atmospheric grids and dust-mode axes')
    if coalescence not in (0,1,2) or efficiency<=0.:
        raise ValueError('coalescence must be 0/1/2 and efficiency must be positive')
    radius,volume,width=map(jnp.asarray,size_grid());rho=p/(gas_constant*t)
    old_mass,old_number=jnp.maximum(mass,1e-15),jnp.maximum(number,1e-15)
    r0=(.75*old_mass/(jnp.pi*old_number*2500.))**(1./3.)*jnp.exp(-1.5*.63676**2)
    bins=old_number[...,None]*rho[...,None,None]*width/(2.*radius*jnp.sqrt(2.*jnp.pi)*.63676)*jnp.exp(
        -.5*jnp.log(radius/r0[...,None])**2/.63676**2)
    total=jnp.sum(bins,axis=-2)
    initial_number=jnp.sum(bins,axis=-1)/rho[...,None]
    weights=bins/jnp.where(total[...,None,:]>0.,total[...,None,:],1.)
    for mode in range(mass.shape[-1]):
        norm=jnp.sum(weights,axis=-2)
        weights=weights.at[...,mode,:].set(weights[...,mode,:]/jnp.where(norm>0.,norm,1.))
    path=mean_free_path(t,rho)
    if direct:
        kernels=pair_kernels(t[...,None,None],path[...,None,None],radius[:,None],radius[None,:],
            rho[...,None,None],efficiency=efficiency,coalescence=coalescence,gravity=gravity)
    else:
        tables=_tables(efficiency,coalescence,gravity,gas_constant)
        kernels=tuple(_lookup(table,t,p if index==2 else path,pressure=index==2)
            for index,table in enumerate(tables))
    kernel=sum(value for enabled,value in zip((brownian,gravitational,diffusion_enhancement),kernels,strict=True) if enabled)
    kernel=jnp.broadcast_to(jnp.asarray(kernel),(*t.shape,25,25))
    fractions=jnp.asarray(_fractions());indices=jnp.arange(25)
    def update(k,new):
        selected=(indices[:,None]<k)&(indices[None,:]<=k)
        term1=jnp.sum(jnp.where(selected,fractions[:,:,k]*volume[:,None]*kernel*
            new[..., :,None]*total[...,None,:],0.),axis=(-2,-1))*dt_seconds/volume[k]
        term2=jnp.sum((1.-fractions[k,:,k])*kernel[...,k,:]*total,axis=-1)*dt_seconds
        return new.at[...,k].set((total[...,k]+factor*term1)/(1.+factor*term2))
    new=jax.lax.fori_loop(0,25,update,jnp.zeros_like(total))
    denominator=jnp.sum(new*volume,axis=-1)
    new=new*(jnp.sum(total*volume,axis=-1)/jnp.where(denominator!=0.,denominator,1.))[...,None]
    active=jnp.sum(total,axis=-1)/rho>1000.
    new=jnp.where(active[...,None],new,0.)
    modal=new[...,None,:]*weights
    new_number=jnp.sum(modal,axis=-1)/rho[...,None]
    new_mass=jnp.sum(modal*volume,axis=-1)*2500./rho[...,None]
    new_mass=new_mass/(1e-15+jnp.sum(new_mass,axis=-1,keepdims=True))*jnp.sum(old_mass,axis=-1,keepdims=True)
    dm=(new_mass-old_mass)/dt_seconds;dn=(new_number-initial_number)/dt_seconds
    dm=jnp.where(mass+dt_seconds*dm<=0.,-mass/dt_seconds+1e-15,dm)
    dn=jnp.where(number+dt_seconds*dn<=0.,-number/dt_seconds+1e-14,dn)
    return CoagulationResult(dm,dn,dict(active=active,
        dust_mass_change_kg_kg=jnp.sum(dm,axis=-1)*dt_seconds,
        initial_discretized_number_per_kg=initial_number))
