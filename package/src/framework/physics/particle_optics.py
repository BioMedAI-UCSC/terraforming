"""Size-dependent dust, coated water ice and bulk cloud optical depths.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from src.framework.physics.tracer_transport import tracer_to_nodal
from functools import lru_cache
from pathlib import Path
from typing import NamedTuple,Any

import numpy as np
from src.framework.gcm._dinosaur import jax,jnp


class ParticleOptics(NamedTuple):
    visible: tuple[Any,...]
    infrared: tuple[Any,...]
    diagnostics: dict[str,Any]


@lru_cache(maxsize=1)
def load_tables():
    with np.load(Path(__file__).with_name('particle_optical_tables.npz'),allow_pickle=False) as archive:
        return {name:archive[name] for name in archive.files}


def _radius(mass,number,density,active,*,deviation=0.):
    volume=jnp.where(active,mass,1.)/jnp.where(active,number,1.)*.75/(jnp.pi*density)
    radius=volume**(1./3.)*jnp.exp(-.5*deviation**2)
    area=jnp.where(active,number*jnp.pi*radius**2,0.)
    return radius,area


def _distribution(radius,edges,deviation):
    inverse=1./radius[...,None]
    argument=jnp.log(jnp.asarray(edges)*inverse)/(jnp.sqrt(2.)*deviation)
    return .5*jnp.diff(jax.scipy.special.erf(argument),axis=-1)


def _average(table,distribution):
    value=jnp.einsum('...r,...rpb->...pb',distribution,jnp.asarray(table))
    return value.at[...,1,:].set(jnp.minimum(value[...,1,:],.99999*value[...,0,:]))


def _depths(area,visible,infrared,pressure,gravity,diagnostics,*,subinterval=False):
    p=jnp.asarray(pressure)
    n=area.shape[0]
    if p.shape!=(2*n+3,*area.shape[1:]):
        raise ValueError('particle pressure grid must have 2*N+3 source levels')
    factor=jnp.repeat(area,2,axis=0)
    factor=jnp.concatenate([jnp.zeros_like(factor[:2]),factor],axis=0)*jnp.diff(p,axis=0)/gravity
    def spectral(values):
        if not subinterval:
            values=jnp.repeat(values,2,axis=0)
        values=jnp.concatenate([jnp.zeros_like(values[:2]),values],axis=0)
        ext=factor[...,None]*values[...,0,:]
        scat=factor[...,None]*values[...,1,:]
        moment=scat*values[...,2,:]
        return tuple(jnp.moveaxis(x,-1,1) for x in (ext,scat,moment))
    return ParticleOptics(spectral(visible),spectral(infrared),diagnostics)


def dust_optics(mass_ratios,number_ratios,pressure,*,bands=12,gravity=3.72):
    """Original opt_dst/calc_opts for any number of non-tagged dust populations.

    Inputs have levels first and populations last, in kg/kg and particles/kg.
    Source asymmetry averages are retained, including their area weighting.
    """
    mass,number=map(jnp.asarray,(mass_ratios,number_ratios))
    if mass.shape!=number.shape or mass.ndim<2 or mass.shape[-1]<1:
        raise ValueError('dust mass and number require matching population axes')
    if bands not in (12,15): raise ValueError('optical tables support 12 or 15 bands')
    data=load_tables();active=(mass>1.e-12)&(number>1.)
    radius,area=_radius(mass,number,2500.,active,deviation=.63676)
    effective=jnp.clip(radius*jnp.exp(1.5*.63676**2),1.e-7,50.e-6)
    distribution=_distribution(effective,data['moment_edges_m'],.63676)
    total=jnp.sum(area,axis=-1)
    weights=area/jnp.where(total[...,None]>0.,total[...,None],1.)
    mixed=jnp.sum(weights[...,None]*distribution,axis=-2)
    visible=_average(data[f'dust_{bands}_sw'][0],mixed)
    infrared=_average(data[f'dust_{bands}_ir'][0],mixed)
    return _depths(total,visible,infrared,pressure,gravity,
        dict(area_m2_kg=total,effective_radius_m=effective,size_distribution=mixed))


def co2_cloud_optics(ice_ratio,pressure,*,nuclei_per_kg=1.e5,bands=15,gravity=3.72):
    """Original opt_co2cld lognormal size integration with native 15-band tables.

    initcld does not load CO2 cloud tables in its 12-band branch. Reject that
    combination rather than inventing band values or reading uninitialized data.
    """
    if bands!=15: raise ValueError('reference CO2 cloud optics requires 15-band radiation')
    data=load_tables();mass=jnp.asarray(ice_ratio)
    threshold=(1.e-7*jnp.exp(.5*.3087**2))**3*nuclei_per_kg*jnp.pi*1620./.75
    active=mass>threshold
    radius,area=_radius(mass,nuclei_per_kg,1620.,active,deviation=.3087)
    effective=jnp.clip(radius*jnp.exp(1.5*.3087**2),1.e-7,100.e-6)
    distribution=_distribution(effective,data['co2_edges_m'],.3087)
    visible=_average(data['co2_15_sw'][0],distribution)
    infrared=_average(data['co2_15_ir'][0],distribution)
    return _depths(area,visible,infrared,pressure,gravity,
        dict(area_m2_kg=area,effective_radius_m=effective,size_distribution=distribution))


def water_bulk_optics(ice_ratio,pressure,temperature,*,nuclei_per_kg=1.e5,bands=15,gravity=3.72):
    """Original opt_cld_blk single-radius interpolation and 273 K ice/liquid switch.

    Temperature is the original 2*N+3 driver grid, before fillpt interpolation.
    The source reads the first five IR entries of its 15-band water tables when
    using 12 bands; this behavior is preserved without remapping wavelengths.
    """
    if bands not in (12,15): raise ValueError('optical tables support 12 or 15 bands')
    data=load_tables();mass=jnp.asarray(ice_ratio);t=jnp.asarray(temperature)
    if t.shape!=jnp.shape(pressure): raise ValueError('water optics needs the original temperature grid')
    active=mass>1.e-21*nuclei_per_kg*jnp.pi*917./.75
    radius,area=_radius(mass,nuclei_per_kg,917.,active)
    radius=jnp.clip(radius,1.e-7,100.e-6)
    centers=jnp.asarray(data['bulk_radii_m'])
    index=jnp.clip(jnp.searchsorted(centers,radius,side='left')-1,0,len(centers)-2)
    weight=jnp.log(radius/centers[index+1])/jnp.log(centers[index]/centers[index+1])
    def interpolate(key):
        table=jnp.asarray(data[key][0])
        value=table[index+1]+(table[index]-table[index+1])*weight[...,None,None]
        return value.at[...,1,:].set(jnp.minimum(value[...,1,:],.99999*value[...,0,:]))
    def phase(key):
        ice=jnp.repeat(interpolate(f'water_ice_15_{key}'),2,axis=0)
        liquid=jnp.repeat(interpolate(f'water_liquid_15_{key}'),2,axis=0)
        values=jnp.where((t[3:]<273.)[...,None,None],ice,liquid)
        return values[...,:5] if bands==12 and key=='ir' else values
    return _depths(area,phase('sw'),phase('ir'),pressure,gravity,
        dict(area_m2_kg=area,effective_radius_m=radius),subinterval=True)


def water_moment_optics(ice_ratio,core_ratio,number_ratio,pressure,*,bands=12,gravity=3.72):
    """Original opt_cld: coated-sphere radius-ratio selection and size moments.

    Core-to-mantle table selection retains the source REAL*4 rounding. It is a
    discrete physical switch; derivatives within a selected table are retained.
    """
    if bands not in (12,15): raise ValueError('optical tables support 12 or 15 bands')
    ice,core,number=map(jnp.asarray,(ice_ratio,core_ratio,number_ratio))
    if ice.shape!=core.shape or ice.shape!=number.shape:
        raise ValueError('water ice, core and number shapes must match')
    data=load_tables();mass=ice+core;active=(mass>1.e-7)&(number>1.)
    total=jnp.where(active,mass,1.)
    density=jnp.where(active,(ice*917.+core*2500.)/total,917.)
    fraction=jnp.where(active,core/2500./jnp.where(active,ice/917.+core/2500.,1.),0.)
    fraction=fraction.astype(jnp.float32).astype(jnp.float64)**(1./3.)
    fraction=fraction.astype(jnp.float32)
    index=jnp.clip(jnp.searchsorted(jnp.asarray(data['core_radius_ratios']),fraction,side='right')-1,0,14)
    radius,area=_radius(mass,number,density,active,deviation=.3087)
    effective=jnp.clip(radius*jnp.exp(1.5*.3087**2),1.e-7,100.e-6)
    distribution=_distribution(effective,data['moment_edges_m'],.3087)
    visible=_average(jnp.asarray(data[f'water_moment_{bands}_sw'])[index],distribution)
    infrared=_average(jnp.asarray(data[f'water_moment_{bands}_ir'])[index],distribution)
    correction=jnp.moveaxis(jnp.asarray(data['water_absorption_correction']),0,-1)[index]
    ir_table=jnp.asarray(data[f'water_moment_{bands}_ir'])[index]
    brightness=jnp.sum(distribution[...,None]*correction[...,None]*(ir_table[...,0,:]-ir_table[...,1,:]),axis=-2)
    return _depths(area,visible,infrared,pressure,gravity,
        dict(area_m2_kg=area,effective_radius_m=effective,size_distribution=distribution,
            core_table_index=index,brightness_absorption=brightness))


def make_particle_provider(*,bands=12,dust_tracers=(),co2_cloud_tracer=None,
                           water_bulk_tracer=None,water_moment_tracers=None,
                           co2_nuclei_per_kg=1.e5,water_nuclei_per_kg=1.e5):
    """Connect explicit particle tracer names to the native radiation callback.

    Dust populations use (mass, number) name pairs. Moment clouds use an
    (ice mass, core mass, number) tuple. Missing selected tracers are errors.
    Return None only when the caller explicitly selects no active species.
    """
    if bands not in (12,15): raise ValueError('particle provider requires 12 or 15 bands')
    if co2_cloud_tracer is not None and bands!=15:
        raise ValueError('reference CO2 cloud optics requires 15-band radiation')
    if any(len(pair)!=2 for pair in dust_tracers):
        raise ValueError('each dust population requires mass and number tracer names')
    if water_moment_tracers is not None and len(water_moment_tracers)!=3:
        raise ValueError('moment clouds require ice, core and number tracer names')
    if co2_nuclei_per_kg<=0. or water_nuclei_per_kg<=0.:
        raise ValueError('cloud condensation nuclei must be positive')
    def provider(state,coords,specs,body,forcing,grid):
        def tracer(name):
            return tracer_to_nodal(coords.horizontal,state.dynamics.tracers[name])
        species=[];options=dict(bands=bands,gravity=body.gravity_m_s2)
        if dust_tracers:
            species.append(dust_optics(jnp.stack([tracer(pair[0]) for pair in dust_tracers],axis=-1),
                jnp.stack([tracer(pair[1]) for pair in dust_tracers],axis=-1),grid.pressure,**options))
        if co2_cloud_tracer is not None:
            species.append(co2_cloud_optics(tracer(co2_cloud_tracer),grid.pressure,
                nuclei_per_kg=co2_nuclei_per_kg,**options))
        if water_bulk_tracer is not None:
            full_t=grid.temperature[3::2]
            original_t=grid.temperature.at[4:-1:2].set(.5*(full_t[:-1]+full_t[1:]))
            original_t=original_t.at[-1].set(.5*(full_t[-1]+state.surface_temperature[0]))
            species.append(water_bulk_optics(tracer(water_bulk_tracer),grid.pressure,original_t,
                nuclei_per_kg=water_nuclei_per_kg,**options))
        if water_moment_tracers is not None:
            species.append(water_moment_optics(*(tracer(name) for name in water_moment_tracers),
                grid.pressure,**options))
        if not species: return None
        return tuple(tuple(sum(item[spectrum][component] for item in species)
            for component in range(3)) for spectrum in range(2))
    return provider
