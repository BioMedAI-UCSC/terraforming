"""Reference wind-stress and convective dust lifting on level-first arrays.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jax,jnp


def stress_flux(stress,threshold,density,latitude_rad,snow,frost,opacity,*,scheme=1,
                efficiency=.0065,gravity=3.72,north_snow_threshold=1.,south_snow_threshold=1.,
                frost_threshold=1.,opacity_threshold=100.):
    """Original flux_stress in kg/m2/s; keep strict source mask boundaries."""
    stress,threshold,rho,lat,snow,frost,tau=map(jnp.asarray,
        (stress,threshold,density,latitude_rad,snow,frost,opacity))
    if scheme in (2,3,4):
        active=stress>=threshold
        safe_stress=jnp.where(active,stress,1.)
        flux=efficiency/gravity*jnp.sqrt(safe_stress**3/rho)
        if scheme!=3:
            ratio=jnp.sqrt(threshold/safe_stress)
            flux=flux*(1.-ratio)*(1.+ratio)**2*(2.61 if scheme==4 else 1.)
    else:
        flux=efficiency*(2.3e-3 if scheme==5 else 2.43e-3)*stress**2*(stress-threshold)/threshold
    blocked=(stress<threshold)|((snow>north_snow_threshold)&(lat>0.))|(
        (snow>south_snow_threshold)&(lat<0.))|(frost>frost_threshold)|(tau>opacity_threshold)
    return jnp.where(blocked,0.,jnp.maximum(flux,0.))


def devil_flux(temperature_bottom,temperature_surface,pressure_bottom,pressure_pbl,pbl_top_layer,
               bottom_layer_index,sensible_heat_flux,latitude_rad,snow,frost,*,dda=True,dth=False,
               fv3=False,heat_flux_threshold=0.,dda_efficiency=1.4e-10,dth_efficiency=.5,
               grain_diameter_m=6e-6,fv3_source=.3,fv3_radius_m=2.5e-6,
               temperature_threshold=22.,north_snow_threshold=1.,south_snow_threshold=1.,
               frost_threshold=1.,gas_constant=189.02,cp=735.94,gravity=3.72):
    """Original three dust-devil flux choices, including their literal units."""
    t,ts,p,pp,top,heat,lat,snow,frost=map(jnp.asarray,(temperature_bottom,temperature_surface,
        pressure_bottom,pressure_pbl,pbl_top_layer,sensible_heat_flux,latitude_rad,snow,frost))
    xsi=gas_constant/cp+1.
    flux=jnp.zeros_like(t)
    if dda or dth:
        # (1-x**a)/(1-x) has the finite limit a at a collapsed PBL.
        ratio=pp/p;collapsed=ratio==1.
        safe=jnp.where(collapsed,.5,ratio)
        b=-jnp.expm1(xsi*jnp.log(safe))/((1.-safe)*xsi*p)
        b=jnp.where(collapsed,1./p,b)
    if dda:
        flux=flux+jnp.where((heat>heat_flux_threshold)&(top<bottom_layer_index),
            dda_efficiency*(1.-b)*heat,0.)
    if dth:
        rho=p/(gas_constant*t)
        limit=jnp.sqrt(1.+15./(2500.*gravity*grain_diameter_m))*jnp.sqrt(2500.*gravity*grain_diameter_m/rho)
        eta=1.-b;eta_h=(t-ts)/ts
        dp=p*(-jnp.expm1(.5*eta/(.5*eta-1.)*(eta_h/(xsi-1.))))
        wind=dp/rho
        flux=flux+jnp.where(wind>=limit,dth_efficiency*(rho*wind**2-15.)/gravity,0.)
    if fv3:
        scaling=(gravity/600.)*1.33*(2500./2.5)*1e-6
        amount=fv3_source*scaling*(fv3_radius_m/1e-6)**2*1.2e-2/200.
        flux=flux+jnp.where(ts-t<temperature_threshold,0.,amount)
    blocked=((snow>north_snow_threshold)&(lat>0.))|((snow>south_snow_threshold)&(lat<0.))|(frost>frost_threshold)
    return jnp.where(blocked,0.,flux)


class LiftingResult(NamedTuple):
    mass_ratio: object
    number_ratio: object
    surface_dust_kg_m2: object
    diagnostics: object


def reference_efficiencies(effective_radius_m=2e-6,*,bands=12):
    """Original initracer fixed-radius VIS extinction and IR absorption."""
    from src.framework.physics.particle_optics import load_tables,_distribution,_average
    if bands not in (12,15): raise ValueError('dust optical tables require 12 or 15 bands')
    data=load_tables()
    radius=jnp.clip(jnp.asarray(effective_radius_m),1e-7,50e-6)
    distribution=_distribution(radius,data['moment_edges_m'],.63676)
    visible=_average(data[f'dust_{bands}_sw'][0],distribution)
    infrared=_average(data[f'dust_{bands}_ir'][0],distribution)
    index=3 if bands==12 else 6
    return jnp.stack([visible[...,0,5],infrared[...,0,index]-infrared[...,1,index]])


def _inject(q,n,layer_mass,top,amount,particle_mass,pbl,fixed_layers):
    level=jnp.arange(q.shape[0]).reshape((q.shape[0],)+(1,)*(q.ndim-1))
    chosen=level>=top if pbl else ((level>=q.shape[0]-1-fixed_layers[1])&
        (level<=q.shape[0]-1-fixed_layers[0]))
    carrier=jnp.sum(jnp.where(chosen,layer_mass,0.),axis=0)
    addition=jnp.where(chosen,amount/jnp.where(carrier>0.,carrier,1.),0.)
    valid=(carrier>0.)&(top>=0)&(top<q.shape[0]) if pbl else carrier>0.
    addition=jnp.where(valid,addition,jnp.nan)
    return q+addition,n+addition/particle_mass


def interactive_step(mass_ratio,number_ratio,temperature,pressure_half,pressure_full,
                     temperature_surface,stress,sensible_heat_flux,pbl_top_layer,pressure_pbl,
                     latitude_rad,snow,frost,opacity,surface_dust_kg_m2,dt_seconds,*,
                     stress_lift=True,convective_lift=True,stress_options=None,devil_options=None,
                     inject_pbl_stress=False,inject_pbl_devils=True,fixed_layers=(0,0),
                     stress_radius_m=2e-6,devil_radius_m=2e-6,limited_reservoir=False,
                     initial_surface_dust_kg_m2=30.,reservoir_factor=20.,threshold_stress=.022,
                     threshold_decrease=1e-11,devil_threshold_increase=.2,gravity=3.72,
                     gas_constant=189.02,cp=735.94,dual_mode=0,dual_scale=1.,scenario_opacity=None):
    """One physical dust mode: source devil/stress order and reservoir feedback.

    The unlimited source reservoir may become negative, as in the reference.
    Limited-reservoir mode changes the stress threshold; it does not cap mass.
    Dual modes 1 and 21 suppress lifting above the scenario opacity target.
    Background injection follows this step and uses the same old opacity.
    """
    q,n,t,p,full,ts,top,reservoir=map(jnp.asarray,(mass_ratio,number_ratio,temperature,
        pressure_half,pressure_full,temperature_surface,pbl_top_layer,surface_dust_kg_m2))
    if q.shape!=t.shape or n.shape!=t.shape or full.shape!=t.shape or p.shape!=(t.shape[0]+1,*t.shape[1:]):
        raise ValueError('dust and atmosphere grids must match')
    if len(fixed_layers)!=2 or not 0<=fixed_layers[0]<=fixed_layers[1]<t.shape[0]:
        raise ValueError('fixed dust layers must be ordered bottom offsets within the column')
    if not isinstance(dual_mode,int) or dual_mode<0:
        raise ValueError('dual dust mode must be a nonnegative integer')
    if dual_mode in (1,21) and scenario_opacity is None:
        raise ValueError('opacity-constrained dust lifting requires a scenario opacity')
    blocked=(dual_scale*jnp.asarray(scenario_opacity)-opacity<0.) if dual_mode in (1,21) else False
    q,n=jnp.maximum(q,1e-31),jnp.maximum(n,1e-31)
    layer_mass=jnp.diff(p,axis=0)/gravity
    def inject(q,n,amount,radius,pbl):
        particle_mass=4./3.*jnp.pi*2500.*(radius*jnp.exp(-.5*.63676**2))**3
        return _inject(q,n,layer_mass,top,amount,particle_mass,pbl,fixed_layers)
    devil=jnp.zeros_like(reservoir);wind=jnp.zeros_like(reservoir)
    if convective_lift:
        devil=devil_flux(t[-1],ts,full[-1],pressure_pbl,top,t.shape[0]-1,sensible_heat_flux,
            latitude_rad,snow,frost,gas_constant=gas_constant,cp=cp,gravity=gravity,
            **({} if devil_options is None else devil_options))
        devil=jnp.where(blocked,0.,devil)
        if limited_reservoir:
            increase=threshold_stress*jnp.expm1(2.*reservoir_factor*(initial_surface_dust_kg_m2-reservoir))
            devil=jnp.where(increase>devil_threshold_increase,0.,devil)
        devil=jnp.maximum(devil*dt_seconds,0.)
        q,n=inject(q,n,devil,devil_radius_m,inject_pbl_devils)
        reservoir=reservoir-devil
    threshold=jnp.full_like(reservoir,threshold_stress)
    if stress_lift:
        if limited_reservoir:
            threshold=threshold_stress*jnp.exp(2.*reservoir_factor*(initial_surface_dust_kg_m2-reservoir))-threshold_decrease*dt_seconds
        wind=stress_flux(stress,threshold,full[-1]/(gas_constant*t[-1]),latitude_rad,snow,frost,opacity,
            gravity=gravity,**({} if stress_options is None else stress_options))
        wind=jnp.where(blocked,0.,wind)
        wind=jnp.maximum(wind*dt_seconds,0.)
        q,n=inject(q,n,wind,stress_radius_m,inject_pbl_stress)
        reservoir=reservoir-wind
    return LiftingResult(q,n,reservoir,dict(devil_lift_kg_m2=devil,stress_lift_kg_m2=wind,
        stress_threshold_pa=threshold))


def interactive_modes_step(mass_ratio,number_ratio,*arguments,surface_dust_kg_m2,**options):
    """Source repeats the first mode's flux and reservoir gating for every mode."""
    mass,number,reservoir=map(jnp.asarray,(mass_ratio,number_ratio,surface_dust_kg_m2))
    if mass.shape!=number.shape or reservoir.shape!=mass.shape[1:]:
        raise ValueError('dust modes and surface reservoirs must share their last mode axis')
    first=interactive_step(mass[...,0],number[...,0],*arguments,
        surface_dust_kg_m2=reservoir[...,0],**options)
    results=[first]
    for mode in range(1,mass.shape[-1]):
        result=interactive_step(mass[...,mode],number[...,mode],*arguments,
            surface_dust_kg_m2=reservoir[...,0],**options)
        results.append(result._replace(surface_dust_kg_m2=reservoir[...,mode]+
            result.surface_dust_kg_m2-reservoir[...,0]))
    return LiftingResult(*(jnp.stack([getattr(r,key) for r in results],axis=-1)
        for key in ('mass_ratio','number_ratio','surface_dust_kg_m2')),
        {key:jnp.stack([r.diagnostics[key] for r in results],axis=-1) for key in first.diagnostics})


def background_step(mass_ratio,number_ratio,pressure_half,pbl_top_layer,temperature_surface,
                    snow,current_opacity,scenario_opacity,scenario_change,surface_dust_kg_m2,
                    dt_seconds,*,extinction_efficiency,effective_radius_m=2e-6,opacity_fraction=1.,
                    opacity_scale=1.,injection_scale=1.,injection_type=1,inject_through_pbl=True,
                    fixed_layers=(0,0),minimum_temperature_k=120.,exclude_caps=True,
                    minimum_scenario_change=-999.,local_time_fraction=None,local_time_scale=0.,
                    remove_excess=False,sink_scale=1.,bottom_sink=False,
                    model_sol_seconds=88704.,gravity=3.72):
    """Source scenario-driven dust injection and removal for one physical mode.

    `current_opacity` is the source 610 Pa diagnostic; `scenario_change` is the
    difference between the two bounding seasonal maps. Coefficients are the
    original fixed-radius VIS extinction or IR absorption efficiencies.
    Bottom removal tallies material before clearing a cell. Surface deposition
    uses the local column, replacing the source's stale loop-index access.
    These repairs give finite conservative meanings to undefined source cases.
    """
    q,n,p,top,ts,snow,current,scenario,change,reservoir=map(jnp.asarray,(mass_ratio,number_ratio,
        pressure_half,pbl_top_layer,temperature_surface,snow,current_opacity,scenario_opacity,
        scenario_change,surface_dust_kg_m2))
    if q.shape!=n.shape or p.shape!=(q.shape[0]+1,*q.shape[1:]):
        raise ValueError('background dust and pressure interfaces must match')
    if len(fixed_layers)!=2 or not 0<=fixed_layers[0]<=fixed_layers[1]<q.shape[0]:
        raise ValueError('fixed dust layers must be ordered bottom offsets within the column')
    if injection_type not in range(1,8): raise ValueError('dust injection type must be 1 through 7')
    q,n=jnp.maximum(q,1e-31),jnp.maximum(n,1e-31)
    current=jnp.maximum(current,1e-10)
    ratio=scenario*opacity_scale/current
    scale=jnp.broadcast_to(jnp.asarray(injection_scale),ratio.shape)
    if injection_type==2:
        average=jnp.sum(jnp.where(ratio>1.,ratio,0.))/jnp.maximum(jnp.sum(ratio>1.),1.)
        scale=jnp.full_like(ratio,jnp.maximum(average,1.))
    elif injection_type==3: scale=jnp.where(ratio>1.,ratio,scale)
    elif injection_type==4: scale=jnp.where(ratio>1.,jnp.exp(ratio-1.),scale)
    elif injection_type>=5: scale=jnp.where(ratio>1.,ratio**(injection_type-3),scale)
    delta=scenario*opacity_scale-current
    if local_time_fraction is not None:
        delta=jnp.where(jnp.asarray(local_time_fraction)>0.,
            (jnp.asarray(local_time_fraction)*local_time_scale+scale)/scale*delta,0.)
    blocked=(change<=minimum_scenario_change)|(ts<minimum_temperature_k)
    if exclude_caps: blocked=blocked|(snow>.9)
    delta=jnp.where(blocked,0.,delta)
    rm=effective_radius_m*jnp.exp(-2.5*.63676**2)
    rss=rm*jnp.exp(.63676**2)
    mass_particle=4./3.*jnp.pi*2500.*rm**3*jnp.exp(4.5*.63676**2)
    injected=jnp.where(delta>0.,opacity_fraction*4./3.*scale*delta*2500.*rm**3*
        jnp.exp(4.5*.63676**2)*dt_seconds/(extinction_efficiency*rss**2*model_sol_seconds),0.)
    layer_mass=jnp.diff(p,axis=0)/gravity
    q,n=_inject(q,n,layer_mass,top,injected,mass_particle,inject_through_pbl,fixed_layers)
    reservoir=reservoir-injected
    deposited=jnp.zeros_like(injected);sink_repair=jnp.zeros_like(injected)
    if remove_excess:
        column_mass=jnp.sum(layer_mass,axis=0)
        sink=jnp.minimum(opacity_fraction*sink_scale*(scenario*opacity_scale-current)/
            (extinction_efficiency*jnp.pi*rss**2)/column_mass,0.)
        requested_number=-sink*dt_seconds/model_sol_seconds
        requested_mass=requested_number*mass_particle
        mean_mass=jnp.sum(q*layer_mass,axis=0)/column_mass
        mean_number=jnp.sum(n*layer_mass,axis=0)/column_mass
        removed_mass=jnp.minimum(requested_mass,mean_mass)
        removed_number=jnp.minimum(requested_number,mean_number)
        # The source adds 1e-32 to these caps, allowing a tiny negative ratio.
        sink_repair=jnp.maximum(jnp.minimum(requested_mass,mean_mass+1e-32)-mean_mass,0.)*column_mass
        if bottom_sink:
            def remove(values,amount):
                def cell(remaining,row):
                    value,carrier=row
                    removed=jnp.minimum(value*carrier,remaining)
                    return remaining-removed,value-removed/carrier
                return jax.lax.scan(cell,amount*column_mass,(values[::-1],layer_mass[::-1]))[1][::-1]
            q,n=remove(q,removed_mass),remove(n,removed_number)
        else:
            q=q*(1.-removed_mass/jnp.where(mean_mass>0.,mean_mass,1.))
            n=n*(1.-removed_number/jnp.where(mean_number>0.,mean_number,1.))
        deposited=removed_mass*column_mass
        reservoir=reservoir+deposited
    return LiftingResult(q,n,reservoir,dict(injected_kg_m2=injected,deposited_kg_m2=deposited,
        injection_scale=scale,opacity_delta=delta,negative_sink_repair_kg_m2=sink_repair))


def background_modes_step(mass_ratio,number_ratio,*arguments,surface_dust_kg_m2,**options):
    """Source physical-mode loop with per-mode radius and opacity fractions."""
    mass,number,reservoir=map(jnp.asarray,(mass_ratio,number_ratio,surface_dust_kg_m2))
    if mass.shape!=number.shape or reservoir.shape!=mass.shape[1:]:
        raise ValueError('dust modes and surface reservoirs must share their last mode axis')
    count=mass.shape[-1]
    efficiency=options.pop('extinction_efficiency')
    modal={key:jnp.asarray(options.pop(key,default)) for key,default in
        (('effective_radius_m',2e-6),('extinction_efficiency',efficiency),
         ('opacity_fraction',jnp.asarray([1.]+[0.]*(count-1))))}
    for key,value in modal.items():
        if value.ndim>0 and value.shape!=(count,):
            raise ValueError(f'{key} must be scalar or have one entry per physical dust mode')
    fractions=jnp.broadcast_to(modal['opacity_fraction'],(count,))
    modal['opacity_fraction']=jnp.where(jnp.sum(fractions)==1.,fractions,
        jnp.asarray([1.]+[0.]*(count-1)))
    results=[]
    for mode in range(count):
        selected={key:value if value.ndim==0 else value[mode] for key,value in modal.items()}
        results.append(background_step(mass[...,mode],number[...,mode],*arguments,
            surface_dust_kg_m2=reservoir[...,mode],**selected,**options))
    return LiftingResult(*(jnp.stack([getattr(r,key) for r in results],axis=-1)
        for key in ('mass_ratio','number_ratio','surface_dust_kg_m2')),
        {key:jnp.stack([r.diagnostics[key] for r in results],axis=-1) for key in results[0].diagnostics})
