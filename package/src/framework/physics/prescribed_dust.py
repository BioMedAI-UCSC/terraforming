"""Reference prescribed dust profiles, seasonal fields and optical properties.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.particle_optics import (load_tables,_distribution,_average,ParticleOptics)


def storm_opacity(solar_longitude_deg,longitude_rad,latitude_rad,*,slope_deg,latitude_coefficient,
                  amplitude=0.,longitude_width_deg=60.,latitude_width_deg=20.,
                  center_longitude_deg=10.,center_latitude_deg=15.,onset_deg=180.):
    """Released add_storm equations with its two undefined locals supplied.

    The source never initializes slope or c. Both are required here; c is the
    latitude quadratic coefficient in inverse radians squared. The two source
    longitude images and its asymmetric cross coefficient are retained.
    """
    lon,lat=jnp.asarray(longitude_rad),jnp.asarray(latitude_rad)
    theta=jnp.deg2rad(jnp.asarray(slope_deg))
    sx,sy=jnp.deg2rad(jnp.asarray(longitude_width_deg)),jnp.deg2rad(jnp.asarray(latitude_width_deg))
    valid=jnp.isfinite(sx)&(sx>0.)&jnp.isfinite(sy)&(sy>0.)
    for value in (slope_deg,latitude_coefficient,amplitude,center_longitude_deg,center_latitude_deg,onset_deg):
        valid=valid&jnp.isfinite(jnp.asarray(value))
    sx=jnp.where(valid,sx,1.);sy=jnp.where(valid,sy,1.)
    center=jnp.deg2rad(jnp.asarray(center_longitude_deg))
    alternate=center+jnp.where(jnp.asarray(center_longitude_deg)<180.,2.*jnp.pi,-2.*jnp.pi)
    dy=lat-jnp.deg2rad(jnp.asarray(center_latitude_deg))
    a=(jnp.cos(theta)/(jnp.sqrt(2.)*sx))**2+(jnp.sin(theta)/(jnp.sqrt(2.)*sy))**2
    b=-jnp.sin(theta)/(2.*sx)**2+jnp.sin(2.*theta)/(2.*sy)**2
    def image(origin):
        dx=lon-origin
        return jnp.exp(-(a*dx**2+2.*b*dx*dy+jnp.asarray(latitude_coefficient)*dy**2))
    opacity=jnp.asarray(amplitude)*(image(center)+image(alternate))*.5*(
        1.+jnp.tanh(jnp.asarray(solar_longitude_deg)-jnp.asarray(onset_deg)))
    return jnp.where(valid,opacity,jnp.nan)


def pulse_layers(pressure_half,pressure_full,*,center_pa=30.,width=1.):
    """Original dust_distrib_pulse2; the profile integrates to one.

    The source comment labels the center as mbar, but the routine uses the
    supplied pressure directly. Preserve its numerical 30 Pa default here.
    """
    half,full=map(jnp.asarray,(pressure_half,pressure_full))
    profile=jnp.exp(-(jnp.log(full/center_pa)/width)**2)
    cumulative=jnp.concatenate([jnp.zeros_like(half[:1]),jnp.cumsum(profile*jnp.diff(half,axis=0),axis=0)],axis=0)
    total=cumulative[-1]
    scale=jnp.where(total>0.,1./jnp.where(total>0.,total,1.),0.)
    return jnp.diff(cumulative*scale,axis=0)


def layer_opacities(pressure_half,pressure_full,optical_depth,*,reference_pressure_pa=610.,
                    latitude_rad=0.,conrath_type=1,conrath=.003,scenario_opacity=None,
                    top_height_km=None,latitude_variation=False,aerosol_ratio=None,cycle_scheme=0,
                    pulse_fraction=0.,pulse_opacity=0.,pulse_center_pa=30.,pulse_width=1.,column_scale=1.):
    """Original dust_distrib_fixed2 normalization, tracer profiles and pulses.

    The fixed opacity scales with surface/reference pressure. Additive elevated
    pulses do not. Retain the source's second normalization and its pressure
    weighting of the optional fractional pulse.
    """
    half,full=map(jnp.asarray,(pressure_half,pressure_full))
    if half.shape!=(full.shape[0]+1,*full.shape[1:]):
        raise ValueError('dust pressure interfaces and centers must match')
    if not isinstance(cycle_scheme,int) or cycle_scheme<0:
        raise ValueError('dust cycle scheme must be a nonnegative integer')
    if cycle_scheme>0 and aerosol_ratio is None:
        raise ValueError('tracer dust cycle scheme requires an aerosol profile')
    if cycle_scheme>1 and full.shape[0]<5:
        raise ValueError('tracer-profile dust requires at least five layers')
    lat=jnp.asarray(latitude_rad)
    if scenario_opacity is not None:
        opacity=jnp.asarray(scenario_opacity)+jnp.maximum(jnp.asarray(optical_depth),0.)
        height=35. if top_height_km is None else jnp.asarray(top_height_km)
    elif latitude_variation:
        opacity=jnp.asarray(optical_depth)*(.3+.7*jnp.cos(lat))
        height=70.-40.*jnp.sin(lat)**2
    else:
        opacity=jnp.asarray(optical_depth);height=70.
    if cycle_scheme==0:
        profile=(jnp.exp(conrath*(1.-reference_pressure_pa/full)) if conrath_type==0 else
            jnp.exp(.007*(1.-(700./full)**(70./height))))
    elif cycle_scheme==1:
        aerosol=jnp.asarray(aerosol_ratio)
        if aerosol.shape!=full.shape: raise ValueError('dust tracer profile must match pressure centers')
        selected=aerosol>.5*jnp.mean(aerosol,axis=0)
        index=jnp.argmax(selected,axis=0)
        top=jnp.take_along_axis(half,index[None],axis=0)[0]
        height=jnp.maximum(10.,jnp.where(top>0.,10.*jnp.log(reference_pressure_pa/
            jnp.where(top>0.,top,1.)),jnp.inf))
        profile=jnp.exp(conrath*(1.-(reference_pressure_pa/full)**(70./height)))
        profile=jnp.where(jnp.any(selected,axis=0)[None],profile,jnp.nan)
    else:
        aerosol=jnp.asarray(aerosol_ratio)
        if aerosol.shape!=full.shape: raise ValueError('dust tracer profile must match pressure centers')
        profile=aerosol.at[-5:].set(aerosol[-5]).at[:3].set(0.)
    dp=jnp.diff(half,axis=0)
    def integrate(values):
        return jnp.concatenate([jnp.zeros_like(half[:1]),jnp.cumsum(values*dp,axis=0)],axis=0)
    def scale_to(total,target):
        return jnp.where(total>0.,target/jnp.where(total>0.,total,1.),0.)
    target=opacity*half[-1]/reference_pressure_pa
    cumulative=integrate(profile)
    cumulative=cumulative*scale_to(cumulative[-1],target)
    def fractional(_):
        mixed=profile*scale_to(cumulative[-1],half[-1]/reference_pressure_pa)
        mixed=mixed+pulse_fraction*jnp.cos(lat)**2*pulse_layers(half,full,
            center_pa=pulse_center_pa,width=pulse_width)
        integrated=integrate(mixed)
        # The original fractional-pulse path does not guard its final normalization.
        return integrated*(target/integrated[-1])
    def background(_):
        return cumulative*scale_to(cumulative[-1],target)
    cumulative=jax.lax.cond(jnp.asarray(pulse_fraction)>0.,fractional,background,operand=None)
    result=jnp.diff(cumulative,axis=0)*column_scale
    result=result+jnp.where(pulse_opacity>0.,pulse_opacity,0.)*jnp.cos(lat)**2*pulse_layers(
        half,full,center_pa=pulse_center_pa,width=pulse_width)
    return result


def seasonal_fields(solar_longitude_deg,seasons,opacities,heights_km):
    """Original season_table linear interpolation; include the 360-degree endpoint."""
    season=jnp.asarray(seasons);ls=jnp.mod(solar_longitude_deg,360.)
    upper=jnp.clip(jnp.searchsorted(season,ls,side='right'),0,season.shape[0]-1)
    lower=jnp.mod(upper-1,season.shape[0])
    width=season[upper]-season[lower]
    width=jnp.where(width<0.,width+360.,width)
    weight=(season[upper]-ls)/width
    return tuple(weight*jnp.asarray(values)[lower]+(1.-weight)*jnp.asarray(values)[upper]
        for values in (opacities,heights_km))


def fixed_optics(layer_opacity,pressure,*,effective_radius_m=1.5e-6,bands=12,optical_properties=None):
    """Original ini_optdst fixed-size or user-specified optical properties.

    Optional (visible, infrared) arrays contain extinction, scattering and
    asymmetry rows, with 7 visible and 5/8 infrared bands. As in the source,
    user coefficients bypass size averaging and its scattering cap.
    """
    if bands not in (12,15): raise ValueError('fixed dust requires 12 or 15 bands')
    tau,p=map(jnp.asarray,(layer_opacity,pressure))
    n=tau.shape[0]
    if p.shape!=(2*n+3,*tau.shape[1:]):
        raise ValueError('fixed dust pressure grid must have 2*N+3 source levels')
    if optical_properties is None:
        data=load_tables();radius=jnp.clip(jnp.asarray(effective_radius_m),1e-7,50e-6)
        distribution=_distribution(radius,data['moment_edges_m'],.63676)
        visible=_average(data[f'dust_{bands}_sw'][0],distribution)
        infrared=_average(data[f'dust_{bands}_ir'][0],distribution)
    else:
        if len(optical_properties)!=2: raise ValueError('fixed dust requires visible and infrared optical properties')
        visible,infrared=map(jnp.asarray,optical_properties)
        if visible.shape!=(3,7) or infrared.shape!=(3,5 if bands==12 else 8):
            raise ValueError('fixed dust optical properties require 3 rows and 7 visible / 5 or 8 infrared bands')
        distribution=jnp.zeros(20,dtype=visible.dtype)
    qref=visible[...,0,5]
    per_pa=tau/jnp.diff(p[2::2],axis=0)
    amount=jnp.concatenate([jnp.zeros_like(tau[:1]),jnp.zeros_like(tau[:1]),
        jnp.repeat(per_pa,2,axis=0)],axis=0)*jnp.diff(p,axis=0)
    def spectrum(values):
        ext=amount[...,None]*values[...,0,:]/qref[...,None]
        scat=amount[...,None]*values[...,1,:]/qref[...,None]
        moment=scat*values[...,2,:]
        return tuple(jnp.moveaxis(v,-1,1) for v in (ext,scat,moment))
    return ParticleOptics(spectrum(visible),spectrum(infrared),
        dict(reference_extinction=qref,size_distribution=distribution))


def make_fixed_dust_provider(*,bands=12,effective_radius_m=1.5e-6,reference_pressure_pa=610.,
                             scenario=None,aerosol_tracer=None,storm_options=None,optical_properties=None,
                             fixed_calendar_date=-1,**profile_options):
    """Connect prescribed visible opacity and native size optics to radiation.

    The scenario contains seasons, opacity and top_height_km on the nodal grid.
    Its opacity includes the source input-file conversion and map scale.
    Select a fixed date for diagnostic calls outside the radiation callback.
    """
    from src.framework.gcm._dinosaur import scales
    from src.framework.physics.gcm import _true_anomaly
    from src.framework.physics.solar_geometry import radiation_clock
    if not isinstance(fixed_calendar_date,int): raise ValueError(
        'fixed calendar date must be an integer')
    if storm_options is not None and {'slope_deg','latitude_coefficient'}-set(storm_options):
        raise ValueError('the source storm slope and latitude coefficient are undefined; supply both explicitly')
    def provider(state,coords,specs,body,forcing,grid):
        options=dict(profile_options)
        if scenario is not None:
            time=state.dynamics.sim_time*float(specs.dimensionalize(1.,scales.units.second).magnitude)
            calendar,_,_=radiation_clock(time,forcing.rotation_period_s,fixed_calendar_date)
            ls=jnp.mod((_true_anomaly(calendar,forcing)+forcing.ls_perihelion_rad)*180./jnp.pi,360.)
            opacity,height=seasonal_fields(ls,scenario['seasons'],scenario['opacity'],scenario['top_height_km'])
            if storm_options is not None:
                opacity=opacity+storm_opacity(ls,jnp.asarray(coords.horizontal.longitudes)[:,None],
                    jnp.asarray(coords.horizontal.latitudes)[None,:],**storm_options)
            options.update(scenario_opacity=opacity,top_height_km=height)
        if aerosol_tracer is not None:
            options['aerosol_ratio']=coords.horizontal.to_nodal(state.dynamics.tracers[aerosol_tracer])
        tau=layer_opacities(grid.pressure[2::2],grid.pressure[3::2],
            forcing.dust_visible_optical_depth,reference_pressure_pa=reference_pressure_pa,
            latitude_rad=jnp.asarray(coords.horizontal.latitudes)[None,:],**options)
        optics=fixed_optics(tau,grid.pressure,effective_radius_m=effective_radius_m,bands=bands,
            optical_properties=optical_properties)
        return optics.visible,optics.infrared
    return provider
