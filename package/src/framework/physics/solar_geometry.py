"""Differentiable sol88/zenith geometry from the reference astronomy patch.

Copyright 2023 United States Government as represented by NASA.
NASA Ames GCM patches: Apache License 2.0, https://www.apache.org/licenses/LICENSE-2.0
"""
from src.framework.gcm._dinosaur import jnp


def daylight_geometry(declination,latitude):
    """Reference sunset angle, daylight-mean cosine and daylight fraction."""
    latitude=jnp.asarray(latitude)
    ss=jnp.sin(latitude)*jnp.sin(declination)
    cc=jnp.cos(latitude)*jnp.cos(declination)
    polar=jnp.abs(jnp.abs(latitude)-jnp.pi*.5)<=1.e-7
    ratio=-ss/jnp.where(cc!=0.,cc,1.)
    magnitude=jnp.abs(ratio)
    interior=magnitude<1.-1.e-7
    angle=jnp.arccos(jnp.where(interior,ratio,0.))
    angle=jnp.where(interior,angle,jnp.where(ratio<0.,jnp.pi,0.))
    angle=jnp.where(magnitude-1.+1.e-7==0.,(magnitude-ratio)*.5*jnp.pi,angle)
    polar_angle=jnp.where(latitude*declination>0.,jnp.pi,0.)
    angle=jnp.where(polar,polar_angle,angle)
    cc=jnp.where(polar,0.,cc)
    angle=jnp.where(declination==0.,jnp.pi*.5,angle)
    ss=jnp.where(declination==0.,0.,ss)
    cc=jnp.where(declination==0.,jnp.cos(latitude),cc)
    safe=jnp.where(angle>0.,angle,1.)
    cosine=jnp.where(angle>0.,jnp.maximum(ss+cc*jnp.sin(angle)/safe,0.),0.)
    return angle,cosine,angle/jnp.pi


def averaged_zenith(day_fraction,declination,longitude,latitude,half_angle):
    """Reference zenith, including partial sunrise/sunset integration windows.

    half_angle must be positive, in radians. The source driver sets it to
    0.5*pi*radiation_interval/model_sol. Inactive branches have finite algebra
    so reverse derivatives can pass through valid solar geometry.
    """
    longitude,latitude=jnp.broadcast_arrays(longitude,latitude)
    sunset,_,_=daylight_geometry(declination,latitude)
    ss=jnp.sin(latitude)*jnp.sin(declination)
    cc=jnp.cos(latitude)*jnp.cos(declination)
    arg=jnp.asarray(half_angle)
    safe_arg=jnp.where(arg>0.,arg,1.)
    hloc=jnp.fmod(day_fraction*2.*jnp.pi+longitude+jnp.pi,2.*jnp.pi)
    hloc=jnp.where(hloc>jnp.pi,hloc-2.*jnp.pi,hloc)
    hlpar=hloc+arg
    armhl=arg-hloc
    sets=hlpar>sunset
    rises=armhl>sunset
    safe_sunset=jnp.where(sunset>0.,sunset,1.)
    entire=jnp.where(sunset>0.,ss+cc*jnp.sin(sunset)/safe_sunset,0.)
    both=rises&sets

    left_wrap=hlpar>jnp.pi
    wrapped=left_wrap|(armhl>jnp.pi)
    dele=.5*jnp.maximum(jnp.where(left_wrap,hlpar+sunset-2.*jnp.pi,sunset+hlpar),0.)
    delw=.5*jnp.maximum(jnp.where(left_wrap,sunset+armhl,armhl+sunset-2.*jnp.pi),0.)
    span=dele+delw
    safe_span=jnp.where(span>0.,span,1.)
    wrapped_cos=jnp.where(span>0.,ss+cc*(jnp.cos(sunset-dele)*jnp.sin(dele)+
        jnp.cos(sunset-delw)*jnp.sin(delw))/safe_span,0.)
    delsh=.5*jnp.where(rises,sunset+hlpar,sunset+armhl)
    safe_delsh=jnp.where(delsh>0.,delsh,1.)
    partial_cos=jnp.where(delsh>0.,ss+cc*jnp.cos(sunset-delsh)*jnp.sin(delsh)/safe_delsh,0.)
    partial_frac=jnp.where(delsh>0.,delsh/safe_arg,0.)
    normal_cos=ss+cc*jnp.cos(hloc)*jnp.sin(arg)/safe_arg
    cosine=jnp.where(rises|sets,partial_cos,normal_cos)
    fraction=jnp.where(rises|sets,partial_frac,1.)
    cosine=jnp.where(wrapped,wrapped_cos,cosine)
    fraction=jnp.where(wrapped,span/safe_arg,fraction)
    cosine=jnp.where(both,entire,cosine)
    fraction=jnp.where(both,jnp.where(sunset>0.,sunset/safe_arg,0.),fraction)
    cosine=jnp.clip(cosine,0.,1.)
    fraction=jnp.minimum(1.,fraction)
    return jnp.where(arg>0.,cosine,jnp.nan),jnp.where(arg>0.,fraction,jnp.nan)
