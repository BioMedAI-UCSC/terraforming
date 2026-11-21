"""Bulk CO2 cloud phase changes and sedimentation on level-first arrays.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple,Any
from jax import Array

from src.framework.gcm._dinosaur import jnp
from src.framework.physics.vertical_solve import solve_tridiagonal

LATENT_HEAT = 590200.
HEAT_CAPACITY = 735.94
ICE_DENSITY = 1620.
LOG_RADIUS_DEVIATION = .3087
# initracer_mod, using KBOLTZ from the released FMS constants patch.
THERMAL_SPEED_SCALE = 3.*(1.38064852e-23/1.66054e-27)/44.


class CloudResult(NamedTuple):
    temperature: Array
    ice_ratio: Array
    gas_mass_loss_kg_m2: Array
    deposition_kg_m2: Array
    radius_m: Array
    diagnostics: dict[str,Any]


def phase_change(temperature, pressure_full, ice_ratio):
    """Original co2condsub: evaporate old ice, then adjust to saturation.

    Ratios are kg ice/kg initial gas. Gas mass rescaling happens after this
    operator. As in the source driver, negative input ice is set to zero.
    """
    temperature,pressure_full,ice_ratio=map(jnp.asarray,
        (temperature,pressure_full,ice_ratio))
    if temperature.shape!=pressure_full.shape or ice_ratio.shape!=temperature.shape:
        raise ValueError('CO2 cloud temperature, pressure and ice shapes must match')
    frost=3182.48/(23.3494-jnp.log(pressure_full/100.))
    evaporated=temperature-jnp.maximum(ice_ratio,0.)*LATENT_HEAT/HEAT_CAPACITY
    ice=HEAT_CAPACITY*jnp.maximum(frost-evaporated,0.)/LATENT_HEAT
    return jnp.maximum(evaporated,frost),ice


def particle_radius(ice_ratio, nuclei_per_kg=1.e5):
    """Mass-derived sedimentation radius, including the source lognormal factors."""
    variance=LOG_RADIUS_DEVIATION**2
    volume=jnp.maximum(ice_ratio,0.)/nuclei_per_kg*.75/jnp.pi/ICE_DENSITY
    # Apply the source radius floor before the cube root for finite clear-air AD.
    minimum_volume=(1.e-7*jnp.exp(.5*variance))**3
    return jnp.maximum(volume,minimum_volume)**(1./3.)*jnp.exp(variance)


def fall_velocity(radius_m,temperature,density,*,gravity=3.72,particle_density=ICE_DENSITY):
    """Original Stokes/Cunningham fallvel with CO2 viscosity, in m/s."""
    thermal=jnp.sqrt(THERMAL_SPEED_SCALE*temperature)
    viscosity=1.59e-6*temperature**1.5/(temperature+244.4)
    knudsen=2.*viscosity/(density*thermal*radius_m)
    alpha=1.246+.42*jnp.exp(-.87/knudsen)
    return 2.*particle_density*gravity*radius_m**2/9.*(1.+alpha*knudsen)/viscosity


def sedimentation(ice_ratio,pressure_half,pressure_full,temperature,temperature_surface,radius_m,
                  heat_diffusivity,dt_seconds,*,gas_constant=189.02,gravity=3.72,
                  particle_density=ICE_DENSITY,density_temperature=None,bottom_density_temperature=None):
    """Original sedimco2 explicit/implicit flux switch and open-bottom deposit.

    Temperature supplies pre-phase interface averages. The driver supplies
    post-phase full-level density separately. CO2 also updates the bottom
    boundary density; bulk water retains its pre-phase bottom temperature.
    Heat diffusivity has N-1 internal interfaces. The original fall-speed
    stencil samples the boundary below the lower cell, including at the ground.
    Return ice ratio, deposited mass and the transport coefficients used.
    """
    q,p,full,t,ts,r,k=map(jnp.asarray,(ice_ratio,pressure_half,pressure_full,temperature,
        temperature_surface,radius_m,heat_diffusivity))
    n=t.shape[0]
    if n<2 or q.shape!=t.shape or r.shape!=t.shape or full.shape!=t.shape or p.shape!=(n+1,*t.shape[1:]):
        raise ValueError('CO2 sedimentation requires matching grids and at least two layers')
    if k.shape!=(n-1,*t.shape[1:]):
        raise ValueError('heat diffusivity must have N-1 internal interfaces')
    return _sediment(q,p,full,t,ts,r,k,dt_seconds,gas_constant,gravity,
        particle_density,density_temperature,bottom_density_temperature)


def _sediment(q,p,full,t,ts,r,k,dt,gas_constant,gravity,particle_density,density_temperature,bottom_density_temperature):
    density=full/(gas_constant*(t if density_temperature is None else density_temperature))
    thickness=jnp.diff(p,axis=0)/gravity/density
    concentration=q*density
    boundary_t=jnp.concatenate([.5*(t[:-1]+t[1:]),.5*(ts+t[-1])[None]])
    bottom_t=t[-1] if bottom_density_temperature is None else bottom_density_temperature
    boundary_rho=jnp.concatenate([p[1:-1]/(gas_constant*boundary_t[:-1]),
        (full[-1]/(gas_constant*bottom_t))[None]])
    w=-fall_velocity(r[1:],boundary_t[1:],boundary_rho[1:],gravity=gravity,particle_density=particle_density)
    spacing=.5*(thickness[:-1]+thickness[1:])
    active=k!=0.
    safe_k=jnp.where(active,k,1.)
    log_density=jnp.log(density[:-1]/density[1:])
    theta=.5*(w*spacing/safe_k+log_density)
    safe_theta=jnp.where(theta!=0.,theta,1.)
    # Series avoids cancellation in coth(theta)-1/theta near zero.
    fitted=jnp.where(jnp.abs(safe_theta)<1.e-3,
        safe_theta/3.-safe_theta**3/45.+2.*safe_theta**5/945.,
        1./jnp.tanh(safe_theta)-1./safe_theta)
    sigma=jnp.where(active & (theta!=0.),fitted,1.)
    lower=concentration[1:]
    ratio=jnp.where(lower==0.,jnp.where(concentration[:-1]==0.,1.,10.),
        jnp.clip(concentration[:-1]/jnp.where(lower!=0.,lower,1.),.1,10.))
    uncorrected=((ratio>.9)&(ratio<1.1))|(jnp.abs(w*dt)>thickness[1:])
    safe_ratio=jnp.where(uncorrected,2.,ratio)
    safe_distance=jnp.where(uncorrected,-spacing,w*dt)
    factor=spacing/safe_distance*jnp.expm1(-safe_distance*jnp.log(safe_ratio)/spacing)/(1.-safe_ratio)
    corrected=-jnp.exp(jnp.log(-w)+jnp.abs(sigma)*jnp.log(factor))
    w1=jnp.where(uncorrected,w,corrected)
    numerator=w1+log_density*k/spacing
    # Equivalent exponential fitting without overflowing exp(2*theta).
    positive=jnp.maximum(safe_theta,0.)
    negative=jnp.minimum(safe_theta,0.)
    fs_positive=numerator/(-jnp.expm1(-2.*jnp.where(positive!=0.,positive,1.)))
    ft_negative=numerator/jnp.expm1(2.*jnp.where(negative!=0.,negative,-1.))
    ft=jnp.where(theta>0.,fs_positive*jnp.exp(-2.*positive),ft_negative)
    fs=jnp.where(theta>0.,fs_positive,ft_negative*jnp.exp(2.*negative))
    ft=jnp.where(active,jnp.where(theta==0.,k/spacing,ft),-w1)
    fs=jnp.where(active,jnp.where(theta==0.,k/spacing,fs),0.)
    zero=jnp.zeros_like(ft[:1])
    ft=jnp.concatenate([zero,ft,-w1[-1:]])
    fs=jnp.concatenate([zero,fs,zero])
    diagonal=ft[1:]+fs[:-1]-thickness/dt
    implicit=jnp.any(diagonal>0.,axis=0)
    diagnostics=dict(implicit=implicit,ft=ft,fs=fs,
        thickness_m=thickness,density_kg_m3=density)
    new,deposit=transport_concentration(concentration,diagnostics,dt)
    return new/density,deposit,diagnostics


def transport_concentration(concentration,coefficients,dt):
    """Carry another extensive particle quantity with the diagnosed ice fluxes.

    Coefficients stay fixed so momentum uses the same mass transport operator.
    This diagnoses deposited momentum without changing the source cloud winds.
    """
    ft,fs,thickness,implicit=(coefficients[k] for k in ('ft','fs','thickness_m','implicit'))
    diagonal=ft[1:]+fs[:-1]-thickness/dt
    zero=jnp.zeros_like(concentration[:1])
    below=jnp.concatenate([zero,concentration[:-1]])
    above=jnp.concatenate([concentration[1:],zero])
    explicit=(diagonal*concentration-ft[:-1]*below-fs[1:]*above)/(-thickness/dt)
    solution=solve_tridiagonal(fs[1:][::-1],-(ft[1:]+fs[:-1]+thickness/dt)[::-1],
        ft[:-1][::-1],(-thickness/dt*concentration)[::-1],reciprocal_pivot=True)[::-1]
    new=jnp.where(implicit[None],solution,explicit)
    deposit=jnp.where(implicit,solution[-1],concentration[-1])*ft[-1]*dt
    return new,deposit


def cloud_step(temperature,pressure_half,pressure_full,ice_ratio,temperature_surface,
               heat_diffusivity,dt_seconds,*,gas_constant=189.02,gravity=3.72,
               nuclei_per_kg=1.e5):
    """Source driver phase changes, diagnosed size, sedimentation and gas source."""
    if isinstance(nuclei_per_kg,(int,float)) and nuclei_per_kg<=0.:
        raise ValueError('cloud condensation nuclei must be positive')
    t,p,full,q=map(jnp.asarray,(temperature,pressure_half,pressure_full,ice_ratio))
    adjusted,formed=phase_change(t,full,q)
    radius=particle_radius(formed,nuclei_per_kg)
    # Interface temperatures precede phase change; full-level density follows it.
    settled,deposit,diagnostics=sedimentation(formed,p,full,t,temperature_surface,
        radius,heat_diffusivity,dt_seconds,gas_constant=gas_constant,gravity=gravity,
        density_temperature=adjusted,bottom_density_temperature=adjusted[-1])
    loss=(formed-jnp.maximum(q,0.))*jnp.diff(p,axis=0)/gravity
    return CloudResult(adjusted,settled,loss,deposit,radius,diagnostics)
