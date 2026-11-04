"""Conservative moment/screening microphysics with explicit uncertainty inputs.

Charge correction follows roadmap Appendix A: W=(exp(kappa)-1)/kappa.
K here is the binary collision kernel; identical-particle number loss is
dN/dt=-K*N^2/2. This convention makes the factor of two explicit.
"""
from __future__ import annotations

import math
from src.framework.gcm._dinosaur import jnp
from .transport import K_B,particle_drag

E_CHARGE=1.602176634e-19
EPSILON_0=8.8541878128e-12


def charge_factor(charge1_e,charge2_e,radius1_m,radius2_m,temperature_k):
    k=charge1_e*charge2_e*E_CHARGE**2/(4*math.pi*EPSILON_0*1.0005*(radius1_m+radius2_m)*K_B*temperature_k)
    pos=jnp.maximum(k,1e-6)
    neg=jnp.minimum(k,-1e-6)
    repulsive=pos*jnp.exp(-pos)/(-jnp.expm1(-pos))
    attractive=(-neg)/(-jnp.expm1(neg))
    return jnp.where(jnp.abs(k)<1e-4,1-k/2+k*k/12,jnp.where(k>0,repulsive,attractive))


def brownian_kernel(particle,temperature_k,pressure_pa,multiplicity=1.,charge_e=0.):
    radius=particle.volume_radius_m*multiplicity**(1/3)
    drag=particle_drag(particle,temperature_k,pressure_pa,multiplicity)
    diffusion=K_B*temperature_k/drag
    diffusive=4*math.pi*(2*diffusion)*(2*radius)
    particle_mass=particle.mass_kg*multiplicity
    relative_speed=jnp.sqrt(16*K_B*temperature_k/(math.pi*particle_mass))
    ballistic=math.pi*(2*radius)**2*relative_speed
    # Harmonic interpolation bounds the low-pressure diffusion expression by
    # the collisionless encounter rate. Shape-dependent sticking is separate.
    neutral=1/(1/diffusive+1/ballistic)
    return neutral*charge_factor(charge_e,charge_e,radius,radius,temperature_k)


def settling_collision_kernel(radius1,radius2,velocity1,velocity2,efficiency):
    return math.pi*(radius1+radius2)**2*jnp.abs(velocity1-velocity2)*efficiency


def constant_kernel_coagulation(mass,number,kernel_m3_s,thickness_m,dt_s):
    """Exact moment solution at fixed binary kernel; material is unchanged."""
    new_number=number/(1+.5*kernel_m3_s*(number/thickness_m)*dt_s)
    return mass,new_number,number-new_number


def collision_scavenging(mass,number,collector_number_m3,kernel_m3_s,dt_s):
    """Transfer engineered material to an explicit collected reservoir.

    Collector population and collision efficiency are supplied, not inferred
    from an ambiguous background dust optical depth.
    """
    fraction=-jnp.expm1(-collector_number_m3*kernel_m3_s*dt_s)
    return mass*(1-fraction),number*(1-fraction),mass*fraction,number*fraction


def re_loft(deposited_mass,deposited_number,friction_velocity,threshold,rate_s,dt_s):
    """Bounded source from a finite surface reservoir; threshold/rate are inputs."""
    active=friction_velocity>threshold
    fraction=jnp.where(active,-jnp.expm1(-rate_s*dt_s),0.)
    lifted_mass=deposited_mass*fraction
    lifted_number=deposited_number*fraction
    return deposited_mass-lifted_mass,deposited_number-lifted_number,lifted_mass,lifted_number
