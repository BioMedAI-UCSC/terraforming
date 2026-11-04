"""Conservative size-resolved aggregation for isolated screening studies.

Binary collision convention: identical-particle events occur at K*n*n/2.
Each binary channel is advanced analytically, then product number is split
between adjacent fixed mass pivots. Both mass and product number are retained.
This sequential operator is first-order in time and requires refinement.

Products above the last pivot remain in a separate *suspended, unresolved*
reservoir. They are never called deposited or discarded. A study must bound
that reservoir and refine its grid before interpreting aggregate lifetimes.
No aggregate optical properties are supplied by this model.
"""
from typing import NamedTuple
import math
import numpy as np
from src.framework.gcm._dinosaur import jax,jnp
from .transport import K_B,particle_drag
from .microphysics import charge_factor


class SectionalResult(NamedTuple):
    number_m3: object
    unresolved_mass_kg_m3: object
    unresolved_number_m3: object
    valid: object


def validate_pivots(pivots_kg):
    pivots=np.asarray(pivots_kg,dtype=float)
    if pivots.ndim!=1 or len(pivots)<2 or not np.all(np.isfinite(pivots)) or np.any(pivots<=0) or np.any(np.diff(pivots)<=0):
        raise ValueError('mass pivots must be a finite positive increasing one-dimensional grid')
    if not np.isclose(pivots[1],2*pivots[0],rtol=1e-12,atol=0):
        raise ValueError('first two pivots must be monomers and dimers to preserve monomer counts')
    return pivots


def coagulator(pivots_kg):
    """Build a JIT-safe number-density operator with explicit overflow.

    Inputs are particles/m³, a symmetric binary kernel in m³/s, and seconds.
    Invalid numerical inputs are flagged; they are not silently clipped into
    a plausible population. Existing unresolved material is carried by callers.
    """
    pivots=jnp.asarray(validate_pivots(pivots_kg))
    size=len(pivots)
    pair_i,pair_j=(jnp.asarray(x) for x in np.triu_indices(size))

    @jax.jit
    def step(number_m3,kernel_m3_s,dt_s):
        number=jnp.asarray(number_m3)
        kernel=jnp.asarray(kernel_m3_s)
        if number.shape!=(size,) or kernel.shape!=(size,size):
            raise ValueError('population/kernel shapes must match the mass pivots')
        valid=(jnp.all(jnp.isfinite(number))&jnp.all(number>=0)&
            jnp.all(jnp.isfinite(kernel))&jnp.all(kernel>=0)&jnp.isfinite(dt_s)&(dt_s>=0)&
            jnp.all(jnp.abs(kernel-kernel.T)<=1e-12*jnp.maximum(jnp.abs(kernel),1e-300)))

        def channel(carry,pair):
            n,overflow_mass,overflow_number=carry
            i,j=pair;ni=n[i];nj=n[j];k=kernel[i,j]
            large=jnp.maximum(ni,nj);small=jnp.minimum(ni,nj)
            difference=large-small
            fraction=-jnp.expm1(-k*difference*dt_s)
            denominator=difference+small*fraction
            unequal=small*(large*fraction/jnp.maximum(denominator,1e-300))
            equal=small*(-jnp.expm1(-jnp.log1p(k*small*dt_s)))
            collisions=jnp.where(difference<=1e-10*jnp.maximum(large,1.),equal,unequal)
            # A self-collision consumes two particles of that species.
            collisions=jnp.where(i==j,equal/2,collisions)
            # Enforce the analytic stoichiometric bound against roundoff only;
            # neither encounter speeds nor the physical kernel are capped.
            collisions=jnp.clip(collisions,0.,jnp.where(i==j,ni/2,small))
            n=n.at[i].add(-collisions)
            n=n.at[j].add(-collisions)
            product_mass=pivots[i]+pivots[j]
            overflow=product_mass>pivots[-1]
            upper=jnp.clip(jnp.searchsorted(pivots,product_mass,side='right'),1,size-1)
            lower=upper-1
            high_weight=jnp.clip((product_mass-pivots[lower])/(pivots[upper]-pivots[lower]),0.,1.)
            resolved=jnp.where(overflow,0.,collisions)
            n=n.at[lower].add(resolved*(1-high_weight))
            n=n.at[upper].add(resolved*high_weight)
            return (n,overflow_mass+jnp.where(overflow,collisions*product_mass,0.),
                    overflow_number+jnp.where(overflow,collisions,0.)),None

        (result,overflow_mass,overflow_number),_=jax.lax.scan(channel,
            (number,jnp.asarray(0.,dtype=number.dtype),jnp.asarray(0.,dtype=number.dtype)),(pair_i,pair_j))
        valid=valid&jnp.all(jnp.isfinite(result))&jnp.all(result>=0)&jnp.isfinite(overflow_mass)&jnp.isfinite(overflow_number)
        return SectionalResult(result,overflow_mass,overflow_number,valid)
    return step


def collision_kernels(particle,pivots_kg,temperature_k,pressure_pa,gravity_m_s2,*,
                      charges_e=0.,sticking_efficiency=1.,fractal_dimension=3.,settling_efficiency=0.):
    """Brownian and differential-settling kernels with labelled screening inputs.

    Effective radius grows as monomer_count**(1/D). The same geometric scale
    enters the existing continuum/Epstein mobility; actual aggregate mass is
    retained in inertia and weight. This is a morphology sensitivity closure,
    not measured rod-aggregate drag or optics. Charge applies to thermal
    encounters only; charged differential-settling capture needs another law.
    """
    pivots=validate_pivots(pivots_kg)
    charges=np.broadcast_to(np.asarray(charges_e,dtype=float),pivots.shape)
    if not np.all(np.isfinite(charges)):
        raise ValueError('finite per-bin particle charges required')
    if not 100<=temperature_k<=350 or not 1<=pressure_pa<=2000 or not math.isfinite(gravity_m_s2) or gravity_m_s2<=0:
        raise ValueError('kernel inputs lie outside the declared Mars screening domain')
    if not 0<=sticking_efficiency<=1 or not 0<=settling_efficiency<=1 or not 1.5<=fractal_dimension<=3:
        raise ValueError('efficiencies must lie in [0,1] and morphology dimension in [1.5,3]')
    if settling_efficiency and np.any(charges!=0):
        raise ValueError('charged differential-settling capture requires a supplied validated force law')
    if not np.isclose(pivots[0],particle.mass_kg,rtol=1e-12,atol=0):
        raise ValueError('first mass pivot must match the source-backed monomer')
    mass=jnp.asarray(pivots)
    multiplicity=mass/particle.mass_kg
    radius=particle.volume_radius_m*multiplicity**(1/fractal_dimension)
    drag=particle_drag(particle,temperature_k,pressure_pa,multiplicity**(3/fractal_dimension))
    diffusion=K_B*temperature_k/drag
    distance=radius[:,None]+radius[None,:]
    diffusive=4*math.pi*distance*(diffusion[:,None]+diffusion[None,:])
    relative_speed=jnp.sqrt(8*K_B*temperature_k/math.pi*(1/mass[:,None]+1/mass[None,:]))
    ballistic=math.pi*distance**2*relative_speed
    neutral=1/(1/diffusive+1/ballistic)
    charge=jnp.asarray(charges)
    brownian=neutral*charge_factor(charge[:,None],charge[None,:],radius[:,None],radius[None,:],temperature_k)
    speed=mass*gravity_m_s2/drag
    settling=math.pi*distance**2*jnp.abs(speed[:,None]-speed[None,:])*settling_efficiency
    return brownian*sticking_efficiency,settling*sticking_efficiency
