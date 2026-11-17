"""Palmer terrain gravity-wave drag from the released Mars GCM.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's reference license.
"""
from typing import NamedTuple
from src.framework.gcm._dinosaur import jax,jnp


class TerrainDragResult(NamedTuple):
    u_tendency: object
    v_tendency: object
    surface_stress_pa: object
    unresolved_stress_pa: object
    interface_stress_pa: object


def infrared_damping(pressure_pa):
    """Eckermann 10 km vertical-wavelength fit; rates per Mars sol."""
    x=jnp.log(jnp.asarray(pressure_pa)/100.)
    coefficients=(.74464,.088451,.0087765,-.059745,-.014485,.0032386,
        .0020174,.00014703,-.000020184,-.0000019778)
    return sum(c*x**i for i,c in enumerate(coefficients)),.625


def tendencies(u,v,temperature,pressure_full,pressure_half,height_full,height_half,
               terrain_std_m,*,gas_constant=189.02,cp=735.94,gravity=3.72,
               coefficient=5e-6,ir_damping=True):
    """Original palmer_drag, top-first layers and SI arrays.

    Preserve the original N/shear² Richardson number, bottom four-layer
    suppression, early exits and missing impulse on zero saturation stress.
    Report its force imbalance instead of silently altering the equations.
    Calm wind and zero shear use continuous limits of the finite equations.
    Heights must be supplied by the core, including its finite upper boundary.
    """
    u,v,t,p,ph,z,zh=map(jnp.asarray,(u,v,temperature,pressure_full,pressure_half,height_full,height_half))
    if u.shape!=v.shape or u.shape!=t.shape or u.shape!=p.shape or u.shape!=z.shape:
        raise ValueError('terrain drag full-level fields must have identical shapes')
    n=u.shape[0]
    if n<2 or ph.shape!=(n+1,*u.shape[1:]) or zh.shape!=ph.shape:
        raise ValueError('terrain drag requires at least two layers and matching interfaces')
    speed2=u[-1]**2+v[-1]**2
    moving=speed2>0.
    speed=jnp.sqrt(jnp.where(moving,speed2,1.))
    ex=jnp.where(moving,u[-1]/speed,0.);ey=jnp.where(moving,v[-1]/speed,0.)
    parallel=u*ex+v*ey
    # The original top projection is the surface speed, even in sheared flow.
    parallel=parallel.at[0].set(jnp.where(moving,speed,0.))
    theta=t*(p[-1]/p)**(gas_constant/cp)
    rho=.5*(p[1:]+p[:-1])/(gas_constant*.5*(t[1:]+t[:-1]))
    stress0=coefficient*rho[-1]*.01*jnp.where(moving,speed,0.)*jnp.minimum(jnp.asarray(terrain_std_m)**2,1.e6)
    dp=jnp.diff(ph,axis=0)
    stress=jnp.zeros_like(ph).at[-1].set(stress0)
    acceleration=jnp.zeros_like(u)
    active=jnp.ones_like(stress0,dtype=bool)

    def propagate(carry,k):
        stress,acceleration,active=carry
        lower=stress[k+1]
        avg=.5*(parallel[k]+parallel[k-1])
        dz=z[k-1]-z[k]
        nsq=gravity/(.5*(theta[k]+theta[k-1]))*(theta[k-1]-theta[k])/dz
        buoyancy=jnp.where(nsq>0.,jnp.sqrt(jnp.where(nsq>0.,nsq,1.)),1.e-8)
        shear2=((u[k-1]-u[k])/dz)**2+((v[k-1]-v[k])/dz)**2
        # r=1/sqrt(Ri) keeps the zero-shear limit finite in both AD modes.
        r=jnp.sqrt(jnp.where(shear2>0.,shear2/buoyancy,1.))
        r=jnp.where(shear2>0.,r,0.)
        damper,wave_speed=infrared_damping(p[k]) if ir_damping else (0.,1.)
        attenuated=lower*jnp.exp(-(zh[k-1]-zh[k])/wave_speed*damper/88775.5)
        safe_avg=jnp.where(avg!=0.,avg,1.)
        displacement2=jnp.abs(attenuated/(rho[k-1]*coefficient*buoyancy*safe_avg))
        displacement=jnp.where(displacement2>0.,
            jnp.sqrt(jnp.where(displacement2>0.,displacement2,1.)),0.)
        a=buoyancy*displacement/jnp.abs(safe_avg)
        denom=(r+a)**2
        rimin=jnp.where(denom>0.,(1.-a)/jnp.where(denom>0.,denom,1.),jnp.inf)
        quiet=rimin>.25
        x=2.+jnp.minimum(r,2.)
        eps=jnp.maximum(2.*jnp.sqrt(x)-x,0.)
        saturated=eps**2*coefficient*rho[k-1]*avg**3/buoyancy
        upper=jnp.where(quiet,attenuated,saturated)
        drop=jnp.where(quiet,lower*(1.-jnp.exp(-30000.*damper/88775.5)),
            jnp.where(((saturated>=0.)&(lower>0.)&(saturated<lower))|
                      ((saturated<=0.)&(lower<0.)&(saturated>lower)),lower-saturated,0.))
        critical=(parallel[k]*lower<0.)|(avg*lower<0.)
        stopped=(~quiet)&(saturated==0.)
        suppressed=k>=n-4
        upper=jnp.where(suppressed,lower,jnp.where(critical,0.,upper))
        accel=jnp.where(suppressed|((~critical)&stopped),0.,
            -jnp.where(critical,lower,drop)*gravity/dp[k])
        stress=stress.at[k].set(jnp.where(active,upper,0.))
        acceleration=acceleration.at[k].set(jnp.where(active,accel,0.))
        active=active&(~((~suppressed)&(critical|stopped)))
        return (stress,acceleration,active),None

    (stress,acceleration,active),_=jax.lax.scan(propagate,(stress,acceleration,active),
        jnp.arange(n-1,0,-1))
    top=jnp.where(active&(jnp.abs(stress[1])>1.e-9),-stress[1]*gravity/dp[0],0.)
    acceleration=acceleration.at[0].set(top)
    residual=stress0+jnp.sum(acceleration*dp/gravity,axis=0)
    return TerrainDragResult(acceleration*ex,acceleration*ey,stress0,residual,stress)


def hydrostatic_heights(temperature,pressure_half,pressure_full,*,gas_constant=189.02,
                       gravity=3.72,top_pressure_pa):
    """Reference mars_physics_update hydrostatic heights above the surface.

    A zero-pressure Dinosaur top has infinite height. The terrain scheme needs
    a finite upper height; its diagnostic pressure is an explicit input and
    does not add atmospheric mass or change the core pressure grid.
    """
    t,ph,p=map(jnp.asarray,(temperature,pressure_half,pressure_full))
    top=jnp.where(ph[0]>0.,ph[0],top_pressure_pa)
    effective=ph.at[0].set(top)
    valid=(top>0.)&jnp.all((p>effective[:-1])&(p<effective[1:]),axis=0)
    safe=jnp.where(valid[None],effective,1.)
    dz=gas_constant*t/gravity*jnp.log(safe[1:]/safe[:-1])
    zh=jnp.concatenate([jnp.cumsum(dz[::-1],axis=0)[::-1],jnp.zeros_like(dz[:1])],axis=0)
    z=zh[1:]+gas_constant*t/gravity*(1.-safe[:-1]/jnp.where(valid[None],p,1.))
    return jnp.where(valid[None],z,jnp.nan),jnp.where(valid[None],zh,jnp.nan)
