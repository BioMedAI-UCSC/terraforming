"""Conservative hemispheric layer adding with thermal scattering/emission.

Diffuse two-stream closure: a=(1-omega*(1+g)/2)/mu,
b=omega*(1-g)/(2*mu). Thermal emission obeys Kirchhoff's law.
The conservative-scattering limit is evaluated analytically, not with an
artificial absorption floor. Layer adding avoids growing exponentials.
"""
from src.framework.gcm._dinosaur import jax,jnp


def layer_coefficients(tau,omega,asymmetry,mu=.5):
    a=(1-omega*(1+asymmetry)/2)/mu
    b=omega*(1-asymmetry)/(2*mu)
    k=jnp.sqrt(jnp.maximum(a*a-b*b,1e-24))
    e=jnp.exp(-k*tau)
    denominator=(k+a)+(k-a)*e*e
    r=b*(1-e*e)/jnp.maximum(denominator,1e-300)
    t=2*k*e/jnp.maximum(denominator,1e-300)
    conservative=k<1e-10
    return (jnp.where(conservative,b*tau/(1+a*tau),r),
            jnp.where(conservative,1/(1+a*tau),t))


def add_layers(a,b):
    """Top/bottom reflectivity and shared transmission, including asymmetry."""
    ra_top,ra_bottom,ta=a
    rb_top,rb_bottom,tb=b
    d=jnp.maximum(1-ra_bottom*rb_top,1e-15)
    return ra_top+ta*ta*rb_top/d,rb_bottom+tb*tb*ra_bottom/d,ta*tb/d


def symmetric_embed(background_r,background_t,particle_r,particle_t):
    """Symmetric sublayer composition retaining the exact zero-particle path.

    The half-background operator is the analytic square root of its adding
    operator. Aerosols are embedded in each layer without treating extinction
    as absorption, or changing the legacy gas/background-dust coefficients.
    """
    rh=background_r/(1+background_t)
    th=jnp.sqrt(jnp.maximum(background_t*(1-rh*rh),0.))
    top,bottom,t=add_layers(add_layers((rh,rh,th),(particle_r,particle_r,particle_t)),(rh,rh,th))
    return (top+bottom)/2,t


def solve_fluxes(reflect,transmit,blackbody,down_top,surface_emission,surface_reflectivity=0.):
    """Return (up,down) at interfaces; axis 0 is top-to-surface layer index."""
    emission=jnp.maximum(1-reflect-transmit,0.)*blackbody
    shape=reflect.shape[1:]
    lower_r=jnp.broadcast_to(jnp.asarray(surface_reflectivity),shape)
    lower_e=jnp.broadcast_to(jnp.asarray(surface_emission),shape)
    def add(carry,layer):
        below_r,below_e=carry
        r,t,e=layer
        denominator=jnp.maximum(1-r*below_r,1e-15)
        new_r=r+t*t*below_r/denominator
        new_e=e+t*(below_e+below_r*e)/denominator
        return (new_r,new_e),(below_r,below_e)
    (top_r,top_e),(saved_r,saved_e)=jax.lax.scan(
        add,(lower_r,lower_e),(reflect[::-1],transmit[::-1],emission[::-1]))
    def propagate(down,layer):
        r,t,e,below_r,below_e=layer
        next_down=(t*down+r*below_e+e)/jnp.maximum(1-r*below_r,1e-15)
        return next_down,(below_r*next_down+below_e,next_down)
    down_top=jnp.broadcast_to(jnp.asarray(down_top),shape)
    _,(up_tail,down_tail)=jax.lax.scan(
        propagate,down_top,(reflect,transmit,emission,saved_r[::-1],saved_e[::-1]))
    return (jnp.concatenate([(top_r*down_top+top_e)[None],up_tail]),
            jnp.concatenate([down_top[None],down_tail]))
