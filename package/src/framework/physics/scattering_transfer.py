"""Differentiable Ames v3.2.1 rotated two-stream radiation solvers.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
Vertical levels are axis zero; trailing dimensions are independent channels.
"""
from src.framework.gcm._dinosaur import jax,jnp
from src.framework.physics.vertical_solve import solve_tridiagonal


def dsolver(gamma,cp,cm,cpm1,cmm1,e1,e2,e3,e4,btop,bsurf,reflectivity):
    """Ames mixed-coefficient matrix and bottom-up tridiagonal solve."""
    gamma,cp,cm,cpm1,cmm1,e1,e2,e3,e4=map(jnp.asarray,
        (gamma,cp,cm,cpm1,cmm1,e1,e2,e3,e4))
    n=gamma.shape[0]
    shape=(2*n,)+gamma.shape[1:]
    lower,diagonal,upper,rhs=(jnp.zeros(shape,dtype=gamma.dtype) for _ in range(4))
    diagonal=diagonal.at[0].set(gamma[0]+1.)
    upper=upper.at[0].set(gamma[0]-1.)
    rhs=rhs.at[0].set(btop-cmm1[0])
    lower=lower.at[1:-1:2].set((e1[:-1]+e3[:-1])*(gamma[1:]-1.))
    diagonal=diagonal.at[1:-1:2].set((e2[:-1]+e4[:-1])*(gamma[1:]-1.))
    upper=upper.at[1:-1:2].set(2.*(1.-gamma[1:]**2))
    rhs=rhs.at[1:-1:2].set((gamma[1:]-1.)*(cpm1[1:]-cp[:-1])+
        (1.-gamma[1:])*(cm[:-1]-cmm1[1:]))
    lower=lower.at[2::2].set(2.*(1.-gamma[:-1]**2))
    diagonal=diagonal.at[2::2].set((e1[:-1]-e3[:-1])*(1.+gamma[1:]))
    upper=upper.at[2::2].set((e1[:-1]+e3[:-1])*(gamma[1:]-1.))
    rhs=rhs.at[2::2].set(e3[:-1]*(cpm1[1:]-cp[:-1])+e1[:-1]*(cm[:-1]-cmm1[1:]))
    lower=lower.at[-1].set(e1[-1]-reflectivity*e3[-1])
    diagonal=diagonal.at[-1].set(e2[-1]-reflectivity*e4[-1])
    rhs=rhs.at[-1].set(bsurf-cp[-1]+reflectivity*cm[-1])
    # dtridgl eliminates from the bottom. Reverse the common Thomas solve.
    mixed=solve_tridiagonal(upper[::-1],diagonal[::-1],lower[::-1],rhs[::-1],
        reciprocal_pivot=True)[::-1]
    positive=mixed[::2]+mixed[1::2]
    negative=mixed[::2]-mixed[1::2]
    negative=jnp.where(jnp.abs(negative/(mixed[::2]+1.e-20))<1.e-30,0.,negative)
    return positive,negative


def _exponent_coefficients(gamma,rate,optical_depth,taumax):
    ep=jnp.exp(jnp.minimum(taumax,rate*optical_depth))
    em=1./ep
    return ep+gamma*em,ep-gamma*em,gamma*ep+em,gamma*ep-em


def _infrared_layer(depth,w0,g,planck_top,planck_bottom,taumax,*,capped_exponent=None):
    """Boundary response of the original hemispheric equations, without B'/tau cancellation."""
    alpha=jnp.sqrt((1.-w0)/(1.-w0*g))
    gamma=(1.-alpha)/(1.+alpha)
    x=alpha*(1.-w0*g)/.5*depth
    capped=jnp.minimum(x,taumax) if capped_exponent is None else capped_exponent
    transmission=jnp.exp(-capped)
    denominator=1.-gamma*gamma*transmission**2
    reflection=gamma*(-jnp.expm1(-2.*capped))/denominator
    through=(1.-gamma*gamma)*transmission/denominator
    absorption=(1.-gamma)*(-jnp.expm1(-capped))/(1.+gamma*transmission)
    # These remainders start at x**2. Direct subtraction loses thin-layer AD.
    c=capped
    f=jnp.where(c<1e-3,c*c*(.5-c/3.+c*c/8.-c**3/30.+c**4/144.),
        -jnp.expm1(-c)-c*transmission)+(c-x)*transmission
    h=jnp.where(c<1e-3,c*c*(-.5+c/6.-c*c/24.+c**3/120.-c**4/720.),
        -jnp.expm1(-c)-c)+(c-x)
    coefficient=(1.-gamma)*(f+gamma*transmission*h)/denominator/jnp.where(x!=0.,x,1.)
    slope_term=(planck_bottom-planck_top)*coefficient
    return reflection,through,planck_top*absorption+slope_term,planck_bottom*absorption-slope_term


def infrared_fluxes(optical_depth,scattering_albedo,asymmetry,planck_boundaries,
                    evaluation_depth,reflectivity,btop,bsurf,*,taumax=35.):
    """Original IR hemispheric equations with stable layer adding and linear sources.

    Integrate the same linear Planck profile and retain the source exponent cap.
    Boundary source terms use exponential remainders instead of subtracting
    terms proportional to 1/tau. This keeps thin-layer reverse derivatives usable.
    """
    dtau,w0,g,planck,offset=map(jnp.asarray,
        (optical_depth,scattering_albedo,asymmetry,planck_boundaries,evaluation_depth))
    if planck.shape!=(dtau.shape[0]+1,*dtau.shape[1:]) or offset.shape!=dtau.shape:
        raise ValueError('Planck levels and evaluation depths must match the radiation grid')
    reflection,through,source_up,source_down=_infrared_layer(dtau,w0,g,planck[:-1],planck[1:],taumax)
    shape=dtau[0].shape
    def add_layer(below,row):
        a,b=below;r,t,su,sd=row
        inverse=1./(1.-a*r)
        combined=(r+t*a*t*inverse,su+t*(a*sd+b)*inverse)
        return combined,(combined,below)
    _,((a,b),(next_a,next_b))=jax.lax.scan(add_layer,
        (jnp.broadcast_to(jnp.asarray(reflectivity),shape),jnp.broadcast_to(jnp.asarray(bsurf),shape)),
        tuple(x[::-1] for x in (reflection,through,source_up,source_down)))
    a,b,next_a,next_b=(x[::-1] for x in (a,b,next_a,next_b))
    top_down=jnp.broadcast_to(jnp.asarray(btop),shape)
    def propagate(down_top,row):
        r,t,sd,an,bn=row
        up_bottom=(an*t*down_top+an*sd+bn)/(1.-an*r)
        down_bottom=t*down_top+r*up_bottom+sd
        return down_bottom,(down_top,up_bottom)
    _,(down_top,up_bottom)=jax.lax.scan(propagate,top_down,
        (reflection,through,source_down,next_a,next_b))
    middle=planck[:-1]+(planck[1:]-planck[:-1])*offset/dtau
    ru,tu,_,sdu=_infrared_layer(offset,w0,g,planck[:-1],middle,taumax)
    rate=jnp.sqrt((1.-w0)*(1.-w0*g))/.5
    # Sampling must share the full layer's cap; independently capping both pieces changes gfluxi.
    lower_exponent=jnp.minimum(rate*dtau,taumax)-jnp.minimum(rate*offset,taumax)
    rl,tl,sul,_=_infrared_layer(dtau-offset,w0,g,middle,planck[1:],taumax,
        capped_exponent=lower_exponent)
    up=(rl*(tu*down_top+sdu)+tl*up_bottom+sul)/(1.-rl*ru)
    down=tu*down_top+ru*up+sdu
    return up*jnp.pi,down*jnp.pi,(a[0]*top_down+b[0]-top_down)*jnp.pi


def visible_fluxes(optical_depth,scattering_albedo,asymmetry,cumulative_sublevels,
                   mu0,solar_intensity,reflectivity,btop,bsurf,*,taumax=35.,maxexp=35.):
    """Ames gfluxv, including its delta-Eddington subgrid treatment.

    cumulative_sublevels has 2*layers+1 entries, including Ames's special
    boundary levels. mu0 must be positive; the outer driver handles night.
    Return up/down layer samples, top up/down, surface diffuse down and the
    delta-scaled column optical depth. bsurf is reflected direct-beam flux.
    """
    dtau,w,g,cumulative=map(jnp.asarray,
        (optical_depth,scattering_albedo,asymmetry,cumulative_sublevels))
    if cumulative.shape!=(2*dtau.shape[0]+1,*dtau.shape[1:]):
        raise ValueError("visible radiation requires the Ames 2*layers+1 subgrid")
    factor=1.-w*g*g
    w0=w*(1.-g*g)/factor
    cosbar=g/(1.+g)
    delta=dtau*factor
    tau=jnp.concatenate([jnp.zeros_like(delta[:1]),jnp.cumsum(delta,axis=0)],axis=0)
    scaled=jnp.zeros_like(cumulative)
    scaled=scaled.at[1].set(cumulative[1]*factor[0])
    scaled=scaled.at[2].set(scaled[1]+(cumulative[2]-cumulative[1])*factor[0])
    scaled=scaled.at[3::2].set(tau[1:-1])
    scaled=scaled.at[4::2].set(tau[1:-1]+(cumulative[4::2]-cumulative[3::2])*factor[:-1])
    scaled=scaled.at[-1].set(tau[-1])
    g1=jnp.sqrt(3.)*.5*(2.-w0*(1.+cosbar))
    g2=jnp.sqrt(3.)*.5*w0*(1.-cosbar)
    g3=.5*(1.-jnp.sqrt(3.)*cosbar*mu0)
    rate=jnp.sqrt(g1*g1-g2*g2)
    # Algebraically identical to (g1-rate)/g2, with the finite w0=0 limit.
    gamma=g2/(g1+rate)
    denominator=rate**2-1./mu0**2
    denominator=jnp.where(denominator==0.,1.e-10,denominator)
    am=solar_intensity*w0*((1.-g3)*(g1+1./mu0)+g2*g3)/denominator
    ap=solar_intensity*w0*(g3*(g1-1./mu0)+g2*(1.-g3))/denominator
    at_top=jnp.exp(-jnp.minimum(tau[:-1]/mu0,maxexp))
    at_bottom=jnp.exp(-jnp.minimum(tau[1:]/mu0,maxexp))
    cp,cm=ap*at_bottom,am*at_bottom
    cpm1,cmm1=ap*at_top,am*at_top
    e1,e2,e3,e4=_exponent_coefficients(gamma,rate,delta,taumax)
    positive,negative=dsolver(gamma,cp,cm,cpm1,cmm1,e1,e2,e3,e4,btop,bsurf,reflectivity)
    offset=scaled[2::2]-scaled[1::2]
    absolute=scaled[2::2].at[-1].set(jnp.minimum(scaled[-1],taumax))
    ep=jnp.exp(jnp.minimum(taumax,rate*offset))
    em=1./ep
    direct_factor=jnp.exp(-jnp.minimum(absolute/mu0,maxexp))
    up=positive*ep+gamma*negative*em+ap*direct_factor
    diffuse=positive*ep*gamma+negative*em+am*direct_factor
    down=diffuse+mu0*solar_intensity*direct_factor
    top_up=positive[0]+gamma[0]*negative[0]+ap[0]
    top_down=positive[0]*gamma[0]+negative[0]+am[0]+mu0*solar_intensity
    return up,down,top_up,top_down,diffuse[-1],scaled[-1]
