"""Differentiable Ames v3.2.1 rotated two-stream radiation solvers.

NASA-derived algorithms: Copyright 2023 United States Government as represented
by NASA. NASA Open Source Agreement 1.3; see the repository's Ames reference.
Vertical levels are axis zero; trailing dimensions are independent channels.
"""
from src.framework.gcm._dinosaur import jnp
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
    mixed=solve_tridiagonal(upper[::-1],diagonal[::-1],lower[::-1],rhs[::-1])[::-1]
    positive=mixed[::2]+mixed[1::2]
    negative=mixed[::2]-mixed[1::2]
    negative=jnp.where(jnp.abs(negative/(mixed[::2]+1.e-20))<1.e-30,0.,negative)
    return positive,negative


def _exponent_coefficients(gamma,rate,optical_depth,taumax):
    ep=jnp.exp(jnp.minimum(taumax,rate*optical_depth))
    em=1./ep
    return ep+gamma*em,ep-gamma*em,gamma*ep+em,gamma*ep-em


def infrared_fluxes(optical_depth,scattering_albedo,asymmetry,planck_boundaries,
                    evaluation_depth,reflectivity,btop,bsurf,*,taumax=35.):
    """Ames gfluxi with supplied absolute band Planck intensities.

    ``planck_boundaries`` has one more level than optical_depth. The caller
    supplies the original radiative subgrid, including its special lower
    layer. evaluation_depth is measured from each radiation layer's top.
    Return upward/downward flux at those depths and the net upward TOA flux.
    Intensities are converted to hemispheric flux by pi, as in gfluxi.
    """
    dtau,w0,g,planck,offset=map(jnp.asarray,
        (optical_depth,scattering_albedo,asymmetry,planck_boundaries,evaluation_depth))
    if planck.shape!=(dtau.shape[0]+1,*dtau.shape[1:]) or offset.shape!=dtau.shape:
        raise ValueError("Planck levels and evaluation depths must match the radiation grid")
    alpha=jnp.sqrt((1.-w0)/(1.-w0*g))
    rate=alpha*(1.-w0*g)/.5
    slope=(planck[1:]-planck[:-1])/dtau
    base=planck[:-1]
    gamma=(1.-alpha)/(1.+alpha)
    term=.5/(1.-w0*g)
    cp=base+slope*dtau+slope*term
    cm=base+slope*dtau-slope*term
    cpm1=base+slope*term
    cmm1=base-slope*term
    e1,e2,e3,e4=_exponent_coefficients(gamma,rate,dtau,taumax)
    positive,negative=dsolver(gamma,cp,cm,cpm1,cmm1,e1,e2,e3,e4,btop,bsurf,reflectivity)
    ep=jnp.exp(jnp.minimum(taumax,rate*offset))
    em=1./ep
    up=(positive*ep+gamma*negative*em+base+slope*offset+slope*term)*jnp.pi
    down=(positive*ep*gamma+negative*em+base+slope*offset-slope*term)*jnp.pi
    top_up=positive[0]+gamma[0]*negative[0]+cpm1[0]
    top_down=positive[0]*gamma[0]+negative[0]+cmm1[0]
    return up,down,(top_up-top_down)*jnp.pi


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
