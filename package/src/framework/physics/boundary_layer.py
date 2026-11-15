"""JAX kernels from Ames v3.2.1 BLcode/pblmod_mgcm.F90.

NASA Ames-derived algorithms: Copyright 2023 United States Government as
represented by the Administrator of NASA. NASA Open Source Agreement 1.3;
see iclr-results/06-co2-physics/ames-reference/NOSA.pdf in this repository.

These reproduce the active legacy Mellor--Yamada closure. The column driver
uses the reference's implicit timestep and no-sublimation branch.
"""
from src.framework.gcm._dinosaur import jnp


def _sqrt_with_zero_subgradient(value):
    return jnp.where(value == 0.,0.,jnp.sqrt(jnp.where(value == 0.,1.,value)))


def layer_heights(temperature, pressure_half, pressure_full, gas_constant, gravity):
    """newpbl's pressure-weighted lower-column temperature and physical heights.

    Arrays have levels first. Only interior interfaces are returned, since
    Dinosaur's zero-pressure model top has no finite hydrostatic height.
    """
    temperature,pressure_half,pressure_full = map(jnp.asarray,
        (temperature,pressure_half,pressure_full))
    if pressure_full.shape != temperature.shape or pressure_half.shape != (
            temperature.shape[0]+1,*temperature.shape[1:]):
        raise ValueError("temperature and pressure grids do not match")
    dp = jnp.diff(pressure_half,axis=0)
    weighted = jnp.cumsum((temperature*dp)[::-1],axis=0)[::-1]
    mean = weighted/(pressure_half[-1]-pressure_half[:-1])
    scale_height = gas_constant*mean/gravity
    midpoint = scale_height*jnp.log(pressure_half[-1]/pressure_full)
    interface = scale_height[1:]*jnp.log(pressure_half[-1]/pressure_half[1:-1])
    return midpoint,interface


def surface_exchange(theta_air, theta_ground, u, v, height, roughness, gravity,
                     von_karman=.4):
    """Ames bndcond with its hard-coded legacy_stability=.true. branch.

    Return heat and momentum coefficients, friction velocity, temperature
    scale, resolved wind speed and bulk Richardson number, in that order.
    """
    theta_air,theta_ground,u,v,height,roughness = map(jnp.asarray,
        (theta_air,theta_ground,u,v,height,roughness))
    speed = _sqrt_with_zero_subgradient(u*u+v*v)
    dphi = theta_air-theta_ground
    rib = gravity*height/(theta_air*speed*speed+1.e-9)*dphi
    stable = rib >= 0.
    # Keep inactive square-root branches finite for reverse-mode derivatives.
    stable_ri = jnp.where(stable,rib,0.)
    unstable_ri = jnp.where(stable,0.,rib)
    stable_root = jnp.sqrt(1.+5.*stable_ri)
    frih = jnp.where(stable,1./(1.+15.*stable_ri/stable_root),
        jnp.sqrt(1.-64.*unstable_ri))
    frim = jnp.where(stable,1./(1.+10.*stable_ri/stable_root),
        jnp.sqrt(1.-16.*unstable_ri))
    neutral = von_karman/jnp.log(height/roughness)
    cdh = jnp.sqrt(frih)*neutral
    cdm = frim*neutral**2
    return cdh,cdm,jnp.sqrt(cdm)*speed,cdh*dphi,speed,rib


def eddy_coefficients(upper, lower, dxi_upper, dxi_lower, inverse_height_spacing,
                      height, previous_shear_squared, dt_seconds, gravity,
                      critical_richardson=.195, von_karman=.4):
    """Ames eddycoef on interior interfaces, with persistent shear history.

    ``upper`` and ``lower`` end in the (u, v, potential-temperature) axis.
    The other arrays are the matching interior-interface values from newpbl;
    inverse_height_spacing is its signed ``xinum``. Return momentum/heat
    diffusivity, diagnosed Richardson number and the updated shear history.
    """
    upper,lower = jnp.asarray(upper),jnp.asarray(lower)
    if upper.shape != lower.shape or upper.shape[-1] != 3:
        raise ValueError("adjacent PBL fields require a matching (u,v,theta) axis")
    dxi_upper,dxi_lower,inverse_height_spacing,height,previous_shear_squared = map(
        jnp.asarray,(dxi_upper,dxi_lower,inverse_height_spacing,height,previous_shear_squared))
    mixing_length = 150.*von_karman*height/(150.+von_karman*height)
    beta = (dxi_upper+dxi_lower)/(dxi_upper*upper[...,2]+dxi_lower*lower[...,2])
    gradient = (lower-upper)*inverse_height_spacing[...,None]
    shear = gradient[...,0]**2+gradient[...,1]**2
    memory = previous_shear_squared-(previous_shear_squared-shear)*dt_seconds/1.e4
    richardson = beta*gravity*gradient[...,2]/(memory+1.e-9)
    neutral = _sqrt_with_zero_subgradient(memory/.153)*mixing_length**2
    unstable_ri = jnp.minimum(richardson,0.)
    kh = neutral*.493*jnp.where(richardson <= 0.,
        (1.-15.*unstable_ri)**.5,1.-richardson/critical_richardson)
    km = neutral*.393*jnp.where(richardson <= 0.,
        (1.-15.*unstable_ri)**.25,1.-richardson/critical_richardson)
    return jnp.maximum(km,.001),jnp.maximum(kh,.001),richardson,memory


def column_step(temperature,u,v,tracers,pressure_half,pressure_full,surface_temperature,
                dt_seconds,gas_constant,cp,gravity,*,surface_co2_ice=0.,surface_water_ice=0.):
    """Ames boundary_layer/pbl_driver/newpbl for predicted gas fields.

    Levels come first and tracer species last. As in ames_pbl_interface,
    radiation is already included in the prediction, shear history resets,
    and the hard-coded internal water-sublimation switch is off. Only gas
    tracers belong here; ice/aerosol selection is the caller's responsibility.
    Return updated T/u/v/tracers plus old-state surface heat/stress diagnostics.
    """
    from src.framework.physics.vertical_solve import solve_tridiagonal
    temperature,u,v,tracers,pressure_half,pressure_full=map(jnp.asarray,
        (temperature,u,v,tracers,pressure_half,pressure_full))
    if temperature.shape[0]<2 or u.shape!=temperature.shape or v.shape!=temperature.shape:
        raise ValueError("PBL requires matching fields with at least two layers")
    if tracers.shape[:-1]!=temperature.shape:
        raise ValueError("PBL tracers require a trailing species axis")
    ps=pressure_half[-1]
    dp=jnp.diff(pressure_half,axis=0)
    dxi_mid=dp/ps
    dxi_interface=jnp.diff(pressure_full,axis=0)/ps
    # pbl_driver uses this fixed exponent, not constants_mod::kappa.
    exner=(pressure_full/ps)**.25684
    theta=temperature/exner
    fields=jnp.concatenate([u[...,None],v[...,None],theta[...,None],tracers],axis=-1)
    midpoint_z,interface_z=layer_heights(temperature,pressure_half,pressure_full,gas_constant,gravity)
    temperature_interface=.5*(temperature[:-1]+temperature[1:])
    rho_interface=pressure_half[1:-1]/(gas_constant*temperature_interface)
    rnum_interface=-ps/(rho_interface*gravity)
    xinum=1./dxi_interface/rnum_interface
    km,kh,richardson,shear=eddy_coefficients(fields[:-1,...,:3],fields[1:,...,:3],
        dxi_mid[:-1],dxi_mid[1:],xinum,interface_z,jnp.zeros_like(xinum),dt_seconds,gravity)
    roughness=jnp.where((jnp.asarray(surface_co2_ice)>0.)|(jnp.asarray(surface_water_ice)>100.),1.e-4,.01)
    cdh,cdm,ustar,thstar,speed,rib=surface_exchange(theta[-1],surface_temperature,
        u[-1],v[-1],midpoint_z[-1],roughness,gravity)
    diffusivity=jnp.concatenate([jnp.stack([km,km,kh],axis=-1),
        jnp.broadcast_to(kh[...,None],kh.shape+(tracers.shape[-1],))],axis=-1)
    pc=diffusivity/rnum_interface[...,None]**2
    upper_coeff=-dt_seconds*pc/(dxi_mid[:-1]*dxi_interface)[...,None]
    lower_coeff=-dt_seconds*pc/(dxi_mid[1:]*dxi_interface)[...,None]
    zeros=jnp.zeros_like(fields[:1])
    lower=jnp.concatenate([zeros,lower_coeff],axis=0)
    upper=jnp.concatenate([upper_coeff,zeros],axis=0)
    diagonal=1.-lower-upper
    rho_low=pressure_full[-1]/(gas_constant*temperature[-1])
    rnum_low=-ps/(rho_low*gravity)
    surface_rate=-ustar/rnum_low*dt_seconds/dxi_mid[-1]
    bottom_coeff=jnp.concatenate([jnp.stack([surface_rate*jnp.sqrt(cdm),
        surface_rate*jnp.sqrt(cdm),surface_rate*cdh],axis=-1),
        jnp.zeros(surface_rate.shape+(tracers.shape[-1],))],axis=-1)
    diagonal=diagonal.at[-1].add(bottom_coeff)
    rhs=fields.at[-1,...,2].add(surface_rate*cdh*surface_temperature)
    result=solve_tridiagonal(lower,diagonal,upper,rhs)
    rho_surface=ps/(gas_constant*.5*(surface_temperature+temperature[-1]))
    heat_flux=-rho_surface*cp*ustar*thstar
    stress_factor=rho_surface*ustar**2/jnp.where(speed>0.,speed,1.)
    stress_x=stress_factor*u[-1]
    stress_y=stress_factor*v[-1]
    diagnostics=dict(heat_flux_w_m2=heat_flux,stress_x_pa=stress_x,stress_y_pa=stress_y,
        heat_flux_surface_derivative_w_m2_k=rho_surface*cp*cdh*ustar,
        heat_flux_atmosphere_derivative_w_m2_k=-rho_surface*cp*cdh*ustar/exner[-1],
        heat_diffusivity_m2_s=kh,momentum_diffusivity_m2_s=km,
        surface_heat_coefficient=cdh,surface_momentum_coefficient=cdm,
        friction_velocity_ms=ustar,temperature_scale_k=thstar,
        richardson=richardson,shear_squared=shear,bulk_richardson=rib,wind_speed_ms=speed)
    return result[...,2]*exner,result[...,0],result[...,1],result[...,3:],diagnostics
