"""Independent offline angular reference for homogeneous thermal slabs.

Solves mu*dI/dtau = -I + omega/2*integral(P*I dmu) + (1-omega)*B/pi
with SciPy's boundary-value solver, not the production layer-adding algorithm.
Double-Gauss angular quadrature integrates each hemisphere separately. The
Henyey--Greenstein phase moments g**l are an explicit hypothetical control:
the source rod tables supply g, not the complete measured phase function.

This checks angular-closure error under that stated phase model. It does not
by itself validate a rod phase function, band averaging or a Mars climatology.
"""
import math
import numpy as np
from importlib.metadata import version


def homogeneous_slab(*,optical_depth,single_scattering_albedo,asymmetry,
                     blackbody_w_m2,down_top_w_m2,surface_emission_w_m2,
                     surface_reflectivity=0.,angles=32,samples=9,tolerance=1e-8):
    """Diffuse boundaries and Lambertian surface; return fluxes in W/m²."""
    from scipy.integrate import solve_bvp
    tau=float(optical_depth);omega=float(single_scattering_albedo);g=float(asymmetry)
    source=float(blackbody_w_m2);incoming=float(down_top_w_m2);emission=float(surface_emission_w_m2)
    reflect=float(surface_reflectivity)
    if not all(math.isfinite(x) for x in (tau,omega,g,source,incoming,emission,reflect,tolerance)):
        raise ValueError('finite optical and boundary inputs required')
    if tau<0 or not 0<=omega<=1 or not -1<g<=1 or min(source,incoming,emission)<0 or not 0<=reflect<=1 or tolerance<=0:
        raise ValueError('inputs outside the slab reference domain')
    if not isinstance(angles,int) or angles<4 or angles>128 or angles%2:
        raise ValueError('angular stream count must be even and in 4..128')
    if not isinstance(samples,int) or samples<2: raise ValueError('at least two flux interfaces required')
    nodes,weights=np.polynomial.legendre.leggauss(angles//2)
    positive=(nodes+1)/2;half_weights=weights/2
    mu=np.r_[-positive[::-1],positive]
    weight=np.r_[half_weights[::-1],half_weights]
    upward=mu<0;downward=mu>0
    x=np.linspace(0.,tau,samples)
    phase_description='Henyey-Greenstein Legendre moments; source rod angular phase unverified'
    if tau==0 or omega==0 or g==1:
        absorption=1-omega if g==1 else 1.
        down=source/math.pi+(incoming-source)/math.pi*np.exp(-absorption*x[None,:]/positive[:,None])
        bottom_down=2*math.pi*np.sum(half_weights*positive*down[:,-1])
        surface=(emission+reflect*bottom_down)/math.pi
        up=source/math.pi+(surface-source/math.pi)*np.exp(-absorption*(tau-x)[None,:]/positive[:,None])
        down_flux=2*math.pi*np.sum((half_weights*positive)[:,None]*down,axis=0)
        up_flux=2*math.pi*np.sum((half_weights*positive)[:,None]*up,axis=0)
        solver_nodes=0
        if g==1: phase_description='exact forward-delta scattering limit'
    else:
        # Full-interval polynomial moments remain exact: each hemisphere's
        # Gauss rule integrates through degree angles-1. Symmetry then gives
        # both energy conservation and isotropic LTE without renormalization.
        polynomials=np.polynomial.legendre.legvander(mu,angles-1)
        coefficients=(2*np.arange(angles)+1)*g**np.arange(angles)
        phase=(polynomials*coefficients)@polynomials.T
        if np.min(phase)<-1e-12:
            raise ValueError('phase truncation is not positive; increase angular resolution')
        if not np.allclose(phase@weight,2.,rtol=1e-10,atol=1e-10):
            raise ValueError('angular phase quadrature does not conserve energy')
        operator=(-np.eye(angles)+omega*.5*phase*weight[None,:])/mu[:,None]
        thermal=(1-omega)*source/math.pi/mu
        def derivative(_,intensity): return operator@intensity+thermal[:,None]
        def jacobian(depth,_): return np.broadcast_to(operator[:,:,None],(angles,angles,len(depth)))
        def boundary(top,bottom):
            flux_down=2*math.pi*np.sum(weight[downward]*mu[downward]*bottom[downward])
            residual=np.empty(angles)
            residual[downward]=top[downward]-incoming/math.pi
            residual[upward]=bottom[upward]-(emission+reflect*flux_down)/math.pi
            return residual
        top_jac=np.zeros((angles,angles));bottom_jac=np.zeros((angles,angles))
        pos=np.flatnonzero(downward);neg=np.flatnonzero(upward)
        top_jac[pos,pos]=1.;bottom_jac[neg,neg]=1.
        bottom_jac[np.ix_(neg,pos)]=-2*reflect*(weight[pos]*mu[pos])[None,:]
        mesh=np.linspace(0.,tau,max(16,samples))
        guess=np.full((angles,len(mesh)),source/math.pi)
        solution=solve_bvp(derivative,boundary,mesh,guess,fun_jac=jacobian,
            bc_jac=lambda _top,_bottom:(top_jac,bottom_jac),tol=tolerance,max_nodes=20000)
        if not solution.success: raise RuntimeError(f'angular reference did not converge: {solution.message}')
        intensity=solution.sol(x)
        if np.min(intensity)<-1e-8*max(source,incoming,emission,1.):
            raise RuntimeError('angular reference produced negative radiance')
        down_flux=2*math.pi*np.sum((weight[downward]*mu[downward])[:,None]*intensity[downward],axis=0)
        up_flux=-2*math.pi*np.sum((weight[upward]*mu[upward])[:,None]*intensity[upward],axis=0)
        solver_nodes=len(solution.x)
    net=up_flux-down_flux
    if not np.all(np.isfinite(up_flux)) or not np.all(np.isfinite(down_flux)):
        raise RuntimeError('angular reference produced nonfinite flux')
    return dict(optical_depth=x,up_w_m2=up_flux,down_w_m2=down_flux,
        layer_convergence_w_m2=np.diff(net),surface_net_w_m2=float(-net[-1]),
        toa_net_down_w_m2=float(-net[0]),angles=angles,solver_nodes=solver_nodes,
        phase_assumption=phase_description,tolerance=tolerance,
        quadrature='double Gauss-Legendre; source B is hemispheric blackbody flux')


def compare_two_stream(*,max_flux_error_fraction,max_layer_heating_error_fraction,**case):
    """Evaluate a caller-frozen error gate; no default acceptance threshold.

    Normalize absolute flux and layer-integrated heating errors by the largest
    imposed source/boundary flux. Interface count and angular resolution are
    recorded so tolerances cannot silently change with the comparison grid.
    A pass applies only to this homogeneous HG control, not a source climate.
    """
    for value in (max_flux_error_fraction,max_layer_heating_error_fraction):
        if not math.isfinite(value) or value<0: raise ValueError('finite nonnegative frozen error limits required')
    from src.framework.physics.twostream import layer_coefficients,solve_fluxes
    from src.framework.gcm._dinosaur import jnp
    reference=homogeneous_slab(**case)
    count=len(reference['optical_depth'])-1
    r,t=layer_coefficients(jnp.full(count,case['optical_depth']/count),case['single_scattering_albedo'],case['asymmetry'])
    up,down=solve_fluxes(r,t,jnp.full(count,case['blackbody_w_m2']),
        case['down_top_w_m2'],case['surface_emission_w_m2'],case.get('surface_reflectivity',0.))
    up=np.asarray(up);down=np.asarray(down)
    if not np.all(np.isfinite(up)) or not np.all(np.isfinite(down)):
        raise RuntimeError('two-stream solver produced nonfinite flux in the reference comparison')
    scale=max(case['blackbody_w_m2'],case['down_top_w_m2'],case['surface_emission_w_m2'],1e-12)
    flux_error=max(np.max(np.abs(up-reference['up_w_m2'])),np.max(np.abs(down-reference['down_w_m2'])))/scale
    heating_error=np.max(np.abs(np.diff(up-down)-reference['layer_convergence_w_m2']))/scale
    return dict(passes_frozen_gate=bool(flux_error<=max_flux_error_fraction and heating_error<=max_layer_heating_error_fraction),
        maximum_flux_error_fraction=float(flux_error),maximum_layer_heating_error_fraction=float(heating_error),
        frozen_limits=dict(flux_fraction=max_flux_error_fraction,layer_heating_fraction=max_layer_heating_error_fraction),
        normalization_w_m2=scale,angles=reference['angles'],interfaces=count+1,
        reference_solver_tolerance=reference['tolerance'],reference_mesh_nodes=reference['solver_nodes'],
        scipy_version=version('scipy'),numpy_version=np.__version__,
        reference_profile=dict(up_w_m2=reference['up_w_m2'].tolist(),down_w_m2=reference['down_w_m2'].tolist(),
            layer_convergence_w_m2=reference['layer_convergence_w_m2'].tolist()),
        two_stream_profile=dict(up_w_m2=up.tolist(),down_w_m2=down.tolist(),layer_convergence_w_m2=np.diff(up-down).tolist()),
        phase_assumption=reference['phase_assumption'],scope='homogeneous thermal HG control; not source warming validation')


def layered_column(*,optical_depths,single_scattering_albedos,asymmetries,blackbody_w_m2,
                   down_top_w_m2,surface_emission_w_m2,surface_reflectivity=0.,angles=16,tolerance=1e-8):
    """Independent angular BVP with separate layer temperatures/properties.

    Each physical layer occupies its own copy of normalized depth [0,1].
    Interface continuity is imposed as boundary conditions between copies.
    This avoids smoothing discontinuous layer properties or multiplying
    exponentially growing transfer matrices in optically thick columns.
    """
    from scipy.integrate import solve_bvp
    arrays=[np.atleast_1d(np.asarray(x,dtype=float)) for x in
        (optical_depths,single_scattering_albedos,asymmetries,blackbody_w_m2)]
    tau,omega,g,blackbody=arrays
    layers=len(tau)
    if not layers or layers>12 or any(x.shape!=(layers,) for x in arrays):
        raise ValueError('reference requires matching one-dimensional arrays with 1..12 layers')
    if any(not np.all(np.isfinite(x)) for x in arrays) or np.any(tau<0) or np.any((omega<0)|(omega>1)) or np.any(np.abs(g)>=1) or np.any(blackbody<0):
        raise ValueError('layer inputs outside the HG column reference domain')
    if not isinstance(angles,int) or angles<4 or angles%2 or angles*layers>256:
        raise ValueError('reference requires an even angular count and at most 256 angular/layer unknowns')
    boundary_values=(down_top_w_m2,surface_emission_w_m2,surface_reflectivity,tolerance)
    if not all(math.isfinite(x) for x in boundary_values) or min(down_top_w_m2,surface_emission_w_m2)<0 or not 0<=surface_reflectivity<=1 or tolerance<=0:
        raise ValueError('invalid boundary fluxes or solver tolerance')
    nodes,weights=np.polynomial.legendre.leggauss(angles//2)
    positive=(nodes+1)/2;half_weights=weights/2
    mu=np.r_[-positive[::-1],positive];weight=np.r_[half_weights[::-1],half_weights]
    up=np.flatnonzero(mu<0);down=np.flatnonzero(mu>0)
    polynomial=np.polynomial.legendre.legvander(mu,angles-1)
    dimension=angles*layers
    operator=np.zeros((dimension,dimension));thermal=np.empty(dimension)
    for layer in range(layers):
        phase=(polynomial*((2*np.arange(angles)+1)*g[layer]**np.arange(angles)))@polynomial.T
        if phase.min()<-1e-12 or not np.allclose(phase@weight,2.,rtol=1e-10,atol=1e-10):
            raise ValueError('phase angular resolution is not positive/conservative')
        section=slice(layer*angles,(layer+1)*angles)
        operator[section,section]=tau[layer]*(-np.eye(angles)+omega[layer]*.5*phase*weight[None,:])/mu[:,None]
        thermal[section]=tau[layer]*(1-omega[layer])*blackbody[layer]/math.pi/mu
    top_jac=np.zeros((dimension,dimension));bottom_jac=np.zeros_like(top_jac)
    target=np.zeros(dimension)
    for layer in range(layers):
        offset=layer*angles
        top_jac[offset+down,offset+down]=1.
        bottom_jac[offset+up,offset+up]=1.
        if layer==0: target[offset+down]=down_top_w_m2/math.pi
        else: bottom_jac[offset+down,offset-angles+down]=-1.
        if layer==layers-1:
            target[offset+up]=surface_emission_w_m2/math.pi
            bottom_jac[np.ix_(offset+up,offset+down)]=-2*surface_reflectivity*(weight[down]*mu[down])[None,:]
        else: top_jac[offset+up,offset+angles+up]=-1.
    def derivative(_,intensity): return operator@intensity+thermal[:,None]
    def jacobian(depth,_): return np.broadcast_to(operator[:,:,None],(dimension,dimension,len(depth)))
    def boundary(top,bottom): return top_jac@top+bottom_jac@bottom-target
    mesh=np.linspace(0.,1.,16)
    guess=np.repeat((np.repeat(blackbody,angles)/math.pi)[:,None],len(mesh),axis=1)
    # Bound reference-solver memory; failure to converge remains explicit.
    max_nodes=min(5000,max(64,20_000_000//dimension**2))
    result=solve_bvp(derivative,boundary,mesh,guess,fun_jac=jacobian,
        bc_jac=lambda _a,_b:(top_jac,bottom_jac),tol=tolerance,max_nodes=max_nodes)
    if not result.success: raise RuntimeError(f'layered angular reference did not converge: {result.message}')
    top=result.sol(0.).reshape(layers,angles);bottom=result.sol(1.).reshape(layers,angles)
    scale=max(down_top_w_m2,surface_emission_w_m2,float(blackbody.max()),1.)
    if min(top.min(),bottom.min())<-1e-8*scale: raise RuntimeError('layered reference produced negative radiance')
    down_flux=np.r_[2*math.pi*np.sum(weight[down]*mu[down]*top[0,down]),
        2*math.pi*np.sum((weight[down]*mu[down])[None,:]*bottom[:,down],axis=1)]
    up_flux=np.r_[-2*math.pi*np.sum(weight[up]*mu[up]*top[0,up]),
        -2*math.pi*np.sum((weight[up]*mu[up])[None,:]*bottom[:,up],axis=1)]
    net=up_flux-down_flux
    if not np.all(np.isfinite(up_flux)) or not np.all(np.isfinite(down_flux)):
        raise RuntimeError('layered reference produced nonfinite flux')
    return dict(optical_depth=np.r_[0.,np.cumsum(tau)],up_w_m2=up_flux,down_w_m2=down_flux,
        layer_convergence_w_m2=np.diff(net),surface_net_w_m2=float(-net[-1]),toa_net_down_w_m2=float(-net[0]),
        boundary_residual=float(np.max(np.abs(boundary(result.sol(0.),result.sol(1.))))),
        angles=angles,solver_nodes=len(result.x),tolerance=tolerance,
        phase_assumption='piecewise homogeneous HG moments; source rod full phase function not supplied')


def compare_two_stream_column(*,max_flux_error_fraction,max_layer_heating_error_fraction,**case):
    """Layer-profile comparison with caller-frozen source-normalized limits."""
    for value in (max_flux_error_fraction,max_layer_heating_error_fraction):
        if not math.isfinite(value) or value<0: raise ValueError('finite nonnegative frozen error limits required')
    from src.framework.physics.twostream import layer_coefficients,solve_fluxes
    from src.framework.gcm._dinosaur import jnp
    reference=layered_column(**case)
    r,t=layer_coefficients(jnp.asarray(case['optical_depths']),jnp.asarray(case['single_scattering_albedos']),jnp.asarray(case['asymmetries']))
    up,down=solve_fluxes(r,t,jnp.asarray(case['blackbody_w_m2']),case['down_top_w_m2'],
        case['surface_emission_w_m2'],case.get('surface_reflectivity',0.))
    up=np.asarray(up);down=np.asarray(down)
    if not np.all(np.isfinite(up)) or not np.all(np.isfinite(down)):
        raise RuntimeError('two-stream solver produced nonfinite flux in the column comparison')
    scale=max(float(np.max(case['blackbody_w_m2'])),case['down_top_w_m2'],case['surface_emission_w_m2'],1e-12)
    flux=float(max(np.max(np.abs(up-reference['up_w_m2'])),np.max(np.abs(down-reference['down_w_m2'])))/scale)
    heating=float(np.max(np.abs(np.diff(up-down)-reference['layer_convergence_w_m2']))/scale)
    return dict(passes_frozen_gate=bool(flux<=max_flux_error_fraction and heating<=max_layer_heating_error_fraction),
        maximum_flux_error_fraction=flux,maximum_layer_heating_error_fraction=heating,
        frozen_limits=dict(flux_fraction=max_flux_error_fraction,layer_heating_fraction=max_layer_heating_error_fraction),
        normalization_w_m2=scale,angles=reference['angles'],interfaces=len(up),phase_assumption=reference['phase_assumption'],
        reference_solver_tolerance=reference['tolerance'],reference_mesh_nodes=reference['solver_nodes'],
        scipy_version=version('scipy'),numpy_version=np.__version__,
        reference_profile=dict(up_w_m2=reference['up_w_m2'].tolist(),down_w_m2=reference['down_w_m2'].tolist(),
            layer_convergence_w_m2=reference['layer_convergence_w_m2'].tolist()),
        two_stream_profile=dict(up_w_m2=up.tolist(),down_w_m2=down.tolist(),layer_convergence_w_m2=np.diff(up-down).tolist()),
        scope='layered thermal HG control; not source warming validation')
