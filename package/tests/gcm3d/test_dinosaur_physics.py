import dataclasses
import numpy as np
import pytest

pytest.importorskip('dinosaur')
import jax
import jax.numpy as jnp

from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY
from src.framework.physics.dinosaur_physics import (
    remap_gas,initial_state,build_step,axial_angular_momentum,exchange_energy)

jax.config.update('jax_enable_x64',True)


def test_sigma_remap_conserves_energy_momentum_and_passive_material():
    weights=jnp.array([.1,.2,.3,.4])
    temperature=jnp.array([140.,160.,180.,200.])
    u=jnp.array([50.,30.,10.,-10.]);v=u*.2
    q=jnp.array([[.001],[.002],[.003],[.004]])
    loss=jnp.array([.01,.02,.1,.2])
    t,unew,vnew,qnew,ps=remap_gas(temperature,u,v,q,jnp.asarray(610.),loss,weights,BODY)
    gas=610.*weights/BODY.gravity_m_s2
    donor=gas-loss
    target=ps*weights/BODY.gravity_m_s2
    np.testing.assert_allclose(jnp.sum(target),jnp.sum(donor),rtol=2e-14)
    np.testing.assert_allclose(jnp.sum(target*(BODY.cp_j_kg_k*t+.5*(unew**2+vnew**2))),
        jnp.sum(donor*(BODY.cp_j_kg_k*temperature+.5*(u*u+v*v))),rtol=2e-14)
    for a,b in ((unew,u),(vnew,v)):
        np.testing.assert_allclose(jnp.sum(target*a),jnp.sum(donor*b),rtol=2e-14)
    np.testing.assert_allclose(jnp.sum(target[:,None]*qnew,axis=0),jnp.sum(gas[:,None]*q,axis=0),rtol=2e-14)
    assert np.min(np.asarray(qnew))>=0.


def test_impossible_donor_mass_is_nonfinite_instead_of_clipped():
    result=remap_gas(jnp.full(4,200.),jnp.zeros(4),jnp.zeros(4),jnp.empty((4,0)),
        jnp.asarray(610.),jnp.ones(4)*1000.,jnp.ones(4)*.25,BODY)
    assert not np.isfinite(np.asarray(result[0])).all()
    assert not np.isfinite(float(result[-1]))


def test_explicit_co2_remap_changes_only_co2_material_and_preserves_gas_closure():
    weights=jnp.array([.1,.2,.3,.4]);gas=610.*weights/BODY.gravity_m_s2
    co2=jnp.array([.9,.8,.7,.6]);q=jnp.stack([co2,1.-co2,jnp.full(4,.001)],axis=-1)
    loss=jnp.array([.01,-.02,.1,.2])
    def apply(factor):
        return remap_gas(jnp.full(4,180.),jnp.zeros(4),jnp.zeros(4),q,
            jnp.asarray(610.),loss*factor,weights,BODY,co2_index=0)
    *_,ratios,pressure=apply(1.)
    target=pressure*weights/BODY.gravity_m_s2
    expected=jnp.sum(gas[:,None]*q,axis=0).at[0].add(-jnp.sum(loss))
    np.testing.assert_allclose(jnp.sum(target[:,None]*ratios,axis=0),expected,rtol=3e-14)
    np.testing.assert_allclose(jnp.sum(ratios[:,:2],axis=-1),1.,rtol=3e-14)
    def objective(factor):
        *_,r,p=apply(factor)
        return jnp.sum(p*weights/BODY.gravity_m_s2*r[:,0])
    np.testing.assert_allclose(jax.grad(objective)(1.),-jnp.sum(loss),rtol=2e-12)
    impossible=remap_gas(jnp.full(4,180.),jnp.zeros(4),jnp.zeros(4),q,
        jnp.asarray(610.),gas*.95,weights,BODY,co2_index=0)
    assert np.isnan(impossible[-1])


def test_sigma_remap_reverse_derivative():
    temperature=jnp.array([140.,160.,180.,200.])
    loss=jnp.array([.01,.02,.1,.2])
    def objective(scale):
        result=remap_gas(temperature,jnp.array([50.,30.,10.,-10.]),jnp.zeros(4),
            jnp.full((4,1),.001),jnp.asarray(610.),loss*scale,jnp.ones(4)*.25,BODY)
        return jnp.sum(result[0])+jnp.sum(result[1])+result[-1]
    auto=jax.grad(objective)(1.)
    for eps in (1e-4,1e-5,1e-6):
        finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=2e-7,atol=1e-7)


def test_sigma_remap_carries_suspended_mass_momentum_and_energy():
    weights=jnp.array([.1,.2,.3,.4]);gas=610.*weights/BODY.gravity_m_s2
    t=jnp.array([140.,160.,180.,200.]);u=jnp.array([50.,30.,10.,-10.]);v=.2*u
    q=jnp.array([[.01],[.02],[.03],[.04]]);loss=jnp.array([.01,.02,.1,.2])
    tn,un,vn,qn,ps=remap_gas(t,u,v,q,jnp.asarray(610.),loss,weights,BODY,cloud_index=0)
    target=ps*weights/BODY.gravity_m_s2
    old_mass=gas-loss+gas*q[:,0];new_mass=target*(1.+qn[:,0])
    np.testing.assert_allclose(jnp.sum(target*qn[:,0]),jnp.sum(gas*q[:,0]),rtol=3e-14)
    for before,after in ((u,un),(v,vn)):
        np.testing.assert_allclose(jnp.sum(old_mass*before),jnp.sum(new_mass*after),rtol=3e-14)
    np.testing.assert_allclose(jnp.sum((gas-loss)*BODY.cp_j_kg_k*t+.5*old_mass*(u*u+v*v)),
        jnp.sum(target*BODY.cp_j_kg_k*tn+.5*new_mass*(un*un+vn*vn)),rtol=3e-14)


def backend_case(temperature_k=140.,surface_temperature_k=148.,return_diagnostics=False,
                 radiation_component=None,wind_speed=0.,suspended_co2_clouds=False,
                 bulk_water=False,initial_water_surface_kg_m2=0.,moment_water=False,
                 dust_lifting_options=None,initial_water_vapor_ratio=1e-6,coagulation_options=None,
                 dust_background_options=None,nonorographic_drag_options=None,effective_input_mode=0,
                 convection_scheme='fv3',moist_convection_options=None,dust_mode_count=1,sigma_boundaries=None,
                 initial_co2_gas_ratio=None,do_qmass=False,thermal_relaxation_options=None):
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm._dinosaur import primitive_equations,scales
    from src.framework.gcm.dynamics import reference_temperature
    from src.celestials.planets.mars.gcm import radiative_forcing
    from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
    from src.framework.physics.gcm import two_stream_radiative_fluxes
    coords=coordinate_system('T21',4,sigma_boundaries=sigma_boundaries);specs=physics_specs(BODY);grid=coords.horizontal
    zeros=jnp.zeros((4,)+grid.modal_shape)
    pressure=float(specs.nondimensionalize(610.*scales.units.pascal))
    ref=jnp.asarray(reference_temperature(coords,BODY))[:,None,None]
    temperature=jnp.full((4,)+grid.nodal_shape,temperature_k)
    dyn=primitive_equations.State(vorticity=zeros,divergence=zeros,
        temperature_variation=grid.to_modal(temperature-ref),
        log_surface_pressure=grid.to_modal(jnp.full((1,)+grid.nodal_shape,np.log(pressure))),sim_time=0.)
    if wind_speed:
        from src.framework.gcm._dinosaur import spherical_harmonic
        speed_scale=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
        wind=jnp.broadcast_to(wind_speed*jnp.asarray(grid.cos_lat),temperature.shape)/speed_scale
        vor,div=spherical_harmonic.uv_nodal_to_vor_div_modal(grid,wind,jnp.zeros_like(wind))
        dyn=dataclasses.replace(dyn,vorticity=vor,divergence=div)
    state=initial_state(dyn,coords,specs,temperature_k=surface_temperature_k,bulk_water=bulk_water,
        initial_water_surface_kg_m2=initial_water_surface_kg_m2,initial_water_vapor_ratio=initial_water_vapor_ratio,
        moment_water=moment_water,initial_dust_mass_ratio=1e-5,initial_dust_number_per_kg=1e8,
        nonorographic_drag_options=nonorographic_drag_options,dust_background_options=dust_background_options,
        moment_cloud_options=dict(effective_input_mode=effective_input_mode),
        moist_convection_options=moist_convection_options,dust_mode_count=dust_mode_count,
        initial_co2_gas_ratio=initial_co2_gas_ratio)
    forcing=(radiative_forcing(co2_radiation_enabled=True) if radiation_component is None else
        reference_radiative_forcing())
    radiation=two_stream_radiative_fluxes if radiation_component is None else radiation_component
    if thermal_relaxation_options is not None: radiation=None
    step=build_step(coords,specs,BODY,forcing,5.,radiation_component=radiation,
        soil_diffusivity=(250./(1500.*627.9))**2,return_diagnostics=return_diagnostics,
        suspended_co2_clouds=suspended_co2_clouds,bulk_water=bulk_water,moment_water=moment_water,
        moment_cloud_options=dict(latent_heat=True,microtimestep=1.,effective_input_mode=effective_input_mode),
        dust_lifting_options=dust_lifting_options,dust_optical_bands=15,coagulation_options=coagulation_options,
        dust_background_options=dust_background_options,nonorographic_drag_options=nonorographic_drag_options,
        convection_scheme=convection_scheme,moist_convection_options=moist_convection_options,
        dust_mode_count=dust_mode_count,do_qmass=do_qmass,
        thermal_relaxation_options=thermal_relaxation_options)
    return state,step,grid


@pytest.mark.parametrize('height',[False,True])
def test_thermal_relaxation_reaches_global_rollout_and_reverse_derivative(height):
    def objective(target):
        options=dict(equilibrium=target,timescale=100.)
        if height:
            from src.framework.gcm.coordinates import grid as spectral_grid
            shape=(4,*spectral_grid('T21').nodal_shape)
            source=jnp.broadcast_to(jnp.array([1e6,1e5,1e4,-1000.])[:,None,None],shape)
            options.update(source_height=source,equilibrium=target+source*1e-5,top_pressure_pa=1e-6)
        state,step,grid=backend_case(temperature_k=180.,surface_temperature_k=180.,convection_scheme='none',
            thermal_relaxation_options=options)
        for _ in range(3): state=step(state)
        return jnp.mean(grid.to_nodal(state.dynamics.temperature_variation))
    automatic=jax.grad(objective)(185.)
    finite=(objective(185.001)-objective(184.999))/.002
    assert np.isfinite(automatic) and automatic>0.
    np.testing.assert_allclose(automatic,finite,rtol=2e-5,atol=1e-7)


def test_simple_radiation_reaches_global_rollout_and_reverse_derivative():
    from src.framework.physics.simple_radiation import make_radiation_component
    def objective(infrared_depth):
        component=make_radiation_component(tau_infrared=infrared_depth,forget_shortwave_heating=True)
        state,step,grid=backend_case(temperature_k=180.,surface_temperature_k=180.,radiation_component=component)
        for _ in range(3): state=step(state)
        return jnp.mean(grid.to_nodal(state.dynamics.temperature_variation))
    automatic=jax.grad(objective)(1.)
    finite=(objective(1.0001)-objective(.9999))/.0002
    assert np.isfinite(automatic) and automatic!=0.
    np.testing.assert_allclose(automatic,finite,rtol=2e-5,atol=1e-7)


def test_explicit_gas_global_species_budget_restart_and_reverse_derivative(tmp_path):
    from src.framework.physics.dinosaur_physics import CO2_GAS_TRACER,ARGON_GAS_TRACER
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm._dinosaur import scales
    from src.framework.gcm.restart import save_restart,load_restart
    state,step,grid=backend_case(initial_co2_gas_ratio=.9,do_qmass=True,
        surface_temperature_k=148.037)
    scale=float(physics_specs(BODY).dimensionalize(1.,scales.units.pascal).magnitude)
    def inventory(s,name):
        pressure=jnp.exp(grid.to_nodal(s.dynamics.log_surface_pressure))[0]*scale
        profile=grid.to_nodal(s.dynamics.tracers[name])
        mass=jnp.sum(pressure[None]*.25/BODY.gravity_m_s2*profile,axis=0)
        if name==CO2_GAS_TRACER: mass=mass+s.co2_ice[0]*scale/BODY.gravity_m_s2
        return jnp.sum(mass*jnp.asarray(grid.quadrature_weights))
    def evolve(s):
        return jax.lax.scan(lambda s,_:(step(s),None),s,None,length=3)[0]
    final=jax.jit(evolve)(state)
    for name in (CO2_GAS_TRACER,ARGON_GAS_TRACER):
        np.testing.assert_allclose(inventory(final,name),inventory(state,name),rtol=3e-12)
    assert float(jnp.mean(final.co2_ice))>float(jnp.mean(state.co2_ice))
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(final))
    restored=load_restart(save_restart(final,tmp_path/'gas.npz'))
    tick=jax.jit(step)
    for a,b in zip(jax.tree.leaves(tick(final)),jax.tree.leaves(tick(restored)),strict=True):
        np.testing.assert_array_equal(a,b)
    def objective(offset):
        final=evolve(state._replace(surface_temperature=state.surface_temperature+offset))
        return jnp.mean(jnp.exp(grid.to_nodal(final.dynamics.log_surface_pressure)))*scale
    auto=jax.jit(jax.grad(objective))(0.);forward=jax.jit(objective);epsilon=1e-4
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    np.testing.assert_allclose(auto,(forward(epsilon)-forward(-epsilon))/(2*epsilon),rtol=3e-5,atol=1e-7)
    missing,invalid,_=backend_case(do_qmass=True)
    with pytest.raises(ValueError,match='CO2 and argon'):
        invalid(missing)


def test_nonorographic_drag_global_timer_and_restart(tmp_path):
    from src.framework.gcm.restart import save_restart,load_restart
    options=dict(top_pressure_pa=.03,update_seconds=15,offset_seconds=5,conserve_energy=True)
    state,step,_=backend_case(temperature_k=200.,surface_temperature_k=200.,wind_speed=40.,
        nonorographic_drag_options=options)
    tick=jax.jit(step);first=tick(state)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(first))
    saved=first.auxiliary_fields['nonorographic_u_tendency_m_s2']
    assert float(jnp.max(jnp.abs(saved)))>0.
    assert first.auxiliary_fields['nonorographic_alarm_seconds']==15.
    second=tick(first)
    np.testing.assert_array_equal(saved,second.auxiliary_fields['nonorographic_u_tendency_m_s2'])
    assert second.auxiliary_fields['nonorographic_alarm_seconds']==10.
    restored=load_restart(save_restart(second,tmp_path/'gravity-wave.npz'))
    for a,b in zip(jax.tree.leaves(tick(second)),jax.tree.leaves(tick(restored)),strict=True):
        np.testing.assert_array_equal(a,b)
    assert tick(tick(second)).auxiliary_fields['nonorographic_alarm_seconds']==15.


def test_nonorographic_drag_global_multistep_reverse_derivative():
    state,step,_=backend_case(temperature_k=200.,surface_temperature_k=200.,wind_speed=40.,
        nonorographic_drag_options=dict(top_pressure_pa=.03,update_seconds=15,offset_seconds=5,
            conserve_energy=True))
    def objective(offset):
        initial=state._replace(surface_temperature=state.surface_temperature+offset)
        final,_=jax.lax.scan(lambda s,_:(step(s),None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)+1e3*jnp.mean(final.auxiliary_fields['nonorographic_u_tendency_m_s2'])
    auto=jax.jit(jax.grad(objective))(0.);forward=jax.jit(objective)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for epsilon in (1e-3,1e-4):
        fd=(forward(epsilon)-forward(-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,fd,rtol=2e-5,atol=2e-7)


def test_dinosaur_backend_runs_cold_column_and_preserves_total_inventory():
    state,step,_=backend_case(return_diagnostics=True)
    final,d=jax.jit(step)(state)
    assert bool(d.valid_inventory)
    np.testing.assert_allclose(d.corrected_total_mass_kg,d.target_total_mass_kg,rtol=2e-13)
    assert float(jnp.max(final.co2_ice))>0.
    assert all(np.all(np.isfinite(np.asarray(x))) for x in jax.tree_util.tree_leaves(final))
    assert np.isfinite(float(d.raw_axial_angular_momentum_kg_m2_s))
    assert np.isfinite(float(d.corrected_axial_angular_momentum_kg_m2_s))
    assert np.isfinite(float(d.raw_exchange_energy_j))
    assert np.isfinite(float(d.corrected_exchange_energy_j))


def test_passive_transport_keeps_material_through_dynamics_and_column_physics():
    from src.framework.gcm.coordinates import coordinate_system
    coords=coordinate_system('T21',4)
    state,step,grid=backend_case(temperature_k=200.,surface_temperature_k=200.,wind_speed=40.,
        return_diagnostics=True)
    lon=jnp.asarray(grid.longitudes)[:,None];lat=jnp.asarray(grid.latitudes)[None,:]
    tracer=jnp.broadcast_to(1e-5*(1.+.99*jnp.cos(20.*lon)*jnp.cos(lat)**20),
        (4,*grid.nodal_shape))
    state=state._replace(dynamics=dataclasses.replace(state.dynamics,
        tracers={**state.dynamics.tracers,'passive':grid.to_modal(tracer)}))
    def material(s):
        return jnp.sum(jnp.exp(grid.to_nodal(s.dynamics.log_surface_pressure))*
            grid.to_nodal(s.dynamics.tracers['passive'])*grid.quadrature_weights*
            jnp.diff(jnp.asarray(coords.vertical.boundaries))[:,None,None])
    initial=material(state);tick=jax.jit(step)
    for _ in range(3):
        state,diagnostics=tick(state)
        assert np.min(grid.to_nodal(state.dynamics.tracers['passive']))>=0.
        np.testing.assert_allclose(material(state),initial,rtol=3e-13)
        for stage in diagnostics.tracer_transport.values():
            report=stage['passive']
            assert report['valid']
            np.testing.assert_allclose(report['corrected_inventory_kg'],report['target_inventory_kg'],rtol=3e-14)


@pytest.mark.parametrize('scheme',['none','legacy'])
def test_convection_source_switch_runs_globally_and_restarts(tmp_path,scheme):
    from src.framework.gcm.restart import save_restart,load_restart
    state,step,grid=backend_case(temperature_k=200.,surface_temperature_k=220.,wind_speed=40.,
        convection_scheme=scheme)
    shape=(4,*grid.nodal_shape)
    variation=state.dynamics.temperature_variation+grid.to_modal(jnp.broadcast_to(
        jnp.array([0.,30.,100.,160.])[:,None,None],shape))
    state=state._replace(dynamics=dataclasses.replace(state.dynamics,temperature_variation=variation))
    tick=jax.jit(step);first=tick(state)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(first))
    restored=load_restart(save_restart(first,tmp_path/'convection.npz'))
    for a,b in zip(jax.tree.leaves(tick(first)),jax.tree.leaves(tick(restored)),strict=True):
        np.testing.assert_array_equal(a,b)
    if scheme=='legacy':
        def objective(offset):
            initial=state._replace(surface_temperature=state.surface_temperature+offset)
            final,_=jax.lax.scan(lambda s,_:(step(s),None),initial,None,length=3)
            return jnp.mean(final.surface_temperature)+jnp.mean(final.dynamics.temperature_variation)
        auto=jax.jit(jax.grad(objective))(0.);forward=jax.jit(objective)
        assert np.isfinite(auto) and abs(float(auto))>1e-6
        for epsilon in (1e-3,1e-4):
            fd=(forward(epsilon)-forward(-epsilon))/(2.*epsilon)
            np.testing.assert_allclose(auto,fd,rtol=3e-5,atol=1e-7)


def moist_backend_case(moment_water=False):
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER,WATER_VAPOR_TRACER
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    provider=(make_particle_provider(bands=15,dust_tracers=(('dust_mass','dust_number'),),
        water_moment_tracers=(WATER_CLOUD_TRACER,'water_core_mass','water_ice_number'))
        if moment_water else make_particle_provider(bands=15,water_bulk_tracer=WATER_CLOUD_TRACER))
    radiation=make_radiation_component(provider,radiation_mode='15band',water_vapor_tracer=WATER_VAPOR_TRACER)
    # Keep the initial emitting surface inside a 0.1 K Planck interpolation cell.
    state,step,grid=backend_case(temperature_k=250.3,surface_temperature_k=250.037,wind_speed=20.,
        bulk_water=not moment_water,moment_water=moment_water,initial_water_surface_kg_m2=100.,
        initial_water_vapor_ratio=.011,radiation_component=radiation,return_diagnostics=True,
        moist_convection_options={})
    profile=jnp.array([225.3,250.7,275.1,299.9])[:,None,None]
    dyn=dataclasses.replace(state.dynamics,
        temperature_variation=state.dynamics.temperature_variation+grid.to_modal(
            jnp.broadcast_to(profile-250.3,(4,*grid.nodal_shape))),
        log_surface_pressure=state.dynamics.log_surface_pressure+grid.to_modal(
            jnp.full((1,*grid.nodal_shape),jnp.log(50000./610.))))
    return state._replace(dynamics=dyn),step,grid


@pytest.mark.parametrize('moment_water',[False,True])
def test_moist_adjustment_global_water_budget_and_restart(tmp_path,moment_water):
    from src.framework.gcm.restart import save_restart,load_restart
    state,step,_=moist_backend_case(moment_water)
    first,diagnostics=jax.jit(step)(state)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(first))
    assert np.all(diagnostics.moist_convection['valid'])
    assert not np.any(diagnostics.moist_convection['unconverged'])
    assert jnp.max(diagnostics.moist_convection['iterations'])>0
    np.testing.assert_allclose(diagnostics.moist_convection['vapor_column_change_kg_m2']+
        5.*(diagnostics.moist_convection['rain_kg_m2_s']+diagnostics.moist_convection['snow_kg_m2_s']),
        0.,atol=1e-12)
    np.testing.assert_allclose(diagnostics.raw_water_mass_kg,diagnostics.target_water_mass_kg,rtol=3e-11)
    np.testing.assert_allclose(diagnostics.corrected_water_mass_kg,diagnostics.target_water_mass_kg,rtol=3e-11)
    if moment_water:
        assert jnp.max(first.auxiliary_fields['moist_bulk_surface_water_kg_m2'])>0.
        np.testing.assert_allclose(first.auxiliary_fields['moist_bulk_surface_water_kg_m2'][0],
            5.*(diagnostics.moist_convection['rain_kg_m2_s']+diagnostics.moist_convection['snow_kg_m2_s']),
            rtol=3e-13,atol=1e-13)
    else:
        assert jnp.max(first.auxiliary_fields['water_surface_kg_m2'])>100.
    restored=load_restart(save_restart(first,tmp_path/'moist-adjustment.npz'))
    tick=jax.jit(step);continued,_=tick(first);restarted,_=tick(restored)
    for a,b in zip(jax.tree.leaves(continued),jax.tree.leaves(restarted),strict=True):
        np.testing.assert_array_equal(a,b)


def test_moist_adjustment_global_reverse_derivative():
    state,step,_=moist_backend_case()
    def objective(offset):
        initial=state._replace(surface_temperature=state.surface_temperature+offset)
        final,_=jax.lax.scan(lambda s,_:(step(s)[0],None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)+jnp.mean(final.auxiliary_fields['water_surface_kg_m2'])
    reverse=jax.jit(jax.grad(objective)).lower(0.).compile()
    memory=reverse.memory_analysis()
    if memory is not None:
        assert memory.temp_size_in_bytes<8*1024**3
    auto=reverse(0.);forward=jax.jit(objective)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    for epsilon in (1e-3,1e-4):
        fd=(forward(epsilon)-forward(-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,fd,rtol=3e-5,atol=1e-7)


def test_reference_cloud_backend_forms_suspended_ice_and_closes_raw_mass():
    from src.framework.physics.co2_clouds import CO2_CLOUD_TRACER
    state,step,grid=backend_case(return_diagnostics=True,suspended_co2_clouds=True)
    final,d=jax.jit(step)(state)
    assert bool(d.valid_inventory)
    assert float(jnp.max(grid.to_nodal(final.dynamics.tracers[CO2_CLOUD_TRACER])))>0.
    np.testing.assert_allclose(d.raw_total_mass_kg,d.target_total_mass_kg,rtol=2e-11)
    np.testing.assert_allclose(d.corrected_total_mass_kg,d.target_total_mass_kg,rtol=2e-13)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(final))


def test_stretched_sigma_global_gradient_and_invalid_geometry(tmp_path):
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.gcm.restart import save_restart,load_restart
    state,step,grid=backend_case(temperature_k=170.,surface_temperature_k=180.037,
        radiation_component=make_radiation_component(lambda *args:None),
        sigma_boundaries=[0.,.001,.05,.5,1.])
    tick=jax.jit(step);first=tick(state)
    restored=load_restart(save_restart(first,tmp_path/'stretched.npz'))
    for a,b in zip(jax.tree.leaves(tick(first)),jax.tree.leaves(tick(restored)),strict=True):
        np.testing.assert_array_equal(a,b)
    def objective(offset):
        initial=state._replace(surface_temperature=state.surface_temperature+offset)
        final,_=jax.lax.scan(lambda s,_:(step(s),None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)
    auto=jax.jit(jax.grad(objective))(0.);forward=jax.jit(objective);epsilon=1e-4
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    np.testing.assert_allclose(auto,(forward(epsilon)-forward(-epsilon))/(2*epsilon),rtol=3e-5,atol=1e-7)
    fields={**state.auxiliary_fields,'vertical_sigma_boundaries':jnp.array([0.,.002,.05,.5,1.])}
    invalid=tick(state._replace(auxiliary_fields=fields))
    assert np.isnan(np.asarray(invalid.dynamics.temperature_variation)).all()


@pytest.mark.parametrize('native_radiation',[False,True])
def test_reference_cloud_backend_multistep_reverse_derivative(native_radiation):
    from src.framework.physics.co2_clouds import CO2_CLOUD_TRACER
    from src.framework.physics.radiative_transfer import make_radiation_component
    radiation=make_radiation_component(lambda *args:None) if native_radiation else None
    state,step,grid=backend_case(suspended_co2_clouds=True,radiation_component=radiation)
    def objective(offset):
        initial=state._replace(surface_temperature=state.surface_temperature+offset)
        final,_=jax.lax.scan(lambda s,_:(step(s),None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)+100.*jnp.mean(
            grid.to_nodal(final.dynamics.tracers[CO2_CLOUD_TRACER]))
    value,auto=jax.jit(jax.value_and_grad(objective))(0.)
    assert np.isfinite(value) and np.isfinite(auto) and abs(float(auto))>1e-6
    forward=jax.jit(objective)
    for eps in (1e-2,1e-3):
        finite=(forward(eps)-forward(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-5,atol=1e-7)


def test_suspended_cloud_radiation_feedback_and_rollout_gradient():
    from src.framework.physics.co2_clouds import CO2_CLOUD_TRACER
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    clear=make_radiation_component(make_particle_provider(bands=15),radiation_mode='extended')
    cloudy=make_radiation_component(make_particle_provider(bands=15,
        co2_cloud_tracer=CO2_CLOUD_TRACER),radiation_mode='extended')
    state,step,grid=backend_case(suspended_co2_clouds=True,radiation_component=cloudy)
    _,clear_step,_=backend_case(suspended_co2_clouds=True,radiation_component=clear)
    def rollout(current,fn):
        return jax.lax.scan(lambda s,_:(fn(s),None),current,None,length=3)[0]
    def objective(offset):
        final=rollout(state._replace(surface_temperature=state.surface_temperature+offset),step)
        return jnp.mean(final.surface_temperature)+jnp.mean(grid.to_nodal(final.dynamics.temperature_variation))
    final=jax.jit(lambda s:rollout(s,step))(state)
    control=jax.jit(lambda s:rollout(s,clear_step))(state)
    assert np.max(np.abs(final.surface_temperature-control.surface_temperature))>1e-8
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(final))
    value,auto=jax.jit(jax.value_and_grad(objective))(0.)
    assert np.isfinite(value) and np.isfinite(auto)
    forward=jax.jit(objective)
    for eps in (1e-2,1e-3):
        finite=(forward(eps)-forward(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-5,atol=1e-7)


def test_bulk_water_backend_closes_material_and_restarts_full_cloud_feedback(tmp_path):
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER,WATER_VAPOR_TRACER
    from src.framework.physics.co2_clouds import CO2_CLOUD_TRACER
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.gcm.restart import save_restart,load_restart
    radiation=make_radiation_component(make_particle_provider(bands=15,
        co2_cloud_tracer=CO2_CLOUD_TRACER,water_bulk_tracer=WATER_CLOUD_TRACER),
        radiation_mode='extended',water_vapor_tracer=WATER_VAPOR_TRACER)
    state,step,grid=backend_case(bulk_water=True,suspended_co2_clouds=True,
        return_diagnostics=True,radiation_component=radiation)
    tick=jax.jit(step)
    first,d=tick(state)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(first))
    assert float(jnp.max(grid.to_nodal(first.dynamics.tracers[WATER_CLOUD_TRACER])))>0.
    np.testing.assert_allclose(d.raw_water_mass_kg,d.target_water_mass_kg,rtol=3e-11)
    np.testing.assert_allclose(d.corrected_water_mass_kg,d.target_water_mass_kg,rtol=3e-11)
    restored=load_restart(save_restart(first,tmp_path/'water-feedback.npz'))
    continued,_=tick(first);restarted,_=tick(restored)
    for a,b in zip(jax.tree.leaves(continued),jax.tree.leaves(restarted),strict=True):
        np.testing.assert_array_equal(a,b)


def test_surface_water_exchange_uses_previous_step_coefficient():
    state,step,_=backend_case(temperature_k=200.,surface_temperature_k=250.,wind_speed=20.,
        bulk_water=True,initial_water_surface_kg_m2=100.)
    tick=jax.jit(step)
    first=tick(state);second=tick(first)
    np.testing.assert_allclose(first.auxiliary_fields['water_surface_kg_m2'],100.,rtol=3e-13,atol=3e-13)
    assert float(jnp.max(first.auxiliary_fields['water_exchange_coefficient_kg_m2_s']))>0.
    assert float(jnp.min(second.auxiliary_fields['water_surface_kg_m2']))<99.999999


def test_bulk_water_backend_multistep_reverse_derivative():
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER,WATER_VAPOR_TRACER
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    radiation=make_radiation_component(make_particle_provider(bands=15,water_bulk_tracer=WATER_CLOUD_TRACER),
        radiation_mode='extended',water_vapor_tracer=WATER_VAPOR_TRACER)
    # A derivative is defined within a source gas-mixture bin, away from 1e-6.
    state,step,_=backend_case(temperature_k=200.,surface_temperature_k=250.,wind_speed=20.,
        bulk_water=True,initial_water_surface_kg_m2=100.,radiation_component=radiation,
        initial_water_vapor_ratio=1.1e-6)
    def objective(offset):
        initial=state._replace(surface_temperature=state.surface_temperature+offset)
        final,_=jax.lax.scan(lambda s,_:(step(s),None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)+jnp.mean(final.auxiliary_fields['water_surface_kg_m2'])
    value,auto=jax.jit(jax.value_and_grad(objective))(0.)
    assert np.isfinite(value) and np.isfinite(auto) and abs(float(auto))>1e-6
    forward=jax.jit(objective)
    for eps in (1e-2,1e-3):
        finite=(forward(eps)-forward(-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-5,atol=1e-7)


@pytest.mark.parametrize('interactive,coagulation,background,effective_input_mode',[
    (False,False,False,0),(True,False,False,0),(True,True,False,0),(False,False,True,0),
    (True,True,True,0),(True,True,'local',0),(True,True,'local',1)])
def test_moment_water_cloud_feedback_material_and_restart(tmp_path,interactive,coagulation,background,effective_input_mode,dust_mode_count=1):
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER,WATER_VAPOR_TRACER
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.gcm.restart import save_restart,load_restart
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.physics.water_moments import moment_particle_tracers
    moment_names=moment_particle_tracers(dust_mode_count)
    dust_pairs=(('dust_mass','dust_number'),)+tuple(zip(moment_names[4::3],moment_names[5::3]))
    radiation=make_radiation_component(make_particle_provider(bands=15,
        dust_tracers=dust_pairs,
        water_moment_tracers=(WATER_CLOUD_TRACER,'water_core_mass','water_ice_number')),
        radiation_mode='extended',water_vapor_tracer=WATER_VAPOR_TRACER)
    coords=coordinate_system('T21',4)
    scenario=dict(seasons=jnp.array([0.,90.,180.,270.,360.]),
        opacity=jnp.full((5,*coords.horizontal.nodal_shape),2.))
    background_options=dict(scenario=scenario) if background else None
    if background=='local':
        background_options['local_time_scenario']=dict(seasons=scenario['seasons'],bins_hours=[0.,12.,24.],
            fractions=jnp.full((5,*coords.horizontal.nodal_shape,2),.5))
    state,step,grid=backend_case(temperature_k=170.,surface_temperature_k=180.,
        moment_water=True,return_diagnostics=True,radiation_component=radiation,
        wind_speed=40. if interactive else 0.,
        dust_lifting_options=(dict(dual_mode=1) if background else {}) if interactive else None,
        coagulation_options={} if coagulation else None,
        dust_background_options=background_options,effective_input_mode=effective_input_mode,
        dust_mode_count=dust_mode_count)
    tick=jax.jit(step);first,d=tick(state)
    assert all(np.isfinite(x).all() for x in jax.tree.leaves(first))
    if background=='local':
        assert first.auxiliary_fields['dust_local_time_initialized']==1.
        assert np.any(first.auxiliary_fields['dust_local_time_opacity']>0.)
    if effective_input_mode:
        assert first.auxiliary_fields['moment_previous_initialized']==1.
        assert np.all(first.auxiliary_fields['moment_previous_temperature_k']>0.)
    assert float(jnp.max(grid.to_nodal(first.dynamics.tracers[WATER_CLOUD_TRACER])))>0.
    np.testing.assert_allclose(d.raw_water_mass_kg,d.target_water_mass_kg,rtol=3e-11)
    np.testing.assert_allclose(d.corrected_water_mass_kg,d.target_water_mass_kg,rtol=3e-11)
    coords=coordinate_system('T21',4);specs=physics_specs(BODY)
    def material(s,names):
        from src.framework.gcm._dinosaur import scales
        scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
        ps=jnp.exp(grid.to_nodal(s.dynamics.log_surface_pressure))*scale
        q=sum(grid.to_nodal(s.dynamics.tracers[name]) for name in names)
        ds=jnp.asarray(np.diff(coords.vertical.boundaries),dtype=jnp.float64)[:,None,None]
        air=jnp.sum(ps*ds*q*grid.quadrature_weights)
        reservoir=s.auxiliary_fields['dust_surface_kg_m2']
        if dust_mode_count>1: reservoir=jnp.sum(reservoir,axis=-1)
        surface=jnp.sum(reservoir*grid.quadrature_weights)*BODY.gravity_m_s2
        source=jnp.sum(s.auxiliary_fields['dust_coagulation_mass_change_kg_m2']*grid.quadrature_weights)*BODY.gravity_m_s2
        return air+surface-source
    names=('dust_mass','water_core_mass')+tuple(name for pair in
        zip(moment_names[4::3],moment_names[6::3]) for name in pair)
    np.testing.assert_allclose(material(first,names),material(state,names),rtol=3e-11)
    if coagulation:
        assert float(jnp.max(jnp.abs(first.auxiliary_fields['dust_coagulation_mass_change_kg_m2'])))>0.
    difference=first.auxiliary_fields['dust_surface_kg_m2']-state.auxiliary_fields['dust_surface_kg_m2']
    if interactive or background:
        assert float(jnp.min(difference))<0.
        assert float(jnp.max(first.auxiliary_fields['dust_reference_opacity']))>0.
    else:
        assert float(jnp.max(difference))>0.
    restored=load_restart(save_restart(first,tmp_path/'moment-feedback.npz'))
    continued,_=tick(first);restarted,_=tick(restored)
    for a,b in zip(jax.tree.leaves(continued),jax.tree.leaves(restarted),strict=True):
        np.testing.assert_array_equal(a,b)


@pytest.mark.parametrize('interactive,background,effective_input_mode',[
    (False,False,0),(True,False,0),(False,True,0),(True,True,0),(True,'local',0),(True,'local',1)])
def test_moment_cloud_multistep_reverse_derivative(interactive,background,effective_input_mode,dust_mode_count=1):
    from src.framework.physics.water_bulk import WATER_CLOUD_TRACER,WATER_VAPOR_TRACER
    from src.framework.physics.particle_optics import make_particle_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.physics.water_moments import moment_particle_tracers
    moment_names=moment_particle_tracers(dust_mode_count)
    dust_pairs=(('dust_mass','dust_number'),)+tuple(zip(moment_names[4::3],moment_names[5::3]))
    radiation=make_radiation_component(make_particle_provider(bands=15,
        dust_tracers=dust_pairs,
        water_moment_tracers=(WATER_CLOUD_TRACER,'water_core_mass','water_ice_number')),
        radiation_mode='extended',water_vapor_tracer=WATER_VAPOR_TRACER)
    from src.framework.gcm.coordinates import coordinate_system
    coords=coordinate_system('T21',4)
    scenario=dict(seasons=jnp.array([0.,90.,180.,270.,360.]),
        opacity=jnp.full((5,*coords.horizontal.nodal_shape),2.))
    background_options=dict(scenario=scenario) if background else None
    if background=='local':
        background_options['local_time_scenario']=dict(seasons=scenario['seasons'],bins_hours=[0.,12.,24.],
            fractions=jnp.full((5,*coords.horizontal.nodal_shape,2),.5))
    state,step,grid=backend_case(temperature_k=170.,surface_temperature_k=180.,
        moment_water=True,radiation_component=radiation,wind_speed=40. if interactive else 0.,
        dust_lifting_options=(dict(dual_mode=1) if background else {}) if interactive else None,
        dust_background_options=background_options,
        # Keep the derivative probe within one source water-mixture bin.
        initial_water_vapor_ratio=1.1e-6,effective_input_mode=effective_input_mode,dust_mode_count=dust_mode_count,
        coagulation_options={} if dust_mode_count>1 else None)
    def objective(amplitude):
        mass_name,number_name=dust_pairs[-1]
        dyn=dataclasses.replace(state.dynamics,tracers={**state.dynamics.tracers,
            mass_name:state.dynamics.tracers[mass_name]*amplitude,
            number_name:state.dynamics.tracers[number_name]*amplitude})
        final,_=jax.lax.scan(lambda s,_:(step(s),None),state._replace(dynamics=dyn),None,length=3)
        return jnp.mean(final.surface_temperature)+jnp.mean(
            grid.to_nodal(final.dynamics.tracers[WATER_CLOUD_TRACER]))*1e6
    reverse=jax.jit(jax.grad(objective))
    if dust_mode_count>1:
        reverse=reverse.lower(1.).compile()
        memory=reverse.memory_analysis()
        if memory is not None:
            assert memory.temp_size_in_bytes<8*1024**3
    auto=reverse(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-8
    forward=jax.jit(objective)
    for eps in (1e-3,1e-4):
        finite=(forward(1.+eps)-forward(1.-eps))/(2.*eps)
        np.testing.assert_allclose(auto,finite,rtol=3e-5,atol=1e-7)


def test_multimode_cloud_dust_feedback_material_and_restart(tmp_path):
    test_moment_water_cloud_feedback_material_and_restart(tmp_path,True,True,'local',1,dust_mode_count=2)


def test_multimode_cloud_dust_multistep_reverse_derivative():
    test_moment_cloud_multistep_reverse_derivative(True,'local',1,dust_mode_count=2)


def test_surface_drag_transfers_axial_momentum_to_ground_and_reports_projection():
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    state,step,grid=backend_case(temperature_k=180.,surface_temperature_k=185.,
        return_diagnostics=True,wind_speed=20.)
    coords=coordinate_system('T21',4);specs=physics_specs(BODY)
    before=axial_angular_momentum(state,coords,specs,BODY)
    final,d=jax.jit(step)(state)
    ground=jnp.sum(final.surface_momentum[0]*grid.quadrature_weights)
    assert float(ground)>0.
    np.testing.assert_allclose(d.raw_axial_angular_momentum_kg_m2_s,before,rtol=2e-11)
    np.testing.assert_allclose(d.corrected_axial_angular_momentum_kg_m2_s,
        axial_angular_momentum(final,coords,specs,BODY),rtol=2e-14)


def test_exchange_energy_surface_derivative_uses_soil_node_capacity():
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.physics.soil_frost import soil_layer_thickness
    coords=coordinate_system('T21',4);specs=physics_specs(BODY)
    state,_,_=backend_case(temperature_k=180.,surface_temperature_k=185.)
    objective=lambda delta:exchange_energy(state._replace(
        surface_temperature=state.surface_temperature+delta),coords,specs,BODY)
    derivative=jax.grad(objective)(0.)
    expected=1500.*627.9*soil_layer_thickness(16)[0]*BODY.surface_area_m2
    np.testing.assert_allclose(derivative,expected,rtol=2e-14)


@pytest.mark.parametrize('native_radiation',[False,True])
def test_multistep_dinosaur_reverse_derivative_reaches_surface_state(native_radiation):
    from src.framework.physics.radiative_transfer import make_radiation_component
    radiation=(make_radiation_component(lambda *args:None) if native_radiation else None)
    state,step,grid=backend_case(temperature_k=180.,surface_temperature_k=185.,
        radiation_component=radiation)

    def objective(perturbation):
        initial=state._replace(surface_temperature=state.surface_temperature+perturbation)
        final,_=jax.lax.scan(lambda carry,_:(step(carry),None),initial,None,length=3)
        return jnp.mean(final.surface_temperature)+jnp.mean(
            grid.to_nodal(final.dynamics.temperature_variation))

    value_and_grad=jax.jit(jax.value_and_grad(objective))
    value,auto=value_and_grad(0.)
    assert np.isfinite(value) and np.isfinite(auto) and abs(float(auto))>1e-6
    forward=jax.jit(objective)
    for epsilon in (1e-2,1e-3):
        finite=(forward(epsilon)-forward(-epsilon))/(2.*epsilon)
        np.testing.assert_allclose(auto,finite,rtol=2e-5,atol=1e-7)


def test_nonlte_callback_keeps_interface_fluxes_and_reports_heating_separately():
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
    from src.framework.physics.radiative_transfer import make_radiation_component
    from src.framework.physics.nonlte_cooling import cooling,solar_heating_factor,infrared_blend
    from src.framework.physics.radiation_grid import radiation_grid
    coords=coordinate_system('T21',4);specs=physics_specs(BODY);grid=coords.horizontal
    state,_,_=backend_case(temperature_k=180.,surface_temperature_k=200.)
    shape=(4,*grid.nodal_shape)
    state=state._replace(dynamics=dataclasses.replace(state.dynamics,tracers={
        'co2':grid.to_modal(jnp.full(shape,.9)),'atomic_o':grid.to_modal(jnp.full(shape,.01))}))
    forcing=reference_radiative_forcing()
    clear=lambda *args:None
    base=make_radiation_component(clear,solar_nonlte=False)(state,coords,specs,BODY,forcing)
    callback=make_radiation_component(clear,nonlte_tracers=('co2','atomic_o'))
    corrected=jax.jit(lambda state:callback(state,coords,specs,BODY,forcing))(state)
    for actual,expected in zip(corrected[:7],base[:7],strict=True):
        np.testing.assert_allclose(actual,expected,rtol=3e-13,atol=1e-11)
    half=jnp.asarray(coords.vertical.boundaries,dtype=jnp.float64)[:,None,None]*610.
    full=jnp.asarray(coords.vertical.centers,dtype=jnp.float64)[:,None,None]*610.
    t=jnp.full(shape,180.);radgrid=radiation_grid(half+jnp.zeros_like(t[:1]),
        full+jnp.zeros_like(t),t,jnp.full(grid.nodal_shape,200.))
    capacity=BODY.cp_j_kg_k*jnp.diff(half,axis=0)/BODY.gravity_m_s2
    sw=jnp.diff(base.shortwave_up_w_m2-base.shortwave_down_w_m2,axis=0)/capacity
    lw=jnp.diff(base.longwave_up_w_m2-base.longwave_down_w_m2,axis=0)/capacity
    nlte=cooling(t,full,.9,.01,radiation_pressure_pa=radgrid.pressure)
    expected=(solar_heating_factor(full)-1.)*sw+infrared_blend(lw,nlte,half[:-1])-lw
    np.testing.assert_allclose(corrected.nonlte_heating_k_s,expected,rtol=2e-11,atol=2e-14)


def scheduled_backend(amplitude=1.):
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.gcm.specs import physics_specs
    from src.framework.gcm._dinosaur import scales
    from src.framework.physics.gcm import RadiativeFluxDiagnostics
    from src.celestials.planets.mars.reference_constants import reference_radiative_forcing
    coords=coordinate_system('T21',4);specs=physics_specs(BODY);grid=coords.horizontal
    state,_,_=backend_case(temperature_k=180.,surface_temperature_k=180.)
    state=initial_state(state.dynamics,coords,specs,180.,radiation_update_seconds=20.)
    time_scale=float(specs.dimensionalize(1.,scales.units.second).magnitude)
    pressure_scale=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    def radiation(state,*args):
        seconds=state.dynamics.sim_time*time_scale
        ps=jnp.exp(grid.to_nodal(state.dynamics.log_surface_pressure))[0]*pressure_scale
        capacity=BODY.cp_j_kg_k*jnp.asarray(np.diff(coords.vertical.boundaries),dtype=jnp.float64)[:,None,None]*ps/BODY.gravity_m_s2
        interfaces=jnp.zeros((5,*grid.nodal_shape))
        surface=jnp.zeros(grid.nodal_shape)
        return RadiativeFluxDiagnostics(interfaces,interfaces,interfaces,interfaces,
            capacity*seconds*1e-6*amplitude,surface,surface,
            stratosphere_flux_convergence_w_m2=100.*amplitude)
    step=build_step(coords,specs,BODY,reference_radiative_forcing(),5.,radiation_component=radiation,
        radiation_update_seconds=20.,soil_diffusivity=(250./(1500.*627.9))**2)
    return state,step,coords,specs


def test_radiation_schedule_holds_kelvin_rate_and_stratosphere_until_source_clock_refresh(tmp_path):
    from src.framework.gcm.restart import save_restart,load_restart
    state,step,_,_=scheduled_backend()
    step=jax.jit(step)
    for count,expected in enumerate((5.,5.,5.,20.,20.),1):
        state=step(state)
        np.testing.assert_allclose(state.auxiliary_fields['radiation_heating_k_s'],expected*1e-6,rtol=2e-13)
        updates=1 if count<4 else 2
        np.testing.assert_allclose(state.auxiliary_fields['radiation_stratosphere_k'],
            170.+updates*20.*BODY.gravity_m_s2/BODY.cp_j_kg_k,rtol=3e-14)
        if count==2:
            perturbed=state._replace(dynamics=dataclasses.replace(state.dynamics,
                log_surface_pressure=state.dynamics.log_surface_pressure+
                    jnp.zeros_like(state.dynamics.log_surface_pressure).at[0,0,0].set(.01)))
            np.testing.assert_array_equal(step(perturbed).auxiliary_fields['radiation_heating_k_s'],
                state.auxiliary_fields['radiation_heating_k_s'])
            path=tmp_path/'scheduled.npz';save_restart(state,path)
            restored=load_restart(path)
            for a,b in zip(jax.tree_util.tree_leaves(step(state)),jax.tree_util.tree_leaves(step(restored)),strict=True):
                np.testing.assert_array_equal(a,b)


def test_held_radiation_retains_multistep_reverse_derivative():
    def objective(amplitude):
        state,step,_,_=scheduled_backend(amplitude)
        final=jax.lax.scan(lambda state,_:(step(state),None),state,None,length=5)[0]
        return jnp.mean(final.dynamics.temperature_variation)+jnp.mean(final.surface_temperature)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-10
    eps=1e-3
    finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
    np.testing.assert_allclose(auto,finite,rtol=4e-6,atol=1e-9)


def test_prescribed_dust_scenario_and_size_optics_reach_multistep_reverse_gradient():
    from src.framework.gcm.coordinates import coordinate_system
    from src.framework.physics.prescribed_dust import make_fixed_dust_provider
    from src.framework.physics.radiative_transfer import make_radiation_component
    coords=coordinate_system('T21',4);shape=(5,*coords.horizontal.nodal_shape)
    def objective(amplitude):
        scenario=dict(seasons=jnp.array([0.,90.,180.,270.,360.]),
            opacity=jnp.full(shape,.2)*amplitude,top_height_km=jnp.full(shape,35.))
        provider=make_fixed_dust_provider(scenario=scenario,effective_radius_m=1.5e-6*amplitude)
        state,step,_=backend_case(temperature_k=180.,surface_temperature_k=200.,
            radiation_component=make_radiation_component(provider))
        final=jax.lax.scan(lambda state,_:(step(state),None),state,None,length=3)[0]
        return jnp.mean(final.surface_temperature)
    auto=jax.jit(jax.grad(objective))(1.)
    assert np.isfinite(auto) and abs(float(auto))>1e-6
    eps=1e-4
    finite=(objective(1.+eps)-objective(1.-eps))/(2.*eps)
    np.testing.assert_allclose(auto,finite,rtol=2e-5,atol=1e-7)
