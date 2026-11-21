#!/usr/bin/env python3
"""Replay an archived reference checkpoint through public build_step controls.

Run with PYTHONPATH=package. Instantaneous replay and archived time means are
reported separately. This diagnostic does not alter the archived run.
"""
import argparse
import dataclasses
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time

os.environ.setdefault('JAX_ENABLE_X64','true')

import numpy as np
from src.framework.gcm._dinosaur import jax,jnp,scales,spherical_harmonic
from src.framework.gcm.coordinates import coordinate_system
from src.framework.gcm.specs import physics_specs
from src.framework.gcm.restart import load_restart,save_restart
from src.framework.physics.dinosaur_physics import build_step,initial_state
from src.framework.physics.dinosaur_physics import water_inventory
from src.framework.physics.vertical_grid import pressure_interfaces,radiation_interfaces
from src.celestials.planets.mars.reference_constants import AMES_BODY_3D as BODY,reference_radiative_forcing
from src.celestials.planets.mars.topography import regrid_to_nodal,mola_modal_orography


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path,help='reference run containing config.json and restart.npz')
    parser.add_argument('data',type=Path,help='released Ames data directory, including MOLA')
    parser.add_argument('--restart',type=Path,help='specific checkpoint; defaults to run/restart.npz')
    parser.add_argument('--native',type=Path,help='directory containing C24/C48-first-average.npz')
    parser.add_argument('--reference-physics',action='store_true',help='also compare held and refreshed radiation')
    parser.add_argument('--damping-matrix',action='store_true',help='compare native/balanced cold starts and coupled D2 off/on')
    parser.add_argument('--assert-mean-ratio',type=float,help='fail if archived mean u RMS exceeds this times both native means')
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    if args.assert_mean_ratio is not None and (args.native is None or not np.isfinite(args.assert_mean_ratio) or args.assert_mean_ratio<=0.):
        parser.error('--assert-mean-ratio requires --native and a finite positive ratio')
    config=json.loads((args.run/'config.json').read_text())
    if config.get('body')!=dataclasses.asdict(BODY) or config.get('physics')!='reference':
        parser.error('replay requires the released Mars reference body and physics')
    restart=args.run/'restart.npz' if args.restart is None else args.restart
    state=load_restart(restart)
    coords=coordinate_system(config['truncation'],config['layers'],
        sigma_boundaries=np.asarray(state.auxiliary_fields['vertical_sigma_boundaries']))
    specs=physics_specs(BODY);grid=coords.horizontal
    weights=np.broadcast_to(np.asarray(grid.quadrature_weights),grid.nodal_shape)
    weights=weights/weights.sum()
    speed=float(specs.dimensionalize(1.,scales.units.meter/scales.units.second).magnitude)
    pascal=float(specs.dimensionalize(1.,scales.units.pascal).magnitude)
    kelvin=float(specs.dimensionalize(1.,scales.units.kelvin).magnitude)
    elevation=regrid_to_nodal(coords,mola_path=args.data/'mars_topo_mola16.nc')
    orography=mola_modal_orography(coords,specs,elevation_nodal_m=elevation)
    tau=config['hyperdiffusion_tau_sols']*BODY.solar_day_seconds
    def rms(value):
        return float(np.sqrt(np.sum(np.asarray(value)**2*weights)))
    def stats(value):
        dyn=value.dynamics
        u,v=spherical_harmonic.vor_div_to_uv_nodal(grid,dyn.vorticity,dyn.divergence)
        div_u,div_v=spherical_harmonic.vor_div_to_uv_nodal(grid,jnp.zeros_like(dyn.vorticity),dyn.divergence)
        low=(jnp.asarray(grid.modal_axes[1])<=3)[None,None,:]
        low_u,low_v=spherical_harmonic.vor_div_to_uv_nodal(grid,jnp.zeros_like(dyn.vorticity),dyn.divergence*low)
        temperature=(grid.to_nodal(dyn.temperature_variation)[0]*kelvin+BODY.reference_temperature_k)
        return dict(top_u_rms_m_s=rms(u[0]*speed),top_vector_rms_m_s=rms(jnp.hypot(u[0],v[0])*speed),
            top_divergent_vector_rms_m_s=rms(jnp.hypot(div_u[0],div_v[0])*speed),
            top_low_degree_divergent_rms_m_s=rms(jnp.hypot(low_u[0],low_v[0])*speed),
            top_temperature_k=float(np.sum(np.asarray(temperature)*weights)),
            top_min_temperature_k=float(np.min(np.asarray(temperature))),max_wind_m_s=float(jnp.hypot(u,v).max())*speed,
            surface_mean_pressure_pa=float(np.sum(np.exp(np.asarray(grid.to_nodal(dyn.log_surface_pressure)))[0]*pascal*weights)))
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    source=Path(__file__).resolve().parents[1]
    changed_sources={name for name,expected in config['source_sha256'].items()
        if not (source/name).exists() or digest(source/name)!=expected}
    report=dict(diagnostic_sha256=digest(Path(__file__)),checkpoint_path=str(restart),checkpoint_sha256=digest(restart),config_sha256=digest(args.run/'config.json'),
        changed_archived_sources=sorted(changed_sources),changed_source_sha256={name:digest(source/name)
            if (source/name).exists() else None for name in sorted(changed_sources)},
        dt_seconds=config['dt'],before=stats(state),controls={})
    for name,expected in config['reference_input_sha256'].items():
        path=args.data/('mars_topo_mola16.nc' if name=='mola' else name)
        if digest(path)!=expected: raise ValueError(f'archived input hash differs: {name}')
    common=dict(radiation_component=None,soil_diffusivity=1.e-7,no_forcing=True,
        hyperdiffusion_tau_seconds=tau,orography=orography)
    diffusivity=config.get('native_top_divergence_damping',{}).get('diffusivity_m2_s')
    if diffusivity is not None: common['divergence_diffusivity_m2_s']=diffusivity
    controls={'dry_terrain':{},'dry_flat':dict(orography=None),
        'dry_no_hyperdiffusion':dict(hyperdiffusion_tau_seconds=None)}
    baseline=None
    for name,control in controls.items():
        step=jax.jit(build_step(coords,specs,BODY,reference_radiative_forcing(),config['dt'],**(common|control)))
        start=time.monotonic();result=step(state);jax.block_until_ready(result)
        elapsed=time.monotonic()-start
        start=time.monotonic();repeat=step(state);jax.block_until_ready(repeat)
        replay_seconds=time.monotonic()-start
        for a,b in zip(jax.tree.leaves(result),jax.tree.leaves(repeat),strict=True):
            np.testing.assert_array_equal(a,b)
            assert np.isfinite(np.asarray(a)).all(),f'{name}: nonfinite replay'
        report['controls'][name]=dict(seconds_compile_and_step=elapsed,seconds_replay=replay_seconds,**stats(result))
        if name=='dry_terrain': baseline=step
    from run_gcm3d_ablation import reference_rest_state
    cold=initial_state(reference_rest_state(coords,specs,BODY,elevation),coords,specs,temperature_k=170.)
    report['cold_start']=dict(before=stats(cold),after=stats(baseline(cold)))
    native_mean_pressure=stats(cold)['surface_mean_pressure_pa']
    cold_steps=round(BODY.solar_day_seconds/config['dt'])
    report['cold_start_evolution']=dict(steps=cold_steps,seconds=cold_steps*config['dt'],controls={})
    length=float(specs.dimensionalize(1.,scales.units.meter).magnitude)
    resolved_height=np.asarray(grid.to_nodal(orography))*length
    cold_cases=[('mola',elevation,orography,False),('flat',np.zeros_like(elevation),None,False),
        ('mola_exponential_pressure',elevation,orography,True),
        ('mola_resolved_exponential_pressure',resolved_height,orography,True)]
    for name,height,terrain,exponential in cold_cases:
        value=initial_state(reference_rest_state(coords,specs,BODY,height),coords,specs,temperature_k=170.)
        if exponential:
            logp=-BODY.gravity_m_s2*height/(BODY.gas_constant_j_kg_k*170.)
            modal=grid.to_modal(jnp.asarray(logp)[None])
            projected=np.exp(np.asarray(grid.to_nodal(modal)))[0]
            logp=logp+np.log(native_mean_pressure/pascal/np.sum(projected*weights))
            value=value._replace(dynamics=dataclasses.replace(value.dynamics,
                log_surface_pressure=grid.to_modal(jnp.asarray(logp)[None])))
        if name=='mola_resolved_exponential_pressure':
            value=initial_state(reference_rest_state(coords,specs,BODY,elevation,
                pressure_scheme='sigma-balanced'),coords,specs,temperature_k=170.)
        step=build_step(coords,specs,BODY,reference_radiative_forcing(),config['dt'],**(common|dict(orography=terrain)))
        advance=jax.jit(lambda initial:jax.lax.fori_loop(0,cold_steps,lambda _,carry:step(carry),initial))
        final=advance(value);jax.block_until_ready(final)
        assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(final)),name
        report['cold_start_evolution']['controls'][name]=dict(before=stats(value),after=stats(final))
        if name=='mola_resolved_exponential_pressure':
            assert stats(final)['top_vector_rms_m_s']<1.e-6,'sigma-balanced cold start launched false winds'
            np.testing.assert_allclose(stats(final)['top_temperature_k'],170.,atol=1.e-7)
            np.testing.assert_allclose(stats(value)['surface_mean_pressure_pa'],native_mean_pressure,rtol=1.e-12)
    # Check the replay seam, including the complete saved state and a directional derivative.
    with tempfile.TemporaryDirectory() as directory:
        first=baseline(state);save_restart(first,Path(directory)/'restart.npz')
        restarted=baseline(load_restart(Path(directory)/'restart.npz'))
        continuous=baseline(first)
        for a,b in zip(jax.tree.leaves(restarted),jax.tree.leaves(continuous),strict=True):
            np.testing.assert_array_equal(a,b)
    def objective(scale):
        modified=state._replace(dynamics=dataclasses.replace(state.dynamics,divergence=state.dynamics.divergence*scale))
        final=baseline(modified).dynamics
        return jnp.sum(final.divergence**2)
    derivative=float(jax.jit(jax.grad(objective))(1.))
    epsilon=1.e-4
    finite_difference=float((objective(1.+epsilon)-objective(1.-epsilon))/(2*epsilon))
    np.testing.assert_allclose(derivative,finite_difference,rtol=1.e-5,atol=1.e-10)
    report['checks']=dict(deterministic=True,finite=True,restart_bitwise_equal=True,
        divergence_derivative=derivative,divergence_finite_difference=finite_difference)
    dsigma=jnp.asarray(np.diff(coords.vertical.boundaries))[:,None,None]
    def inventory(value):
        ps=jnp.exp(grid.to_nodal(value.dynamics.log_surface_pressure))[0]*pascal
        gas=ps[None]*dsigma/BODY.gravity_m_s2
        totals={name:float(jnp.sum(gas*ratio*weights)*BODY.surface_area_m2)
            for name,ratio in value.dynamics.tracers.items()}
        totals['water_with_surface']=float(water_inventory(value,coords,specs,BODY))
        totals['gas_with_surface_co2']=float(jnp.sum((ps+value.co2_ice[0]*pascal)*weights)*BODY.surface_area_m2/BODY.gravity_m_s2)
        dust=jnp.sum(gas*value.dynamics.tracers.get('dust_mass',jnp.zeros_like(gas)),axis=0)
        dust=dust+value.auxiliary_fields.get('dust_surface_kg_m2',jnp.zeros_like(ps))[0]
        totals['dust_with_surface']=float(jnp.sum(dust*weights)*BODY.surface_area_m2)
        return totals
    before_inventory=inventory(state);after_inventory=inventory(first)
    drift={name:(after_inventory[name]-value)/max(abs(value),1.) for name,value in before_inventory.items()}
    assert all(abs(value)<1.e-10 for value in drift.values()),drift
    assert all(np.asarray(value).min()>=0. for value in first.dynamics.tracers.values())
    report['checks'].update(dry_material_relative_drift=drift,local_tracers_nonnegative=True)
    surface=np.exp(np.asarray(grid.to_nodal(state.dynamics.log_surface_pressure)))[0]*pascal
    sigma=np.asarray(coords.vertical.boundaries)
    native_half=np.asarray(pressure_interfaces(config['layers'],705.,surface))
    radiation_half=np.asarray(radiation_interfaces(sigma,surface))
    report['pressure_pa']=dict(surface_min=float(surface.min()),surface_max=float(surface.max()),
        sigma_top_center_min=float((sigma[1]*surface/2).min()),sigma_top_center_max=float((sigma[1]*surface/2).max()),
        native_top_interface_min=float(native_half[0].min()),native_top_interface_max=float(native_half[0].max()),
        native_top_lower_min=float(native_half[1].min()),native_top_lower_max=float(native_half[1].max()),
        radiation_top_interface_min=float(radiation_half[0].min()),radiation_top_interface_max=float(radiation_half[0].max()))
    if args.reference_physics or args.damping_matrix:
        if (config.get('reference_protocol')!='present-Mars-clouds-v3.2' or config.get('initial_ls')!=0. or
                config.get('radiation_mode')!='12band' or any(key.endswith('_controls') or key.endswith('_switches') for key in config)):
            parser.error('full-physics replay requires the unchanged default reference preset')
        from run_gcm3d_ablation import reference_configuration
        _,forcing,radiation,options,_=reference_configuration(grid,args.data,config['dt'])
        options={key:value for key,value in options.items() if not key.startswith('initial_')}
        if diffusivity is not None: options['divergence_diffusivity_m2_s']=diffusivity
        step=jax.jit(build_step(coords,specs,BODY,forcing,config['dt'],radiation_component=radiation,
            orography=orography,hyperdiffusion_tau_seconds=tau,**options))
        # The archived checkpoint is on an update boundary; shift both clocks equally.
        off_clock=state._replace(dynamics=dataclasses.replace(state.dynamics,
            sim_time=state.dynamics.sim_time+specs.nondimensionalize(config['dt']*scales.units.second)))
        report['radiation_control_clock_shift_seconds']=config['dt']
        for name,value in [('held_radiation',off_clock),('refreshed_radiation',off_clock._replace(auxiliary_fields={
                **off_clock.auxiliary_fields,'radiation_initialized':jnp.zeros_like(state.auxiliary_fields['radiation_initialized'])}))]:
            result=step(value);jax.block_until_ready(result)
            assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(result)),name
            report['controls'][name]=stats(result)
    if args.damping_matrix:
        from run_gcm3d_ablation import reference_divergence_diffusivity
        profile,metadata=reference_divergence_diffusivity(config['truncation'],config['layers'])
        eigen=-np.asarray(grid.laplacian_eigenvalues)
        alpha=np.asarray(profile[:2])[:,None]*metadata['native_advective_substep_seconds']*eigen/BODY.radius_m**2
        assert np.all(1.-alpha>0.),'native explicit background comparison is unstable at this resolution'
        native_steps=metadata['native_advective_substeps']
        exact=np.exp(-alpha*native_steps)
        errors=[float(np.max(abs((1.-alpha/refinement)**(native_steps*refinement)-exact))) for refinement in (1,2,4,8)]
        assert all(a>b for a,b in zip(errors,errors[1:])),errors
        matrix=dict(metadata=metadata,native_euler_refinement_max_error=errors,cold_one_sol={},coupled={})
        report['damping_matrix']=matrix
        for scheme in ('native','sigma-balanced'):
            for enabled in (False,True):
                name=f'{scheme}_d2_{enabled}'
                value=initial_state(reference_rest_state(coords,specs,BODY,elevation,pressure_scheme=scheme),
                    coords,specs,temperature_k=170.)
                selected=profile if enabled else None
                step=build_step(coords,specs,BODY,forcing,config['dt'],**(common|dict(divergence_diffusivity_m2_s=selected)))
                final=jax.jit(lambda value:jax.lax.fori_loop(0,cold_steps,lambda _,carry:step(carry),value))(value)
                jax.block_until_ready(final)
                matrix['cold_one_sol'][name]=stats(final)
                assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(final)),name
        for enabled in (False,True):
            name=f'd2_{enabled}'
            selected=profile if enabled else None
            step=jax.jit(build_step(coords,specs,BODY,forcing,config['dt'],radiation_component=radiation,
                orography=orography,hyperdiffusion_tau_seconds=tau,
                **(options|dict(divergence_diffusivity_m2_s=selected))))
            value=state;records={}
            for count in range(1,33):
                value=step(value)
                if count in (1,8,32): records[str(count)]=stats(value)
            assert all(np.isfinite(np.asarray(leaf)).all() for leaf in jax.tree.leaves(value)),name
            totals=inventory(value)
            records['material_relative_drift']={key:(totals[key]-before_inventory[key])/max(abs(before_inventory[key]),1.)
                for key in ('water_with_surface','gas_with_surface_co2','dust_with_surface')}
            assert all(abs(value)<1.e-10 for value in records['material_relative_drift'].values()),records
            assert all(np.asarray(value).min()>=0. for value in value.dynamics.tracers.values())
            matrix['coupled'][name]=records
    if args.native:
        import xarray as xr
        means=sorted((args.run/'checkpoints').glob('mean_*.nc'))
        with xr.open_dataset(means[-1]) as mean:
            archived_rms=rms(np.asarray(mean.eastward_wind)[0,0].T)
        native={}
        for case in ('C24','C48'):
            path=args.native/f'{case}-first-average.npz'
            with np.load(path,allow_pickle=False) as opened:
                native[case]=dict(top_mean_u_rms_m_s=float(np.sqrt(np.sum(opened['ucomp'][:,0]**2*opened['area'])/opened['area'].sum())),sha256=digest(path))
        report['archived_time_means']=dict(mean_path=str(means[-1].relative_to(args.run)),mean_sha256=digest(means[-1]),
            top_mean_u_rms_m_s=archived_rms,native=native)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
    print(json.dumps(report,indent=2,allow_nan=False))
    if args.assert_mean_ratio is not None:
        assert archived_rms<=args.assert_mean_ratio*max(item['top_mean_u_rms_m_s'] for item in native.values()),'archived top mean zonal wind exceeds native threshold; one-step replay cannot certify a climate fix'


if __name__=='__main__':
    main()
