"""Shared continuous aerosol run adapter for CLI and HTTP server."""
from pathlib import Path
import json
import math
import numpy as np


def run(config,planet,output_dir,*,callback=lambda report,fields:None,
        should_stop=lambda:False,resume=None,diurnal=True,initial_ls_deg=0.,
        co2_lw_scale=1.,dust_lw_scale=1.,surface_exchange_multiplier=1.):
    if planet.composition is not None:
        raise ValueError('aerosol pilot currently requires the fixed Mars CO2 gas composition')
    import jax
    jax.config.update('jax_enable_x64',True)
    from src.interventions.aerosols.pilot import prepare
    from src.interventions.aerosols.checkpoint import save,load,read_configuration
    from src.interventions.aerosols.reporting import stream_run
    from src.interventions.aerosols.state import burdens,gas_mass
    from src.framework.gcm._dinosaur import scales
    options=config.model_dump(exclude={'duration_s','paired'})
    # Resume restores all prognostic fields from the experiment archive, so it
    # must not depend on reopening the original physical-start file.
    archived=read_configuration(resume) if resume is not None else None
    if resume is not None: options['initial_climate']=None
    builder=prepare
    if config.paired:
        from src.interventions.aerosols.paired import prepare_pair
        builder=prepare_pair
        options.pop('mode')
    state,step,context=builder(**options,surface_temperature_k=planet.surface_temperature,
        surface_pressure_pa=planet.surface_pressure,albedo=planet.albedo,diurnal=diurnal,
        initial_ls_deg=initial_ls_deg,co2_lw_scale=co2_lw_scale,dust_lw_scale=dust_lw_scale,
        surface_exchange_multiplier=surface_exchange_multiplier)
    identity=context.pop('configuration')
    if archived is not None:
        seeds=('initial_climate_sha256','initial_absolute_sim_time_nd',
               'initial_surface_temperature_k','initial_surface_pressure_pa')
        current_cases=identity.get('cases',{'single':identity})
        archived_cases=archived.get('cases',{'single':archived})
        for name,current_case in current_cases.items():
            original=archived_cases.get(name,{})
            if original.get('initial_climate_sha256') is not None:
                # Only initial-state provenance is restored. Body/grid/forcing,
                # particle assets and schedules still match the caller's config.
                for key in seeds: current_case[key]=original[key]
    if resume is not None: state=load(resume,state,identity)
    current=state.active if config.paired else state
    remaining=config.duration_s-float(current.elapsed_seconds)
    if remaining<=0: raise ValueError('requested end time must follow checkpoint time')
    # Full fixed steps only; duration denotes a requested minimum end time.
    n_steps=math.ceil(remaining/config.dt_s)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=True)
    (out/'configuration.json').write_text(json.dumps(identity,indent=2,allow_nan=False))
    def fields(current):
        coords=context['coords'];specs=context['specs'];body=context['body']
        mass,_=burdens(current.climate,coords,specs,body)
        gas=gas_mass(current.climate,coords,specs,body)
        def field(value,label,units):
            array=np.asarray(value).T
            return dict(label=label,units=units,min=float(array.min()),max=float(array.max()),data=array.tolist())
        return dict(lon=np.degrees(coords.horizontal.longitudes).tolist(),
            lat=np.degrees(coords.horizontal.latitudes).tolist(),sigma=np.asarray(coords.vertical.centers).tolist(),
            maps=dict(surface_temperature=field(current.climate.surface_temperature[0],'Surface temperature','K'),
                surface_pressure=field(np.sum(gas,axis=0)*body.gravity_m_s2,'Surface pressure','Pa'),
                aerosol_burden=field(np.sum(mass,axis=0),'Suspended aerosol','kg/m²'),
                aerosol_deposit=field(current.ledger.deposited_mass_kg_m2,'Deposited aerosol','kg/m²')),
            sections={})
    report_every=max(1,math.ceil(n_steps/500))
    final_report=None
    with (out/'history.ndjson').open('a' if resume else 'w') as stream:
        runner=stream_run;extra={}
        if config.paired:
            from src.interventions.aerosols.paired import stream_pair
            runner=stream_pair
            extra['orbital_period_s']=identity['cases']['active']['forcing']['orbital_period_s']
        for state,report in runner(state,step,n_steps,**context,report_every=report_every,
                output=stream,should_stop=should_stop,**extra):
            current=state.active if config.paired else state
            from src.interventions.aerosols.state import quadrature_weights
            weights=np.asarray(quadrature_weights(context['coords']))
            gas=gas_mass(current.climate,context['coords'],context['specs'],context['body'])
            report['pressure_pa']=float(np.sum(np.sum(gas,axis=0)*context['body'].gravity_m_s2*weights))
            pressure_scale=float(context['specs'].dimensionalize(1.,scales.units.pascal).magnitude)
            report['ice_mass_kg']=float(np.sum(np.asarray(current.climate.co2_ice)[0]*pressure_scale*
                weights)*context['body'].surface_area_m2/context['body'].gravity_m_s2)
            save(out/'restart.npz',state,identity)
            final_report=report
            headline=fields(current)
            callback(report,headline)
            if report['status'] in ('completed','stopped','invalid_regime','numerical_failure'):
                from cli.aerosol_results import write_results
                case_fields={mode:fields(getattr(state,mode)) for mode in ('off','passive','active')} if config.paired else {config.mode:headline}
                final_report=write_results(out,report,case_fields,identity)
            if report['status'] in ('invalid_regime','numerical_failure'):
                details=report.get('failure_diagnostics',{})
                reasons=details.get('reasons') or {key:value.get('reasons',[]) for key,value in details.items() if isinstance(value,dict)}
                raise RuntimeError(f"aerosol run stopped: {report['status']}; reasons={reasons}; "
                    f"last valid state saved; see {out/'summary.json'}")
    return final_report
