"""Shared continuous aerosol run adapter for CLI and HTTP server."""
from pathlib import Path
import json
import math
import numpy as np


def run(config,planet,output_dir,*,callback=lambda report,fields:None,
        should_stop=lambda:False,resume=None,diurnal=True,initial_ls_deg=0.,
        co2_lw_scale=1.,dust_lw_scale=1.,surface_exchange_multiplier=1.):
    import jax
    jax.config.update('jax_enable_x64',True)
    from src.terraforming.aerosols.pilot import prepare
    from src.terraforming.aerosols.checkpoint import save,load
    from src.terraforming.aerosols.reporting import stream_run
    from src.terraforming.aerosols.state import burdens,gas_mass
    from src.framework.gcm._dinosaur import scales
    options=config.model_dump(exclude={'duration_s'})
    state,step,context=prepare(**options,surface_temperature_k=planet.surface_temperature,
        surface_pressure_pa=planet.surface_pressure,albedo=planet.albedo,diurnal=diurnal,
        initial_ls_deg=initial_ls_deg,co2_lw_scale=co2_lw_scale,dust_lw_scale=dust_lw_scale,
        surface_exchange_multiplier=surface_exchange_multiplier)
    identity=context.pop('configuration')
    if resume is not None: state=load(resume,state,identity)
    remaining=config.duration_s-float(state.elapsed_seconds)
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
        for state,report in stream_run(state,step,n_steps,**context,report_every=report_every,
                output=stream,should_stop=should_stop):
            from src.terraforming.aerosols.state import quadrature_weights
            weights=np.asarray(quadrature_weights(context['coords']))
            gas=gas_mass(state.climate,context['coords'],context['specs'],context['body'])
            report['pressure_pa']=float(np.sum(np.sum(gas,axis=0)*context['body'].gravity_m_s2*weights))
            pressure_scale=float(context['specs'].dimensionalize(1.,scales.units.pascal).magnitude)
            report['ice_mass_kg']=float(np.sum(np.asarray(state.climate.co2_ice)[0]*pressure_scale*
                weights)*context['body'].surface_area_m2/context['body'].gravity_m_s2)
            save(out/'restart.npz',state,identity)
            final_report=report
            callback(report,fields(state))
            if report['status'] in ('invalid_regime','numerical_failure'):
                raise RuntimeError(f"aerosol run stopped: {report['status']}; last valid state saved")
    return final_report
