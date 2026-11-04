"""Bounded paired aerosol pilots; outputs cannot certify long-term warming.

Run with the GCM extra: python scripts/run_aerosol_pilot_matrix.py --output DIR
Every case uses the same uniform initial climate and declared radiation setup.
"""
from pathlib import Path
import argparse
import json
import math
import jax
from src.terraforming.aerosols.pilot import prepare
from src.terraforming.aerosols.reporting import stream_run
from src.terraforming.aerosols.checkpoint import save


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seconds',type=float,default=300.)
    args=parser.parse_args()
    if not 0<args.seconds<=88775.244:
        parser.error('bounded pilot duration must be within one sol')
    jax.config.update('jax_enable_x64',True)
    args.output.mkdir(parents=True,exist_ok=True)
    matrix=[]
    cases=[('off','off',60.,4,1.),('passive','passive',60.,4,1.),
           ('active','active',60.,4,1.),('double_dose','active',60.,4,2.),
           ('off_half_dt','off',30.,4,1.),('half_dt','active',30.,4,1.),
           ('off_double_layers','off',60.,8,1.),('double_layers','active',60.,8,1.)]
    for name,mode,dt,layers,dose in cases:
        state,step,context=prepare(mode=mode,dt_s=dt,n_layers=layers,sources=[
            dict(start_s=0.,stop_s=args.seconds/2,rate=5*dose,unit='kg/s')])
        config=context.pop('configuration')
        folder=args.output/name;folder.mkdir(exist_ok=True)
        with (folder/'history.ndjson').open('w') as stream:
            for state,report in stream_run(state,step,math.ceil(args.seconds/dt),
                    **context,report_every=1,output=stream):
                if report['status'] in ('invalid_regime','numerical_failure'): break
        save(folder/'restart.npz',state,config)
        matrix.append(dict(case=name,mode=mode,dt_s=dt,n_layers=layers,dose=dose,**report))
    baselines={(x['dt_s'],x['n_layers']):x['surface_temperature_k'] for x in matrix if x['mode']=='off'}
    for entry in matrix:
        entry['paired_final_surface_difference_k']=entry['surface_temperature_k']-baselines[(entry['dt_s'],entry['n_layers'])]
        entry['confirmed_long_term_warming']=False
        entry['interpretation']='short numerical pilot; no equilibrated-climate/source-warming validation'
    (args.output/'matrix.json').write_text(json.dumps(matrix,indent=2,allow_nan=False))
    print(json.dumps(matrix,indent=2,allow_nan=False))


if __name__=='__main__': main()
