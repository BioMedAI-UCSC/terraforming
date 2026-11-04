"""Explicit finite parameter sweeps for isolated aerosol lifetime screens."""
import copy
import itertools
import json
from .specs import aluminium_rod,iron_rod
from .screening import CollectorScenario,lifetime_screen


def screening_cases(definition):
    """Yield at most 64 source-labelled cases; unsupported cases remain visible.

    Input is a JSON-compatible mapping with material, base parameters and sweep
    axes. This generator retains no time histories and makes no warming claim.
    """
    json.dumps(definition,allow_nan=False)
    if not isinstance(definition,dict) or set(definition)-{'material','base','sweep'}:
        raise ValueError('study accepts material, base and sweep fields')
    material=definition.get('material','aluminium')
    if material not in ('aluminium','iron'): raise ValueError('source-backed screening baseline must be aluminium or iron rods')
    base=definition.get('base');sweep=definition.get('sweep',{})
    if not isinstance(base,dict) or not isinstance(sweep,dict): raise ValueError('base and sweep must be mappings')
    allowed={'charge_e','sticking_efficiency','fractal_dimension','collector_number_m3','max_monomers','dt_s'}
    if set(sweep)-allowed: raise ValueError('unsupported sweep axis')
    count=1
    for values in sweep.values():
        if not isinstance(values,list) or not values: raise ValueError('sweep axes must be nonempty lists')
        count*=len(values)
    if count>64: raise ValueError('study is limited to 64 explicit cases')
    particle=aluminium_rod() if material=='aluminium' else iron_rod()
    keys=list(sweep)
    for index,values in enumerate(itertools.product(*(sweep[k] for k in keys))):
        parameters=copy.deepcopy(base)
        axes=dict(zip(keys,values))
        for key,value in axes.items():
            if key!='collector_number_m3': parameters[key]=value
        try:
            if 'collector_number_m3' in axes:
                if not isinstance(parameters.get('collector'),dict): raise ValueError('collector-number sweep requires a sourced collector scenario')
                parameters['collector']['number_m3']=axes['collector_number_m3']
            options=copy.deepcopy(parameters)
            if options.get('collector') is not None: options['collector']=CollectorScenario(**options['collector'])
            result=lifetime_screen(particle,**options)
        except (ValueError,TypeError,OverflowError) as exc:
            result=dict(status='unsupported_configuration',error=str(exc),
                radiatively_effective_residence_time_s=None)
        yield dict(case=index,axes=axes,parameters=parameters,result=result)
