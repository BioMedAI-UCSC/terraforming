"""Matched climate experiments and fixed-memory annual warming accounting.

Intervals use linear interpolation between area-weighted endpoint differences.
Each annual integral is split at an exact orbital-period boundary. Three annual
means, a partial-year integral and the previous endpoint suffice for restart;
no per-step history is retained in the experiment state.
"""
from typing import NamedTuple
import dataclasses
import math
import numpy as np
from src.framework.gcm._dinosaur import jnp
from .experiment import diagnostics
from .state import MASS, NUMBER


class AnnualWarming(NamedTuple):
    anchor_s: object
    last_s: object
    previous_difference_k: object
    partial_integral_k_s: object
    completed_years: object
    last_three_means_k: object
    total_integral_k_s: object


class PairedState(NamedTuple):
    off: object
    passive: object
    active: object
    warming: AnnualWarming


def initial_warming(elapsed_s, difference_k=0.):
    zero=jnp.asarray(0.,dtype=jnp.float64)
    return AnnualWarming(jnp.asarray(elapsed_s),jnp.asarray(elapsed_s),
        jnp.asarray(difference_k),zero,jnp.asarray(0,dtype=jnp.int64),jnp.zeros(3),zero)


def accumulate(warming,stop_s,difference_k,orbital_period_s):
    """Integrate nonuniform intervals, including those crossing annual bounds."""
    start=float(warming.last_s);stop=float(stop_s);year=float(orbital_period_s)
    previous=float(warming.previous_difference_k);difference=float(difference_k)
    if not all(math.isfinite(x) for x in (start,stop,year,previous,difference)) or year<=0 or stop<=start:
        raise ValueError('warming requires a finite forward interval and positive orbital period')
    anchor=float(warming.anchor_s);completed=int(warming.completed_years)
    partial=float(warming.partial_integral_k_s);means=np.asarray(warming.last_three_means_k).copy()
    cursor=start
    while cursor<stop:
        boundary=anchor+(completed+1)*year
        if boundary<=cursor:
            raise ValueError('annual accumulator is inconsistent with elapsed time')
        end=min(stop,boundary)
        a=previous+(difference-previous)*(cursor-start)/(stop-start)
        b=previous+(difference-previous)*(end-start)/(stop-start)
        partial+=(end-cursor)*(a+b)/2
        if end==boundary:
            means=np.r_[means[1:],partial/year]
            partial=0.;completed+=1
        cursor=end
    total=float(warming.total_integral_k_s)+(stop-start)*(previous+difference)/2
    return AnnualWarming(warming.anchor_s,jnp.asarray(stop),jnp.asarray(difference),
        jnp.asarray(partial),jnp.asarray(completed,dtype=jnp.int64),jnp.asarray(means),jnp.asarray(total))


def warming_report(warming,orbital_period_s):
    completed=int(warming.completed_years)
    elapsed=float(warming.last_s-warming.anchor_s)
    annual=np.asarray(warming.last_three_means_k)[-min(completed,3):].tolist() if completed else []
    ready=completed>=3
    return dict(completed_mars_years=completed,partial_mars_year=(
        elapsed-completed*orbital_period_s)/orbital_period_s,
        paired_time_mean_warming_k=None if elapsed==0 else float(warming.total_integral_k_s)/elapsed,
        consecutive_annual_warming_k=annual,complete_three_mars_years=ready,
        interpretation='model comparison only; baseline climate validation is a separate requirement',
        sustained_thresholds_k={str(k):bool(ready and all(x>=k for x in annual)) for k in (5,10,20)},
        source_comparison_thresholds_k={str(k):dict(three_annual_means_reached=bool(
            ready and all(x>=k for x in annual)),definition='three area/time-mean annual differences from matched control; source setup equivalence unverified') for k in (4,35)})


def prepare_pair(**options):
    """Fork off/passive/active from exactly the same initial climate fields."""
    from .pilot import prepare
    if 'mode' in options: raise ValueError('paired mode controls off/passive/active internally')
    states={};steps={};configurations={};context=None
    for mode in ('off','passive','active'):
        initial=dict(options)
        if mode=='off': initial['initial_burden_kg_m2']=0.
        states[mode],steps[mode],current=prepare(mode=mode,**initial)
        configurations[mode]=current.pop('configuration')
        if context is None: context=current
    # Copy the common climate explicitly. Only the seeded particle tracers and
    # their initial material ledgers differ in the control.
    common=states['active'].climate
    for mode in ('off','passive'):
        tracers=dict(common.dynamics.tracers)
        for key in (MASS,NUMBER): tracers[key]=states[mode].climate.dynamics.tracers[key]
        climate=common._replace(dynamics=dataclasses.replace(common.dynamics,tracers=tracers))
        states[mode]=states[mode]._replace(climate=climate)
    configuration=dict(schema='paired-aerosol-v1',cases=configurations,
        annual_definition='orbital periods from fork time; trapezoidal endpoint integration split at year boundaries')
    state=PairedState(states['off'],states['passive'],states['active'],initial_warming(0.))
    return state,steps,{**context,'configuration':configuration}


def paired_report(state,*,orbital_period_s,**context):
    cases={mode:diagnostics(getattr(state,mode),**context) for mode in ('off','passive','active')}
    active=dict(cases['active'])
    return {**active,'cases':cases,'warming':warming_report(state.warming,orbital_period_s)}


def stream_pair(state,steps,n_steps,coords,specs,body,particle,*,orbital_period_s,
                report_every=100,should_stop=lambda:False,output=None):
    """Advance all cases together and retain all three last-valid states on failure."""
    import json
    if n_steps<0 or report_every<1: raise ValueError('invalid run/report interval')
    context=dict(coords=coords,specs=specs,body=body,particle=particle)
    for index in range(n_steps):
        if should_stop():
            report=paired_report(state,orbital_period_s=orbital_period_s,**context)
            report['status']='stopped'
            yield state,report
            return
        candidates={mode:steps[mode](getattr(state,mode)) for mode in ('off','passive','active')}
        reports={mode:diagnostics(candidate,**context) for mode,candidate in candidates.items()}
        failed=[mode for mode,r in reports.items() if r['status']=='invalid_regime' or any(
            not math.isfinite(v) for v in r.values() if isinstance(v,float))]
        if failed:
            report=paired_report(state,orbital_period_s=orbital_period_s,**context)
            report.update(status='invalid_regime' if any(reports[m]['status']=='invalid_regime' for m in failed)
                else 'numerical_failure',failed_cases=failed,last_valid_elapsed_seconds=float(state.active.elapsed_seconds))
            report['warming']=warming_report(state.warming,orbital_period_s)
            yield state,report
            return
        elapsed=float(candidates['active'].elapsed_seconds)
        if any(float(c.elapsed_seconds)!=elapsed for c in candidates.values()):
            raise ValueError('paired clocks diverged')
        difference=reports['active']['surface_temperature_k']-reports['off']['surface_temperature_k']
        warming=accumulate(state.warming,elapsed,difference,orbital_period_s)
        state=PairedState(**candidates,warming=warming)
        if (index+1)%report_every==0 or index==n_steps-1:
            report={**reports['active'],'cases':reports,'warming':warming_report(warming,orbital_period_s),
                'status':'completed' if index==n_steps-1 else 'running'}
            if output is not None:
                output.write(json.dumps(report,allow_nan=False)+'\n');output.flush()
            yield state,report
