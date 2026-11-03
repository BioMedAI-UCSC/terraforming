"""Bounded streaming reports and complete-window paired warming metrics."""
import json
import math
from collections import deque
from .experiment import diagnostics


class WarmingWindow:
    """Time-integrated paired differences; incomplete three-year windows fail."""
    def __init__(self,orbital_period_s):
        if not math.isfinite(orbital_period_s) or orbital_period_s<=0:
            raise ValueError('orbital period must be positive SI seconds')
        self.duration=3*orbital_period_s
        self.year=orbital_period_s
        self.samples=deque()
        self.last_time=None

    def add(self,start_s,stop_s,off_k,active_k):
        if not all(math.isfinite(x) for x in (start_s,stop_s,off_k,active_k)) or stop_s<=start_s:
            raise ValueError('finite forward interval required')
        if self.last_time is not None and start_s!=self.last_time:
            raise ValueError('paired warming intervals must be continuous')
        self.samples.append((start_s,stop_s,active_k-off_k));self.last_time=stop_s
        cutoff=stop_s-self.duration
        while self.samples and self.samples[0][1]<=cutoff: self.samples.popleft()
        # Samples must be output intervals (e.g. sols), never individual GCM steps.
        covered=sum(max(0,b-max(a,cutoff)) for a,b,_ in self.samples)
        mean=sum(max(0,b-max(a,cutoff))*x for a,b,x in self.samples)/covered
        complete=covered>=self.duration*(1-1e-12)
        annual=[]
        if complete:
            for i in range(3):
                end=stop_s-i*self.year;start=end-self.year
                annual.append(sum(max(0,min(b,end)-max(a,start))*x
                    for a,b,x in self.samples)/self.year)
        return dict(complete_three_mars_years=complete,paired_warming_k=mean,
            consecutive_annual_warming_k=annual,
            sustained_thresholds_k={str(k):bool(complete and all(x>=k for x in annual)) for k in (5,10,20)})

    def snapshot(self):
        return dict(orbital_period_s=self.year,samples=list(self.samples),last_time=self.last_time)

    @classmethod
    def restore(cls,snapshot):
        window=cls(snapshot['orbital_period_s'])
        for a,b,x in snapshot['samples']: window.add(a,b,0.,x)
        if window.last_time!=snapshot['last_time']: raise ValueError('invalid warming checkpoint')
        return window


def stream_run(state,step,n_steps,coords,specs,body,particle,*,report_every=100,
               output=None,should_stop=lambda:False):
    """Yield bounded reports; state advances continuously and no step history is kept.

    Invalid steps do not replace the last-valid state. Caller owns checkpoints
    and final-state storage. Each yield contains the current complete state.
    """
    if n_steps<0 or report_every<1: raise ValueError('invalid run/report interval')
    for index in range(n_steps):
        if should_stop():
            report=diagnostics(state,coords,specs,body,particle);report['status']='stopped'
            yield state,report
            return
        candidate=step(state)
        report=diagnostics(candidate,coords,specs,body,particle)
        if report['status']=='invalid_regime' or not all(
            math.isfinite(v) for v in report.values() if isinstance(v,float)):
            failure='invalid_regime' if bool(candidate.ledger.invalid) else 'numerical_failure'
            attempted_time=float(candidate.elapsed_seconds)
            report=diagnostics(state,coords,specs,body,particle)
            report['status']=failure
            report['attempted_elapsed_seconds']=attempted_time
            report['last_valid_elapsed_seconds']=float(state.elapsed_seconds)
            yield state,report
            return
        state=candidate
        if (index+1)%report_every==0 or index==n_steps-1:
            report['status']='completed' if index==n_steps-1 else 'running'
            if output is not None:
                output.write(json.dumps(report,allow_nan=False)+'\n');output.flush()
            yield state,report
