"""Exercise fitting/checkpoint/evaluation wiring with a known identifiable oracle."""
from pathlib import Path
import json
import sys
from types import SimpleNamespace

import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[3]/'scripts'))
import run_differentiable_experiments as runner


def test_physical_runner_writes_convergence_and_actual_selected_controls(tmp_path,monkeypatch):
    d=runner.d
    d.jax.config.update('jax_enable_x64',True)
    truth=d.jnp.asarray([.6,.8,1.4])
    class Model:
        initial=d.jnp.asarray(0.)
        forcing=None
        weights=d.jnp.ones((1,1))
        coords=None
        specs=None
        def trajectory(self,seconds,dt):
            return lambda controls:d.jnp.broadcast_to(controls[None,:,None,None,None],(1,3,12,1,1))
        def loss(self,forecast,target,scale):
            return d.jnp.mean(((forecast-target)/scale[None,:,None,None,None])**2)
    model=Model()
    target=model.trajectory([1.],1.)(truth)
    monkeypatch.setattr(runner,'target_arrays',lambda *args:(np.array([1.]),target))
    monkeypatch.setattr(runner,'state_features',lambda *args:d.jnp.zeros((1,1,21)))
    windows=[dict(id=str(i),split=s,block=f'my{i}_q0',reference_kind='synthetic')
             for i,s in enumerate(['train','train','validation','test'])]
    args=SimpleNamespace(dt=1.,seed=0,heating_bound=2e-4,epsilons=[1e-3,1e-4],
                         output=tmp_path,optimizer='lbfgsb',updates=100,max_evaluations=100,
                         gtol=1e-6,ftol=1e-12,starts=1,learning_rate=.001)
    runner.fit(model,args,dict(windows=windows),False)
    report=json.loads((tmp_path/'acceptance.json').read_text())
    assert report['optimization']['stationary']
    np.testing.assert_allclose(report['physical_controls'],truth,atol=1e-6)
    assert report['uncertainty']['interval_95_k'] is None  # Only one synthetic test block.
    assert (tmp_path/'optimization.json').is_file()
    with np.load(tmp_path/'checkpoint.npz') as checkpoint:
        np.testing.assert_allclose(d.LOWER_BOUNDS+checkpoint['best']*(d.UPPER_BOUNDS-d.LOWER_BOUNDS),truth,atol=1e-6)
