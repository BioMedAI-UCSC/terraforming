"""Run in a fresh process so JAX can create four virtual CPU devices."""
import os
from pathlib import Path
import subprocess
import sys


def test_four_device_average_and_resume(tmp_path):
    root = Path(__file__).resolve().parents[3]
    code = r'''
import pickle
import sys
import numpy as np
sys.path.insert(0, "scripts")
from train_neural_pbl import synchronized_update
from neural_macda import atomic_bytes
from src.framework.gcm._dinosaur import jax, jnp
jax.config.update("jax_enable_x64", True)
devices = jax.local_devices()
assert len(devices) == 4
loss = lambda p, x: jnp.sum((p - x)**2)
update = synchronized_update(loss, .01, devices)
p = jnp.zeros((4, 2))
z = jnp.zeros_like(p)
count = jnp.zeros(4, dtype=int)
x = jnp.array([[1., -2.], [3., 4.], [-5., 6.], [7., -8.]])
first = update(p, z, z, count, x)
for value in first[:4]:
    np.testing.assert_allclose(value, jnp.broadcast_to(value[0], value.shape), atol=1e-14)
# Compare against the gradient of the global mean loss on a single device.
single = synchronized_update(lambda p, x: jnp.mean(jax.vmap(lambda y: loss(p, y))(x)), .01, devices[:1])
reference = single(p[:1], z[:1], z[:1], count[:1], x[None])
for parallel, serial in zip(first, reference):
    np.testing.assert_allclose(parallel[0], serial[0], rtol=1e-12, atol=1e-14)
path = sys.argv[1]
atomic_bytes(path, pickle.dumps(jax.device_get(first[:4])))
restored = pickle.loads(open(path, "rb").read())
continued = update(*first[:4], x)
resumed = update(*restored, x)
for a, b in zip(continued, resumed):
    np.testing.assert_array_equal(a, b)
print("four-device gradient averaging and optimizer resume passed")
'''
    env = dict(os.environ, XLA_FLAGS="--xla_force_host_platform_device_count=4", JAX_PLATFORMS="cpu")
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path / "state.pkl")],
                            cwd=root, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 0, result.stdout + result.stderr
