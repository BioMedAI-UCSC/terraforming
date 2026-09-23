"""Closure invariants, native cache and coupled reverse-mode training checks."""
import dataclasses
from pathlib import Path
import sys

import numpy as np
import pytest

pytest.importorskip("dinosaur")
from src.framework.gcm._dinosaur import jax, jnp
from src.framework.physics import gcm as physics
from src.framework.physics.neural_pbl import Closure, initialize, multipliers

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))
from train_neural_pbl import BODY, Experiment, synchronized_update
from neural_macda import NativeCache, windows, validate, UNITS, InvalidNativeWindow

jax.config.update("jax_enable_x64", True)


def synthetic(experiment, count=3):
    import xarray as xr
    grid = experiment.coords.horizontal
    shape = (count, experiment.coords.vertical.layers, *grid.nodal_shape[::-1])
    t = np.broadcast_to(np.linspace(190., 225., shape[1])[None, :, None, None], shape).copy()
    u = np.broadcast_to(np.linspace(2., 20., shape[1])[None, :, None, None], shape).copy()
    data = {name: (("time", "lev", "lat", "lon"), value, {"units": UNITS[name]})
            for name, value in (("temp", t), ("uwind", u), ("vwind", u / 3))}
    for name, value in (("psurf", 610.), ("tsurf", 220.), ("co2ice", 0.), ("coldust", .3)):
        data[name] = (("time", "lat", "lon"), np.full((count, *shape[2:]), value), {"units": UNITS[name]})
    data.update(Ls=("time", np.zeros(count)), MY_Ls=("time", np.full(count, 24)))
    return xr.Dataset(data, coords={"time": np.arange(count) / 12,
                      "lev": experiment.coords.vertical.centers,
                      "lat": np.degrees(grid.latitudes), "lon": np.degrees(grid.longitudes)})


def test_identity_bounds_and_disabled():
    e = Experiment(spinup=0, layers=4)
    state, context = e.snapshot(synthetic(e), 0)
    closure = Closure(initialize(jax.random.key(0)), jnp.zeros(13), jnp.ones(13))
    features = e.features(state, context)
    np.testing.assert_array_equal(multipliers(closure, features), np.ones(features.shape[:-1] + (2,)))
    baseline = physics.pbl_vertical_diffusion_tendencies(state, e.coords, e.specs, BODY, e.forcing(context))
    corrected = physics.pbl_vertical_diffusion_tendencies(state, e.coords, e.specs, BODY, e.forcing(context), neural_closure=closure)
    for a, b in zip(jax.tree_util.tree_leaves(baseline), jax.tree_util.tree_leaves(corrected)):
        np.testing.assert_array_equal(a, b)
    off = dataclasses.replace(e.forcing(context), pbl_diffusion_enabled=False)
    disabled = physics.pbl_vertical_diffusion_tendencies(state, e.coords, e.specs, BODY, off, neural_closure=closure)
    assert all(np.all(np.asarray(x) == 0) for x in jax.tree_util.tree_leaves(disabled))
    for sign in (-1, 1):
        values = multipliers(Closure({"constant": jnp.full(2, sign * 1e6)}, closure.mean, closure.scale), features)
        assert np.all(np.asarray(values) >= .5) and np.all(np.asarray(values) <= 2.)


def test_conservative_exchange_with_corrected_rates():
    e = Experiment(layers=4)
    field = jnp.arange(24.).reshape(4, 3, 2)
    weights = jnp.asarray([.1, .2, .3, .4])
    for factor in (.5, 1., 2.):
        tendency, result = physics._implicit_vertical_diffusion_tendency(
            field, jnp.ones((3, 3, 2)) * factor / 1800, weights, 300., e.specs)
        np.testing.assert_allclose(np.sum(np.asarray(tendency) * np.asarray(weights)[:, None, None], 0), 0, atol=1e-11)
        assert np.asarray(result).min() >= field.min() and np.asarray(result).max() <= field.max()


def test_split_and_cache(tmp_path):
    e = Experiment(spinup=0)
    ds = synthetic(e)
    validate(ds, "train", prepared=True)
    with pytest.raises(ValueError, match="split"):
        validate(ds, "validation")
    bad = ds.assign_coords(time=[0., 1., 2.])
    with pytest.raises(ValueError, match="contiguous"):
        validate(bad, "train")
    with pytest.raises(ValueError, match="immutable"):
        NativeCache(tmp_path, "main")
    cache = NativeCache(tmp_path, "a" * 40)
    cache.source = ds  # Offline source exercises regridding and actual atomic cache writes.
    loaded = cache.get((0, 3), "train")
    assert loaded.sizes == ds.sizes
    cache.source = None
    np.testing.assert_allclose(cache.get((0, 3), "train").temp, loaded.temp)
    path = next(cache.root.glob("*.nc"))
    with open(path, "ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        cache.get((0, 3), "train")
    assert all(stop <= 32092 or start >= 40320 for start, stop in windows("train"))


def test_source_discontinuities_are_excluded_without_hiding_other_errors(tmp_path, monkeypatch):
    e = Experiment(spinup=0)
    ds = synthetic(e)
    # Reproduce the pinned archive's index 56159 -> 56160 forward jump.
    bad = ds.assign_coords(time=[4049.9166666666665, 4050., 4680.083333333333])
    with pytest.raises(InvalidNativeWindow, match="delta=630.083"):
        validate(bad, "train")
    backward = ds.assign_coords(time=[4680., 4020.0833333333335, 4020.1666666666665])
    with pytest.raises(InvalidNativeWindow, match="non-contiguous"):
        validate(backward, "train")
    cache = NativeCache(tmp_path, "a" * 40)
    def get(window, split):
        value = bad if window == (3, 6) else ds
        validate(value, split)
        return value
    monkeypatch.setattr(cache, "get", get)
    selections = [(0, 3), (3, 6), (6, 9)]
    indexed = list(cache.iterate(selections, "train", indexed=True))
    assert [window for window, _ in indexed] == selections
    assert indexed[1][1] is None
    assert len(list(cache.iterate(selections, "train"))) == 2
    def broken(*args):
        raise ValueError("cache checksum/provenance mismatch")
    monkeypatch.setattr(cache, "get", broken)
    with pytest.raises(ValueError, match="checksum"):
        list(cache.iterate(selections, "train"))


@pytest.mark.slow
def test_coupled_reverse_mode_and_training_update():
    e = Experiment(spinup=0, layers=4)
    # Two actual coupled steps keep this a derivative test, not a climate run.
    e.steps, e.dt = 2, 30.
    state, context = e.snapshot(synthetic(e), 0)
    features = e.features(state, context)
    mean = jnp.mean(features, axis=(0, 1, 2))
    scale = jnp.maximum(jnp.std(features, axis=(0, 1, 2)), 1.)
    params = initialize(jax.random.key(0), constant=True)
    example = (state, context, e.fields(state)[None], jnp.ones((1, 4, *e.coords.horizontal.nodal_shape)), features)
    loss = lambda p, batch: e.loss(p, batch, mean, scale, .001)
    value_grad = jax.jit(jax.value_and_grad(lambda x: loss({"constant": jnp.array([x, 0.])}, example)))
    value, grad = value_grad(0.)
    eps = 1e-3
    fd = (value_grad(eps)[0] - value_grad(-eps)[0]) / (2 * eps)
    assert np.isfinite(value) and abs(float(grad)) > 1e-12
    np.testing.assert_allclose(grad, fd, rtol=.01, atol=1e-9)
    devices = jax.local_devices()[:1]
    replicate = lambda x: jax.tree_util.tree_map(lambda leaf: jnp.broadcast_to(leaf, (len(devices),) + np.shape(leaf)), x)
    params = initialize(jax.random.key(1))
    zeros = jax.tree_util.tree_map(jnp.zeros_like, params)
    result = synchronized_update(loss, .001, devices)(replicate(params), replicate(zeros), replicate(zeros), replicate(jnp.array(0)), replicate(example))
    assert all(np.isfinite(np.asarray(x)).all() for x in jax.tree_util.tree_leaves(result))
    assert not np.array_equal(np.asarray(result[0]["w3"])[0], params["w3"])
