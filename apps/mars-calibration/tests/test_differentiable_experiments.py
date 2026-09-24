"""Scientific data-contract regressions for the experiment runner."""
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_differentiable_experiments import climate, matchup_metrics
from run_differentiable_experiments import load_manifest, digest


def test_trajectory_captures_window_restart_and_forcing(monkeypatch):
    from dataclasses import dataclass
    from mars_calibration import paper as p
    @dataclass
    class Forcing:
        ames_co2_longwave_opacity_scale: float = 1.
        ames_dust_longwave_opacity_scale: float = 1.
        surface_exchange_multiplier: float = 1.
        offset: float = 2.
    model = p.Model.__new__(p.Model)
    model.coords = model.specs = model.orography = None
    model.initial = p.d.jnp.asarray(3.)
    model.forcing = Forcing()
    model.observe = lambda state: state
    monkeypatch.setattr(p.d, "_build_step", lambda coords, specs, forcing, dt, **kw: lambda state: state + forcing.offset)
    first = model.trajectory([1.], 1.)
    model.initial = p.d.jnp.asarray(10.)
    model.forcing = Forcing(offset=7.)
    second = model.trajectory([1.], 1.)
    np.testing.assert_allclose(first(p.d.jnp.ones(3)), [5.])
    np.testing.assert_allclose(second(p.d.jnp.ones(3)), [17.])


def test_paired_matchups_reject_different_reference_samples():
    frame = pd.DataFrame([
        dict(configuration=c, field="T", units="K", sample_id=i, block=str(i),
             observed=200., predicted=201., weight=1.)
        for c in ("T21", "T42") for i in (0, 1)])
    result = matchup_metrics(frame)
    np.testing.assert_allclose(result.rmse, 1.)
    frame.loc[3, "observed"] = 202.
    with pytest.raises(ValueError, match="identical paired"):
        matchup_metrics(frame)


def test_climate_requires_complete_coverage_and_reports_repeatability():
    time = np.arange(0., 31.)
    frame = pd.DataFrame(dict(elapsed_sols=time))
    for field in ("mean_surface_pressure_pa", "mean_co2_ice_pa", "mean_surface_temperature_k", "mean_deep_soil_temperature_k"):
        frame[field] = 200 + np.sin(time * 2*np.pi/10)
    result = climate(frame, 10., bin_sols=2.)
    assert result.year.nunique() == 3
    with pytest.raises(ValueError, match="initialization"):
        climate(frame.iloc[3:], 10., bin_sols=2.)
    with pytest.raises(ValueError, match="two complete"):
        climate(frame.iloc[:10], 10., bin_sols=2.)


def test_manifest_overlap_hash_and_block_isolation(tmp_path):
    rows = []
    for i, split in enumerate(("train", "test")):
        restart = tmp_path / f"restart{i}.npz"
        target = tmp_path / f"target{i}.npz"
        np.savez(restart, data=[i])
        np.savez(target, fields=[i])
        rows.append(dict(id=str(i), split=split, block=str(i), source="test", reference_kind="synthetic",
            start_seconds=i*10., end_seconds=i*10.+5., restart=restart.name, target=target.name,
            restart_sha256=digest(restart), target_sha256=digest(target)))
    manifest = tmp_path / "windows.json"
    def write(): manifest.write_text(json.dumps(dict(schema_version=1, windows=rows)))
    write()
    assert len(load_manifest(manifest)["windows"]) == 2
    rows[1]["start_seconds"] = 1.
    write()
    with pytest.raises(ValueError, match="Overlapping"):
        load_manifest(manifest)
    rows[1]["start_seconds"] = 10.
    rows[1]["block"] = "0"
    write()
    with pytest.raises(ValueError, match="blocks"):
        load_manifest(manifest)
    rows[1]["block"] = "1"
    rows[1]["target_sha256"] = "bad"
    write()
    with pytest.raises(ValueError, match="Hash mismatch"):
        load_manifest(manifest)
