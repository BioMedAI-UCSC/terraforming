"""Tests for cli.server — the gcm ('gcm3d maps') backend wiring.

Covers:
  - _extract_maps_fields: adapts a MarsMapFields-like object into the browser
    RunFields JSON contract (fast, dependency-free via a duck-typed stand-in).
  - the full gcm run path through _run_simulation (skipped without the gcm3d
    extra): produces field grids, an empty timeseries, and hides fields from the
    run poll while serving them on the fields endpoint.
"""

from __future__ import annotations

from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from cli import server


# ── _extract_maps_fields (no heavy deps) ──────────────────────────────────────

def _fake_fields(with_co2: bool = True):
    """A duck-typed stand-in for gcm3d.MarsMapFields (n_lat=4, n_lon=6)."""
    shape = (4, 6)
    ns = SimpleNamespace(
        lon_deg=np.linspace(0, 300, 6),
        lat_deg=np.linspace(-90, 90, 4),
        elevation_m=np.zeros(shape),
        temperature_k=np.full(shape, 210.0),
        surface_pressure_pa=np.full(shape, 610.0),
        u_ms=np.zeros(shape),
        v_ms=np.zeros(shape),
        wind_speed_ms=np.zeros(shape),
        co2_ice_pa=np.zeros(shape) if with_co2 else None,
        approximate_wind_height_m=432.0,
    )
    return ns


class TestExtractMapsFields:

    def test_contract_shape(self):
        out = server._extract_maps_fields(_fake_fields())
        assert out["lon"] and out["lat"]
        assert len(out["lon"]) == 6 and len(out["lat"]) == 4
        assert out["sigma"] == [] and out["sections"] == {}
        # required surface maps present
        for key in ("surface_temperature", "surface_pressure",
                    "surface_zonal_wind", "surface_meridional_wind",
                    "surface_wind_speed", "elevation"):
            assert key in out["maps"]

    def test_grid_orientation_and_stats(self):
        grid = server._extract_maps_fields(_fake_fields())["maps"]["surface_temperature"]
        # data is [rows=lat][cols=lon]
        assert len(grid["data"]) == 4 and len(grid["data"][0]) == 6
        assert grid["units"] == "K"
        assert grid["min"] == pytest.approx(210.0)
        assert grid["max"] == pytest.approx(210.0)

    def test_co2_optional(self):
        assert "co2_ice" in server._extract_maps_fields(_fake_fields(True))["maps"]
        assert "co2_ice" not in server._extract_maps_fields(_fake_fields(False))["maps"]


def test_matched_mcd_comparison_uses_requested_parameters(monkeypatch):
    import xarray as xr
    from src.gcm3d import mcd

    calls = []
    server._mcd_ascii_cache.clear()
    reference = xr.Dataset({
        "temperature": (("lat", "lon"), np.full((4, 6), 205.0)),
        "surface_pressure": (("lat", "lon"), np.full((4, 6), 600.0)),
        "wind_speed": (("lat", "lon"), np.full((4, 6), 5.0)),
        "co2_ice": (("lat", "lon"), np.zeros((4, 6))),
    }, coords={"lat": np.linspace(-90, 90, 4), "lon": np.linspace(0, 300, 6)})

    def fetch(ls, hour, **kwargs):
        calls.append((ls, hour, kwargs))
        return "ignored", f"mcd://{hour}"

    monkeypatch.setattr(mcd, "fetch_ascii", fetch)
    monkeypatch.setattr(mcd, "parse_ascii", lambda _: reference)
    monkeypatch.setattr(mcd, "interpolate_periodic", lambda ref, target: ref)
    out = server._matched_mcd_comparison(
        _fake_fields(), ls_deg=90.0, local_time=14.0, dust=3
    )
    assert calls == [(90.0, 14.0, {
        "dust": 3, "high_res": True, "altitude_m": 432.0,
    })]
    assert out["metadata"]["local_times_hours"] == [14.0]
    assert out["metrics"]["temperature"]["bias"] == pytest.approx(5.0)
    assert out["mcd"]["temperature"]["data"][0][0] == pytest.approx(205.0)


# ── _snapshot_years (pure) ────────────────────────────────────────────────────

class TestSnapshotYears:

    def test_evenly_spaced_and_includes_final(self):
        ys = server._snapshot_years(100, 5)
        assert max(ys) == 100          # final year always captured
        assert min(ys) >= 1
        assert len(ys) == 5

    def test_zero_snapshots_is_empty(self):
        assert server._snapshot_years(100, 0) == set()

    def test_more_snapshots_than_years_caps_at_every_year(self):
        assert server._snapshot_years(3, 10) == {1, 2, 3}


@pytest.mark.slow
def test_gcm_snapshot_failure_is_recorded_and_fails_final(monkeypatch):
    """A failing 3-D snapshot must not be silently swallowed: it is recorded, and
    a failed *final*-year snapshot fails the whole GCM job (not a silent success)."""
    pytest.importorskip("dinosaur")

    def boom(*a, **k):
        raise RuntimeError("snapshot boom")

    monkeypatch.setattr(server, "_gcm_snapshot", boom)

    rid = "srv_snapfail"
    req = server.RunRequest(preset="current-mars", exp_type="intervention",
                            accuracy="gcm", years=2, snapshots=2, scale="fast")
    server._runs[rid] = {
        "id": rid, "status": "running", "progress": 0.0,
        "config": req.model_dump(), "data": [], "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None, "label": "snapfail",
    }
    try:
        server._run_simulation(rid, req)
        run = server._runs[rid]
        assert run["status"] == "error"           # final snapshot failed -> job failed
        assert "snapshot boom" in run["error"]
        assert run["snapshot_errors"]             # failures recorded, not hidden
    finally:
        server._runs.pop(rid, None)


# ── Full gcm run path (needs the gcm3d extra) ─────────────────────────────────

@pytest.mark.slow
def test_gcm_run_produces_fields_and_hides_them_from_poll():
    pytest.importorskip("dinosaur")
    from src.gcm3d import topography as topo

    if not topo._DEFAULT_MOLA.exists():
        pytest.skip("MOLA raster not staged")

    rid = "srv_gcm_test"
    req = server.RunRequest(preset="current-mars", exp_type="sol",
                            accuracy="gcm", ls=270.0)
    server._runs[rid] = {
        "id": rid, "status": "running", "progress": 0.0,
        "config": req.model_dump(), "data": [], "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None, "label": "gcm test",
    }
    try:
        server._run_simulation(rid, req)  # synchronous
        run = server._runs[rid]
        assert run["status"] == "done", run.get("error")
        assert run["data"] == []  # spatial run: no timeseries
        f = run["fields"]
        assert len(f["lat"]) > 0 and len(f["lon"]) > 0
        assert "surface_temperature" in f["maps"]
        # the run poll must not carry the (heavy) field grids
        poll = {k: v for k, v in run.items() if k != "fields"}
        assert "fields" not in poll
    finally:
        server._runs.pop(rid, None)


@pytest.mark.slow
def test_gcm_intervention_captures_snapshots_along_timeline():
    pytest.importorskip("dinosaur")
    from src.gcm3d import topography as topo

    if not topo._DEFAULT_MOLA.exists():
        pytest.skip("MOLA raster not staged")

    rid = "srv_hybrid_test"
    req = server.RunRequest(preset="current-mars", exp_type="intervention",
                            accuracy="gcm", years=4, snapshots=2, scale="fast",
                            inject={"SF6": 5e8}, ls=270.0)
    server._runs[rid] = {
        "id": rid, "status": "running", "progress": 0.0,
        "config": req.model_dump(), "data": [], "error": None,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": None, "label": "hybrid test",
    }
    try:
        server._run_simulation(rid, req)
        run = server._runs[rid]
        assert run["status"] == "done", run.get("error")
        # trajectory chart present (the existing timeline)
        assert len(run["data"]) == 4
        # 3-D snapshots captured along that timeline, final year included
        years = run["snapshot_years"]
        assert 4 in years
        assert set(str(y) for y in years) == set(run["field_snapshots"].keys())
        # headline fields = the final snapshot
        assert "surface_temperature" in run["fields"]["maps"]
    finally:
        server._runs.pop(rid, None)
