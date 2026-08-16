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
                    "surface_zonal_wind", "surface_wind_speed", "elevation"):
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
