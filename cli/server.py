"""FastAPI visualisation server for tform.

Started by ``tform serve``.  Runs simulations in a thread pool, streams every
physics step to the browser via Server-Sent Events, and serves the pre-built
React UI from cli/static/.

Callback chain
--------------
TimeController.run(callback=step_cb)
    └── step_cb(planet, elapsed)          per integration step
         └── _runs[id]["data"].append()   thread-safe via CPython GIL
              └── SSE generator           polls list with cursor
                   └── EventSource        browser updates Recharts in real-time

For intervention runs the InterventionController provides a per-year callback
(InterventionSnapshot) which is the natural chart granularity.  For sol/year
runs the per-step callback is throttled to at most MAX_CHART_POINTS points.
"""

from __future__ import annotations

import asyncio
import csv as _csv
import json
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

app = FastAPI(title="tform visualizer", docs_url="/api/docs")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_executor = ThreadPoolExecutor(max_workers=4)
_runs: dict[str, dict[str, Any]] = {}

MAX_CHART_POINTS = 2000  # throttle cap for sol/year runs
_STATIC_DIR  = Path(__file__).parent / "static"
_OUTPUTS_DIR = Path(__file__).parent.parent / "outputs" / "server"


# ── Request model ──────────────────────────────────────────────────────────────

class RunRequest(BaseModel):
    preset: str = "current-mars"
    exp_type: str = "intervention"
    years: int = 100
    sols: float = 1.0
    accuracy: str = "fast"
    dt: float = 3600.0
    lat: float | None = None
    lon: float | None = None
    elevation: float | None = None
    ls: float | None = None
    surface_temp: float | None = None
    surface_pressure: float | None = None
    albedo: float | None = None
    greenhouse_factor: float | None = None
    ice_mass: float | None = None
    inject: dict[str, float] = {}
    label: str | None = None
    # gcm ('gcm3d maps') options
    scale: str = "fast"      # runtime resolution preset (fast/balanced/high/ultra)
    snapshots: int = 5       # 3-D map snapshots along an intervention timeline


# ── Helpers ────────────────────────────────────────────────────────────────────

def _v(t) -> float:
    return float(t.item())


def _default_label(req: RunRequest) -> str:
    if req.exp_type == "intervention":
        compounds = "+".join(req.inject.keys()) or "baseline"
        return f"{req.years}yr {compounds}"
    if req.exp_type == "year":
        return f"{req.preset} {req.years}yr"
    if req.exp_type == "sol":
        return f"{req.preset} {req.sols}sol"
    return f"{req.preset} {req.exp_type}"


def _save_run_csv(run_id: str, run: dict) -> None:
    if not run["data"]:
        return
    _OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    slug = run["label"][:40].replace("/", "-").replace(" ", "_")
    path = _OUTPUTS_DIR / f"{slug}_{run_id}.csv"
    with open(path, "w", newline="") as f:
        writer = _csv.DictWriter(f, fieldnames=list(run["data"][0].keys()))
        writer.writeheader()
        writer.writerows(run["data"])
    run["csv_path"] = str(path)


# ── Simulation thread ──────────────────────────────────────────────────────────

def _extract_maps_fields(fields) -> dict:
    """Adapt a gcm3d ``MarsMapFields`` into the browser field-grid contract.

    Returns lon/lat axes plus 2-D maps (row = lat, col = lon) each with its own
    min/max/units, matching the ``RunFields`` shape the ``FieldMap`` component
    consumes. gcm3d already stores every field as ``(n_lat, n_lon)``.
    """
    import numpy as np

    def _map(arr, label, units):
        a = np.asarray(arr, dtype=float)
        return {"label": label, "units": units,
                "min": float(np.nanmin(a)), "max": float(np.nanmax(a)),
                "data": np.round(a, 3).tolist()}

    maps = {
        "surface_temperature": _map(fields.temperature_k, "Surface temperature", "K"),
        "surface_pressure": _map(fields.surface_pressure_pa, "Surface pressure", "Pa"),
        "surface_zonal_wind": _map(fields.u_ms, "Surface zonal wind", "m/s"),
        "surface_wind_speed": _map(fields.wind_speed_ms, "Surface wind speed", "m/s"),
        "elevation": _map(fields.elevation_m, "MOLA elevation", "m"),
    }
    if getattr(fields, "co2_ice_pa", None) is not None:
        maps["co2_ice"] = _map(fields.co2_ice_pa, "CO2 surface frost", "Pa-equiv")

    return {
        "lon": np.round(np.asarray(fields.lon_deg), 2).tolist(),
        "lat": np.round(np.asarray(fields.lat_deg), 2).tolist(),
        "sigma": [],
        "maps": maps,
        "sections": {},
    }


def _gcm_snapshot(scale: str, albedo: float, greenhouse: float, ls_deg: float,
                  pressure_pa: float | None) -> dict:
    """Run one 3-D gcm3d map at the given atmosphere state; return field grids.

    ``pressure_pa`` (if given) sets the reference surface pressure, so a snapshot
    taken partway through a terraforming run reflects that epoch's evolved (thicker)
    atmosphere and greenhouse factor.
    """
    import dataclasses
    import math

    from src.gcm3d.maps import resolve_scale, run_maps
    from src.gcm3d.physics import mars_co2_forcing, mars_radiative_forcing

    from src.gcm3d.physics import mean_anomaly_for_ls

    forcing = mars_radiative_forcing(albedo=albedo, greenhouse_factor=greenhouse,
                                     diurnal=False)
    forcing = dataclasses.replace(
        forcing, init_orbital_angle_rad=mean_anomaly_for_ls(math.radians(ls_deg), forcing),
    )
    cfg = resolve_scale(scale)
    fields = run_maps(
        forcing=forcing, co2_forcing=mars_co2_forcing(),
        p0_pa=pressure_pa, **cfg,
    )
    return _extract_maps_fields(fields)


def _run_gcm_maps(run, req: RunRequest, cfg) -> None:
    """Run a single gcm3d 3-D map (no timeseries) and attach the field grids.

    ``run["data"]`` stays empty, so the browser shows the ``FieldMap`` view.
    """
    p = cfg.planet
    run["fields"] = _gcm_snapshot(
        req.scale, p.albedo, p.greenhouse_factor, p.initial_ls_deg or 0.0,
        pressure_pa=p.surface_pressure,
    )
    run["progress"] = 1.0


def _snapshot_years(n_years: int, n_snapshots: int) -> set[int]:
    """The intervention years at which to capture a 3-D snapshot.

    Evenly spaced across 1..n_years, always including the final year (so the last
    snapshot is the terraformed end-state used as the headline heatmap).
    """
    if n_snapshots <= 0 or n_years <= 0:
        return set()
    if n_snapshots >= n_years:
        return set(range(1, n_years + 1))
    import numpy as np
    years = {int(y) for y in np.linspace(1, n_years, n_snapshots).round().astype(int)}
    years.add(n_years)
    return years


def _run_simulation(run_id: str, req: RunRequest) -> None:
    """Execute the simulation in a background thread.

    Every step writes one data point to _runs[run_id]["data"].  The SSE
    generator reads that list with a cursor so the browser sees live updates.

    List.append is GIL-safe in CPython; no explicit lock is needed.
    """
    run = _runs[run_id]
    try:
        from cli import config_loader
        from cli.models import Accuracy, ExpType, RunFlags
        from src.celestials import Mars, MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
        from src.engine import Accuracy as SrcAccuracy, TimeController
        from src.interventions import InterventionController

        flags = RunFlags(
            exp_type=ExpType(req.exp_type),
            accuracy=Accuracy(req.accuracy),
            dt=req.dt,
            lat=req.lat,
            lon=req.lon,
            elevation=req.elevation,
            ls=req.ls,
            surface_temp=req.surface_temp,
            pressure=req.surface_pressure,
            albedo=req.albedo,
            greenhouse_factor=req.greenhouse_factor,
            ice_mass=req.ice_mass,
            sols=req.sols if req.exp_type == "sol" else None,
            n_years=req.years if req.exp_type == "intervention" else None,
            inject=req.inject or None,
            no_save=True,
        )
        cfg = config_loader.load(planet="mars", preset=req.preset)
        cfg = config_loader.merge_overrides(cfg, flags)
        p = cfg.planet

        # The 'gcm' accuracy runs the 3-D gcm3d maps backend (dinosaur dycore +
        # radiation + CO2). A one-off gcm map (no timeseries) short-circuits here;
        # a gcm *intervention* runs the 100-yr trajectory and captures 3-D snapshots
        # along that existing timeline (handled below, after Mars is built).
        if req.accuracy == "gcm" and req.exp_type != "intervention":
            _run_gcm_maps(run, req, cfg)
            run["status"] = "done"
            run["progress"] = 1.0
            run["completed_at"] = datetime.now(timezone.utc).isoformat()
            return

        mars = Mars(
            surface_temperature=p.surface_temperature,
            surface_pressure=p.surface_pressure,
            albedo=p.albedo,
            greenhouse_factor=p.greenhouse_factor,
            ice_mass=p.ice_mass,
            latitude=p.latitude,
            longitude=p.longitude,
            elevation_m=p.elevation_m,
            initial_ls_deg=p.initial_ls_deg,
        )
        # gcm-flavoured intervention drives the (torch) trajectory with the FAST
        # reduced-order kernel and layers 3-D snapshots on top.
        accuracy = SrcAccuracy.ACCURATE if req.accuracy == "accurate" else SrcAccuracy.FAST

        if req.exp_type == "intervention":
            _run_intervention(run, req, mars, cfg, accuracy,
                              capture_gcm=(req.accuracy == "gcm"))
        else:
            _run_timeseries(run, req, mars, cfg, accuracy)

        run["status"] = "done"
        run["progress"] = 1.0
        run["completed_at"] = datetime.now(timezone.utc).isoformat()
        _save_run_csv(run_id, run)

    except Exception as exc:
        import traceback
        run["status"] = "error"
        run["error"] = f"{type(exc).__name__}: {exc}\n{traceback.format_exc()}"


def _run_intervention(run, req, mars, cfg, accuracy, capture_gcm: bool = False) -> None:
    """One data point per Mars year via InterventionController callback.

    When ``capture_gcm`` is set, a 3-D gcm3d snapshot is also captured at a handful
    of years along this same (existing) timeline — using each year's evolved
    pressure + greenhouse factor — so the browser can show the spatial evolution
    of the terraforming run, with the final year as the headline heatmap.
    """
    from src.interventions import InterventionController

    ic = InterventionController(
        mars,
        injection_schedule_kg_yr=req.inject or {},
        dt=cfg.engine.dt,
        accuracy=accuracy,
    )
    n_years = req.years
    ls_deg = cfg.planet.initial_ls_deg or 0.0
    snap_years = _snapshot_years(n_years, req.snapshots) if capture_gcm else set()
    if capture_gcm:
        run["snapshot_years"] = sorted(snap_years)
        run["field_snapshots"] = {}

    def iv_cb(snap) -> None:
        run["data"].append({
            "year": snap.year,
            "temperature_k": _v(snap.surface_temperature),
            "temp_min_k":    _v(snap.temp_min),
            "temp_max_k":    _v(snap.temp_max),
            "pressure_pa":   _v(snap.surface_pressure),
            "ice_mass_kg":   _v(snap.ice_mass),
            "delta_F":       _v(snap.delta_F),
            "greenhouse_factor": _v(snap.greenhouse_factor),
        })
        # 3-D snapshot at this year's evolved atmosphere (thicker air + stronger
        # greenhouse as the terraforming proceeds).
        if snap.year in snap_years:
            try:
                fld = _gcm_snapshot(
                    req.scale,
                    albedo=cfg.planet.albedo,
                    greenhouse=_v(snap.greenhouse_factor),
                    ls_deg=ls_deg,
                    pressure_pa=_v(snap.surface_pressure),
                )
                run["field_snapshots"][str(snap.year)] = fld
                run["fields"] = fld  # latest snapshot is the current headline
            except Exception:  # noqa: BLE001 — snapshots are optional, never fail the run
                pass
        # Progress: trajectory year plus the extra weight of snapshot rendering.
        run["progress"] = snap.year / n_years

    ic.run(n_years=n_years, callback=iv_cb)


def _run_timeseries(run, req, mars, cfg, accuracy) -> None:
    """Per-step callback throttled to MAX_CHART_POINTS for sol/year runs.

    The TimeController calls callback(planet, elapsed) after every integration
    step.  We emit every Nth step so the browser receives at most
    MAX_CHART_POINTS data points regardless of run length.
    """
    from src.celestials import MARS_ROTATION_PERIOD, MARS_ORBITAL_PERIOD
    from src.engine import TimeController

    sol_s = float(MARS_ROTATION_PERIOD.item())

    if req.exp_type == "year":
        duration = float(MARS_ORBITAL_PERIOD.item()) * req.years
    else:
        duration = cfg.experiment.sols * sol_s

    dt_s = float(cfg.engine.dt)
    total_steps = max(1, int(duration / dt_s))
    emit_every = max(1, total_steps // MAX_CHART_POINTS)

    tc = TimeController(mars, dt=dt_s, accuracy=accuracy)
    step_n = [0]

    orbital_s = float(MARS_ORBITAL_PERIOD.item())

    def step_cb(planet, elapsed) -> None:
        step_n[0] += 1
        if step_n[0] % emit_every != 0:
            return
        t_s = _v(elapsed)
        pt: dict = {
            "time_h":        t_s / 3600.0,
            "sol":           t_s / sol_s,
            "temperature_k": _v(planet.thermal.surface_temperature),
            "pressure_pa":   _v(planet.atmosphere.surface_pressure),
            "ice_mass_kg":   _v(planet.water.ice_mass),
            "solar_flux":    _v(planet.radiation.solar_flux),
        }
        if req.exp_type == "year":
            pt["mars_year"] = t_s / orbital_s
        run["data"].append(pt)
        run["progress"] = step_n[0] / total_steps

    tc.run(duration=duration, callback=step_cb)


# ── API routes ─────────────────────────────────────────────────────────────────

@app.post("/api/runs")
async def create_run(req: RunRequest) -> dict:
    run_id = str(uuid.uuid4())[:8]
    _runs[run_id] = {
        "id":           run_id,
        "status":       "running",
        "progress":     0.0,
        "config":       req.model_dump(),
        "data":         [],
        "error":        None,
        "created_at":   datetime.now(timezone.utc).isoformat(),
        "completed_at": None,
        "label":        req.label or _default_label(req),
    }
    _executor.submit(_run_simulation, run_id, req)
    return {"run_id": run_id}


@app.get("/api/runs")
async def list_runs() -> list:
    return [
        {k: v for k, v in run.items() if k != "data"}
        for run in reversed(list(_runs.values()))
    ]


@app.get("/api/runs/{run_id}")
async def get_run(run_id: str) -> dict:
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    # Field grids (3-D maps) are served separately so run polling stays light.
    heavy = {"fields", "field_snapshots"}
    return {k: v for k, v in _runs[run_id].items() if k not in heavy}


@app.get("/api/runs/{run_id}/fields")
async def get_run_fields(run_id: str, year: int | None = None) -> dict:
    """The lat/lon field grids for a gcm run.

    With no ``year`` this returns the headline snapshot (the final/most-recent one).
    Pass ``year`` to fetch a specific snapshot from a terraforming timeline.
    """
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    if year is not None:
        snap = (run.get("field_snapshots") or {}).get(str(year))
        if not snap:
            raise HTTPException(404, f"No snapshot for year {year}")
        return snap
    fields = run.get("fields")
    if not fields:
        raise HTTPException(404, "No field data (not a gcm run, or not finished)")
    return fields


@app.get("/api/runs/{run_id}/snapshots")
async def get_run_snapshots(run_id: str) -> dict:
    """The years for which 3-D snapshots exist (for the timeline slider)."""
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")
    run = _runs[run_id]
    years = sorted(int(y) for y in (run.get("field_snapshots") or {}))
    return {"years": years}


@app.get("/api/runs/{run_id}/events")
async def stream_run(run_id: str) -> StreamingResponse:
    """SSE stream — emits new data points as the simulation progresses.

    The generator polls _runs[run_id]["data"] with a cursor, batching up to
    100 points per tick.  Poll interval is 250 ms.  When the simulation thread
    sets status to "done" or "error" a final event is emitted and the stream
    closes.
    """
    if run_id not in _runs:
        raise HTTPException(404, "Run not found")

    async def generator():
        cursor = 0
        while True:
            run  = _runs[run_id]
            data = run["data"]

            while cursor < len(data):
                batch = data[cursor:cursor + 100]
                for pt in batch:
                    yield f"data: {json.dumps({'type': 'point', 'data': pt})}\n\n"
                cursor += len(batch)

            if run["status"] in ("done", "error"):
                yield f"data: {json.dumps({'type': 'done', 'status': run['status'], 'error': run['error']})}\n\n"
                break

            await asyncio.sleep(0.25)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/api/compounds")
async def get_compounds() -> list[str]:
    from src.interventions.compounds import list_compounds
    return list_compounds()


@app.get("/api/presets")
async def get_presets() -> list[str]:
    from cli.presets import MARS_PRESET_NAMES
    return list(MARS_PRESET_NAMES)


@app.get("/api/presets/{name}")
async def get_preset_config(name: str) -> dict:
    from cli.presets import MARS_PRESET_NAMES
    if name not in MARS_PRESET_NAMES:
        raise HTTPException(404, f"Unknown preset '{name}'")
    from cli import config_loader
    cfg = config_loader.load(planet="mars", preset=name)
    return {"planet": cfg.planet.model_dump(), "engine": cfg.engine.model_dump()}


# ── Static UI — registered last so API routes win ──────────────────────────────

if _STATIC_DIR.exists() and any(_STATIC_DIR.iterdir()):
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
