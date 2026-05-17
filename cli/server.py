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
        accuracy = SrcAccuracy.FAST if req.accuracy == "fast" else SrcAccuracy.ACCURATE

        if req.exp_type == "intervention":
            _run_intervention(run, req, mars, cfg, accuracy)
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


def _run_intervention(run, req, mars, cfg, accuracy) -> None:
    """One data point per Mars year via InterventionController callback."""
    from src.interventions import InterventionController

    ic = InterventionController(
        mars,
        injection_schedule_kg_yr=req.inject or {},
        dt=cfg.engine.dt,
        accuracy=accuracy,
    )
    n_years = req.years

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
    return _runs[run_id]


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
