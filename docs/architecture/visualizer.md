# Architecture — tform Visualizer

> Live web UI for Mars climate simulations, started with a single `tform serve` command.

## Overview

The visualizer adds a browser-based chart dashboard to the tform CLI without
changing the existing simulation or output pipeline.  It consists of two peers:

| Component | Location | Language | Role |
|-----------|----------|----------|------|
| FastAPI server | `cli/server.py` | Python | Runs simulations, streams SSE, serves static UI |
| React app | `ui/` | TypeScript | Renders charts, manages run state |

---

## Directory layout

```
terraforming/
├── cli/
│   ├── server.py       ← FastAPI app + simulation thread pool
│   ├── static/         ← pre-built React app (committed, served by FastAPI)
│   │   └── index.html
│   └── main.py         ← tform serve command lives here
└── ui/                 ← React source (Node.js / npm, not part of the Python package)
    ├── src/
    │   ├── App.tsx
    │   ├── api.ts
    │   ├── types.ts
    │   └── components/
    │       ├── RunForm.tsx
    │       ├── RunList.tsx
    │       └── ChartPanel.tsx
    ├── vite.config.ts   ← builds into cli/static/
    └── package.json
```

`ui/` is a peer module at the repo root, not a subdirectory of `cli/`.  It has
its own package manager (`npm`), its own dependency graph, and its own build
step.  This separation matters because mixing a Node.js project inside a Python
package directory would pollute `node_modules` into the installable package and
couple two unrelated toolchains.

`cli/server.py` is legitimately inside `cli/` because it directly imports
`cli.runner`, `cli.config_loader`, and `cli.models`.  It is the Python half of
the server bridge — not a UI concern.

---

## Why ship the built UI inside the Python package

The pre-built React app (`cli/static/`) is committed alongside the Python source
so that `uv add tform` / `pip install tform` gives users a fully working
`tform serve` with no Node.js runtime required.

This follows the same pattern used by Jupyter, Grafana CLI, and Streamlit: the
frontend is compiled once by the maintainer and bundled as package data.  End
users interact only with Python.

`vite build` is configured to write its output directly to `cli/static/` via
`vite.config.ts`:

```ts
build: {
  outDir: '../cli/static',
  emptyOutDir: true,
}
```

The FastAPI app mounts `cli/static/` as a `StaticFiles` route last, after all
`/api/*` routes, so the API always wins on path conflicts.

---

## Single-command startup

```
tform serve [--port N] [--dev] [--no-browser]
```

1. `tform serve` calls `uvicorn.run("cli.server:app", ...)` — one Python process.
2. FastAPI serves both the simulation API and the static React bundle.
3. A `threading.Timer` opens the browser after 1.5 s (giving uvicorn time to bind).
4. `--dev` additionally spawns `npm run dev` in `ui/` as a subprocess.
   - Vite dev server runs on port 5173 and proxies `/api/*` to FastAPI on port 8000.
   - The browser opens `localhost:5173` so it gets Vite HMR.
   - The subprocess is terminated when the uvicorn process exits.

---

## Callback chain — how data flows from engine to chart

Every physics integration step in the engine fires a callback.  The server
replaces the CLI's progress-bar callback with one that writes to an in-memory
list, which the SSE generator then streams to the browser.

```
TimeController.run(duration, callback=step_cb)
      │
      │  called once per dt (e.g. 3600 s)
      ▼
  step_cb(planet: Planet, elapsed: Tensor)
      │
      │  reads planet.thermal.surface_temperature
      │        planet.atmosphere.surface_pressure
      │        planet.water.ice_mass
      │        planet.radiation.solar_flux
      ▼
  _runs[run_id]["data"].append(point)    ← list.append is GIL-safe in CPython
      │
      │  polled every 250 ms by the SSE generator
      ▼
  GET /api/runs/{id}/events  (text/event-stream)
      │
      │  EventSource in the browser
      ▼
  React: setData(prev => [...prev, point])
      │
      ▼
  Recharts re-renders all four charts (syncId keeps crosshair in sync)
```

For **intervention runs** the natural granularity is one point per Mars year,
provided by `InterventionController`'s `callback(snap: InterventionSnapshot)`.
The snapshot includes annual-mean temperature (plus min/max band), pressure, ice
mass, radiative forcing (ΔF), and greenhouse factor.

For **sol / year runs** the per-step `TimeController` callback is throttled:

```python
emit_every = max(1, total_steps // MAX_CHART_POINTS)  # MAX_CHART_POINTS = 2000
if step_n[0] % emit_every == 0:
    run["data"].append(point)
```

This caps the browser at 2 000 data points regardless of run length, keeping
Recharts responsive even for decade-long hourly runs.

---

## Thread safety

Simulations run in a `ThreadPoolExecutor` thread.  The SSE generator runs in the
FastAPI async event loop.  They share `_runs[run_id]["data"]`, a plain Python
list.

CPython's GIL guarantees that `list.append` is atomic at the bytecode level —
no explicit lock is needed for the producer (simulation thread) / consumer (SSE
generator) pattern used here.  The SSE generator advances a `cursor` integer
that is local to the generator coroutine, so there is no shared mutable state
between concurrent SSE consumers of the same run.

---

## UI layout

```
┌─────────────────┬──────────────────────────────────────────────────┐
│  tform          │  100yr SF6+CF4   ● live  [42%]                   │
│  visualizer     ├──────────────────────────────────────────────────┤
│                 │  Temperature (K)                   [full width]  │
│  ─ New Run ──   │  ────────────────────────────────────────────    │
│  Preset         │                                                  │
│  Type           ├──────────────────────────────────────────────────┤
│  Years  100 500 │  Pressure (Pa)                     [full width]  │
│         1000    │  ────────────────────────────────────────────    │
│  [Run]          │                                                  │
│                 ├───────────────────────┬──────────────────────────┤
│  ─ Past Runs ── │  Ice Mass (kg)        │  ΔF Radiative Forcing    │
│  ● 100yr SF6    │  ────────────────     │  ───────────────────     │
│  ○ 500yr CF4    │                       │                          │
└─────────────────┴───────────────────────┴──────────────────────────┘
```

All four charts share a `syncId` — hovering over any chart aligns the tooltip
crosshair across all panels at the same x-axis value (year or hour).

Temperature uses an `AreaChart` with a min/max band for intervention runs, where
`InterventionSnapshot` carries `temp_min` and `temp_max` from the annual
integration.

---

## Development workflow

```bash
# Terminal 1 — FastAPI backend (hot-reloads Python on file change)
tform serve --dev

# Terminal 2 — edit React source; Vite HMR updates browser instantly
# (already started by --dev; or run separately:)
cd ui && npm run dev
```

When the UI changes are ready to ship:

```bash
cd ui && npm run build
# writes to cli/static/ — commit both ui/src/ and cli/static/
```

A `Makefile` target is provided for convenience:

```makefile
ui-build:
    cd ui && npm install && npm run build
```
