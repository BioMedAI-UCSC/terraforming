# CLI Commands

The `tform` command-line tool provides an interactive interface to the Mars climate simulation framework.

## Global Options

```
tform [OPTIONS] COMMAND [ARGS]...

Options:
  --version   Show version and exit.
  --help      Show help message.
```

---

## `tform mars run`

Run a Mars climate simulation.

```bash
tform mars run [OPTIONS]
```

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--preset NAME` | str | `current-mars` | Use a built-in preset as base config |
| `--config FILE` | path | — | YAML config file (merged over preset) |
| `--lat FLOAT` | float | preset value | Override latitude (−90 to 90) |
| `--lon FLOAT` | float | preset value | Override longitude (0 to 360) |
| `--sols N` | int | preset value | Number of sols (Martian days) to simulate |
| `--accuracy` | `fast`\|`accurate` | `accurate` | Integration mode |
| `--type` | `sol`\|`year`\|`multi`\|`spots`\|`intervention` | `year` | Experiment type |
| `--no-plot` | flag | — | Suppress matplotlib output |

### Experiment Types

| Type | Description |
|------|-------------|
| `sol` | Single Martian day (diurnal cycle) at a given lat/lon |
| `year` | One full Martian year (~687 Earth days) with seasonal cycles |
| `multi` | Three canonical latitudes: 45°N, equator, 40°S (all at 137°E) |
| `spots` | Four landmark sites: Olympus Mons, Elysium Mons, Hellas Basin, South Polar Cap |
| `intervention` | Multi-year GHG injection run from the preset's intervention schedule |

### Accuracy Modes

| Mode | Method | Use Case |
|------|--------|----------|
| `accurate` | 4th-order Runge-Kutta on full ODE system | Science runs, publications |
| `fast` | Reduced-order analytic/relaxation updates | Parameter sweeps, interactive exploration |

### Examples

```bash
# Diurnal cycle at Hellas Basin
tform mars run --preset hellas-basin --type sol

# 10-year intervention with accurate integration
tform mars run --preset terraforming-phase1 --type intervention --sols 3435

# Custom latitude, no plots
tform mars run --preset current-mars --type year --lat 60 --no-plot

# Fully custom config
tform mars run --config experiments/my-run.yaml
```

---

## `tform mars config list`

List all available built-in presets.

```bash
tform mars config list
```

---

## `tform mars config show`

Show the full resolved config for a preset.

```bash
tform mars config show <PRESET_NAME>
```

**Example:**

```bash
tform mars config show terraforming-phase1
```

---

## `tform mars config validate`

Validate a YAML config file against the schema without running a simulation.

```bash
tform mars config validate <FILE>
```

**Example:**

```bash
tform mars config validate experiments/my-run.yaml
```

---

## `tform serve`

Start the tform visualisation web server.

```bash
tform serve [OPTIONS]
```

Launches a FastAPI server that accepts simulation run requests from the browser UI,
streams per-step physics data back in real time via Server-Sent Events, and serves
the pre-built React app from `cli/static/`.  A browser tab opens automatically
after one second.

### Options

| Flag | Type | Default | Description |
|------|------|---------|-------------|
| `--port N` | int | `8000` | Port to bind the server to |
| `--host ADDR` | str | `127.0.0.1` | Host address to bind |
| `--no-browser` | flag | — | Don't open a browser tab automatically |
| `--dev` | flag | — | Also start the Vite dev server from `ui/` on port 5173 |

### Examples

```bash
# Start the server and open the UI (default)
tform serve

# Use a custom port
tform serve --port 9000

# Development mode — starts FastAPI on :8000 and Vite on :5173
tform serve --dev

# Headless (useful on a remote machine)
tform serve --no-browser
```

### API endpoints

The server exposes a REST + SSE API consumed by the React UI:

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/api/runs` | Create and start a simulation run |
| `GET` | `/api/runs` | List all runs (metadata, no data arrays) |
| `GET` | `/api/runs/{id}` | Get full run including all data points |
| `GET` | `/api/runs/{id}/events` | SSE stream — emits data points as they are computed |
| `GET` | `/api/presets` | List available Mars presets |
| `GET` | `/api/compounds` | List available GHG compounds |
| `GET` | `/api/docs` | Interactive Swagger UI |

### Data flow

Every integration step in the physics engine fires a callback.  That callback
writes one data point to an in-memory list.  The SSE generator polls the list
with a cursor and pushes new points to the browser, where Recharts re-renders
the chart in real time.

```
TimeController.step() → callback(planet, t) → _runs[id].data.append()
                                                     ↓
                                              SSE /events stream
                                                     ↓
                                            React state → Recharts
```

For intervention runs the natural granularity is one point per Mars year
(via `InterventionController`'s per-year callback).  For sol/year runs the
per-step callback is throttled to at most 2 000 chart points.

### Requirements

`fastapi` and `uvicorn` are included in the standard `cli` dependencies.
No separate install step is required beyond the usual `uv sync`.

To build the UI from source (only needed when modifying the frontend):

```bash
cd ui && npm install && npm run build
# built assets are written to cli/static/ automatically
```

---

## `tform man`

Show reference information for a planet or subsystem.

```bash
tform man [PLANET]
```

**Example:**

```bash
tform man mars
```

---

## Config Priority Chain

When multiple config sources are provided, later sources override earlier ones:

```
Built-in defaults → Preset YAML → --config FILE → CLI flags
```

This means `--lat 45` always wins over whatever latitude is in the YAML file.
