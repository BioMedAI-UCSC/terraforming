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
