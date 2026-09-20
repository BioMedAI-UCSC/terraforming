# Model, Ames, MCD and ARCO comparison

Generate an offline browser report from the staged reference files:

```sh
rtk proxy .venv/bin/python -m cli.main mars compare \
  --config cli/configs/reference-comparison.json \
  --output outputs/reference-comparison-new
```

The equivalent installed command is `tform mars compare`. The output directory
must be new. Open its `index.html` in a browser; it needs no server or external
JavaScript. The existing generated report is `outputs/reference-comparison/index.html`.

The report includes a season selector (Ls 45, 135, 225, 315 degrees), four-source
surface-field grids, model-minus-reference grids, MOLA elevation, configuration
and sampling tables, Gaussian-area-weighted metrics, CSV tables, input hashes,
and AmesCAP NetCDF/template exports. Each field has the same color limits across
sources; difference maps use shared symmetric limits centered at zero.

## Meaning of the comparisons

The included configuration uses the existing physical baseline. It does not use
or imply results from the pending 0.25-sol calibration. That calibration writes
parameters and losses, not a seasonal climate dataset. To compare calibrated
climate, freeze the selected parameters, run a separate physical evaluation with
the same averaging protocol as the baseline, export its fields, and add it to a
new report configuration. An improvement in the short calibration loss does not
establish equilibrium or held-out climate improvement.

The sources have different scientific roles:

| Source | Role | Current sampling |
|---|---|---|
| Physical GCM | Our deterministic model | Sparse instantaneous checkpoints averaged in a seasonal window |
| Ames FV3 | Independent model reference | Duration-weighted five-sol averages |
| MCD 6.1 | Model-derived climatology | Four cached fixed-local-time samples at a nearby season |
| ARCO-MACDA | Reanalysis | MY34–35 daily means in the seasonal window |

These are diagnostic comparisons. Exact local-time/averaging alignment remains
unresolved, winds compare different effective heights, dust scenarios differ,
and MCD frost is a monthly statistic. MCD's four local times are a sampled mean,
not an exact continuous daily mean. Original NetCDF attributes and input hashes
remain in `report.json`. MOLA contours provide common geographic context; they
do not assert identical terrain across the source models.

The parameter table distinguishes framework-specific calibration multipliers
from comparable output quantities. Ames/MCD/ARCO multiplier entries remain
unavailable rather than being guessed. Model multipliers in the supplied config
are declared baseline values from the progress report, not recovered from each
seasonal file.

## Configuration contract

Paths resolve relative to `root`, itself relative to the configuration file.
The first source is the model used in every difference and metric. Each source
must declare its name, role, sampling, vertical level, dust scenario and paths.
Patterns may include `{season:03d}` and glob wildcards. Duplicate source names
are rejected.

For time-dependent inputs, declare `season_variable` (a one-dimensional variable
on `time`), `mean_dims: ["time"]`, and any explicit `isel` choices such as
`{"lev": -1}`. Files are selected within the configured circular Ls half-width.
All selected daily/time samples receive equal weight; unequal-duration averages
must be preprocessed with explicit duration weights, as done by the existing
Ames evaluator. No last-time or last-level choice is made implicitly.

The adapter supports the staged surface-temperature, pressure, wind-speed and
frost schemas. It converts frost mass (kg/m²) to pressure equivalent with Mars
gravity, accepts documented equivalent velocity-unit spellings, and rejects
unknown or missing units. ARCO wind speed is formed from each daily vector
before averaging those speeds. Missing variables are labeled unavailable;
unresolved dimensions and non-finite comparison fields are errors.

Metrics are bias, MAE, RMSE, spatial correlation and model/reference means.
Gaussian latitude quadrature is used on the T21 grid. Correlation is `null`/N/A
for constant fields. Reference interpolation is periodic in longitude.

## AmesCAP presentation

[AmesCAP](https://github.com/NASA-Planetary-Science/AmesCAP) is NASA's **Community
Analysis Pipeline**. Its native workflows target Ames GCM output. Our export
adapter supplies already-selected two-dimensional latitude/longitude NetCDF
fields for the model, Ames, MCD and ARCO, plus the common MOLA field `zsurf`.
It does not invent time series, pressure levels, or native FV3 coordinates.

Each season contains:

```text
amescap/ls045/
  comparison.in
  source1/comparison.nc   # physical GCM
  source2/comparison.nc   # Ames
  source3/comparison.nc   # MCD
  source4/comparison.nc   # ARCO
```

`comparison.in` uses MarsPlot's multi-simulation `@N` notation, `HOLD ON` panels,
shared field ranges, difference expressions, and terrain contours. Paths in the
template are absolute: regenerate or edit them after moving the report.

Install CAP in a separate environment according to its
[official instructions](https://amescap.readthedocs.io/en/latest/installation.html),
then run from a directory where CAP may write its `plots/` folder:

```sh
rtk proxy MarsPlot /absolute/path/to/report/amescap/ls045/comparison.in -ftype png --debug
```

Older CAP installations expose `MarsPlot.py` instead of `MarsPlot` and use
`-o png` instead of `-ftype png`. Use PNG to
avoid a dependency on system PDF/ghostscript tooling. The report's own PNGs and
HTML do not require CAP at all. CAP is a presentation layer: numerical metrics
are computed by the CLI before export. Cite AmesCAP separately when publishing
its figures, along with Ames, MCD, ARCO-MACDA/MACDA and MOLA data sources.

The native Ames archive can also be processed with `MarsInterp`, `MarsVars` and
`MarsPlot`; the CLI's existing `scripts/benchmark_ames.py`,
`scripts/benchmark_mcd.py`, and `scripts/evaluate_arco_seasonal.py` remain the
source-specific evaluation tools. Full pressure-level profile and zonal plots
need matching vertical coordinates and separate exports; the new common grid
covers the shared surface diagnostics only.

### Verification in this workspace

The CLI generated all four seasonal reports and 48 field/reference metric rows.
The generated Ls45 template rendered all 28 panels (eight PNG pages) using
MarsPlot 3.3 from AmesCAP commit
`f3d23713566285709ade7f70198b3448c445b276`, with temporary NumPy 1.26.4 and
netCDF4 dependencies. Its plots are in
`outputs/reference-comparison/amescap-rendered/plots/`.
The project environment was not downgraded.

Current AmesCAP (MarsPlot 3.5, checked commit
`df04a413a9a228fe8aa31b3dccc3acb6c8f5b442`) uses `-ftype png` and requires
`~/.amescap_profile` from the official installation instructions. That user
configuration is absent here, so the current-release renderer was not fully
validated. It can exit zero after reporting a missing profile; verify that
actual plot files were produced. The successful render above is explicitly the
3.3 compatibility test, not a claim that all CAP releases were tested.
