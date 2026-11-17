# Module test jobs

The `Tests` workflow owns the Python regression suite and UI build. The planner
in `scripts/ci.py` gives each test file one owner. It fails if a file is missing,
assigned twice, or added without a category.

| Suite | Coverage |
| --- | --- |
| dynamics | Coordinates, dynamics, maps, initialization, restarts and runner controls |
| radiation | Gas and particle optics, spectral fluxes and radiation solvers |
| surface | Soil, frost, CO2 microphysics, PBL, convection and drag |
| particles | Natural dust, water, settling and tracer transport |
| coupled | Global Dinosaur physics, budgets, feedbacks, restarts and derivatives |
| neural | Neural physics and training budgets |
| aerosols | Engineered aerosol state, optics, transport and experiments |
| legacy | Torch planet model, JAX 0-D prototypes, engine and gas interventions |
| cli | Command line, server and output contracts |
| calibration | Calibration application and analysis regressions |
| runtime | Nautilus source preparation, job templates and experiment runner |

Pull requests select suites from changed paths. Physics changes also select
downstream consumers and coupled tests. Changes to shared model code run all
Python suites except the independent deployment runtime. Dependency, workflow,
shared fixture and unknown changes run the full Python suite. UI changes build
the UI. Documentation-only changes skip numerical tests. The planner summary
lists the selected and skipped suites.

Every push to `main` runs the full suite. Use **Actions → Tests → Run workflow**
to run all suites or one named suite. Select **Include tests marked slow** when
those tests are needed. Data-dependent tests still require their documented
external inputs.

The coupled suite has four parallel partitions. Each coupled case runs in its
own process to release JAX compilation memory. Other suites use one process per
test file. Jobs save JUnit results and slow-test timings as seven-day artifacts.
All numerical jobs install the locked GCM dependencies. The runtime suite only
installs pytest. The stable `tests` and `gcm3d-tests` gates pass only when every
selected job passes; intentionally skipped modules do not leave a pending gate.

The aerosol pilot is a separate manual workflow. It does not duplicate the
regression suite. Paper revision and paper experiment workflows are removed.

Run the same selection locally:

```sh
python scripts/ci.py --plan --base main --head HEAD
python scripts/ci.py --plan --full
env PYTHONPATH=package:.:scripts:apps/nautalis python scripts/ci.py --suite radiation
env PYTHONPATH=package:.:scripts:apps/nautalis python scripts/ci.py --suite coupled --shard 0 --shards 4
python -m unittest discover -s scripts/tests -p test_ci.py -v
```
