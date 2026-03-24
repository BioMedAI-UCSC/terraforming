---
description: >
  Governs how tests are written, where they live, and when they must be created
  or updated. Applies to all code in package/ and cli/. Every approved code
  change must ship with tests. Untestable code must be flagged for refactoring.
paths:
  - package/**
  - cli/**
---

# Testing Rule

## When This Rule Applies

After **every approved code change** in `package/` or `cli/`, you MUST:

1. Write or update tests that cover the changed behaviour.
2. Flag any code that cannot be tested in its current form (see "Untestable Code" below).
3. Never mark a task complete if new public functions/classes have zero test coverage.

---

## Test Layout

Tests mirror the source tree exactly. Every source module gets one test file.

```
package/
  src/
    framework/atmosphere.py      →  package/tests/framework/test_atmosphere.py
    framework/thermal.py         →  package/tests/framework/test_thermal.py
    engine/time_controller.py    →  package/tests/engine/test_time_controller.py   ← exists
    celestials/planets/mars.py   →  package/tests/mars/basic/test_planet_interface.py ← exists

cli/
  config_loader.py   →  cli/tests/test_config_loader.py
  models.py          →  cli/tests/test_models.py
  runner.py          →  cli/tests/test_runner.py
  output.py          →  cli/tests/test_output.py
  presets.py         →  cli/tests/test_presets.py
  main.py            →  cli/tests/test_main.py
```

Each test directory must have an `__init__.py`.

---

## Test Categories and What to Write

### Unit tests  (required for every public function/class/method)

Cover:
- Happy-path: expected inputs produce expected outputs
- Boundary values: zero, negative, extremes, empty collections
- Invariants: physical laws that must always hold (e.g. temperature > 0 K)

### Integration tests  (required when a module calls another module)

Cover:
- CLI → runner → package round-trips (e.g. `run_sol` produces non-empty history)
- Config loading + validation pipeline
- Preset → SimConfig expansion

### Regression tests  (required when fixing a bug)

Name the test after the bug. Include a comment with the issue or symptom.
```python
def test_greenhouse_factor_below_one_raises():
    # Regression: values < 1 silently produced negative flux
```

---

## Test File Template

```python
"""Tests for <module_path>.

Covers:
  - <what is tested, one bullet per class/function>
"""

from __future__ import annotations

import pytest  # or unittest — match the existing style in the submodule

# ── Fixtures ──────────────────────────────────────────────────────────────────

# Put shared setup here (e.g. a default Mars() or default SimConfig).


# ── <ClassName / function_name> ───────────────────────────────────────────────

class Test<Name>:

    def test_<behaviour>_given_<condition>(self):
        """One sentence: what this asserts and why it matters."""
        # arrange
        # act
        # assert

    def test_<edge_case>(self):
        """Describe the boundary or error case."""
        with pytest.raises(ValueError):
            ...
```

Naming convention: `test_<behaviour>_given_<condition>` or `test_<behaviour>_raises_on_<condition>`.

---

## CI Automation Requirements

Every test file must be runnable via:

```
# package
cd package && python -m pytest tests/ -v

# cli
cd cli && python -m pytest tests/ -v
```

Rules for CI-compatible tests:
- **No I/O side effects** — do not write to `outputs/`, do not open matplotlib windows. Use `tmp_path` (pytest fixture) for any file output.
- **No network calls** — all external data must be fixtures or constants.
- **No interactive prompts** — CLI tests must invoke Click commands via `CliRunner`, never via subprocess.
- **Deterministic** — tests must not depend on wall-clock time, random seeds, or filesystem state outside the test.
- **Fast** — unit tests must complete in < 2 s each. Integration tests < 10 s each. Mark slow tests with `@pytest.mark.slow`.

---

## CLI Testing Pattern (Click commands)

Use `click.testing.CliRunner` to test every CLI command. Never invoke via subprocess.

```python
from click.testing import CliRunner
from cli.main import cli

def test_mars_config_list_exits_zero():
    runner = CliRunner()
    result = runner.invoke(cli, ["mars", "config", "list"])
    assert result.exit_code == 0

def test_mars_run_invalid_preset_exits_nonzero():
    runner = CliRunner()
    result = runner.invoke(cli, ["mars", "run", "--preset", "nonexistent"])
    assert result.exit_code != 0

def test_mars_config_validate_valid_file(tmp_path):
    cfg_file = tmp_path / "test.yaml"
    cfg_file.write_text("planet:\n  surface_temperature: 210.0\n")
    runner = CliRunner()
    result = runner.invoke(cli, ["mars", "config", "validate", str(cfg_file)])
    assert result.exit_code == 0
```

---

## Physics / Numerical Test Patterns

For physics modules, always assert on physically meaningful invariants, not just
that the function returned something.

```python
# Good — tests physical correctness
def test_surface_temperature_is_positive_after_step():
    mars = Mars()
    tc = TimeController(mars, dt=3600.0, accuracy=Accuracy.FAST)
    history = tc.run(duration=3600.0 * 24)
    for snap in history:
        assert float(snap.surface_temperature.item()) > 0

# Bad — only tests that it didn't crash
def test_run_does_not_raise():
    Mars(); ...
```

Tolerance guidelines:
- Temperature comparisons: `±1 K` absolute or `±0.5 %` relative
- Pressure comparisons: `±10 Pa`
- Use `pytest.approx(expected, rel=1e-3)` for floating-point equality

---

## Flagging Untestable Code

If you encounter code that cannot be covered by an automated test in its current
form, you MUST:

1. Add an inline comment directly above the offending block:
   ```python
   # UNTESTABLE: <reason> — needs refactor to inject <dependency>
   ```

2. List it in the test file under a dedicated section:
   ```python
   # ── Untestable (requires refactor) ───────────────────────────────────────────
   # cli/main.py :: _wizard_mars — reads from stdin via input(); inject IO stream
   # cli/runner.py :: _styled_temp — depends on click.style side effects; extract pure fn
   ```

3. Common reasons and the required refactor:
   | Reason | Required fix |
   |--------|-------------|
   | Calls `input()` directly | Accept an `io.TextIOBase` parameter |
   | Calls `sys.exit()` | Raise a typed exception instead; catch at top level |
   | Hardcoded file paths | Accept `pathlib.Path` parameter |
   | Global mutable state | Wrap in a class or accept state as parameter |
   | Side effects mixed with logic | Extract pure computation into a separate function |

---

## Coverage Expectations

| Module | Minimum line coverage |
|--------|-----------------------|
| `package/src/framework/` | 95 % |
| `package/src/engine/` | 95 % |
| `package/src/celestials/` | 95 % |
| `cli/config_loader.py` | 90 % |
| `cli/models.py` | 95 % |
| `cli/runner.py` | 75 % |
| `cli/presets.py` | 90 % |
| `cli/output.py` | 70 % |
| `cli/main.py` | 65 % |

Run coverage with:
```
cd package && python -m pytest tests/ --cov=src --cov-report=term-missing
cd cli     && python -m pytest tests/ --cov=cli --cov-report=term-missing
```
