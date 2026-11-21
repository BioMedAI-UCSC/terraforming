#!/usr/bin/env python3
"""Compare fixed-update radiation runs, preserving all physical-time settings."""
import argparse
import json
from pathlib import Path

import numpy as np

TIMING_KEYS = {
    "first_update_compile_and_execution_seconds", "training_seconds",
    "physical_first_call_seconds", "physical_warm_seconds",
    "reference_first_call_seconds", "reference_warm_seconds",
    "first_call_seconds", "warm_seconds",
}


def compare(left, right, path="report"):
    if isinstance(left, dict):
        assert left.keys() == right.keys(), path
        for key in left:
            if key not in TIMING_KEYS and path + "." + key != "report.provenance.arguments.output":
                compare(left[key], right[key], path + "." + key)
    elif isinstance(left, list):
        assert len(left) == len(right), path
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            compare(a, b, f"{path}[{index}]")
    elif isinstance(left, (float, int)) and not isinstance(left, bool):
        np.testing.assert_allclose(left, right, rtol=1e-10, atol=1e-12, equal_nan=False, err_msg=path)
    else:
        assert left == right, path


def verify(first, repeat):
    compare(json.loads((first / "report.json").read_text()),
            json.loads((repeat / "report.json").read_text()))
    names = {p.name for p in first.glob("*.npz")}
    assert names == {p.name for p in repeat.glob("*.npz")} and names, "checkpoint files"
    arrays = 0
    for name in sorted(names):
        with np.load(first / name, allow_pickle=False) as a, np.load(repeat / name, allow_pickle=False) as b:
            assert set(a.files) == set(b.files), name
            for key in a.files:
                if key == "metadata":
                    compare(json.loads(str(a[key])), json.loads(str(b[key])), name + ".metadata")
                else:
                    assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype, (name, key)
                    assert np.isfinite(a[key]).all() and np.isfinite(b[key]).all(), (name, key)
                    np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12,
                                               equal_nan=False, err_msg=name + "." + key)
                    arrays += 1
    return dict(status="passed", numeric_arrays=arrays, rtol=1e-10, atol=1e-12,
                ignored_timing_keys=sorted(TIMING_KEYS),
                limitation="One seed on one CPU; no multi-seed or Mars observational validation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("first", type=Path)
    parser.add_argument("repeat", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.first, args.repeat)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))
