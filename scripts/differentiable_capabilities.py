#!/usr/bin/env python3
"""Run JAX-MD-style capability experiments (sensitivity maps, design, ensembles)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "apps" / "mars-calibration"))
sys.path.insert(0, str(ROOT / "package"))
from mars_calibration.capabilities import main

if __name__ == "__main__":
    raise SystemExit(main())
