#!/usr/bin/env python3
"""Compatibility entry point for the standalone Mars calibration application."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "mars-calibration"))
from mars_calibration.driver import main

if __name__ == "__main__":
    raise SystemExit(main())
