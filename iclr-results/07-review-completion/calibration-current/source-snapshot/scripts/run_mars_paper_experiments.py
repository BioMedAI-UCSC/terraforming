#!/usr/bin/env python3
"""Run versioned trajectory recovery, paired ablations, and timing experiments."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "apps" / "mars-calibration"))
from mars_calibration.paper import main

if __name__ == "__main__":
    raise SystemExit(main())
