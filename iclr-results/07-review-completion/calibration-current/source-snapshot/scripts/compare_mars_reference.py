#!/usr/bin/env python3
"""Compare matched 2-D Mars fields from Dinosaur, Ames MGCM, or MCD.

Inputs must already select the same season, temporal statistic and vertical
level. Native time/level selection is deliberately explicit in preprocessing.
Variable/coordinate renaming is accepted as a JSON object, e.g.
--rename '{"grid_xt":"lon","grid_yt":"lat","ps":"surface_pressure"}'.
No implicit unit conversion or vertical extrapolation is performed.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import xarray as xr
from benchmark_mcd import interpolate_periodic, weighted_metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", type=Path)
    parser.add_argument("reference", type=Path)
    parser.add_argument("--source", choices=["Ames MGCM", "MCD"], required=True)
    parser.add_argument("--rename", default="{}")
    parser.add_argument("--fields", default="surface_pressure,temperature")
    parser.add_argument("--protocol", type=Path, required=True,
                        help="JSON recording season, sampling, vertical coordinate, dust, spinup and source version")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    required = {"season", "sampling", "vertical_coordinate", "dust", "spinup", "source_version"}
    if required - protocol.keys():
        parser.error(f"protocol missing: {sorted(required - protocol.keys())}")
    names = args.fields.split(",")
    with xr.open_dataset(args.model) as opened:
        model = opened.load()
    with xr.open_dataset(args.reference) as opened:
        reference = opened.rename(json.loads(args.rename)).load()
    for name in names:
        for dataset in (model, reference):
            if set(dataset[name].dims) != {"lat", "lon"}:
                parser.error(f"{name}: select/average time and vertical coordinates explicitly before comparing")
        if model[name].attrs.get("units") != reference[name].attrs.get("units") or not model[name].attrs.get("units"):
            parser.error(f"{name}: convert both fields to identical explicit units first")
    regridded = interpolate_periodic(reference[names], model)
    metrics = {}
    for name in names:
        metrics[name] = weighted_metrics(model[name].transpose("lat", "lon").values,
                                         regridded[name].transpose("lat", "lon").values,
                                         model.lat.values)
        metrics[name]["reference_area_mean"] = metrics[name].pop("mcd_area_mean")
    report = dict(status="model_reference_comparison_not_observational_validation",
                  source=args.source, protocol=protocol, metrics=metrics,
                  model_sha256=hashlib.sha256(args.model.read_bytes()).hexdigest(),
                  reference_sha256=hashlib.sha256(args.reference.read_bytes()).hexdigest(),
                  matching="User-supplied protocol; physical matching requires review",
                  regridding="periodic linear horizontal interpolation; Gaussian target quadrature")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
