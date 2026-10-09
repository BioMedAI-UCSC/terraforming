#!/usr/bin/env python3
"""Analyze station/retrieval matchups, scaling and annual convergence.

Matchups must be prepared with an explicit observation operator; this script
does not mistake boundary maps or reanalyses for held-out observations.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def matchup_metrics(frame):
    required = {"configuration", "field", "units", "sample_id", "block", "observed", "predicted", "weight"}
    if not required <= set(frame):
        raise ValueError(f"Missing columns: {required-set(frame)}")
    if frame[list(required)].isna().any().any() or frame.duplicated(["configuration", "field", "sample_id"]).any():
        raise ValueError("Missing values or duplicate matchups")
    if not np.isfinite(frame[["observed", "predicted", "weight"]]).all().all() or (frame.weight <= 0).any():
        raise ValueError("Expected finite matchups and positive weights")
    for field, group in frame.groupby("field"):
        if group.units.nunique() != 1:
            raise ValueError("Mixed units")
        reference = None
        for _, sub in group.groupby("configuration"):
            aligned = sub.sort_values("sample_id").set_index("sample_id")[["observed", "weight", "block"]]
            if reference is not None and not aligned.equals(reference):
                raise ValueError("Configurations must use identical paired samples, observations, weights and blocks")
            reference = aligned
    rows = []
    for (configuration, field), sub in frame.groupby(["configuration", "field"]):
        err = sub.predicted - sub.observed
        rows.append(dict(configuration=configuration, field=field, units=sub.units.iloc[0],
            rmse=float(np.sqrt(np.average(err**2, weights=sub.weight))),
            bias=float(np.average(err, weights=sub.weight)), samples=len(sub), blocks=sub.block.nunique()))
    return pd.DataFrame(rows)


def climate(frame, year_sols, bin_sols=20.):
    if not np.isfinite([year_sols, bin_sols]).all() or min(year_sols, bin_sols) <= 0:
        raise ValueError("Year and bin durations must be finite and positive")
    fields = ["mean_surface_pressure_pa", "mean_co2_ice_pa", "mean_surface_temperature_k", "mean_deep_soil_temperature_k"]
    if not {"elapsed_sols", *fields} <= set(frame):
        raise ValueError("Missing checkpoint fields")
    t = frame.elapsed_sols.to_numpy()
    if not np.isfinite(frame[["elapsed_sols", *fields]]).all().all() or np.any(np.diff(t) <= 0):
        raise ValueError("Nonfinite or unordered diagnostics")
    if t[0] > 0.01 or np.diff(t).max() > bin_sols:
        raise ValueError("Need coverage from initialization and checkpoint spacing <= bin width")
    # Exclude incomplete final years; compare identical elapsed-orbital phase bins.
    n = int(np.floor((t[-1]+1e-6)/year_sols))
    if n < 2:
        raise ValueError("At least two complete orbital years required")
    records = []
    bins = int(np.ceil(year_sols/bin_sols))
    for year in range(n):
        phase = t-year*year_sols
        for b in range(bins):
            lo, hi = b*year_sols/bins, (b+1)*year_sols/bins
            mask = (phase >= lo) & (phase < hi)
            if not mask.any():
                raise ValueError(f"Missing phase bin {b} in year {year+1}")
            for field in fields:
                records.append(dict(year=year+1, phase_sol=(lo+hi)/2, field=field,
                                    mean=float(frame.loc[mask, field].mean()), samples=int(mask.sum())))
    return pd.DataFrame(records)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("task", choices=["observations", "scaling", "climate"])
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, help="Required matchup provenance JSON")
    ap.add_argument("--timing", type=Path, help="Scaling CSV: configuration,median_seconds")
    ap.add_argument("--year-sols", type=float, default=668.6188325204715)
    args = ap.parse_args()
    data = pd.read_csv(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    if args.task == "climate":
        means = climate(data, args.year_sols)
        means.to_csv(args.output / "climate.csv", index=False)
        last = means.year.max()
        a = means[means.year == last-1].set_index(["field", "phase_sol"])["mean"]
        b = means[means.year == last].set_index(["field", "phase_sol"])["mean"]
        delta = (b-a).rename("delta").reset_index()
        delta.to_csv(args.output / "climate-deltas.csv", index=False)
        thresholds = dict(mean_surface_pressure_pa=5., mean_co2_ice_pa=5.,
                          mean_surface_temperature_k=1., mean_deep_soil_temperature_k=.25)
        report = {field: dict(max_absolute_delta=float(sub.delta.abs().max()),
            threshold=thresholds[field], pass_check=bool(sub.delta.abs().max() <= thresholds[field]))
            for field, sub in delta.groupby("field")}
        report["scope"] = "Last two complete years; phase-bin checkpoint means, not observational validation or exact time averages"
    else:
        if args.metadata is None:
            raise ValueError("--metadata required")
        meta = json.loads(args.metadata.read_text())
        for key in ("source", "reference_kind", "observation_operator", "split", "quality_control", "units", "time_alignment"):
            if not meta.get(key):
                raise ValueError(f"Missing provenance: {key}")
        if args.task == "observations" and (meta["reference_kind"] != "observation" or meta["split"] != "test"):
            raise ValueError("Observational validation requires held-out observations")
        metrics = matchup_metrics(data)
        if args.task == "scaling":
            if args.timing is None:
                raise ValueError("--timing required")
            timing = pd.read_csv(args.timing)
            if timing.configuration.duplicated().any() or not np.isfinite(timing.median_seconds).all() or (timing.median_seconds <= 0).any():
                raise ValueError("One finite positive timing per configuration required")
            metrics = metrics.merge(timing, on="configuration", how="left", validate="many_to_one")
            if metrics.median_seconds.isna().any():
                raise ValueError("Missing configuration timing")
        metrics.to_csv(args.output / f"{args.task}.csv", index=False)
        report = dict(provenance=meta, scope="Matched-sample errors; no confidence interval inferred from grid cells")
    import hashlib
    report["input_sha256"] = hashlib.sha256(args.data.read_bytes()).hexdigest()
    (args.output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
