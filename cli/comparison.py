"""Offline, explicit surface comparisons and AmesCAP map exports."""
from __future__ import annotations

import csv
import glob
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

FIELDS = {
    "temperature": ("Surface temperature", "K", "inferno"),
    "surface_pressure": ("Surface pressure", "Pa", "viridis"),
    "wind_speed": ("Near-surface wind speed", "m/s", "magma"),
    "co2_ice": ("Surface CO2 frost", "Pa-equivalent", "Blues"),
}
ALIASES = {
    "temperature": ("temperature", "surface_temperature", "tsurf"),
    "surface_pressure": ("surface_pressure", "psurf", "ps"),
    "wind_speed": ("wind_speed", "surface_wind_speed"),
    "co2_ice": ("co2_ice", "co2_frost", "co2ice"),
}


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def normalize_units(da, field):
    unit = str(da.attrs.get("units", "")).strip()
    accepted = {"temperature": {"K"}, "surface_pressure": {"Pa"},
                "wind_speed": {"m/s", "m s-1", "m s^-1"},
                "co2_ice": {"Pa", "Pa-equivalent", "Pa-equiv"}}
    if field == "co2_ice" and unit in {"kg m-2", "kg/m2", "kg m^-2"}:
        from src.celestials.planets.mars import MARS_BODY_3D
        da = da * MARS_BODY_3D.gravity_m_s2
    elif unit not in accepted[field]:
        raise ValueError(f"{field}: unsupported or missing units {unit!r}")
    da.attrs = {"units": FIELDS[field][1], "long_name": FIELDS[field][0]}
    return da


def load_source(spec, root, season, half_width):
    paths = sorted({Path(p) for pattern in spec["paths"]
                    for p in glob.glob(str(root / pattern.format(season=season)))})
    if not paths:
        raise ValueError(f"{spec['name']}: no files for Ls={season}")
    samples, records = [], []
    for path in paths:
        with xr.open_dataset(path, decode_times=False) as opened:
            ds = opened.load()
        ls_name = spec.get("season_variable")
        season_range = None
        if ls_name:
            mask = np.abs((ds[ls_name] - season + 180) % 360 - 180) <= half_width
            if mask.dims != ("time",):
                raise ValueError("Season selection requires a one-dimensional time coordinate")
            ds = ds.isel(time=np.flatnonzero(mask.values))
            if not ds.sizes["time"]:
                continue
            season_range = [float(ds[ls_name].min()), float(ds[ls_name].max())]
        fields = {}
        for name, aliases in ALIASES.items():
            variable = next((v for v in aliases if v in ds), None)
            if variable:
                fields[name] = normalize_units(ds[variable], name)
        if "wind_speed" not in fields and {"uwind", "vwind"} <= set(ds):
            u = normalize_units(ds.uwind, "wind_speed")
            v = normalize_units(ds.vwind, "wind_speed")
            fields["wind_speed"] = np.hypot(u, v)
            fields["wind_speed"].attrs = {"units": "m/s"}
        selected = xr.Dataset(fields)
        for dim, index in spec.get("isel", {}).items():
            if dim in selected.dims:
                selected = selected.isel({dim: index})
        # Reduction must be declared, never silently take the last time/level.
        for dim in spec.get("mean_dims", []):
            if dim in selected.dims and dim != "time":
                selected = selected.mean(dim, keep_attrs=True, skipna=False)
        if "time" in selected.dims and "time" not in spec.get("mean_dims", []):
            raise ValueError(f"{spec['name']}: explicitly select or average time")
        for da in selected.data_vars.values():
            if set(da.dims) - {"time", "lat", "lon"}:
                raise ValueError(f"{spec['name']}: unresolved dimensions {da.dims}")
        samples.append(selected)
        records.append({"path": str(path), "sha256": sha256(path),
                        "selected_ls_range": season_range,
                        "attributes": {k: str(v) for k, v in ds.attrs.items()}})
    if not samples:
        raise ValueError(f"{spec['name']}: empty seasonal window")
    if any("time" in item.dims for item in samples):
        data = xr.concat(samples, dim="time", join="exact").mean("time", keep_attrs=True, skipna=False)
    elif len(samples) == 1:
        data = samples[0]
    else:
        raise ValueError("Multiple already-averaged files require explicit preprocessing weights")
    data = data.assign_coords(lon=data.lon % 360).sortby("lon").sortby("lat")
    data = data.groupby("lon").mean(keep_attrs=True)
    for name in data:
        if set(data[name].dims) != {"lat", "lon"} or not np.isfinite(data[name]).all():
            raise ValueError(f"{spec['name']}: {name} must be a finite 2-D field")
    return data, records


def metrics(model, reference, lat):
    nodes, weights = np.polynomial.legendre.leggauss(len(lat))
    if not np.allclose(np.sin(np.deg2rad(lat)), nodes, atol=1e-6):
        weights = np.cos(np.deg2rad(lat))
    w = np.broadcast_to(weights[:, None], model.shape).copy()
    if not np.isfinite(model).all() or not np.isfinite(reference).all():
        raise ValueError("Non-finite comparison cells; check spatial coverage")
    w /= w.sum()
    x, y = np.sum(w * model), np.sum(w * reference)
    delta = model - reference
    denominator = np.sqrt(np.sum(w * (model-x)**2) * np.sum(w * (reference-y)**2))
    return {"model_mean": float(x), "reference_mean": float(y),
            "bias": float(np.sum(w * delta)), "mae": float(np.sum(w * abs(delta))),
            "rmse": float(np.sqrt(np.sum(w * delta**2))),
            "correlation": float(np.sum(w * (model-x) * (reference-y)) / denominator)
            if denominator > 0 else None}


def table(rows):
    if not rows:
        return "<p>No comparable fields.</p>"
    def cell(value):
        return html.escape(f"{value:.4g}" if isinstance(value, float) else str(value if value is not None else "N/A"))
    return ("<table><thead><tr>" + "".join(f"<th>{cell(k)}</th>" for k in rows[0]) +
            "</tr></thead><tbody>" + "".join("<tr>" + "".join(f"<td>{cell(v)}</td>" for v in row.values()) +
            "</tr>" for row in rows) + "</tbody></table>")


def write_csv(path, rows):
    if rows:
        with path.open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)


def cap_export(output, sources, terrain, ranges):
    """Export explicitly collapsed 2-D fields; do not invent a time or pressure axis."""
    lines = ["===================== |MarsPlot V3.3| ===================",
             "<<<<<<<<<<<<<<<<<<<<<< Simulations >>>>>>>>>>>>>>>>>>>>>"]
    for index, (name, ds) in enumerate(sources.items(), 1):
        folder = output / f"source{index}"
        folder.mkdir(parents=True, exist_ok=True)
        ds = ds.copy()
        ds["zsurf"] = terrain
        ds.attrs.update(title=name, processing="Preselected 2-D diagnostic maps; no time series or vertical coordinate implied")
        for var in ds:
            ds[var].attrs.setdefault("long_name", var)
        ds.transpose("lat", "lon").to_netcdf(folder / "comparison.nc")
        lines.append(f"{'ref' if index == 1 else index}> {folder.resolve()}")
    lines += ["=======================================================", "START"]
    def panel(title, variable, lo, hi, cmap):
        return ["<<<<<<<<<<<<<<| Plot 2D lon X lat = True |>>>>>>>>>>>>>",
                f"Title = {title}", f"Main Variable = {variable}", f"Cmin, Cmax = {lo},{hi}",
                "Ls 0-360 = None", "Level Pa/m = None", "2nd Variable = comparison.zsurf",
                "Contours Var 2 = 0",
                f"Axis Options : lon = [None,None] | lat = [None,None] | cmap = {cmap} | scale = lin | proj = cart"]
    for field, (label, _, cmap) in FIELDS.items():
        available = [(i, name) for i, (name, ds) in enumerate(sources.items(), 1) if field in ds]
        if not available:
            continue
        lines.append(f"HOLD ON {2 if len(available) > 2 else 1},{(len(available)+1)//2 if len(available) > 2 else len(available)}")
        for index, name in available:
            lines += panel(f"{name} - {label}", f"comparison@{index}.{field}", *ranges[field], cmap)
        lines.append("HOLD OFF")
        if field in next(iter(sources.values())) and len(available) > 1:
            differences = [next(iter(sources.values()))[field] - ds[field] for ds in list(sources.values())[1:] if field in ds]
            limit = max(float(abs(v).max()) for v in differences) or 1
            lines.append(f"HOLD ON {2 if len(differences) > 2 else 1},{(len(differences)+1)//2 if len(differences) > 2 else len(differences)}")
            for index, name in available:
                if index != 1:
                    lines += panel(f"Model minus {name} - {label}", f"[comparison@1.{field}]-[comparison@{index}.{field}]", -limit, limit, "RdBu_r")
            lines.append("HOLD OFF")
    lines.append("STOP")
    (output / "comparison.in").write_text("\n".join(lines) + "\n")


def build_report(config_path, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path("outputs/.matplotlib").resolve()))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.celestials.planets.mars.mcd import interpolate_periodic
    from src.celestials.planets.mars.topography import load_mola_meg

    config_path, output = Path(config_path).resolve(), Path(output)
    config = json.loads(config_path.read_text())
    root = (config_path.parent / config.get("root", ".")).resolve()
    specs = config["sources"]
    if len(specs) < 2 or len({s["name"] for s in specs}) != len(specs):
        raise ValueError("At least two uniquely named sources are required; model must be first")
    for spec in specs:
        for key in ("sampling", "vertical", "dust", "role"):
            if not spec.get(key):
                raise ValueError(f"{spec['name']}: declare {key}")
    output.mkdir(parents=True, exist_ok=False)
    elevation, lat, lon = load_mola_meg(root / config["mola"])
    mola = xr.Dataset({"elevation": (("lat", "lon"), elevation)}, coords={"lat": lat, "lon": lon})
    rows, provenance, sections = [], [], []
    parameters = [{"source": s["name"], "role": s["role"], "sampling": s["sampling"],
                   "wind level": s["vertical"], "dust": s["dust"],
                   "CO2 LW scale": s.get("parameters", {}).get("co2_lw", "not comparable / not supplied"),
                   "dust LW scale": s.get("parameters", {}).get("dust_lw", "not comparable / not supplied"),
                   "exchange scale": s.get("parameters", {}).get("exchange", "not comparable / not supplied")}
                  for s in specs]
    for season in config["seasons"]:
        aligned, source_records = {}, []
        for spec in specs:
            ds, records = load_source(spec, root, season, config.get("half_width", 5))
            if aligned:
                ds = interpolate_periodic(ds, next(iter(aligned.values())))
            aligned[spec["name"]] = ds
            source_records.append({"source": spec["name"], "inputs": records})
        model = next(iter(aligned.values()))
        terrain = interpolate_periodic(mola, model).elevation
        terrain.attrs = {"units": "m", "long_name": "MOLA elevation (common geographic context, not reference-specific terrain)"}
        tag = f"ls{season:03d}"
        ranges = {name: (min(float(ds[name].min()) for ds in aligned.values() if name in ds),
                         max(float(ds[name].max()) for ds in aligned.values() if name in ds))
                  for name in FIELDS if any(name in ds for ds in aligned.values())}
        fig, axes = plt.subplots(4, len(aligned), figsize=(4.4*len(aligned), 11), squeeze=False, constrained_layout=True)
        diff_fig, diff_axes = plt.subplots(4, len(aligned)-1, figsize=(4.4*(len(aligned)-1), 11), squeeze=False, constrained_layout=True)
        season_rows = []
        for row, (field, (label, units, cmap)) in enumerate(FIELDS.items()):
            differences = [model[field] - ds[field] for ds in list(aligned.values())[1:] if field in model and field in ds]
            limit = max([float(abs(v).max()) for v in differences] or [1]) or 1
            for column, (name, ds) in enumerate(aligned.items()):
                ax = axes[row, column]
                ax.set_title(f"{name} · {label}", fontsize=9)
                if field not in ds:
                    ax.text(.5, .5, "Unavailable", transform=ax.transAxes, ha="center")
                else:
                    lo, hi = ranges[field]
                    im = ax.pcolormesh(model.lon, model.lat, ds[field].transpose("lat", "lon"), cmap=cmap, vmin=lo, vmax=hi if hi>lo else lo+1, shading="auto", rasterized=True)
                    ax.contour(model.lon, model.lat, terrain, levels=[0], colors="white", linewidths=.35, alpha=.6)
                    fig.colorbar(im, ax=ax, label=units, shrink=.75)
                ax.set(xlabel="Longitude (°E)", ylabel="Latitude (°N)")
                if column:
                    dax = diff_axes[row, column-1]
                    dax.set(title=f"Model − {name} · {label}", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
                    if field in model and field in ds:
                        values = metrics(model[field].transpose("lat", "lon").values, ds[field].transpose("lat", "lon").values, model.lat.values)
                        record = {"Ls": season, "reference": name, "field": label, "units": units, **values}
                        rows.append(record); season_rows.append(record)
                        im = dax.pcolormesh(model.lon, model.lat, (model[field]-ds[field]).transpose("lat", "lon"), cmap="RdBu_r", vmin=-limit, vmax=limit, shading="auto", rasterized=True)
                        diff_fig.colorbar(im, ax=dax, label=units, shrink=.75)
                    else:
                        dax.text(.5, .5, "Unavailable", transform=dax.transAxes, ha="center")
        fig.suptitle(f"Ls ≈ {season}° · diagnostic comparison · shared scale within each row")
        diff_fig.suptitle(f"Ls ≈ {season}° · model minus reference · sampling/height mismatches remain")
        fig.savefig(output / f"{tag}_grid.png", dpi=150)
        diff_fig.savefig(output / f"{tag}_differences.png", dpi=150)
        plt.close(fig); plt.close(diff_fig)
        cap_export(output / "amescap" / tag, aligned, terrain, ranges)
        provenance.append({"season": season, "sources": source_records})
        sections.append(f'<section data-season="{season}"><h2>Ls ≈ {season}°</h2><img src="{tag}_grid.png" alt="Source comparison grid"><img src="{tag}_differences.png" alt="Difference grid">{table(season_rows)}</section>')
        if season == config["seasons"][0]:
            fig, ax = plt.subplots(figsize=(12, 4), constrained_layout=True)
            im = ax.pcolormesh(model.lon, model.lat, terrain / 1000, cmap="terrain", shading="auto")
            ax.set(title="MOLA elevation on comparison grid — common geographic context", xlabel="Longitude (°E)", ylabel="Latitude (°N)")
            fig.colorbar(im, ax=ax, label="Elevation (km)")
            fig.savefig(output / "mola.png", dpi=170); plt.close(fig)
    write_csv(output / "metrics.csv", rows)
    write_csv(output / "parameters.csv", parameters)
    report = {"status": "diagnostic_comparison_not_observational_validation", "config": config,
              "mola_sha256": sha256(root / config["mola"]), "metrics": rows, "provenance": provenance}
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False))
    (output / "config.json").write_text(json.dumps(config, indent=2))
    notes = "".join(f"<li>{html.escape(note)}</li>" for note in config.get("limitations", []))
    options = "".join(f'<option value="{season}">{season}°</option>' for season in config["seasons"])
    document = f"""<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Mars reference comparison</title><style>body{{font:15px system-ui;background:#10151d;color:#e9eef6;margin:32px;line-height:1.5}}h1,h2{{color:#ffc49c}}table{{border-collapse:collapse;width:100%;margin:20px 0;font-size:13px}}th,td{{border:1px solid #354252;padding:8px;text-align:left}}th{{background:#243040}}img{{width:100%;background:white;margin:12px 0}}select{{padding:8px}}a{{color:#a6d8ff}}.scroll{{overflow:auto}}</style>
<h1>Mars: model, Ames, MCD and ARCO</h1><p>Physical baseline diagnostic. No calibrated or equilibrium improvement is implied.</p>
<ul>{notes}</ul><h2>Configuration and sampling</h2><div class="scroll">{table(parameters)}</div>
<p>Parameter scales are framework-specific; reference values are not inferred.</p><h2>MOLA terrain</h2><img src="mola.png" alt="MOLA terrain">
<label>Season <select id="season">{options}</select></label>{''.join(sections)}
<p><a href="metrics.csv">Metrics CSV</a> · <a href="parameters.csv">Parameters CSV</a> · <a href="report.json">Full report and provenance</a></p>
<script>const select=document.getElementById('season');function show(){{document.querySelectorAll('[data-season]').forEach(s=>s.hidden=s.dataset.season!==select.value)}}select.addEventListener('change',show);show();</script></html>"""
    (output / "index.html").write_text(document)
    return output / "index.html"
