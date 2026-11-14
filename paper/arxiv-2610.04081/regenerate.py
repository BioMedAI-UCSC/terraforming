"""Verify frozen evidence and regenerate the revised numerical tables/figure."""
import csv
import hashlib
import json
from pathlib import Path
import re

import numpy as np

ROOT = Path(__file__).resolve().parent


def read(path):
    return json.loads(path.read_text())


def verify_claims(root):
    claims = read(root / "claims.json")["claims"]
    assert len({row["result"] for row in claims}) == len(claims), "Duplicate claim"
    for row in claims:
        assert row["issues"] and row["disposition"], row["result"]
        for relative in row.get("evidence", []):
            path = (root / relative).resolve()
            assert path.is_relative_to(root.parents[1]) and path.exists(), relative
            assert path.stat().st_size if path.is_file() else any(path.iterdir()), relative
    maps = [root / f"revision/assets/ls{season:03d}_grid.png" for season in (45, 135, 225, 315)]
    assert len({hashlib.sha256(p.read_bytes()).hexdigest() for p in maps}) == 4, "Duplicate seasonal map"


def main():
    verify_claims(ROOT)
    source = read(ROOT / "source.json")
    assert hashlib.sha256((ROOT / source["original_archive"]).read_bytes()).hexdigest() == source["sha256"]
    for entry in read(ROOT / "evidence-manifest.json")["files"]:
        assert hashlib.sha256((ROOT / entry["path"]).read_bytes()).hexdigest() == entry["sha256"], entry["path"]
    report = read(ROOT / "evidence/radiation-fixed/report.json")
    repeat = read(ROOT / "evidence/radiation-repeat/report.json")
    assert report["comparison_budget"] == repeat["comparison_budget"] == dict(
        mode="updates", accuracy_claim="fixed-update single-seed capability comparison",
        trajectory_updates=20, continued_local_updates=20)
    for name in ("local", "trajectory", "continued_local"):
        for split in ("test", "extrapolation"):
            a = report["column_evaluation"][split]["48"]["models"][name]
            b = repeat["column_evaluation"][split]["48"]["models"][name]
            assert a["failed_columns"] == b["failed_columns"] == 0
            np.testing.assert_allclose(a["air_rmse_k"], b["air_rmse_k"], rtol=1e-10, atol=1e-12)
        for filename in (name + ".npz",):
            with np.load(ROOT / "evidence/radiation-fixed" / filename, allow_pickle=False) as a, \
                 np.load(ROOT / "evidence/radiation-repeat" / filename, allow_pickle=False) as b:
                assert a.files == b.files
                for key in a.files:
                    if np.issubdtype(a[key].dtype, np.number):
                        np.testing.assert_allclose(a[key], b[key], rtol=1e-10, atol=1e-12)
                    else:
                        np.testing.assert_array_equal(a[key], b[key])
    tex = (ROOT / "revision/iclr2027_conference.tex").read_text()
    bib = (ROOT / "revision/iclr2027_conference.bib").read_text()
    cited = {key.strip() for group in re.findall(r"\\cite\w*\{([^}]+)\}", tex) for key in group.split(",")}
    keys = re.findall(r"@\w+\s*\{\s*([^,]+),", bib)
    assert len(keys) == len(set(keys)) and cited <= set(keys), sorted(cited - set(keys))
    assert "forget1999improved" not in cited

    assets = ROOT / "revision/assets"
    rows = list(csv.DictReader((ROOT / "evidence/historical-audit/reference-summary.csv").open()))
    fields = {"Surface pressure": "Pressure (Pa)", "Surface temperature": "Temperature (K)",
              "Near-surface wind speed": "Wind (m/s)", "Surface CO2 frost": r"Frost (Pa-eq.)"}
    lines = [r"\begin{table}[t]\centering\small",
        r"\caption{Historical reference characterization, using the arithmetic mean of four seasonal statistics per row (not pooled spatial RMSE). Bias is model minus reference. Temporal sampling and wind heights are unmatched; opacity controls were Ames-informed.}",
        r"\label{tab:reference}\label{mola-table}", r"\begin{tabular}{llrrr}\toprule",
        r"Reference & Field & Bias & RMSE & Correlation \\\midrule"]
    for reference in ("NASA Ames", "MCD 6.1", "ARCO-MACDA"):
        for field, label in fields.items():
            row = next(r for r in rows if r["reference"] == reference and r["field"] == field)
            lines.append(f"{reference} & {label} & {float(row['bias']):+.2f} & {float(row['rmse']):.2f} & {float(row['correlation']):.3f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (assets / "reference.tex").write_text("\n".join(lines) + "\n")
    labels = [("local", "Initial local fit"), ("trajectory", "Trajectory, 20 updates"),
              ("continued_local", "Continued local, 20 updates")]
    values = np.array([[report["column_evaluation"][split]["48"]["models"][name]["air_rmse_k"]
                        for split in ("test", "extrapolation")] for name, _ in labels])
    lines = [r"\begin{table}[t]\centering\small",
        r"\caption{Fixed-update, seed-0, 48-step column air-temperature RMSE (K). Both continuations start from the same 100-update local fit. Generated physical-kernel targets; not observational Mars skill.}",
        r"\label{tab:radiation_results}\begin{tabular}{lrr}\toprule",
        r"Training & Interpolation & Higher dust \\\midrule"]
    for (_, label), pair in zip(labels, values):
        lines.append(f"{label} & {pair[0]:.3f} & {pair[1]:.3f} " + r"\\")
    lines += [r"\bottomrule\end{tabular}\end{table}"]
    (assets / "radiation-fixed.tex").write_text("\n".join(lines) + "\n")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 2, figsize=(7, 2.8), layout="constrained")
    for index, ax in enumerate(axes):
        ax.bar(range(3), values[:, index], color=["#627c96", "#b35c39", "#537756"])
        ax.set_xticks(range(3), ["Local", "Trajectory\n+20", "Local\n+20"])
        ax.set(title=["Interpolation", "Higher dust"][index], ylabel="48-step air RMSE (K)")
    fig.savefig(assets / "neural-radiation-48step.pdf", metadata={"CreationDate": None})
    plt.close(fig)
    print("Evidence hashes, fixed-update repeat, checkpoints, citations and tables verified")


if __name__ == "__main__":
    main()
