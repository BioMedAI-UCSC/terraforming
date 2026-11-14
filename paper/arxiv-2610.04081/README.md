# AEGIS revision against arXiv v1

The user selected [2610.04081v1](https://arxiv.org/html/2610.04081v1) as the
paper of record. `v1-source.tar.gz` is the unchanged source downloaded from
arXiv; `source.json` records its URL, size and SHA-256. `revision/` is the
proposed revision, including the compiled PDF, not a replacement of arXiv v1.
Other ignored local manuscript drafts are untouched.

Start with [the simplified reproduction guide](../../docs/paper-reproduction.md).
From the repository root, run:

```sh
rtk proxy .venv/bin/python paper/arxiv-2610.04081/regenerate.py
rtk proxy latexmk -cd -pdf -interaction=nonstopmode -halt-on-error paper/arxiv-2610.04081/revision/iclr2027_conference.tex
```

Corrections include the surface-frost sign, seasonal labels and Ls225 panel,
duplicate/mis-authored Forget bibliography entry, explicit parameter roles and
Ames-informed tuning, full bias/wind/frost metrics, calibration convergence and
two-year uncertainty limits, raw versus projected conservation wording, missing
atmospheric condensation and mixture limitations, unsupported external speedup,
unverified headline timing/equilibration claims, and a new fixed-update neural
radiation table/figure. The five disputed bibliography keys already identify
the correct papers in v1; `reference-audit.json` records those dispositions
against archived DOI registration metadata rather than changing correct authors.

The radiation run is new: clean commit `1b8e7a8`, seed 0, CPU float64, 100 local
updates and 20 updates for each continuation. An independent repeat agrees in
the saved evaluation metrics and checkpoint arrays to rtol 1e-10/atol 1e-12.
Original CPU model identification was unavailable in this sandbox; no CPU/GPU
or external timing-speedup claim is made. The full reports preserve that fact.

`evidence/historical-audit` is a new analysis of **old model outputs**, not new
current-physics climate evidence. It bundles recovered small-data calibration
logs/checkpoint and reference/timing summaries. Its raw annual NetCDF inputs
remain unavailable in this public bundle. `claims.json` identifies what was
recomputed, retained, withdrawn, and still requires runs. Do not close the
remaining scientific issues solely because their limitations are now disclosed.
