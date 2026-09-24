"""Shared, paper-consistent matplotlib styling for ICLR figures.

Import `apply_style()` once at program start. Use the exported PALETTE and
helper functions so every figure shares fonts, colors, spines and export
settings. Figures are saved as both PNG (raster preview) and PDF (vector, for
LaTeX \\includegraphics).
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

# Colorblind-friendly, print-safe qualitative palette (Okabe-Ito derived).
PALETTE = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "yellow": "#F0E442",
    "grey": "#7F7F7F",
    "ink": "#222222",
}
SEQ = [PALETTE["blue"], PALETTE["orange"], PALETTE["green"], PALETTE["red"],
       PALETTE["purple"], PALETTE["sky"], PALETTE["grey"]]

# Stable per-reference and per-method colors so they match across all figures.
REFERENCE_COLORS = {
    "NASA Ames": PALETTE["blue"],
    "MCD 6.1": PALETTE["orange"],
    "ARCO-MACDA": PALETTE["green"],
}
METHOD_COLORS = {
    "local": PALETTE["blue"],
    "trajectory": PALETTE["orange"],
    "continued_local": PALETTE["green"],
}
METHOD_LABELS = {
    "local": "Local fitting",
    "trajectory": "Trajectory fine-tune",
    "continued_local": "Continued local",
}


def apply_style() -> None:
    """Install the shared rcParams. Idempotent."""
    mpl.rcParams.update({
        "figure.dpi": 130,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.03,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "axes.labelcolor": PALETTE["ink"],
        "axes.edgecolor": "#444444",
        "axes.linewidth": 0.9,
        "axes.grid": True,
        "axes.grid.axis": "y",
        "grid.color": "#DDDDDD",
        "grid.linewidth": 0.8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "xtick.color": PALETTE["ink"],
        "ytick.color": PALETTE["ink"],
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.frameon": False,
        "legend.fontsize": 9.5,
        "lines.linewidth": 2.0,
        "lines.markersize": 5,
        "axes.prop_cycle": mpl.cycler(color=SEQ),
    })


def prettify_case(name: str) -> str:
    """Human-readable ablation case label."""
    return (name.replace("_", " ")
            .replace("co2", "CO2").replace("pbl", "PBL")
            .replace("aam", "AAM").strip().capitalize())


def save(fig, outdir: Path, stem: str) -> list[Path]:
    """Save a figure to PNG+PDF and return the written paths."""
    outdir.mkdir(parents=True, exist_ok=True)
    written = []
    for ext in ("png", "pdf"):
        p = outdir / f"{stem}.{ext}"
        fig.savefig(p)
        written.append(p)
    plt.close(fig)
    return written


def annotate_provenance(fig, text: str) -> None:
    """Small footer noting the data source / caveat."""
    fig.text(0.005, 0.002, text, fontsize=6.5, color=PALETTE["grey"],
             ha="left", va="bottom")
