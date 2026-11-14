#!/usr/bin/env python3
"""Draw a Mars mantle map from local geophysical ND models."""

from __future__ import annotations

from pathlib import Path
import re
import zipfile

import cv2
import numpy as np


ZIP_PATH = Path("data/mantle/Models_Geophysical.zip")
OUTPUT_PATH = Path("outputs/mars_mantle_velocity_map_local.png")


def _model_num(name: str) -> int:
    m = re.search(r"Geophysical_model(\d+)\.nd$", name)
    return int(m.group(1)) if m else 10**9


def load_mantle_profiles(zip_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return depth grid (km), Vp profiles, Vs profiles for mantle interval."""
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".nd")]
        names.sort(key=_model_num)

        depth_grid = np.linspace(50.0, 1549.732, 650, dtype=np.float64)
        vp_profiles = []
        vs_profiles = []

        for name in names:
            lines = zf.read(name).decode("utf-8", errors="replace").splitlines()
            in_mantle = False
            d_vals: list[float] = []
            vp_vals: list[float] = []
            vs_vals: list[float] = []

            for line in lines:
                s = line.strip().lower()
                if not s:
                    continue
                if s == "mantle":
                    in_mantle = True
                    continue
                if s in {"outer-core", "outer core", "core"}:
                    in_mantle = False
                if not in_mantle:
                    continue

                parts = line.split()
                if len(parts) < 4:
                    continue
                try:
                    d, vp, vs = float(parts[0]), float(parts[1]), float(parts[2])
                except ValueError:
                    continue
                d_vals.append(d)
                vp_vals.append(vp)
                vs_vals.append(vs)

            if len(d_vals) < 3:
                continue

            d_arr = np.asarray(d_vals, dtype=np.float64)
            vp_arr = np.asarray(vp_vals, dtype=np.float64)
            vs_arr = np.asarray(vs_vals, dtype=np.float64)

            order = np.argsort(d_arr)
            d_arr, vp_arr, vs_arr = d_arr[order], vp_arr[order], vs_arr[order]

            # Deduplicate repeated depth points at interfaces.
            uniq_d, idx = np.unique(d_arr, return_index=True)
            vp_u = vp_arr[idx]
            vs_u = vs_arr[idx]

            vp_profiles.append(np.interp(depth_grid, uniq_d, vp_u))
            vs_profiles.append(np.interp(depth_grid, uniq_d, vs_u))

    if not vp_profiles:
        raise RuntimeError("No mantle profiles could be parsed from zip.")
    return depth_grid, np.asarray(vp_profiles), np.asarray(vs_profiles)


def draw_panel(canvas: np.ndarray, panel: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
    """Draw one heatmap panel with simple color scale."""
    p_min = float(np.percentile(panel, 2))
    p_max = float(np.percentile(panel, 98))
    if p_max <= p_min:
        p_max = p_min + 1e-6

    norm = np.clip((panel - p_min) / (p_max - p_min), 0.0, 1.0)
    img_u8 = (norm * 255.0).astype(np.uint8)
    heat = cv2.applyColorMap(img_u8, cv2.COLORMAP_INFERNO)
    heat = cv2.resize(heat, (w, h), interpolation=cv2.INTER_CUBIC)
    canvas[y : y + h, x : x + w] = heat
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (240, 240, 240), 1)

    cv2.putText(canvas, title, (x, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (245, 245, 245), 2, cv2.LINE_AA)
    cv2.putText(
        canvas,
        f"scale: {p_min:.2f} to {p_max:.2f} km/s",
        (x + w - 255, y + h + 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing mantle models zip: {ZIP_PATH}")

    depth_km, vp_models, vs_models = load_mantle_profiles(ZIP_PATH)

    # Keep model index order vertically and depth horizontally.
    vp_panel = vp_models
    vs_panel = vs_models

    h, w = 980, 1800
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (18, 18, 24)

    cv2.putText(
        canvas,
        "Mars Mantle Map (Local Geophysical Models)",
        (38, 54),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.1,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Source: data/mantle/Models_Geophysical.zip | Ensemble of ND seismic models",
        (40, 84),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    panel_w, panel_h = 820, 360
    x0 = 80
    y1 = 150
    y2 = 560
    draw_panel(canvas, vp_panel, x0, y1, panel_w, panel_h, "Mantle P-wave velocity (Vp)")
    draw_panel(canvas, vs_panel, x0, y2, panel_w, panel_h, "Mantle S-wave velocity (Vs)")

    # Axes and guide labels.
    n_models = vp_panel.shape[0]
    depth_min, depth_max = float(depth_km[0]), float(depth_km[-1])
    for y in (y1, y2):
        # Vertical model index ticks.
        for frac, label in [(0.0, "1"), (0.5, str(n_models // 2)), (1.0, str(n_models))]:
            yy = int(y + frac * panel_h)
            cv2.line(canvas, (x0 - 6, yy), (x0, yy), (220, 220, 220), 1)
            cv2.putText(canvas, label, (x0 - 54, yy + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)
        cv2.putText(canvas, "Model #", (x0 - 70, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

        # Horizontal depth ticks.
        for frac, label in [(0.0, f"{depth_min:.0f}"), (0.33, "550"), (0.66, "1050"), (1.0, f"{depth_max:.0f}")]:
            xx = int(x0 + frac * panel_w)
            cv2.line(canvas, (xx, y + panel_h), (xx, y + panel_h + 5), (220, 220, 220), 1)
            cv2.putText(canvas, label, (xx - 12, y + panel_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1, cv2.LINE_AA)

    cv2.putText(canvas, "Depth in mantle (km)", (x0 + 300, y2 + panel_h + 46), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (235, 235, 235), 1, cv2.LINE_AA)

    # Summary stats block.
    vp_med = np.median(vp_models, axis=0)
    vs_med = np.median(vs_models, axis=0)
    summary = [
        f"Models parsed: {n_models}",
        f"Mantle depth span: {depth_min:.1f} - {depth_max:.1f} km",
        f"Median Vp range: {vp_med.min():.2f} - {vp_med.max():.2f} km/s",
        f"Median Vs range: {vs_med.min():.2f} - {vs_med.max():.2f} km/s",
    ]
    bx, by = 980, 210
    cv2.rectangle(canvas, (bx - 18, by - 38), (w - 60, by + 180), (28, 28, 36), -1)
    cv2.rectangle(canvas, (bx - 18, by - 38), (w - 60, by + 180), (105, 105, 125), 1)
    cv2.putText(canvas, "Mantle ensemble summary", (bx, by - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (245, 245, 245), 2, cv2.LINE_AA)
    for i, line in enumerate(summary):
        cv2.putText(canvas, line, (bx, by + 28 + i * 34), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (225, 225, 225), 1, cv2.LINE_AA)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"Saved mantle map: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
