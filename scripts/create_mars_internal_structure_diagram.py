#!/usr/bin/env python3
"""Create Mars internal structure diagram (core->crust) from local model."""

from __future__ import annotations

from pathlib import Path
import zipfile

import cv2
import numpy as np


ZIP_PATH = Path("data/mantle/Models_Geophysical.zip")
OUTPUT_IMG = Path("outputs/mars_internal_structure_core_to_crust.png")
OUTPUT_TXT = Path("outputs/mars_internal_structure_core_to_crust_summary.txt")


def extract_depth_boundaries(zip_path: Path) -> tuple[float, float, float, float]:
    """
    Extract key depth boundaries (km) from one local ND model.

    Returns:
      planet_radius_km, moho_depth_km, cmb_depth_km, icb_depth_km
    """
    with zipfile.ZipFile(zip_path) as zf:
        text = zf.read("Models_Geophysical/Geophysical_model1.nd").decode("utf-8", errors="replace").splitlines()

    def next_depth(start_idx: int) -> float:
        for j in range(start_idx + 1, len(text)):
            parts = text[j].split()
            if len(parts) < 4:
                continue
            try:
                return float(parts[0])
            except ValueError:
                continue
        raise RuntimeError("Could not parse numeric depth after marker.")

    moho_depth = None
    cmb_depth = None
    icb_depth = None
    max_depth = 0.0

    for i, line in enumerate(text):
        parts = line.split()
        if parts:
            try:
                max_depth = max(max_depth, float(parts[0]))
            except ValueError:
                pass

        s = line.strip().lower()
        if s == "mantle":
            moho_depth = next_depth(i)
        elif s in {"outer-core", "outer core"}:
            cmb_depth = next_depth(i)
        elif s in {"inner-core", "inner core"}:
            icb_depth = next_depth(i)

    if moho_depth is None or cmb_depth is None or icb_depth is None:
        raise RuntimeError("Missing one or more layer markers in ND model.")

    return max_depth, moho_depth, cmb_depth, icb_depth


def main() -> None:
    if not ZIP_PATH.exists():
        raise FileNotFoundError(f"Missing local model zip: {ZIP_PATH}")

    planet_r, moho_d, cmb_d, icb_d = extract_depth_boundaries(ZIP_PATH)

    crust_th = moho_d
    mantle_th = cmb_d - moho_d
    core_th = planet_r - cmb_d
    core_r = planet_r - cmb_d
    inner_core_r = max(planet_r - icb_d, 0.0)

    # Temperature ranges are estimated bounds commonly used in Mars interior studies.
    # (The local ND seismic files do not include temperature columns.)
    crust_temp = "-63 to ~500 C"
    mantle_temp = "~500 to ~1800 C"
    core_temp = "~1600 to ~2200 C"

    w, h = 1600, 1000
    canvas = np.zeros((h, w, 3), dtype=np.uint8)
    canvas[:] = (16, 14, 20)

    cx, cy = 520, 560
    outer_px = 350
    scale = outer_px / planet_r

    r_planet = int(round(planet_r * scale))
    r_cmb = int(round((planet_r - cmb_d) * scale))
    r_icb = max(2, int(round((planet_r - icb_d) * scale)))

    # Draw concentric layers.
    cv2.circle(canvas, (cx, cy), r_planet, (70, 120, 190), thickness=-1)   # crust shell color base
    cv2.circle(canvas, (cx, cy), int(round((planet_r - moho_d) * scale)), (55, 90, 150), thickness=-1)  # mantle outer
    cv2.circle(canvas, (cx, cy), r_cmb, (45, 70, 110), thickness=-1)        # core
    cv2.circle(canvas, (cx, cy), r_icb, (220, 220, 240), thickness=-1)      # tiny inner core

    # Repaint annuli for visual distinction.
    cv2.circle(canvas, (cx, cy), r_planet, (180, 130, 90), thickness=12)  # crust rim
    cv2.circle(canvas, (cx, cy), int(round((planet_r - moho_d) * scale)), (95, 70, 40), thickness=10)  # mantle top boundary
    cv2.circle(canvas, (cx, cy), r_cmb, (230, 150, 90), thickness=8)  # CMB boundary

    # Header.
    cv2.putText(
        canvas,
        "Mars Internal Structure (Core to Crust)",
        (70, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Layer lengths from local ND model (Models_Geophysical.zip) | Temperatures are estimated ranges",
        (72, 112),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (205, 205, 205),
        1,
        cv2.LINE_AA,
    )

    # Label helpers.
    def label(txt: str, x: int, y: int, color: tuple[int, int, int] = (240, 240, 240)) -> None:
        cv2.putText(canvas, txt, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.68, color, 2, cv2.LINE_AA)

    # Leader lines and labels.
    cv2.line(canvas, (cx + r_planet - 8, cy - 180), (980, 250), (230, 230, 230), 2)
    label(f"Crust thickness: {crust_th:.1f} km", 1000, 245)
    label(f"Crust temperature: {crust_temp}", 1000, 278)

    mantle_r_mid = int(round((planet_r - moho_d - mantle_th * 0.45) * scale))
    cv2.line(canvas, (cx + mantle_r_mid, cy - 20), (980, 430), (230, 230, 230), 2)
    label(f"Mantle thickness: {mantle_th:.1f} km", 1000, 425)
    label(f"Mantle temperature: {mantle_temp}", 1000, 458)

    core_anchor = (cx + max(8, r_cmb // 2), cy + 20)
    cv2.line(canvas, core_anchor, (980, 610), (230, 230, 230), 2)
    label(f"Core radius: {core_r:.1f} km", 1000, 605)
    label(f"Core temperature: {core_temp}", 1000, 638)

    label(f"Planet radius: {planet_r:.1f} km", 1000, 710, color=(210, 220, 255))
    label(f"CMB depth: {cmb_d:.1f} km", 1000, 742, color=(210, 220, 255))
    label(f"Tiny inner core radius (model artifact): {inner_core_r:.1f} km", 1000, 774, color=(210, 220, 255))

    # Small legend blocks.
    ly = 840
    cv2.rectangle(canvas, (80, ly - 20), (105, ly + 5), (180, 130, 90), -1)
    label("Crust", 120, ly)
    cv2.rectangle(canvas, (260, ly - 20), (285, ly + 5), (95, 70, 40), -1)
    label("Mantle", 300, ly)
    cv2.rectangle(canvas, (500, ly - 20), (525, ly + 5), (45, 70, 110), -1)
    label("Core", 540, ly)

    OUTPUT_IMG.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_IMG), canvas)

    summary = [
        "Mars Internal Structure Summary",
        f"Source model: {ZIP_PATH}",
        f"Planet radius (km): {planet_r:.3f}",
        f"Crust thickness (km): {crust_th:.3f}",
        f"Mantle thickness (km): {mantle_th:.3f}",
        f"Core radius (km): {core_r:.3f}",
        f"CMB depth (km): {cmb_d:.3f}",
        f"Inner core radius in model (km): {inner_core_r:.3f}",
        "",
        "Estimated temperature ranges used in drawing:",
        f"- Crust: {crust_temp}",
        f"- Mantle: {mantle_temp}",
        f"- Core: {core_temp}",
    ]
    OUTPUT_TXT.write_text("\n".join(summary), encoding="utf-8")
    print(f"Saved diagram: {OUTPUT_IMG}")
    print(f"Saved summary: {OUTPUT_TXT}")


if __name__ == "__main__":
    main()
