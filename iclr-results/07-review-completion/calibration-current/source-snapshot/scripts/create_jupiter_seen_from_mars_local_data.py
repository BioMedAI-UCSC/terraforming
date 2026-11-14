#!/usr/bin/env python3
"""Render Jupiter as seen from Mars using only local constants/code."""

from __future__ import annotations

from pathlib import Path
import math

import cv2
import numpy as np


OUTPUT_PATH = Path("outputs/jupiter_seen_from_mars_local_data.png")

# Local geometric/planet constants (km, AU).
AU_KM = 149_597_870.7
MARS_SEMI_MAJOR_AU = 1.523679
JUPITER_SEMI_MAJOR_AU = 5.2044
JUPITER_RADIUS_KM = 69_911.0

# Rendering configuration for "naked-eye style" scene.
WIDTH = 1920
HEIGHT = 1080
HORIZONTAL_FOV_DEG = 60.0


def jupiter_apparent_diameter_arcsec(distance_au: float) -> float:
    """Angular diameter (arcsec) from center-to-center distance in AU."""
    distance_km = distance_au * AU_KM
    angular_diameter_rad = 2.0 * math.atan2(JUPITER_RADIUS_KM, distance_km)
    return math.degrees(angular_diameter_rad) * 3600.0


def make_mars_sky(h: int, w: int) -> np.ndarray:
    y = np.linspace(0.0, 1.0, h, dtype=np.float32)[:, None]
    top = np.array([28, 45, 82], dtype=np.float32)  # dusty dusk BGR
    bottom = np.array([75, 118, 180], dtype=np.float32)
    sky = np.zeros((h, w, 3), dtype=np.float32)
    for c in range(3):
        sky[:, :, c] = top[c] * (1.0 - y) + bottom[c] * y

    # Mild noise texture.
    noise = np.random.default_rng(8).normal(0.0, 4.0, size=(h, w, 3)).astype(np.float32)
    sky = np.clip(sky + noise, 0.0, 255.0)
    return sky.astype(np.uint8)


def add_stars(img: np.ndarray, count: int = 150) -> None:
    rng = np.random.default_rng(21)
    h, w = img.shape[:2]
    for _ in range(count):
        x = int(rng.integers(0, w))
        y = int(rng.integers(20, int(h * 0.75)))
        b = int(rng.integers(110, 210))
        cv2.circle(img, (x, y), 1, (b, b, b), -1, lineType=cv2.LINE_AA)


def add_mars_horizon(img: np.ndarray) -> None:
    h, w = img.shape[:2]
    terrain = np.array(
        [
            [0, h],
            [0, int(h * 0.74)],
            [int(w * 0.10), int(h * 0.70)],
            [int(w * 0.23), int(h * 0.73)],
            [int(w * 0.40), int(h * 0.69)],
            [int(w * 0.58), int(h * 0.72)],
            [int(w * 0.78), int(h * 0.68)],
            [w, int(h * 0.71)],
            [w, h],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(img, [terrain], color=(48, 82, 146))

    # Foreground shadow layer.
    overlay = img.copy()
    cv2.rectangle(overlay, (0, int(h * 0.78)), (w, h), (25, 40, 70), -1)
    cv2.addWeighted(overlay, 0.42, img, 0.58, 0.0, img)


def add_jupiter_point(img: np.ndarray, x: int, y: int, apparent_px: float) -> None:
    # Jupiter is usually sub-pixel at naked-eye scene scale from Mars.
    bright = np.zeros_like(img)
    core_radius = max(1, int(round(apparent_px)))
    cv2.circle(bright, (x, y), core_radius, (225, 235, 255), -1, lineType=cv2.LINE_AA)
    cv2.circle(bright, (x, y), core_radius + 4, (140, 160, 220), 1, lineType=cv2.LINE_AA)
    blurred = cv2.GaussianBlur(bright, (0, 0), sigmaX=2.2, sigmaY=2.2)
    cv2.addWeighted(blurred, 0.95, img, 1.0, 0.0, img)


def draw_jupiter_zoom_inset(canvas: np.ndarray, arcsec_now: float) -> None:
    """Inset: illustrative telescope-like Jupiter at fixed zoom."""
    h, w = canvas.shape[:2]
    inset_w, inset_h = 500, 500
    x0, y0 = w - inset_w - 40, 50

    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + inset_w + 1, y0 + inset_h + 1), (210, 210, 215), 1)
    panel = canvas[y0 : y0 + inset_h, x0 : x0 + inset_w]
    panel[:] = (18, 18, 24)

    cx, cy = inset_w // 2, inset_h // 2 + 10
    j_radius = 145
    cv2.circle(panel, (cx, cy), j_radius, (172, 202, 240), -1, lineType=cv2.LINE_AA)

    # Jupiter-like cloud bands.
    for i, yy in enumerate(range(cy - j_radius + 8, cy + j_radius - 8, 26)):
        color = (145 + (i % 2) * 20, 178 + (i % 2) * 12, 218 + (i % 2) * 5)
        cv2.ellipse(panel, (cx, yy), (j_radius - 12, 10), 0, 0, 360, color, -1, lineType=cv2.LINE_AA)
    cv2.ellipse(panel, (cx + 40, cy + 32), (32, 18), -8, 0, 360, (128, 166, 205), -1, lineType=cv2.LINE_AA)

    # Limb darkening.
    vignette = np.zeros((inset_h, inset_w), dtype=np.float32)
    yy, xx = np.indices((inset_h, inset_w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / j_radius
    vignette = np.clip(1.0 - 0.35 * rr, 0.62, 1.0)
    for c in range(3):
        panel[:, :, c] = np.clip(panel[:, :, c].astype(np.float32) * vignette, 0, 255).astype(np.uint8)

    # Galilean moons as points.
    moon_y = cy - 185
    for mx in (cx - 190, cx - 112, cx + 95, cx + 205):
        cv2.circle(panel, (mx, moon_y), 4, (210, 210, 220), -1, lineType=cv2.LINE_AA)

    cv2.putText(
        canvas,
        "Illustrative zoom (not scene scale)",
        (x0 + 18, y0 + inset_h + 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Jupiter apparent diameter now: {arcsec_now:.1f} arcsec",
        (x0 + 18, y0 + inset_h + 52),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (208, 208, 208),
        1,
        cv2.LINE_AA,
    )


def main() -> None:
    # Circular-orbit distance bounds for local estimate:
    # closest roughly when aligned same heliocentric direction, farthest when opposite.
    d_closest_au = JUPITER_SEMI_MAJOR_AU - MARS_SEMI_MAJOR_AU
    d_farthest_au = JUPITER_SEMI_MAJOR_AU + MARS_SEMI_MAJOR_AU
    d_mid_au = 0.5 * (d_closest_au + d_farthest_au)

    arcsec_closest = jupiter_apparent_diameter_arcsec(d_closest_au)
    arcsec_farthest = jupiter_apparent_diameter_arcsec(d_farthest_au)
    arcsec_now = jupiter_apparent_diameter_arcsec(d_mid_au)

    # Convert apparent angular diameter to pixels in scene.
    px_per_deg = WIDTH / HORIZONTAL_FOV_DEG
    apparent_px = (arcsec_now / 3600.0) * px_per_deg

    canvas = make_mars_sky(HEIGHT, WIDTH)
    add_stars(canvas)
    add_mars_horizon(canvas)

    # Place Jupiter in upper-right sky.
    jx, jy = int(WIDTH * 0.70), int(HEIGHT * 0.24)
    add_jupiter_point(canvas, jx, jy, apparent_px)

    cv2.putText(
        canvas,
        "Jupiter seen from Mars (local-data geometric estimate)",
        (46, 62),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (245, 245, 245),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "Scene shows realistic naked-eye scale; inset shows telescopic detail.",
        (48, 92),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.62,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"Distance range Mars-Jupiter: {d_closest_au:.3f}-{d_farthest_au:.3f} AU | "
            f"Jupiter angular size: {arcsec_closest:.1f}\" to {arcsec_farthest:.1f}\""
        ),
        (48, HEIGHT - 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (212, 212, 212),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        f"Current illustrative midpoint size in scene: {arcsec_now:.1f}\" (~{apparent_px:.2f} px at {HORIZONTAL_FOV_DEG:.0f}deg FOV)",
        (48, HEIGHT - 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.54,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    draw_jupiter_zoom_inset(canvas, arcsec_now=arcsec_now)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUTPUT_PATH), canvas)
    print(f"Saved: {OUTPUT_PATH}")
    print(f"Closest angular diameter: {arcsec_closest:.2f} arcsec")
    print(f"Farthest angular diameter: {arcsec_farthest:.2f} arcsec")


if __name__ == "__main__":
    main()
