#!/usr/bin/env python3
"""Create a 3:11 Mars videoclip with two main characters."""

from __future__ import annotations

from pathlib import Path
import subprocess

import cv2
import numpy as np


FPS = 24
DURATION_SECONDS = 191  # exact requested duration (3m11s)
FRAME_COUNT = FPS * DURATION_SECONDS
SIZE = (1280, 720)  # width, height

OUTPUT_PATH = Path("outputs/mars_main_characters_3m11s.mp4")

CHAR1_PATH = Path("data/characters/main_character_1.png")
CHAR2_PATH = Path("data/characters/main_character_2.png")

BACKGROUND_PATHS = [
    Path("outputs/mars_crustal_magnetism_relief_composite.png"),
    Path("outputs/mars_south_pole_melt_map.png"),
    Path("outputs/mars_wider_hellas_melt_fit_global_map.png"),
    Path("outputs/mars_crustal_magnetism_from_tif_with_legend.png"),
    Path("data/mars_mgs_mola_dem_mosaic_global_1024.jpg"),
    Path("outputs/mars_viking_global_clean.png"),
    Path("outputs/mars_crustal_magnetism_pdf_121nT_location.png"),
]

SCENE_SECONDS = [28, 28, 27, 27, 27, 27, 27]  # sums to 191
assert sum(SCENE_SECONDS) == DURATION_SECONDS


def resize_cover(img: np.ndarray, out_w: int, out_h: int) -> np.ndarray:
    """Resize preserving aspect ratio and crop center."""
    h, w = img.shape[:2]
    scale = max(out_w / w, out_h / h)
    nw, nh = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_CUBIC)
    x0 = (nw - out_w) // 2
    y0 = (nh - out_h) // 2
    return resized[y0 : y0 + out_h, x0 : x0 + out_w]


def make_mars_style(bg: np.ndarray, phase: float) -> np.ndarray:
    """Give background a martian sky/soil cinematic style."""
    out_h, out_w = SIZE[1], SIZE[0]
    base = resize_cover(bg, out_w, out_h).astype(np.float32)

    # Slow zoom for scene motion.
    zoom = 1.0 + 0.08 * phase
    zw, zh = int(out_w * zoom), int(out_h * zoom)
    zimg = cv2.resize(base.astype(np.uint8), (zw, zh), interpolation=cv2.INTER_CUBIC)
    cx = (zw - out_w) // 2
    cy = (zh - out_h) // 2
    frame = zimg[cy : cy + out_h, cx : cx + out_w].astype(np.float32)

    # Mars sky/soil gradient overlays.
    y = np.linspace(0.0, 1.0, out_h, dtype=np.float32)[:, None]
    top_tint = np.array([45, 80, 150], dtype=np.float32)   # BGR cool dusk sky
    bottom_tint = np.array([55, 105, 180], dtype=np.float32)  # BGR warm dust/soil
    top_alpha = np.clip(0.28 * (1.0 - y / 0.62), 0.0, 0.28)
    bottom_alpha = np.clip(0.34 * ((y - 0.45) / 0.55), 0.0, 0.34)
    frame = frame * (1.0 - top_alpha[..., None]) + top_tint * top_alpha[..., None]
    frame = frame * (1.0 - bottom_alpha[..., None]) + bottom_tint * bottom_alpha[..., None]

    # Vignette for cinematic depth.
    xx = np.linspace(-1.0, 1.0, out_w, dtype=np.float32)
    yy = np.linspace(-1.0, 1.0, out_h, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(xx, yy)
    rr = np.sqrt(grid_x * grid_x + grid_y * grid_y)
    vignette = np.clip(1.0 - 0.35 * (rr**1.6), 0.55, 1.0)
    frame *= vignette[..., None]

    return np.clip(frame, 0, 255).astype(np.uint8)


def add_card(frame: np.ndarray, char_img: np.ndarray, x: int, y: int, w: int, h: int, title: str) -> None:
    """Place one character portrait card with translucent panel."""
    h_frame, w_frame = frame.shape[:2]
    x = max(0, min(x, w_frame - w))
    y = max(0, min(y, h_frame - h))

    panel = frame.copy()
    cv2.rectangle(panel, (x - 14, y - 50), (x + w + 14, y + h + 16), (12, 20, 36), thickness=-1)
    cv2.addWeighted(panel, 0.40, frame, 0.60, 0.0, dst=frame)

    resized = resize_cover(char_img, w, h)
    frame[y : y + h, x : x + w] = resized
    cv2.rectangle(frame, (x, y), (x + w, y + h), (245, 245, 245), thickness=3)
    cv2.putText(frame, title, (x, y - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 240, 240), 2, cv2.LINE_AA)


def main() -> None:
    for p in [CHAR1_PATH, CHAR2_PATH, *BACKGROUND_PATHS]:
        if not p.exists():
            raise FileNotFoundError(f"Missing required file: {p}")

    char1 = cv2.imread(str(CHAR1_PATH), cv2.IMREAD_COLOR)
    char2 = cv2.imread(str(CHAR2_PATH), cv2.IMREAD_COLOR)
    bgs = [cv2.imread(str(p), cv2.IMREAD_COLOR) for p in BACKGROUND_PATHS]

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{SIZE[0]}x{SIZE[1]}",
        "-r",
        str(FPS),
        "-i",
        "-",
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        str(OUTPUT_PATH),
    ]
    proc = subprocess.Popen(ffmpeg_cmd, stdin=subprocess.PIPE)
    if proc.stdin is None:
        raise RuntimeError("Could not open ffmpeg stdin pipe.")

    scene_starts = []
    acc = 0
    for s in SCENE_SECONDS:
        scene_starts.append(acc)
        acc += s * FPS

    for i in range(FRAME_COUNT):
        # Which scene are we in.
        scene_idx = 0
        for j, st in enumerate(scene_starts):
            if i >= st:
                scene_idx = j
        scene_start = scene_starts[scene_idx]
        scene_len = SCENE_SECONDS[scene_idx] * FPS
        t = (i - scene_start) / max(1, scene_len - 1)  # 0..1 in scene

        bg = make_mars_style(bgs[scene_idx % len(bgs)], t)

        # Move cards gently left-right for life.
        x1 = int(70 + 34 * np.sin(2.0 * np.pi * t))
        y1 = int(160 + 18 * np.sin(2.0 * np.pi * t + 0.7))
        x2 = int(760 + 34 * np.sin(2.0 * np.pi * t + 2.2))
        y2 = int(190 + 18 * np.sin(2.0 * np.pi * t + 1.4))

        add_card(bg, char1, x1, y1, 360, 360, "Main Character 01")
        add_card(bg, char2, x2, y2, 360, 360, "Main Character 02")

        # Headline and footer.
        cv2.rectangle(bg, (0, 0), (SIZE[0], 64), (8, 12, 20), thickness=-1)
        cv2.putText(
            bg,
            "Mars Project Anthem Videoclip - Main Characters on Mars",
            (24, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.90,
            (242, 242, 242),
            2,
            cv2.LINE_AA,
        )
        cv2.rectangle(bg, (0, SIZE[1] - 34), (SIZE[0], SIZE[1]), (8, 12, 20), thickness=-1)
        elapsed = i / FPS
        mm = int(elapsed // 60)
        ss = int(elapsed % 60)
        cv2.putText(
            bg,
            f"Mars sky + soil local backgrounds | Duration target 03:11 | t={mm:02d}:{ss:02d}",
            (20, SIZE[1] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (220, 220, 220),
            1,
            cv2.LINE_AA,
        )

        proc.stdin.write(bg.tobytes())

    proc.stdin.close()
    ret = proc.wait()
    if ret != 0:
        raise RuntimeError(f"ffmpeg failed with exit code {ret}")
    print(f"Saved video: {OUTPUT_PATH}")
    print(f"Frames: {FRAME_COUNT}, FPS: {FPS}, Duration: {FRAME_COUNT / FPS:.3f}s")


if __name__ == "__main__":
    main()
