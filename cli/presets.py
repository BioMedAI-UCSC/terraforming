"""Registry of built-in simulation presets for each supported planet."""

from __future__ import annotations

# ── Mars presets ──────────────────────────────────────────────────────────────

MARS_PRESET_NAMES = [
    "current-mars",
    "gale-crater",
    "early-mars",
    "terraforming-phase1",
    "equatorial",
    "polar",
    "landmark-spots",
    "olympus-mons",
    "elysium-mons",
    "hellas-basin",
    "south-polar-cap",
]

MARS_PRESET_DESCRIPTIONS = {
    "current-mars":       "Present-day Mars, default parameters (baseline)",
    "gale-crater":        "Gale Crater — Curiosity MSL site (4.6°S, 137.4°E, -4500 m)",
    "early-mars":         "Noachian-era Mars: warmer, wetter, thicker CO₂ atmosphere",
    "terraforming-phase1":"Post-CO₂ release: 5 kPa surface pressure, 240 K start",
    "equatorial":         "Multi-latitude survey: 45°N, equator, 40°S at 137°E",
    "polar":              "North polar cap site (85°N) — CO₂ ice sublimation study",
    "landmark-spots":     "All 4 landmark sites simultaneously (type=spots)",
    "olympus-mons":       "Olympus Mons flank (18.65°N, 226.2°E, +2000 m, 210 K, 508 Pa)",
    "elysium-mons":       "Elysium Mons (25.02°N, 147.21°E, +1500 m, 213 K, 533 Pa)",
    "hellas-basin":       "Hellas Basin (39.0°S, 61.0°E, -4000 m, 225 K, 872 Pa)",
    "south-polar-cap":    "South Polar Cap (73.0°S, 305.0°E, +1800 m, 157 K, 519 Pa)",
}

MARS_PRESET_TAGS = {
    "current-mars":        ["baseline", "fast", "single-sol"],
    "gale-crater":         ["accurate", "rems", "single-sol", "ground-truth"],
    "early-mars":          ["noachian", "fast", "year-long", "high-pressure"],
    "terraforming-phase1": ["terraforming", "fast", "year-long", "high-pressure"],
    "equatorial":          ["multi-coord", "accurate", "single-sol"],
    "polar":               ["polar", "fast", "single-sol", "high-albedo"],
    "landmark-spots":      ["spots", "fast", "multi-site", "single-sol"],
    "olympus-mons":        ["landmark", "fast", "single-sol", "volcano", "high-altitude"],
    "elysium-mons":        ["landmark", "fast", "single-sol", "volcano"],
    "hellas-basin":        ["landmark", "fast", "single-sol", "basin", "high-pressure"],
    "south-polar-cap":     ["landmark", "fast", "single-sol", "polar", "high-albedo"],
}

# ── Planet registry ───────────────────────────────────────────────────────────

SUPPORTED_PLANETS = ["mars"]

PLANET_DESCRIPTIONS = {
    "mars": "Mars (fourth planet) — CO₂ atmosphere, polar caps, 687-day year",
}
