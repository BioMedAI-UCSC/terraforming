"""Reference Mars climatology for calibrating / validating the simulator.

**What this is.** A small, cited set of observed seasonal cycles the 0-D model is
compared against. The primary target is the **Viking Lander 1 surface-pressure
annual cycle** — the canonical Mars CO₂-cycle benchmark. VL1 sits at **22.3°N**,
which matches the simulator's default site latitude (22°N), so the comparison is
latitude-consistent without interpolation.

**Provenance & honesty.** These are *observational* reference curves (Viking
Landers), digitised coarsely (every 30° of Ls) from the published daily-mean
pressure cycles in Hess et al. (1980) and Tillman (1993). They are **not** a live
extraction from the Mars Climate Database (MCD) 6.1 — MCD is gated behind its
access tools and is not fetched here. They are, however, exactly the curves MCD
6.1 (and every Mars GCM) is validated against: Guo et al. (2009) fit a GCM to
these same Viking pressure curves. The :class:`ReferenceClimatology.source` field
distinguishes ``"viking-lander"`` from ``"mcd-6.1"`` so a direct MCD 6.1 seasonal
extraction (same shape: ``ls_deg`` + ``pressure_pa`` [+ ``temperature_k``]) can be
dropped in unchanged when available.

References
----------
- Hess, S.L., Henry, R.M., Tillman, J.E. (1980). "The annual cycle of pressure on
  Mars measured by Viking Landers 1 and 2." *Geophys. Res. Lett.* 7(3), 197-200.
  https://doi.org/10.1029/GL007i003p00197
- Tillman, J.E. et al. (1993). "The Martian annual atmospheric pressure cycle:
  Years without great dust storms." *JGR* 98(E6). https://doi.org/10.1029/93JE01084
- Guo, X. et al. (2009). "Fitting the Viking lander surface pressure cycle with a
  Mars General Circulation Model." *JGR* 114, E07006.
  https://doi.org/10.1029/2008JE003302
"""

from __future__ import annotations

import dataclasses

import numpy as np

# Solar-longitude sample points shared by the digitised curves (every 30°).
_LS_DEG = np.arange(0.0, 360.0, 30.0)


@dataclasses.dataclass(frozen=True)
class ReferenceClimatology:
    """A site's observed seasonal cycle, sampled on ``ls_deg``.

    Pressure is the calibration target; ``temperature_k`` is optional (``None``
    when no latitude-matched daily-mean surface-temperature climatology is
    encoded — kept honest rather than fabricated).
    """

    name: str
    latitude_deg: float
    elevation_m: float
    ls_deg: np.ndarray
    pressure_pa: np.ndarray | None
    temperature_k: np.ndarray | None
    source: str  # "viking-lander" | "mcd-6.1" | ...
    citation: str
    notes: str = ""

    def __post_init__(self) -> None:
        for field in ("pressure_pa", "temperature_k"):
            arr = getattr(self, field)
            if arr is not None and len(arr) != len(self.ls_deg):
                raise ValueError(
                    f"{field} length {len(arr)} != ls_deg length {len(self.ls_deg)}"
                )
        if self.pressure_pa is None and self.temperature_k is None:
            raise ValueError("a reference must carry at least one of pressure/temperature")


def viking_lander_1() -> ReferenceClimatology:
    """VL1 (22.3°N, ~ -3.6 km) daily-mean surface-pressure annual cycle.

    The canonical benchmark: two broad maxima preceding each cap's winter
    solstice, deep minimum near Ls ~150 (southern-cap maximum), main maximum near
    Ls ~270. Annual mean ~7.9 hPa, ~26 % peak-to-peak swing driven by the CO₂
    cycle. Values in Pa, digitised from the Hess (1980) / Tillman (1993) curves.
    """
    pressure_pa = np.array(
        [760, 780, 790, 770, 730, 690, 710, 760, 840, 890, 850, 790],
        dtype=float,
    )
    return ReferenceClimatology(
        name="Viking Lander 1",
        latitude_deg=22.3,
        elevation_m=-3627.0,
        ls_deg=_LS_DEG.copy(),
        pressure_pa=pressure_pa,
        temperature_k=None,
        source="viking-lander",
        citation="Hess et al. 1980 (GRL); Tillman et al. 1993 (JGR)",
        notes=(
            "Coarse (30° Ls) digitisation of the published VL1 daily-mean "
            "pressure cycle. Annual mean ~790 Pa; MCD 6.1 reproduces this curve."
        ),
    )


def viking_lander_2() -> ReferenceClimatology:
    """VL2 (47.6°N, ~ -4.5 km) — higher-latitude, larger-amplitude cycle.

    Provided for extension/genericity; values are approximate digitisations.
    """
    pressure_pa = np.array(
        [880, 910, 900, 850, 790, 730, 770, 850, 980, 1080, 1010, 920],
        dtype=float,
    )
    return ReferenceClimatology(
        name="Viking Lander 2",
        latitude_deg=47.6,
        elevation_m=-4505.0,
        ls_deg=_LS_DEG.copy(),
        pressure_pa=pressure_pa,
        temperature_k=None,
        source="viking-lander",
        citation="Hess et al. 1980 (GRL); Tillman et al. 1993 (JGR)",
        notes="Approximate digitisation. Higher-latitude, ~39 % pressure swing.",
    )


# Registry — keyed by short id; extend with an "mcd-6.1" entry when available.
REFERENCES = {
    "vl1": viking_lander_1,
    "vl2": viking_lander_2,
}


def get_reference(key: str) -> ReferenceClimatology:
    """Look up a reference climatology by key (e.g. ``"vl1"``)."""
    if key not in REFERENCES:
        raise KeyError(f"unknown reference {key!r}; known: {sorted(REFERENCES)}")
    return REFERENCES[key]()
