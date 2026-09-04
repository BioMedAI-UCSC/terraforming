"""Tests for the Keplerian orbit in framework GCM physics.

Covers the orbit-correctness items from the physics review:
  - mean anomaly advances uniformly; Kepler's equation is solved for E; the true
    anomaly and heliocentric distance follow from E (not from M directly).
  - perihelion/aphelion flux ratio matches ((1+e)/(1-e))^2.
  - Mars's unequal season lengths are reproduced (the signature of a correct
    eccentric orbit + perihelion at Ls~251 deg).
  - epoch/perihelion/Ls are mutually consistent (mean_anomaly_for_ls inverts the
    chain exactly).
"""

from __future__ import annotations

import math

import numpy as np
import pytest

pytest.importorskip("dinosaur")

from src.celestials.planets.mars import gcm as mars_gcm  # noqa: E402
from src.framework.physics import gcm as P  # noqa: E402


def _forcing():
    return mars_gcm.radiative_forcing()


class TestKeplerSolve:

    def test_eccentric_anomaly_satisfies_keplers_equation(self):
        f = _forcing()
        e = f.eccentricity
        for M in np.linspace(0, 2 * math.pi, 17):
            E = float(P._eccentric_anomaly(M, e))
            assert abs((E - e * math.sin(E)) - M) < 1e-10

    def test_distance_extremes(self):
        f = _forcing()
        e = f.eccentricity
        r_peri = float(P.orbital_distance(0.0, f))                 # M = 0
        r_apo = float(P.orbital_distance(f.orbital_period_s / 2, f))  # M = pi
        assert r_peri == pytest.approx(f.semi_major_axis_m * (1 - e), rel=1e-9)
        assert r_apo == pytest.approx(f.semi_major_axis_m * (1 + e), rel=1e-9)


class TestFlux:

    def test_perihelion_aphelion_ratio(self):
        f = _forcing()
        e = f.eccentricity
        per = float(P.solar_flux(0.0, f))
        apo = float(P.solar_flux(f.orbital_period_s / 2, f))
        assert per / apo == pytest.approx(((1 + e) / (1 - e)) ** 2, rel=1e-6)


class TestSeasonLengths:

    def _t_of_ls(self, ls_deg, f):
        e = f.eccentricity
        nu = math.radians(ls_deg) - f.ls_perihelion_rad
        E = 2 * math.atan2(math.sqrt(1 - e) * math.sin(nu / 2),
                           math.sqrt(1 + e) * math.cos(nu / 2))
        M = (E - e * math.sin(E)) % (2 * math.pi)
        return M / (2 * math.pi) * f.orbital_period_s

    def test_unequal_lengths_match_mars(self):
        """NH season lengths (sols) vs observed 194/178/143/154."""
        f = _forcing()
        sol = f.rotation_period_s

        def seg(a, b):
            return ((self._t_of_ls(b, f) - self._t_of_ls(a, f)) % f.orbital_period_s) / sol

        spring, summer = seg(0, 90), seg(90, 180)
        autumn, winter = seg(180, 270), seg(270, 360)
        assert spring == pytest.approx(194, abs=3)
        assert summer == pytest.approx(178, abs=3)
        assert autumn == pytest.approx(143, abs=3)
        assert winter == pytest.approx(154, abs=3)
        # they are genuinely unequal (would all be ~167 for a circular orbit)
        assert max(spring, summer, autumn, winter) - min(spring, summer, autumn, winter) > 30
        assert spring + summer + autumn + winter == pytest.approx(668.6, abs=1.0)


class TestEpochConsistency:

    def test_mean_anomaly_for_ls_roundtrips(self):
        import dataclasses
        f = _forcing()
        for ls in (0.0, 90.0, 180.0, 270.0):
            f2 = dataclasses.replace(
                f, init_orbital_angle_rad=P.mean_anomaly_for_ls(math.radians(ls), f))
            nu0 = float(P._true_anomaly(0.0, f2))
            ls0 = math.degrees(nu0 + f2.ls_perihelion_rad) % 360.0
            assert ls0 == pytest.approx(ls if ls > 0 else 360.0, abs=0.05) or \
                   ls0 == pytest.approx(ls, abs=0.05)
