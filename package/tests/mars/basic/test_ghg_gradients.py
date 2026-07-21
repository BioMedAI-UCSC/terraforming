"""Gradient regression tests for the GHG intervention path in mars.py.

Covers the P0 differentiability bugs (see docs/ideas/iclr2027-workplan.md,
Phase 0 audit):
  - B1: inject() cast the injected mass to Python float, cutting the
        autograd graph between a schedule and the greenhouse factor.
  - B3: _recompute_greenhouse_factor() early-returned on float(dF.item()),
        a CPU sync + data-dependent branch on the gradient path.
Also asserts the float-input API is unchanged (bit-identical results).
"""

from __future__ import annotations

import pytest
import torch

from src.celestials import Mars
from src.constants import TF_DTYPE


def _val(t: torch.Tensor) -> float:
    return float(t.item())


class TestInjectGradientFlow:

    def test_ghf_has_grad_fn_given_tensor_mass(self):
        # Regression B1: float(kg)*float(g.item())/float(A.item()) created a
        # fresh leaf tensor, so GHF never depended on the injected mass.
        kg = torch.tensor(1e13, dtype=TF_DTYPE, device="cpu", requires_grad=True)
        mars = Mars(device="cpu")
        mars.inject({"SF6": kg})
        assert mars.thermal.greenhouse_factor.grad_fn is not None

    def test_ghf_gradient_wrt_mass_is_finite_and_positive(self):
        """More injected GHG must warm: d(GHF)/d(kg) > 0 and finite."""
        kg = torch.tensor(1e13, dtype=TF_DTYPE, device="cpu", requires_grad=True)
        mars = Mars(device="cpu")
        mars.inject({"SF6": kg})
        mars.thermal.greenhouse_factor.backward()
        assert kg.grad is not None
        assert torch.isfinite(kg.grad)
        assert _val(kg.grad) > 0.0

    def test_delta_F_has_grad_fn_given_tensor_mass(self):
        kg = torch.tensor(1e13, dtype=TF_DTYPE, device="cpu", requires_grad=True)
        mars = Mars(device="cpu")
        mars.inject({"CF4": kg})
        assert mars.delta_F.grad_fn is not None

    def test_float_mass_api_unchanged(self):
        """Plain-float schedules (the CLI path) must behave exactly as before."""
        mars_f = Mars(device="cpu")
        mars_t = Mars(device="cpu")
        mars_f.inject({"SF6": 1e13})
        mars_t.inject({"SF6": torch.tensor(1e13, dtype=TF_DTYPE)})
        assert _val(mars_f.thermal.greenhouse_factor) == pytest.approx(
            _val(mars_t.thermal.greenhouse_factor), rel=1e-12
        )
        assert _val(mars_f.atmosphere.composition["SF6"]) == pytest.approx(
            _val(mars_t.atmosphere.composition["SF6"]), rel=1e-12
        )


class TestRecomputeGreenhouseFactorSemantics:

    def test_zero_forcing_leaves_ghf_at_baseline(self):
        # Regression B3: the relu formulation must reproduce the old
        # "skip when ΔF ≤ 0" semantics — no GHGs in composition → GHF stays
        # at its baseline value.
        mars = Mars(device="cpu")
        ghf_before = _val(mars.thermal.greenhouse_factor)
        mars._init_ghg()
        mars._recompute_greenhouse_factor()
        assert _val(mars.thermal.greenhouse_factor) == pytest.approx(ghf_before)

    def test_ghf_stays_at_least_one_after_injection(self):
        mars = Mars(device="cpu")
        mars.inject({"CF4": 1e13})
        assert _val(mars.thermal.greenhouse_factor) >= 1.0
