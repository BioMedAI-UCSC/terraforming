from pathlib import Path
import sys
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / 'scripts'))
from paper_calibration import bounded_fit, paired_block_interval


def test_converged_box_optimum_and_validation_selection():
    # First control has its constrained optimum at zero; raw gradient is nonzero.
    target = np.array([-.5, .7, .2])
    best, result, trace = bounded_fit(
        lambda x: (np.sum((x-target)**2), 2*(x-target)),
        lambda x: np.sum((x-target)**2), [.5,.5,.5], starts=2)
    np.testing.assert_allclose(best, [0,.7,.2], atol=1e-6)
    assert result['stationary'] and trace[-1]['projected_gradient_inf'] < 1e-6


def test_budget_exhaustion_is_not_convergence():
    _, result, trace = bounded_fit(
        lambda x: (np.sum(np.exp(x)), np.exp(x)), lambda x: np.sum(x*x), [.9,.9,.9], maxeval=1)
    assert len(trace) == 1
    assert not result['stationary']
    assert not result['terminations'][0]['success']


def test_stationary_endpoint_does_not_label_validation_checkpoint_converged():
    best, result, _ = bounded_fit(
        lambda x: (np.sum((x-.8)**2), 2*(x-.8)),
        lambda x: np.sum((x-.2)**2), [.2,.2,.2])
    np.testing.assert_allclose(best, [.2,.2,.2])
    assert result['stationary']
    assert not result['validation_selected_stationary']
    assert result['validation_selected_projected_gradient_inf'] > 1


def test_paired_year_blocks_not_gridcells_and_missing_replication():
    a = paired_block_interval([9,9,16,16], [4,4,9,9], ['a','a','b','b'])
    assert a['blocks'] == 2 and a['resamples'] == 4
    gain_mixed = np.sqrt(12.5) - np.sqrt(6.5)
    np.testing.assert_allclose(a['interval_95_k'], np.quantile([1.,gain_mixed,gain_mixed,1.],[.025,.975]))
    assert paired_block_interval([9], [4], ['a'])['interval_95_k'] is None
