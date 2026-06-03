"""食料品消費税の時限減税分析スクリプトのテスト"""

import numpy as np

from analysis.food_tax_1pct_2yr import (
    CURRENT_EFFECTIVE,
    HOLD_QUARTERS,
    PERIODS,
    SHOCK,
    build_innovation_sequence,
    build_model,
    estimate_revenue_loss,
    propagate,
)
from japan_fiscal_simulator.core.model import VARIABLE_INDICES


def test_model_uses_current_effective_tax_rate_as_baseline() -> None:
    """表示する現行実効税率とモデルの定常状態が一致する"""
    model = build_model()
    assert model.params.government.tau_c == CURRENT_EFFECTIVE


def test_timed_tax_cut_only_changes_inflation_when_tax_rate_changes() -> None:
    """維持用ショックは追加のインフレ転嫁を発生させない"""
    model = build_model()
    pf = model.policy_function
    eps = build_innovation_sequence(SHOCK, model.params.shocks.rho_tau_c, HOLD_QUARTERS, PERIODS)
    x = propagate(pf.P, pf.Q, eps, PERIODS)
    idx = VARIABLE_INDICES

    tau = x[:, idx["tau_c"]]
    pi = x[:, idx["pi"]]
    np.testing.assert_allclose(tau[:HOLD_QUARTERS], SHOCK)
    np.testing.assert_allclose(tau[HOLD_QUARTERS:], 0.0, atol=1e-15)
    np.testing.assert_allclose(pi[1:HOLD_QUARTERS], 0.0, atol=1e-15)
    np.testing.assert_allclose(pi[HOLD_QUARTERS], -pi[0])
    np.testing.assert_allclose(pi[HOLD_QUARTERS + 1 :], 0.0, atol=1e-15)


def test_revenue_loss_uses_aggregate_consumption_share() -> None:
    """実効税率差に食料品シェアを重ねて掛けない"""
    model = build_model()
    tau_path = np.full(HOLD_QUARTERS, SHOCK)
    loss = estimate_revenue_loss(model, tau_path, HOLD_QUARTERS)
    ss = model.steady_state
    elasticity = (
        model.derived_coefficients.compute_impulse_coefficients().consumption_tax_elasticity
    )
    expected = (
        -SHOCK
        * HOLD_QUARTERS
        * (ss.consumption / ss.output)
        * (1.0 - model.params.government.tau_c * elasticity)
    )

    np.testing.assert_allclose(loss, expected)
