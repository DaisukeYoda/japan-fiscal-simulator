"""Phase2 NKソルバー統合テスト"""

import numpy as np

from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


def test_phase2_solution_bk_satisfied() -> None:
    model = NewKeynesianModel(DefaultParameters())
    sol = model.solution

    assert sol.bk_satisfied
    assert sol.P.shape == (5, 5)
    assert sol.Q.shape == (5, 5)
    assert sol.R.shape == (6, 5)
    assert sol.S.shape == (6, 5)
    assert sol.determinacy == "determinate"


def test_wage_shock_affects_state() -> None:
    model = NewKeynesianModel(DefaultParameters())
    sol = model.solution

    # e_w が状態変数に影響することを確認
    assert np.any(np.abs(sol.Q[:, 4]) > 0.0)


def test_monetary_shock_affects_controls_only() -> None:
    model = NewKeynesianModel(DefaultParameters())
    sol = model.solution

    # e_m は状態変数に影響しない
    assert np.allclose(sol.Q[:, 2], 0.0)
    # ただし制御変数には直接効果がある
    assert np.any(np.abs(sol.S[:, 2]) > 0.0)


def test_bk_dimensions_consistent() -> None:
    model = NewKeynesianModel(DefaultParameters())
    sol = model.solution

    # 制御変数の数 = 6
    assert sol.R.shape[0] == 6
    assert sol.S.shape[0] == 6
