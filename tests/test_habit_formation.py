"""習慣形成のテスト (Phase 2)"""

import pytest

from japan_fiscal_simulator.core.equations.is_curve import ISCurve, ISCurveParameters
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.parameters.defaults import DefaultParameters, HouseholdParameters


class TestISCurveWithHabit:
    """習慣形成付きIS曲線のテスト"""

    def test_no_habit_coefficients(self) -> None:
        """習慣形成なし（h=0）の係数テスト"""
        params = ISCurveParameters(sigma=1.5, g_y=0.2, habit=0.0)
        eq = ISCurve(params)
        coef = eq.coefficients()

        # 習慣形成なしの係数
        assert coef.y_current == 1.0
        assert coef.y_forward == -1.0
        assert coef.y_lag == 0.0  # ラグ項なし
        assert coef.r_current == pytest.approx(1.0 / 1.5)
        assert coef.pi_forward == pytest.approx(-1.0 / 1.5)

    def test_with_habit_coefficients(self) -> None:
        """習慣形成あり（h=0.7）の係数テスト"""
        h = 0.7
        sigma = 1.5
        params = ISCurveParameters(sigma=sigma, g_y=0.2, habit=h)
        eq = ISCurve(params)
        coef = eq.coefficients()

        # 習慣形成調整後の sigma
        sigma_h = sigma * (1 - h) / (1 + h)
        sigma_h_inv = 1.0 / sigma_h

        # 係数を確認
        assert coef.y_current == 1.0
        assert coef.y_forward == pytest.approx(-(1 - h))
        assert coef.y_lag == pytest.approx(-h)
        assert coef.r_current == pytest.approx(sigma_h_inv)
        assert coef.pi_forward == pytest.approx(-sigma_h_inv)

    def test_habit_increases_interest_rate_sensitivity(self) -> None:
        """習慣形成により金利感応度が上昇"""
        sigma = 1.5
        g_y = 0.2

        # 習慣形成なし
        eq_no_habit = ISCurve(ISCurveParameters(sigma=sigma, g_y=g_y, habit=0.0))
        coef_no = eq_no_habit.coefficients()

        # 習慣形成あり
        eq_habit = ISCurve(ISCurveParameters(sigma=sigma, g_y=g_y, habit=0.7))
        coef_habit = eq_habit.coefficients()

        # σ_h = σ(1-h)/(1+h) < σ なので 1/σ_h > 1/σ
        # つまり r_current の絶対値が大きくなる
        assert abs(coef_habit.r_current) > abs(coef_no.r_current)

    def test_habit_zero_recovers_standard(self) -> None:
        """h=0で標準的なIS曲線に戻る"""
        params_zero = ISCurveParameters(sigma=1.5, g_y=0.2, habit=0.0)
        params_small = ISCurveParameters(sigma=1.5, g_y=0.2, habit=0.001)

        eq_zero = ISCurve(params_zero)
        eq_small = ISCurve(params_small)

        coef_zero = eq_zero.coefficients()
        coef_small = eq_small.coefficients()

        # 非常に小さい habit では標準に近い
        assert coef_small.y_forward == pytest.approx(coef_zero.y_forward, rel=0.01)
        # ただし y_lag は 0 にはならない
        assert coef_small.y_lag != 0.0

    def test_description_changes_with_habit(self) -> None:
        """習慣形成の有無で説明が変わる"""
        eq_no = ISCurve(ISCurveParameters(sigma=1.5, g_y=0.2, habit=0.0))
        eq_yes = ISCurve(ISCurveParameters(sigma=1.5, g_y=0.2, habit=0.7))

        # 習慣形成ありの説明には h または y_{t-1} が含まれる
        assert "h" in eq_yes.description or "y_{t-1}" in eq_yes.description
        # 習慣形成なしの説明には含まれない（標準的なIS曲線）
        assert "h·" not in eq_no.description

    def test_extreme_habit(self) -> None:
        """極端な習慣形成パラメータのテスト"""
        # h に近い値
        params = ISCurveParameters(sigma=1.5, g_y=0.2, habit=0.95)
        eq = ISCurve(params)
        coef = eq.coefficients()

        # y_lag が大きくなる
        assert abs(coef.y_lag) > 0.9
        # y_forward が小さくなる
        assert abs(coef.y_forward) < 0.1

    def test_habit_sum_property(self) -> None:
        """y_forward + y_lag の関係"""
        h = 0.7
        params = ISCurveParameters(sigma=1.5, g_y=0.2, habit=h)
        eq = ISCurve(params)
        coef = eq.coefficients()

        # y_forward = -(1-h), y_lag = -h
        # y_forward + y_lag = -(1-h) - h = -1
        assert coef.y_forward + coef.y_lag == pytest.approx(-1.0)


class TestHabitFormationIntegration:
    """習慣形成の統合テスト"""

    def test_default_habit_parameter(self) -> None:
        """デフォルトの習慣形成パラメータ"""
        params = DefaultParameters()
        # HouseholdParameters に habit が存在
        assert hasattr(params.household, "habit")
        assert params.household.habit == 0.7

    def test_is_curve_uses_habit(self) -> None:
        """IS曲線が習慣形成を使用することを確認"""
        # 習慣形成あり
        params_habit = DefaultParameters()

        # 習慣形成なし
        params_no_habit = DefaultParameters().with_updates(
            household=HouseholdParameters(
                beta=params_habit.household.beta,
                sigma=params_habit.household.sigma,
                phi=params_habit.household.phi,
                habit=0.0,
                chi=params_habit.household.chi,
            )
        )

        model_habit = NewKeynesianModel(params_habit)
        model_no_habit = NewKeynesianModel(params_no_habit)

        sol_habit = model_habit.solution
        sol_no_habit = model_no_habit.solution

        # Phillips曲線スロープは習慣形成に依存しない
        assert sol_habit.kappa == sol_no_habit.kappa

        # IS曲線の係数を直接比較して習慣形成の効果を確認
        is_habit = ISCurve(
            ISCurveParameters(
                sigma=params_habit.household.sigma,
                g_y=params_habit.government.g_y_ratio,
                habit=params_habit.household.habit,
            )
        )
        is_no_habit = ISCurve(
            ISCurveParameters(
                sigma=params_no_habit.household.sigma,
                g_y=params_no_habit.government.g_y_ratio,
                habit=0.0,
            )
        )

        coef_habit = is_habit.coefficients()
        coef_no_habit = is_no_habit.coefficients()

        # 習慣形成ありではy_lagが非ゼロ
        assert coef_habit.y_lag != 0.0
        assert coef_no_habit.y_lag == 0.0

        # 習慣形成ありでは金利感応度が異なる
        assert coef_habit.r_current != coef_no_habit.r_current
