"""労働市場方程式のテスト (Phase 2)"""

import pytest

from japan_fiscal_simulator.core.equations.labor_demand import (
    LaborDemand,
    LaborDemandParameters,
)
from japan_fiscal_simulator.core.equations.wage_phillips import (
    WagePhillipsCurve,
    WagePhillipsCurveParameters,
    compute_wage_adjustment_speed,
)
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.parameters.defaults import (
    DefaultParameters,
    LaborParameters,
    ShockParameters,
)


class TestWageAdjustmentSpeed:
    """賃金調整速度 λ_w のテスト"""

    def test_compute_basic(self) -> None:
        """基本的な計算のテスト"""
        beta = 0.99
        theta_w = 0.75
        lambda_w = compute_wage_adjustment_speed(beta, theta_w)

        # λ_w = (1-0.75)(1-0.99*0.75)/0.75 = 0.25 * 0.2575 / 0.75 ≈ 0.086
        expected = (1 - 0.75) * (1 - 0.99 * 0.75) / 0.75
        assert lambda_w == pytest.approx(expected)

    def test_high_rigidity(self) -> None:
        """高硬直性（θ_w=0.9）で調整速度が低下"""
        beta = 0.99
        theta_w_high = 0.90
        theta_w_low = 0.50

        lambda_w_high = compute_wage_adjustment_speed(beta, theta_w_high)
        lambda_w_low = compute_wage_adjustment_speed(beta, theta_w_low)

        # 高硬直性 → 低調整速度
        assert lambda_w_high < lambda_w_low

    def test_limiting_cases(self) -> None:
        """極限ケースのテスト"""
        beta = 0.99

        # θ_w が小さいほど λ_w は大きい
        lambda_w_flex = compute_wage_adjustment_speed(beta, 0.1)
        lambda_w_rigid = compute_wage_adjustment_speed(beta, 0.9)

        assert lambda_w_flex > lambda_w_rigid


class TestWagePhillipsCurve:
    """賃金NKPCのテスト"""

    def test_coefficients_basic(self) -> None:
        """基本的な係数のテスト"""
        params = WagePhillipsCurveParameters(
            beta=0.99,
            theta_w=0.75,
            sigma=1.5,
            phi=2.0,
        )
        eq = WagePhillipsCurve(params)
        coef = eq.coefficients()

        # w_current > 0（左辺）
        assert coef.w_current > 0
        # w_forward < 0（前向き期待）
        assert coef.w_forward < 0
        # w_lag < 0（後向きインデクセーション）
        assert coef.w_lag < 0
        # mrs_current < 0（MRS項）
        assert coef.mrs_current < 0
        # e_w = -1（ショック）
        assert coef.e_w == -1.0

    def test_coefficient_sum(self) -> None:
        """係数の整合性テスト"""
        params = WagePhillipsCurveParameters(
            beta=0.99,
            theta_w=0.75,
            sigma=1.5,
            phi=2.0,
        )
        eq = WagePhillipsCurve(params)
        coef = eq.coefficients()

        # w_current + w_forward + w_lag の関係をチェック
        # 定常状態では w_t = w_{t-1} = E[w_{t+1}] なので
        # w_current + w_forward + w_lag + λ_w/(1+β) = 0 に近くなる
        denom = 1 + params.beta
        lambda_w = eq.lambda_w
        expected_w_sum = 1 + lambda_w / denom - params.beta / denom - 1 / denom
        assert coef.w_current + coef.w_forward + coef.w_lag == pytest.approx(
            expected_w_sum
        )

    def test_name_and_description(self) -> None:
        """名前と説明のテスト"""
        params = WagePhillipsCurveParameters(
            beta=0.99, theta_w=0.75, sigma=1.5, phi=2.0
        )
        eq = WagePhillipsCurve(params)

        assert "Wage" in eq.name
        assert "ŵ_t" in eq.description or "w_t" in eq.description

    def test_lambda_w_property(self) -> None:
        """λ_w プロパティのテスト"""
        params = WagePhillipsCurveParameters(
            beta=0.99, theta_w=0.75, sigma=1.5, phi=2.0
        )
        eq = WagePhillipsCurve(params)

        expected = compute_wage_adjustment_speed(0.99, 0.75)
        assert eq.lambda_w == pytest.approx(expected)


class TestLaborDemand:
    """労働需要方程式のテスト"""

    def test_coefficients_basic(self) -> None:
        """基本的な係数のテスト"""
        params = LaborDemandParameters(alpha=0.33)
        eq = LaborDemand(params)
        coef = eq.coefficients()

        labor_share = 1 - 0.33

        # n_current = 1
        assert coef.n_current == 1.0
        # y_current = -1/(1-α)
        assert coef.y_current == pytest.approx(-1.0 / labor_share)
        # a_current = 1/(1-α)
        assert coef.a_current == pytest.approx(1.0 / labor_share)
        # k_lag = α/(1-α)
        assert coef.k_lag == pytest.approx(0.33 / labor_share)

    def test_labor_share_effect(self) -> None:
        """労働シェアの効果テスト"""
        # α=0.25（労働シェア高 = 0.75）
        params_high = LaborDemandParameters(alpha=0.25)
        eq_high = LaborDemand(params_high)
        coef_high = eq_high.coefficients()

        # α=0.40（労働シェア低 = 0.60）
        params_low = LaborDemandParameters(alpha=0.40)
        eq_low = LaborDemand(params_low)
        coef_low = eq_low.coefficients()

        # 労働シェアが低いほど y_current の絶対値が大きい
        assert abs(coef_low.y_current) > abs(coef_high.y_current)

    def test_name_and_description(self) -> None:
        """名前と説明のテスト"""
        params = LaborDemandParameters(alpha=0.33)
        eq = LaborDemand(params)

        assert "Labor" in eq.name
        assert "n̂_t" in eq.description or "n_t" in eq.description


class TestLaborParameters:
    """労働パラメータのテスト"""

    def test_default_parameters(self) -> None:
        """デフォルトパラメータが設定されていることを確認"""
        params = DefaultParameters()

        assert hasattr(params, "labor")
        assert params.labor.theta_w == 0.75
        assert params.labor.epsilon_w == 10.0
        assert params.labor.iota_w == 0.5

    def test_labor_parameters_immutable(self) -> None:
        """LaborParametersが不変であることを確認"""
        params = LaborParameters()

        with pytest.raises(AttributeError):
            params.theta_w = 0.5  # type: ignore[misc]

    def test_shock_parameters(self) -> None:
        """ショックパラメータに賃金マークアップが追加されていることを確認"""
        params = DefaultParameters()

        assert hasattr(params.shocks, "rho_w")
        assert hasattr(params.shocks, "sigma_w")
        assert params.shocks.rho_w == 0.90
        assert params.shocks.sigma_w == 0.01

    def test_with_updates_labor(self) -> None:
        """with_updatesでlaborパラメータを更新できることを確認"""
        params = DefaultParameters()
        new_labor = LaborParameters(theta_w=0.70)
        updated = params.with_updates(labor=new_labor)

        assert updated.labor.theta_w == 0.70
        # 他のパラメータは変更されていない
        assert updated.household.beta == params.household.beta

    def test_wage_markup_persistence_affects_irf(self) -> None:
        """rho_w の違いが e_w ショック応答に反映されることを確認"""
        base = DefaultParameters()
        low = base.with_updates(shocks=ShockParameters(rho_w=0.0))
        high = base.with_updates(shocks=ShockParameters(rho_w=0.9))

        model_low = NewKeynesianModel(low)
        model_high = NewKeynesianModel(high)

        irf_low = model_low.impulse_response("e_w", size=0.01, periods=2)
        irf_high = model_high.impulse_response("e_w", size=0.01, periods=2)

        # 持続性が高いほど、次期以降の反応が大きい
        assert abs(irf_high["w"][1]) > abs(irf_low["w"][1])
        # 状態に持続ショックが入るため、少なくとも1期先の状態が変わる
        assert irf_high["k"][1] != pytest.approx(irf_low["k"][1])
