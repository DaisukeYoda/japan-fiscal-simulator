"""方程式モジュールのユニットテスト

全方程式クラス（IS曲線、Taylor則、Phillips曲線、資源制約、
財政ルール、限界費用、MRS、資本蓄積、投資、労働需要、賃金NKPC）を網羅する。
"""

import pytest

from japan_fiscal_simulator.core.equations import (
    CapitalAccumulation,
    CapitalAccumulationParameters,
    CapitalRentalRateEquation,
    GovernmentSpendingProcess,
    ISCurve,
    ISCurveParameters,
    InvestmentAdjustmentEquation,
    InvestmentAdjustmentParameters,
    LaborDemand,
    LaborDemandParameters,
    MarginalCostEquation,
    MarginalCostParameters,
    MRSEquation,
    MRSEquationParameters,
    PhillipsCurve,
    PhillipsCurveParameters,
    ResourceConstraint,
    ResourceConstraintParameters,
    TaylorRule,
    TaylorRuleParameters,
    TechnologyProcess,
    TobinsQEquation,
    TobinsQParameters,
    WagePhillipsCurve,
    WagePhillipsCurveParameters,
    check_taylor_principle,
    compute_phillips_slope,
    compute_wage_adjustment_speed,
)


# ---------------------------------------------------------------------------
# IS Curve
# ---------------------------------------------------------------------------

class TestISCurve:
    def test_no_habit_coefficients(self) -> None:
        """習慣形成なし（h=0）の場合の係数を検証"""
        sigma = 2.0
        g_y = 0.2
        params = ISCurveParameters(sigma=sigma, g_y=g_y)
        coefs = ISCurve(params).coefficients()

        assert coefs.y_current == pytest.approx(1.0)
        assert coefs.y_forward == pytest.approx(-1.0)
        assert coefs.r_current == pytest.approx(1.0 / sigma)
        assert coefs.pi_forward == pytest.approx(-1.0 / sigma)
        assert coefs.g_current == pytest.approx(-g_y)
        assert coefs.a_current == pytest.approx(-1.0)
        assert coefs.y_lag == pytest.approx(0.0)

    def test_with_habit_formation(self) -> None:
        """習慣形成あり（h>0）の場合のσ_h調整と前期依存を検証"""
        sigma = 2.0
        g_y = 0.2
        h = 0.7
        params = ISCurveParameters(sigma=sigma, g_y=g_y, habit=h)
        coefs = ISCurve(params).coefficients()

        sigma_h = sigma * (1 - h) / (1 + h)
        assert coefs.y_current == pytest.approx(1.0)
        assert coefs.y_forward == pytest.approx(-(1 - h))
        assert coefs.y_lag == pytest.approx(-h)
        assert coefs.r_current == pytest.approx(1.0 / sigma_h)
        assert coefs.pi_forward == pytest.approx(-1.0 / sigma_h)

    def test_name_and_description(self) -> None:
        params = ISCurveParameters(sigma=1.0, g_y=0.2)
        eq = ISCurve(params)
        assert eq.name == "IS Curve"
        assert "y_t" in eq.description


# ---------------------------------------------------------------------------
# Taylor Rule
# ---------------------------------------------------------------------------

class TestTaylorRule:
    def test_coefficients(self) -> None:
        phi_pi = 1.5
        phi_y = 0.125
        params = TaylorRuleParameters(phi_pi=phi_pi, phi_y=phi_y)
        coefs = TaylorRule(params).coefficients()

        assert coefs.r_current == pytest.approx(1.0)
        assert coefs.pi_current == pytest.approx(-phi_pi)
        assert coefs.y_current == pytest.approx(-phi_y)
        assert coefs.e_m == pytest.approx(-1.0)

    def test_taylor_principle_satisfied(self) -> None:
        """Taylor原則が成立するケース"""
        satisfied, criterion = check_taylor_principle(
            phi_pi=1.5, phi_y=0.125, beta=0.99, kappa=0.1
        )
        expected = 1.5 + (1 - 0.99) / 0.1 * 0.125
        assert satisfied is True
        assert criterion == pytest.approx(expected)

    def test_taylor_principle_violated(self) -> None:
        """Taylor原則が破れるケース（φ_π < 1, φ_y = 0）"""
        satisfied, criterion = check_taylor_principle(
            phi_pi=0.8, phi_y=0.0, beta=0.99, kappa=0.1
        )
        assert satisfied is False
        assert criterion == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# Phillips Curve
# ---------------------------------------------------------------------------

class TestPhillipsCurve:
    def test_slope_no_indexation(self) -> None:
        """インデクセーションなしのκ計算"""
        beta = 0.99
        theta = 0.75
        kappa = compute_phillips_slope(beta, theta)
        expected = (1 - theta) * (1 - beta * theta) / theta
        assert kappa == pytest.approx(expected)

    def test_slope_with_indexation(self) -> None:
        """インデクセーションありのκ計算（分母に 1+β*ι_p）"""
        beta = 0.99
        theta = 0.75
        iota_p = 0.5
        kappa = compute_phillips_slope(beta, theta, iota_p)
        base = (1 - theta) * (1 - beta * theta) / theta
        expected = base / (1 + beta * iota_p)
        assert kappa == pytest.approx(expected)

    def test_coefficients_no_indexation(self) -> None:
        """iota_p=0のとき: pi_lag=0, pi_forward=-beta"""
        params = PhillipsCurveParameters(beta=0.99, theta=0.75)
        eq = PhillipsCurve(params)
        coefs = eq.coefficients()

        assert coefs.pi_current == pytest.approx(1.0)
        assert coefs.pi_lag == pytest.approx(0.0)
        assert coefs.pi_forward == pytest.approx(-0.99)
        assert coefs.mc_current == pytest.approx(-eq.kappa)

    def test_coefficients_with_indexation(self) -> None:
        """iota_p>0のとき: pi_lagが非零になる"""
        beta = 0.99
        iota_p = 0.5
        params = PhillipsCurveParameters(beta=beta, theta=0.75, iota_p=iota_p)
        coefs = PhillipsCurve(params).coefficients()

        denom = 1 + beta * iota_p
        assert coefs.pi_lag == pytest.approx(-iota_p / denom)
        assert coefs.pi_forward == pytest.approx(-beta / denom)

    def test_shock_persistence_scaling(self) -> None:
        """rho_p > 0 のとき e_p のスケーリングが変わる"""
        beta = 0.99
        rho_p = 0.5
        params = PhillipsCurveParameters(beta=beta, theta=0.75, rho_p=rho_p)
        coefs = PhillipsCurve(params).coefficients()

        pi_fwd_coef = beta / (1 + beta * 0.0)  # iota_p=0
        expected_scale = 1.0 / (1.0 - pi_fwd_coef * rho_p)
        assert coefs.e_p == pytest.approx(-expected_scale)


# ---------------------------------------------------------------------------
# Resource Constraint
# ---------------------------------------------------------------------------

class TestResourceConstraint:
    def test_coefficients(self) -> None:
        s_c, s_i, s_g = 0.6, 0.2, 0.2
        params = ResourceConstraintParameters(s_c=s_c, s_i=s_i, s_g=s_g)
        coefs = ResourceConstraint(params).coefficients()

        assert coefs.y_current == pytest.approx(1.0)
        assert coefs.c_current == pytest.approx(-s_c)
        assert coefs.i_current == pytest.approx(-s_i)
        assert coefs.g_current == pytest.approx(-s_g)


# ---------------------------------------------------------------------------
# Fiscal Rule (AR(1) processes)
# ---------------------------------------------------------------------------

class TestGovernmentSpendingProcess:
    def test_coefficients(self) -> None:
        rho_g = 0.9
        coefs = GovernmentSpendingProcess(rho_g).coefficients()

        assert coefs.g_current == pytest.approx(1.0)
        assert coefs.g_lag == pytest.approx(-rho_g)
        assert coefs.e_g == pytest.approx(-1.0)


class TestTechnologyProcess:
    def test_coefficients(self) -> None:
        rho_a = 0.95
        coefs = TechnologyProcess(rho_a).coefficients()

        assert coefs.a_current == pytest.approx(1.0)
        assert coefs.a_lag == pytest.approx(-rho_a)
        assert coefs.e_a == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Marginal Cost
# ---------------------------------------------------------------------------

class TestMarginalCostEquation:
    def test_coefficients(self) -> None:
        alpha = 0.33
        params = MarginalCostParameters(alpha=alpha)
        coefs = MarginalCostEquation(params).coefficients()

        assert coefs.mc_current == pytest.approx(1.0)
        assert coefs.rk_current == pytest.approx(-alpha)
        assert coefs.w_current == pytest.approx(-(1 - alpha))
        assert coefs.a_current == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MRS
# ---------------------------------------------------------------------------

class TestMRSEquation:
    def test_coefficients(self) -> None:
        sigma = 2.0
        phi = 1.5
        params = MRSEquationParameters(sigma=sigma, phi=phi)
        coefs = MRSEquation(params).coefficients()

        assert coefs.mrs_current == pytest.approx(1.0)
        assert coefs.c_current == pytest.approx(-sigma)
        assert coefs.n_current == pytest.approx(-phi)


# ---------------------------------------------------------------------------
# Capital Accumulation
# ---------------------------------------------------------------------------

class TestCapitalAccumulation:
    def test_coefficients(self) -> None:
        delta = 0.025
        params = CapitalAccumulationParameters(delta=delta)
        coefs = CapitalAccumulation(params).coefficients()

        assert coefs.k_current == pytest.approx(1.0)
        assert coefs.k_lag == pytest.approx(-(1 - delta))
        assert coefs.i_current == pytest.approx(-delta)


# ---------------------------------------------------------------------------
# Tobin's Q
# ---------------------------------------------------------------------------

class TestTobinsQEquation:
    def test_coefficients(self) -> None:
        beta = 0.99
        delta = 0.025
        params = TobinsQParameters(beta=beta, delta=delta)
        coefs = TobinsQEquation(params).coefficients()

        assert coefs.q_current == pytest.approx(1.0)
        assert coefs.q_forward == pytest.approx(-beta * (1 - delta))
        assert coefs.rk_forward == pytest.approx(-beta)
        assert coefs.r_current == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Investment Adjustment
# ---------------------------------------------------------------------------

class TestInvestmentAdjustmentEquation:
    def test_coefficients(self) -> None:
        S_pp = 5.0
        params = InvestmentAdjustmentParameters(S_double_prime=S_pp)
        coefs = InvestmentAdjustmentEquation(params).coefficients()

        assert coefs.i_current == pytest.approx(1.0)
        assert coefs.i_lag == pytest.approx(-1.0)
        assert coefs.q_current == pytest.approx(-1.0 / S_pp)
        assert coefs.e_i == pytest.approx(-1.0)


# ---------------------------------------------------------------------------
# Capital Rental Rate
# ---------------------------------------------------------------------------

class TestCapitalRentalRateEquation:
    def test_coefficients(self) -> None:
        coefs = CapitalRentalRateEquation().coefficients()

        assert coefs.rk_current == pytest.approx(1.0)
        assert coefs.y_current == pytest.approx(-1.0)
        assert coefs.k_lag == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# Labor Demand
# ---------------------------------------------------------------------------

class TestLaborDemand:
    def test_coefficients(self) -> None:
        alpha = 0.33
        params = LaborDemandParameters(alpha=alpha)
        coefs = LaborDemand(params).coefficients()

        labor_share = 1 - alpha
        assert coefs.n_current == pytest.approx(1.0)
        assert coefs.y_current == pytest.approx(-1.0 / labor_share)
        assert coefs.a_current == pytest.approx(1.0 / labor_share)
        assert coefs.k_lag == pytest.approx(alpha / labor_share)


# ---------------------------------------------------------------------------
# Wage Phillips Curve
# ---------------------------------------------------------------------------

class TestWagePhillipsCurve:
    def test_adjustment_speed(self) -> None:
        """賃金調整速度 λ_w の計算を検証"""
        beta = 0.99
        theta_w = 0.75
        lambda_w = compute_wage_adjustment_speed(beta, theta_w)
        expected = (1 - theta_w) * (1 - beta * theta_w) / theta_w
        assert lambda_w == pytest.approx(expected)

    def test_coefficients(self) -> None:
        beta = 0.99
        theta_w = 0.75
        params = WagePhillipsCurveParameters(
            beta=beta, theta_w=theta_w, sigma=2.0, phi=1.5
        )
        eq = WagePhillipsCurve(params)
        coefs = eq.coefficients()

        lambda_w = eq.lambda_w
        denom = 1 + beta

        assert coefs.w_current == pytest.approx(1.0 + lambda_w / denom)
        assert coefs.w_forward == pytest.approx(-beta / denom)
        assert coefs.w_lag == pytest.approx(-1.0 / denom)
        assert coefs.mrs_current == pytest.approx(-lambda_w / denom)
        assert coefs.e_w == pytest.approx(-1.0)
