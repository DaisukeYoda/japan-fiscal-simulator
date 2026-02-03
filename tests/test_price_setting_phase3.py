"""Phase 3: 価格設定ブロックのテスト"""

import pytest

from japan_fiscal_simulator.core.equations.marginal_cost import (
    MarginalCostEquation,
    MarginalCostParameters,
)
from japan_fiscal_simulator.core.equations.phillips_curve import (
    PhillipsCurve,
    PhillipsCurveParameters,
)
from japan_fiscal_simulator.core.model import DSGEModel
from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.parameters.defaults import DefaultParameters, FirmParameters, ShockParameters


class TestIndexedPhillipsCurve:
    """インデクセーション付きPhillips曲線のテスト"""

    def test_coefficients_signs(self) -> None:
        params = PhillipsCurveParameters(beta=0.99, theta=0.75, iota_p=0.5)
        eq = PhillipsCurve(params)
        coef = eq.coefficients()

        assert coef.pi_current == 1.0
        assert coef.pi_lag < 0.0
        assert coef.pi_forward < 0.0
        assert coef.mc_current < 0.0
        assert coef.e_p == -1.0

    def test_indexation_reduces_kappa(self) -> None:
        no_index = PhillipsCurve(PhillipsCurveParameters(beta=0.99, theta=0.75, iota_p=0.0))
        with_index = PhillipsCurve(PhillipsCurveParameters(beta=0.99, theta=0.75, iota_p=0.5))

        assert with_index.kappa < no_index.kappa


class TestMarginalCostEquation:
    """限界費用方程式のテスト"""

    def test_coefficients(self) -> None:
        eq = MarginalCostEquation(MarginalCostParameters(alpha=0.33))
        coef = eq.coefficients()

        assert coef.mc_current == 1.0
        assert coef.rk_current == pytest.approx(-0.33)
        assert coef.w_current == pytest.approx(-(1.0 - 0.33))
        assert coef.a_current == 1.0


class TestPhase3Parameters:
    """Phase 3追加パラメータのテスト"""

    def test_firm_iota_p_alias(self) -> None:
        params = FirmParameters(psi=0.6)
        assert params.psi == 0.6
        assert params.iota_p == pytest.approx(0.6)

    def test_price_markup_shock_defaults(self) -> None:
        params = ShockParameters()
        assert params.rho_p == pytest.approx(0.90)
        assert params.sigma_p == pytest.approx(0.01)

    def test_price_markup_irf_not_scaled_by_sigma_p(self) -> None:
        params_base = ShockParameters(sigma_p=0.01)
        params_alt = ShockParameters(sigma_p=0.0)

        model_base = DSGEModel(params=DefaultParameters(shocks=params_base))
        model_alt = DSGEModel(params=DefaultParameters(shocks=params_alt))

        pi_base = ImpulseResponseSimulator(model_base).simulate("e_p", shock_size=0.01, periods=1)
        pi_alt = ImpulseResponseSimulator(model_alt).simulate("e_p", shock_size=0.01, periods=1)

        assert pi_base.get_response("pi")[0] == pytest.approx(pi_alt.get_response("pi")[0])
