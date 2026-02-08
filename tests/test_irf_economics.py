"""IRFの経済学的性質テスト

インパルス応答関数が経済理論と整合的な符号・形状を示すことを検証する。
"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.model import DSGEModel
from japan_fiscal_simulator.core.simulation import (
    FiscalMultiplierCalculator,
    ImpulseResponseSimulator,
)
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@pytest.fixture(scope="module")
def model() -> DSGEModel:
    return DSGEModel(DefaultParameters())


@pytest.fixture(scope="module")
def simulator(model: DSGEModel) -> ImpulseResponseSimulator:
    return ImpulseResponseSimulator(model)


# === 政府支出ショック (e_g) ===


class TestGovernmentSpendingShock:
    """政府支出ショックの経済学的性質"""

    @pytest.fixture(scope="class")
    def irf(self, simulator: ImpulseResponseSimulator):
        return simulator.simulate("e_g", shock_size=0.01, periods=40)

    def test_output_increases_on_impact(self, irf) -> None:
        """政府支出増加で産出が即座に増加する"""
        y = irf.get_response("y")
        assert y[0] > 0

    def test_crowding_out(self, irf) -> None:
        """消費の増加は産出の増加より小さい（クラウディングアウト）"""
        y = irf.get_response("y")
        c = irf.get_response("c")
        assert c[0] < y[0]

    def test_inflation_rises(self, irf) -> None:
        """政府支出増加でインフレ率が上昇する"""
        pi = irf.get_response("pi")
        assert pi[0] > 0

    def test_nominal_rate_rises(self, irf) -> None:
        """インフレ上昇に対して名目金利が上昇する（Taylor則）"""
        R = irf.get_response("R")
        assert R[0] > 0

    def test_response_decays(self, irf) -> None:
        """応答がt=40までに十分減衰する"""
        y = irf.get_response("y")
        abs_y = np.abs(y)
        peak_idx = int(np.argmax(abs_y))
        # 最終期の応答がピークの50%未満に減衰
        assert abs_y[-1] < abs_y[peak_idx] * 0.5


# === 技術ショック (e_a) ===


class TestTechnologyShock:
    """技術ショックの経済学的性質"""

    @pytest.fixture(scope="class")
    def irf(self, simulator: ImpulseResponseSimulator):
        return simulator.simulate("e_a", shock_size=0.01, periods=40)

    def test_output_increases(self, irf) -> None:
        """正の技術ショックで産出が増加する"""
        y = irf.get_response("y")
        assert y[0] > 0

    def test_marginal_cost_falls(self, irf) -> None:
        """供給拡大で限界費用が低下する"""
        mc = irf.get_response("mc")
        assert mc[0] < 0


# === 金融引き締めショック (e_m) ===


class TestMonetaryTighteningShock:
    """金融引き締めショックの経済学的性質"""

    @pytest.fixture(scope="class")
    def irf(self, simulator: ImpulseResponseSimulator):
        return simulator.simulate("e_m", shock_size=0.01, periods=40)

    def test_output_contracts(self, irf) -> None:
        """金融引き締めで産出が減少する"""
        y = irf.get_response("y")
        assert y[0] < 0

    def test_inflation_falls(self, irf) -> None:
        """金融引き締めでインフレ率が低下する"""
        pi = irf.get_response("pi")
        assert pi[0] < 0


# === 消費税減税ショック (e_tau) ===


class TestConsumptionTaxCutShock:
    """消費税減税ショックの経済学的性質"""

    def test_output_increases(self, simulator: ImpulseResponseSimulator) -> None:
        """消費税減税で産出が増加する"""
        irf = simulator.simulate("e_tau", shock_size=-0.01, periods=40)
        y = irf.get_response("y")
        assert y[0] > 0


# === 長期収束テスト ===


class TestLongRunConvergence:
    """全ショックが長期的に定常状態へ収束することを検証"""

    def test_government_spending_converges(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """政府支出ショックがt=40までに収束する"""
        irf = simulator.simulate("e_g", shock_size=0.01, periods=40)
        responses_at_end = [
            abs(irf.get_response(var)[-1])
            for var in ("y", "c", "pi", "R")
        ]
        assert max(responses_at_end) < 0.01

    def test_technology_shock_converges(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """技術ショックがt=40までに収束する"""
        irf = simulator.simulate("e_a", shock_size=0.01, periods=40)
        responses_at_end = [
            abs(irf.get_response(var)[-1])
            for var in ("y", "c", "pi", "R")
        ]
        assert max(responses_at_end) < 0.01

    def test_monetary_shock_converges(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """金融政策ショックがt=40までに収束する"""
        irf = simulator.simulate("e_m", shock_size=0.01, periods=40)
        responses_at_end = [
            abs(irf.get_response(var)[-1])
            for var in ("y", "c", "pi", "R")
        ]
        assert max(responses_at_end) < 0.01


# === 財政乗数テスト ===


class TestFiscalMultiplier:
    """財政乗数の経済学的性質"""

    def test_spending_multiplier_positive(self, model: DSGEModel) -> None:
        """政府支出のインパクト乗数が正である"""
        calc = FiscalMultiplierCalculator(model)
        result = calc.compute_spending_multiplier()
        assert result.impact > 0
