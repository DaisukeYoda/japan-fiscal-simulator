"""シミュレーションのテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.model import DSGEModel
from japan_fiscal_simulator.core.simulation import (
    FiscalMultiplierCalculator,
    ImpulseResponseResult,
    ImpulseResponseSimulator,
)
from japan_fiscal_simulator.parameters.calibration import JapanCalibration
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestImpulseResponseSimulator:
    """ImpulseResponseSimulatorのテスト"""

    @pytest.fixture
    def model(self) -> DSGEModel:
        """テスト用モデル"""
        params = DefaultParameters()
        return DSGEModel(params)

    @pytest.fixture
    def simulator(self, model: DSGEModel) -> ImpulseResponseSimulator:
        """テスト用シミュレーター"""
        return ImpulseResponseSimulator(model)

    def test_simulate_technology_shock(self, simulator: ImpulseResponseSimulator) -> None:
        """技術ショックのシミュレーション"""
        result = simulator.simulate("e_a", shock_size=0.01, periods=20)

        assert isinstance(result, ImpulseResponseResult)
        assert result.periods == 21  # t=0を含むので periods + 1
        assert result.shock_name == "e_a"
        assert result.shock_size == 0.01

    def test_simulate_government_spending(self, simulator: ImpulseResponseSimulator) -> None:
        """政府支出ショックのシミュレーション"""
        result = simulator.simulate_government_spending(spending_increase=0.01, periods=40)

        assert result.shock_name == "e_g"
        assert "y" in result.responses
        assert "g" in result.responses

    def test_simulate_consumption_tax_cut(self, simulator: ImpulseResponseSimulator) -> None:
        """消費税減税のシミュレーション"""
        result = simulator.simulate_consumption_tax_cut(tax_cut=0.02, periods=40)

        assert result.shock_name == "e_tau"
        assert result.shock_size == -0.02  # 減税はマイナス

    def test_peak_response(self, simulator: ImpulseResponseSimulator) -> None:
        """ピーク応答の計算"""
        result = simulator.simulate("e_g", shock_size=0.01, periods=40)

        peak_period, peak_value = result.peak_response("y")

        assert isinstance(peak_period, int)
        assert 0 <= peak_period < 40

    def test_cumulative_response(self, simulator: ImpulseResponseSimulator) -> None:
        """累積応答の計算"""
        result = simulator.simulate("e_g", shock_size=0.01, periods=40)

        cumulative = result.cumulative_response("y", horizon=4)

        assert isinstance(cumulative, float)


class TestFiscalMultiplierCalculator:
    """FiscalMultiplierCalculatorのテスト"""

    @pytest.fixture
    def model(self) -> DSGEModel:
        """テスト用モデル"""
        calibration = JapanCalibration.create()
        return DSGEModel(calibration.parameters)

    @pytest.fixture
    def calculator(self, model: DSGEModel) -> FiscalMultiplierCalculator:
        """テスト用計算機"""
        return FiscalMultiplierCalculator(model)

    def test_spending_multiplier(self, calculator: FiscalMultiplierCalculator) -> None:
        """政府支出乗数の計算"""
        result = calculator.compute_spending_multiplier(horizon=40)

        assert hasattr(result, "impact")
        assert hasattr(result, "peak")
        assert hasattr(result, "cumulative_4q")

    def test_tax_multiplier(self, calculator: FiscalMultiplierCalculator) -> None:
        """消費税乗数の計算"""
        result = calculator.compute_tax_multiplier(horizon=40)

        assert hasattr(result, "impact")
        assert hasattr(result, "present_value")


class TestImpulseResponseResult:
    """ImpulseResponseResultのテスト"""

    def test_get_response(self) -> None:
        """応答取得のテスト"""
        responses = {
            "y": np.array([0.01, 0.02, 0.015, 0.01]),
            "c": np.array([0.005, 0.01, 0.008, 0.005]),
        }

        result = ImpulseResponseResult(
            periods=4,
            shock_name="e_g",
            shock_size=0.01,
            responses=responses,
        )

        y_response = result.get_response("y")
        assert len(y_response) == 4
        assert y_response[1] == 0.02

    def test_get_nonexistent_response(self) -> None:
        """存在しない変数の応答取得"""
        result = ImpulseResponseResult(
            periods=4,
            shock_name="e_g",
            shock_size=0.01,
            responses={},
        )

        response = result.get_response("nonexistent")
        assert len(response) == 4
        assert all(v == 0 for v in response)
