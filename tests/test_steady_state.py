"""定常状態計算のテスト"""

from japan_fiscal_simulator.core.steady_state import SteadyState, SteadyStateSolver
from japan_fiscal_simulator.parameters.calibration import JapanCalibration
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestSteadyStateSolver:
    """SteadyStateSolverのテスト"""

    def test_solve_returns_steady_state(self) -> None:
        """定常状態が計算できることを確認"""
        params = DefaultParameters()
        solver = SteadyStateSolver(params)
        ss = solver.solve()

        assert isinstance(ss, SteadyState)
        assert ss.output > 0
        assert ss.consumption > 0
        assert ss.investment > 0

    def test_steady_state_satisfies_euler_equation(self) -> None:
        """定常状態がオイラー方程式を満たすことを確認"""
        params = DefaultParameters()
        solver = SteadyStateSolver(params)
        ss = solver.solve()

        # β(1+r) = 1 in steady state
        expected = params.household.beta * (1 + ss.real_interest_rate)
        assert abs(expected - 1.0) < 1e-6

    def test_steady_state_market_clearing(self) -> None:
        """財市場均衡条件を確認"""
        params = DefaultParameters()
        solver = SteadyStateSolver(params)
        ss = solver.solve()

        # Y = C + I + G
        total_demand = ss.consumption + ss.investment + ss.government_spending
        assert abs(ss.output - total_demand) < 1e-6

    def test_capital_accumulation_steady_state(self) -> None:
        """資本蓄積の定常状態条件を確認"""
        params = DefaultParameters()
        solver = SteadyStateSolver(params)
        ss = solver.solve()

        # I = δK in steady state
        expected_investment = params.firm.delta * ss.capital
        assert abs(ss.investment - expected_investment) < 1e-6

    def test_verify_steady_state(self) -> None:
        """定常状態検証が機能することを確認"""
        params = DefaultParameters()
        solver = SteadyStateSolver(params)
        ss = solver.solve()

        assert solver.verify_steady_state(ss)


class TestJapanCalibration:
    """日本キャリブレーションのテスト"""

    def test_baseline_calibration(self) -> None:
        """ベースラインキャリブレーションが作成できることを確認"""
        calibration = JapanCalibration.create()

        assert calibration.parameters.government.tau_c == 0.10
        assert calibration.parameters.government.b_y_ratio == 2.00
        assert calibration.parameters.household.beta == 0.999

    def test_set_consumption_tax(self) -> None:
        """消費税率変更が機能することを確認"""
        calibration = JapanCalibration.create()
        new_calibration = calibration.set_consumption_tax(0.08)

        assert new_calibration.parameters.government.tau_c == 0.08
        # 元のキャリブレーションは変更されていない
        assert calibration.parameters.government.tau_c == 0.10

    def test_high_debt_scenario(self) -> None:
        """高債務シナリオが作成できることを確認"""
        calibration = JapanCalibration.create_high_debt_scenario()

        assert calibration.parameters.government.b_y_ratio == 2.50

    def test_zlb_scenario(self) -> None:
        """ZLBシナリオが作成できることを確認"""
        calibration = JapanCalibration.create_zlb_scenario()

        assert calibration.parameters.central_bank.r_lower_bound == 0.0
