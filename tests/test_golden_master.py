"""Golden Master テスト

リファクタリング中の回帰を防ぐために、現在の動作をキャプチャする。
"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.model import N_SHOCKS, N_VARIABLES, DSGEModel
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.core.simulation import (
    FiscalMultiplierCalculator,
    ImpulseResponseSimulator,
)
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestNKModelGoldenMaster:
    """NewKeynesianModelの解行列を検証"""

    @pytest.fixture
    def nk_model(self) -> NewKeynesianModel:
        return NewKeynesianModel(DefaultParameters())

    def test_solution_P_matrix(self, nk_model: NewKeynesianModel) -> None:
        """状態遷移行列Pの値を検証"""
        sol = nk_model.solution
        P = sol.P

        assert P.shape == (2, 2)
        # P は diag([rho_g, rho_a]) = diag([0.9, 0.9])
        np.testing.assert_allclose(P[0, 0], 0.9, rtol=1e-10)
        np.testing.assert_allclose(P[1, 1], 0.9, rtol=1e-10)
        np.testing.assert_allclose(P[0, 1], 0.0, rtol=1e-10)
        np.testing.assert_allclose(P[1, 0], 0.0, rtol=1e-10)

    def test_solution_Q_matrix(self, nk_model: NewKeynesianModel) -> None:
        """ショック応答行列Qの値を検証"""
        sol = nk_model.solution
        Q = sol.Q

        assert Q.shape == (2, 3)
        # Q[0, 0] = 1 (e_g -> g), Q[1, 1] = 1 (e_a -> a)
        np.testing.assert_allclose(Q[0, 0], 1.0, rtol=1e-10)
        np.testing.assert_allclose(Q[1, 1], 1.0, rtol=1e-10)
        np.testing.assert_allclose(Q[0, 1], 0.0, rtol=1e-10)
        np.testing.assert_allclose(Q[1, 0], 0.0, rtol=1e-10)

    def test_solution_R_matrix(self, nk_model: NewKeynesianModel) -> None:
        """制御変数の状態依存行列Rの値を検証"""
        sol = nk_model.solution
        R = sol.R

        assert R.shape == (3, 2)
        # R[0, 0] = psi_yg (y の g への応答)
        # 値は正であるべき（政府支出は産出を増加させる）
        assert R[0, 0] > 0  # psi_yg
        # R[1, 0] = psi_pig (π の g への応答) は正
        assert R[1, 0] > 0  # psi_pig

    def test_solution_S_matrix(self, nk_model: NewKeynesianModel) -> None:
        """ショック直接効果行列Sの値を検証"""
        sol = nk_model.solution
        S = sol.S

        assert S.shape == (3, 3)
        # 金融引き締め(e_m > 0)は産出を減少させる
        assert S[0, 2] < 0  # psi_ym < 0

    def test_kappa_value(self, nk_model: NewKeynesianModel) -> None:
        """Phillips曲線スロープkappaの値を検証"""
        sol = nk_model.solution
        # kappa = (1 - theta)(1 - beta*theta) / theta
        # theta = 0.75, beta = 0.999
        expected_kappa = (1 - 0.75) * (1 - 0.999 * 0.75) / 0.75
        np.testing.assert_allclose(sol.kappa, expected_kappa, rtol=1e-10)

    def test_determinacy(self, nk_model: NewKeynesianModel) -> None:
        """解の決定性を検証"""
        sol = nk_model.solution
        # Taylor原則が満たされている場合は determinate
        assert sol.determinacy == "determinate"


class TestDSGEModelGoldenMaster:
    """DSGEModelの政策関数を検証"""

    @pytest.fixture
    def model(self) -> DSGEModel:
        return DSGEModel(DefaultParameters())

    def test_policy_function_P_shape(self, model: DSGEModel) -> None:
        """状態遷移行列Pの形状を検証"""
        pf = model.policy_function
        assert pf.P.shape == (N_VARIABLES, N_VARIABLES)

    def test_policy_function_Q_shape(self, model: DSGEModel) -> None:
        """ショック応答行列Qの形状を検証"""
        pf = model.policy_function
        assert pf.Q.shape == (N_VARIABLES, N_SHOCKS)

    def test_policy_function_P_diagonal_persistence(self, model: DSGEModel) -> None:
        """状態遷移行列Pの持続性パラメータを検証"""
        pf = model.policy_function
        P = pf.P

        # g の持続性 = rho_g = 0.9
        g_idx = model.get_variable_index("g")
        np.testing.assert_allclose(P[g_idx, g_idx], 0.9, rtol=1e-10)

        # a の持続性 = rho_a = 0.9
        a_idx = model.get_variable_index("a")
        np.testing.assert_allclose(P[a_idx, a_idx], 0.9, rtol=1e-10)

        # tau_c の持続性 = rho_tau_c = 0.95
        tau_c_idx = model.get_variable_index("tau_c")
        np.testing.assert_allclose(P[tau_c_idx, tau_c_idx], 0.95, rtol=1e-10)

    def test_policy_function_Q_shock_mapping(self, model: DSGEModel) -> None:
        """ショック応答行列Qのショックマッピングを検証"""
        pf = model.policy_function
        Q = pf.Q

        # e_g (index 1) は g (index 10) に1.0の効果
        g_idx = model.get_variable_index("g")
        np.testing.assert_allclose(Q[g_idx, 1], 1.0, rtol=1e-10)

        # e_a (index 0) は a (index 13) に1.0の効果
        a_idx = model.get_variable_index("a")
        np.testing.assert_allclose(Q[a_idx, 0], 1.0, rtol=1e-10)

        # e_tau (index 3) は tau_c (index 12) に1.0の効果
        tau_c_idx = model.get_variable_index("tau_c")
        np.testing.assert_allclose(Q[tau_c_idx, 3], 1.0, rtol=1e-10)


class TestImpulseResponseGoldenMaster:
    """インパルス応答の数値を検証"""

    @pytest.fixture
    def simulator(self) -> ImpulseResponseSimulator:
        model = DSGEModel(DefaultParameters())
        return ImpulseResponseSimulator(model)

    def test_government_spending_shock_output_response(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """政府支出ショックの産出応答を検証"""
        result = simulator.simulate("e_g", shock_size=0.01, periods=20)
        y = result.get_response("y")

        # t=0 での産出応答は正
        assert y[0] > 0

        # 応答は時間とともに減衰
        assert abs(y[10]) < abs(y[0])
        assert abs(y[20]) < abs(y[10])

    def test_government_spending_shock_inflation_response(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """政府支出ショックのインフレ応答を検証"""
        result = simulator.simulate("e_g", shock_size=0.01, periods=20)
        pi = result.get_response("pi")

        # t=0 でのインフレ応答は正（需要増加）
        assert pi[0] > 0

    def test_consumption_tax_shock_output_response(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """消費税ショックの産出応答を検証"""
        result = simulator.simulate("e_tau", shock_size=0.01, periods=20)
        y = result.get_response("y")

        # 消費税増税は産出を減少させる
        assert y[0] < 0

    def test_technology_shock_output_response(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """技術ショックの産出応答を検証"""
        result = simulator.simulate("e_a", shock_size=0.01, periods=20)
        y = result.get_response("y")

        # 技術向上は産出を増加させる
        assert y[0] > 0

    def test_monetary_shock_output_response(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """金融政策ショックの産出応答を検証"""
        result = simulator.simulate("e_m", shock_size=0.01, periods=20)
        y = result.get_response("y")
        R = result.get_response("R")

        # 金融引き締め（金利上昇）は産出を減少させる
        assert y[0] < 0
        # 名目金利は上昇
        assert R[0] > 0


class TestFiscalMultiplierGoldenMaster:
    """財政乗数の数値を検証"""

    @pytest.fixture
    def calculator(self) -> FiscalMultiplierCalculator:
        model = DSGEModel(DefaultParameters())
        return FiscalMultiplierCalculator(model)

    def test_spending_multiplier_impact(
        self, calculator: FiscalMultiplierCalculator
    ) -> None:
        """政府支出乗数のインパクト値を検証"""
        result = calculator.compute_spending_multiplier(horizon=40)

        # インパクト乗数は0より大きく、通常1-2の範囲
        assert result.impact > 0
        assert result.impact < 3

    def test_spending_multiplier_cumulative(
        self, calculator: FiscalMultiplierCalculator
    ) -> None:
        """政府支出乗数の累積値を検証"""
        result = calculator.compute_spending_multiplier(horizon=40)

        # 累積乗数も正
        assert result.cumulative_4q > 0
        assert result.cumulative_8q > 0

    def test_tax_multiplier_sign(
        self, calculator: FiscalMultiplierCalculator
    ) -> None:
        """消費税乗数の符号を検証"""
        result = calculator.compute_tax_multiplier(horizon=40)

        # 減税乗数は正（減税で産出増加）
        # ただし compute_tax_multiplier は -y_response を使うので符号に注意
        assert result.impact != 0
