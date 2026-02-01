"""資本蓄積と投資方程式のテスト"""

import pytest

from japan_fiscal_simulator.core.equations import (
    CapitalAccumulation,
    CapitalAccumulationParameters,
    CapitalRentalRateEquation,
    InvestmentAdjustmentEquation,
    InvestmentAdjustmentParameters,
    TobinsQEquation,
    TobinsQParameters,
)
from japan_fiscal_simulator.core.model import N_SHOCKS, N_VARIABLES, VARIABLE_INDICES, DSGEModel
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestCapitalAccumulation:
    """資本蓄積方程式のテスト"""

    def test_coefficients_basic(self) -> None:
        """基本的な係数のテスト"""
        delta = 0.025
        params = CapitalAccumulationParameters(delta=delta)
        eq = CapitalAccumulation(params)

        coef = eq.coefficients()

        # k_t - (1-δ)k_{t-1} - δi_t = 0
        assert coef.k_current == 1.0
        assert coef.k_lag == pytest.approx(-(1 - delta))
        assert coef.i_current == pytest.approx(-delta)

    def test_name_and_description(self) -> None:
        """名前と説明のテスト"""
        params = CapitalAccumulationParameters(delta=0.025)
        eq = CapitalAccumulation(params)

        assert "Capital" in eq.name
        assert "k_t" in eq.description


class TestTobinsQ:
    """Tobin's Q方程式のテスト"""

    def test_coefficients_basic(self) -> None:
        """基本的な係数のテスト"""
        beta = 0.99
        delta = 0.025
        params = TobinsQParameters(beta=beta, delta=delta)
        eq = TobinsQEquation(params)

        coef = eq.coefficients()

        # q_t - β(1-δ)E[q_{t+1}] - βE[rk_{t+1}] + r_t = 0
        assert coef.q_current == 1.0
        assert coef.q_forward == pytest.approx(-beta * (1 - delta))
        assert coef.rk_forward == pytest.approx(-beta)
        assert coef.r_current == 1.0

    def test_name_and_description(self) -> None:
        """名前と説明のテスト"""
        params = TobinsQParameters(beta=0.99, delta=0.025)
        eq = TobinsQEquation(params)

        assert "Tobin" in eq.name
        assert "q_t" in eq.description


class TestInvestmentAdjustment:
    """投資調整コスト方程式のテスト"""

    def test_coefficients_basic(self) -> None:
        """基本的な係数のテスト"""
        S_double_prime = 5.0
        params = InvestmentAdjustmentParameters(S_double_prime=S_double_prime)
        eq = InvestmentAdjustmentEquation(params)

        coef = eq.coefficients()

        # i_t - i_{t-1} - (1/S'')q_t - e_i = 0
        assert coef.i_current == 1.0
        assert coef.i_lag == -1.0
        assert coef.q_current == pytest.approx(-1.0 / S_double_prime)
        assert coef.e_i == -1.0

    def test_high_adjustment_cost(self) -> None:
        """高い調整コストのテスト（投資がqに鈍感）"""
        S_double_prime = 10.0
        params = InvestmentAdjustmentParameters(S_double_prime=S_double_prime)
        eq = InvestmentAdjustmentEquation(params)

        coef = eq.coefficients()

        # S''が大きいほど、qの係数が小さくなる
        assert coef.q_current == pytest.approx(-0.1)

    def test_name_and_description(self) -> None:
        """名前と説明のテスト"""
        params = InvestmentAdjustmentParameters(S_double_prime=5.0)
        eq = InvestmentAdjustmentEquation(params)

        assert "Investment" in eq.name
        assert "i_t" in eq.description


class TestCapitalRentalRate:
    """資本収益率方程式のテスト"""

    def test_coefficients_basic(self) -> None:
        """基本的な係数のテスト"""
        eq = CapitalRentalRateEquation()
        coef = eq.coefficients()

        # rk_t - y_t + k_{t-1} = 0
        assert coef.rk_current == 1.0
        assert coef.y_current == -1.0
        assert coef.k_lag == 1.0

    def test_name_and_description(self) -> None:
        """名前と説明のテスト"""
        eq = CapitalRentalRateEquation()

        assert "Capital" in eq.name or "Rental" in eq.name
        assert "rk_t" in eq.description


class TestNewKeynesianModelExpanded:
    """拡張NKモデル（9方程式）のテスト"""

    def test_model_variables(self) -> None:
        """モデル変数が正しく設定されていることを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)

        # 状態変数: g, a, k, i
        assert model.vars.n_state == 4
        assert "g" in model.vars.state_vars
        assert "a" in model.vars.state_vars
        assert "k" in model.vars.state_vars
        assert "i" in model.vars.state_vars

        # 制御変数: y, pi, r, q, rk
        assert model.vars.n_control == 5
        assert "y" in model.vars.control_vars
        assert "pi" in model.vars.control_vars
        assert "r" in model.vars.control_vars
        assert "q" in model.vars.control_vars
        assert "rk" in model.vars.control_vars

        # ショック: e_g, e_a, e_m, e_i
        assert model.vars.n_shock == 4
        assert "e_i" in model.vars.shocks

    def test_solution_matrices_shape(self) -> None:
        """解行列の形状が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)
        sol = model.solution

        # P: (4 x 4) 状態遷移
        assert sol.P.shape == (4, 4)
        # Q: (4 x 4) ショック応答
        assert sol.Q.shape == (4, 4)
        # R: (5 x 4) 制御の状態依存
        assert sol.R.shape == (5, 4)
        # S: (5 x 4) 制御へのショック直接効果
        assert sol.S.shape == (5, 4)

    def test_state_persistence(self) -> None:
        """状態変数の持続性が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)
        sol = model.solution

        # 対角成分が持続性パラメータ
        assert sol.P[0, 0] == pytest.approx(params.shocks.rho_g)  # g
        assert sol.P[1, 1] == pytest.approx(params.shocks.rho_a)  # a
        assert sol.P[2, 2] == pytest.approx(1 - params.firm.delta)  # k
        assert sol.P[3, 3] == pytest.approx(params.shocks.rho_i)  # i

    def test_capital_accumulation_in_P(self) -> None:
        """資本蓄積の係数が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)
        sol = model.solution

        # k_t = (1-δ)k_{t-1} + δi_t
        # P[k, k] = 1 - δ
        # P[k, i] = δ
        assert sol.P[2, 2] == pytest.approx(1 - params.firm.delta)
        assert sol.P[2, 3] == pytest.approx(params.firm.delta)

    def test_investment_shock_response(self) -> None:
        """投資ショックへの応答が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)

        # 投資ショックのインパルス応答
        irf = model.impulse_response("e_i", size=0.01, periods=40)

        # i が正の応答を示す
        assert irf["i"][0] > 0

        # k は徐々に蓄積
        assert irf["k"][0] == pytest.approx(0.0, abs=1e-10)  # 初期は影響なし
        assert irf["k"][10] > 0  # 時間経過で蓄積

        # q は投資調整を反映
        assert "q" in irf

        # rk（資本収益率）も計算される
        assert "rk" in irf

    def test_capital_rental_rate_dynamics(self) -> None:
        """資本収益率のダイナミクスを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)

        # 政府支出ショックへの応答
        irf = model.impulse_response("e_g", size=0.01, periods=20)

        # rk は y と k の関数なので存在する
        assert "rk" in irf

        # 政府支出増加は産出を増加させるので、rk も増加するはず
        assert irf["rk"][0] > 0


class TestDSGEModelExpanded:
    """拡張DSGEモデル（16変数）のテスト"""

    def test_variable_indices(self) -> None:
        """変数インデックスが正しく設定されていることを確認"""
        assert "q" in VARIABLE_INDICES
        assert "rk" in VARIABLE_INDICES
        assert VARIABLE_INDICES["q"] == 14
        assert VARIABLE_INDICES["rk"] == 15

    def test_n_variables(self) -> None:
        """変数数が正しいことを確認"""
        assert N_VARIABLES == 16

    def test_n_shocks(self) -> None:
        """ショック数が正しいことを確認"""
        assert N_SHOCKS == 6

    def test_policy_function_shape(self) -> None:
        """政策関数の形状が正しいことを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)
        pf = model.policy_function

        assert pf.P.shape == (N_VARIABLES, N_VARIABLES)
        assert pf.Q.shape == (N_VARIABLES, N_SHOCKS)

    def test_investment_shock_propagation(self) -> None:
        """投資ショックの波及効果を確認"""
        params = DefaultParameters()
        model = DSGEModel(params)
        pf = model.policy_function

        # e_i (index 5) が i に影響することを確認
        idx_i = VARIABLE_INDICES["i"]
        assert pf.Q[idx_i, 5] != 0.0

        # 投資ショックが資本に波及
        idx_k = VARIABLE_INDICES["k"]
        assert pf.Q[idx_k, 5] != 0.0

    def test_tobins_q_dynamics(self) -> None:
        """Tobin's Qのダイナミクスを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)
        pf = model.policy_function

        idx_q = VARIABLE_INDICES["q"]

        # q は政府支出ショック（金利変化）に反応
        assert pf.Q[idx_q, 1] != 0.0  # e_g

        # q は金融政策ショックに反応
        assert pf.Q[idx_q, 2] != 0.0  # e_m


class TestInvestmentParameters:
    """投資パラメータのテスト"""

    def test_default_parameters(self) -> None:
        """デフォルトパラメータが設定されていることを確認"""
        params = DefaultParameters()

        assert hasattr(params, "investment")
        assert params.investment.S_double_prime == 5.0

    def test_shock_parameters(self) -> None:
        """ショックパラメータに投資が追加されていることを確認"""
        params = DefaultParameters()

        assert params.shocks.rho_i == 0.70
        assert params.shocks.sigma_i == 0.01
