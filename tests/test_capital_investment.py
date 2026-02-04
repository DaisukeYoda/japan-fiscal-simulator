"""資本蓄積と投資方程式のテスト"""

import numpy as np
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
from japan_fiscal_simulator.core.exceptions import ValidationError
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

    def test_low_adjustment_cost(self) -> None:
        """低い調整コストのテスト（投資がqに敏感）"""
        S_double_prime = 0.5
        params = InvestmentAdjustmentParameters(S_double_prime=S_double_prime)
        eq = InvestmentAdjustmentEquation(params)

        coef = eq.coefficients()

        # S''が小さいほど、qの係数が大きくなる
        assert coef.q_current == pytest.approx(-2.0)

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
    """拡張NKモデル（14方程式）のテスト"""

    def test_model_variables(self) -> None:
        """モデル変数が正しく設定されていることを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)

        # 状態変数: g, a, k, i, w (Phase 2で拡張)
        assert model.vars.n_state == 5
        assert "g" in model.vars.state_vars
        assert "a" in model.vars.state_vars
        assert "k" in model.vars.state_vars
        assert "i" in model.vars.state_vars
        assert "w" in model.vars.state_vars  # Phase 2

        # 制御変数: y, pi, r, q, rk, n, c, mc, mrs (Phase 4)
        assert model.vars.n_control == 9
        assert "y" in model.vars.control_vars
        assert "pi" in model.vars.control_vars
        assert "r" in model.vars.control_vars
        assert "q" in model.vars.control_vars
        assert "rk" in model.vars.control_vars
        assert "n" in model.vars.control_vars
        assert "c" in model.vars.control_vars
        assert "mc" in model.vars.control_vars
        assert "mrs" in model.vars.control_vars

        # ショック: e_g, e_a, e_m, e_i, e_w, e_p (Phase 3で拡張)
        assert model.vars.n_shock == 6
        assert "e_i" in model.vars.shocks
        assert "e_w" in model.vars.shocks  # Phase 2
        assert "e_p" in model.vars.shocks  # Phase 3

    def test_solution_matrices_shape(self) -> None:
        """解行列の形状が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)
        sol = model.solution

        # P: (5 x 5) 状態遷移
        assert sol.P.shape == (5, 5)
        # Q: (5 x 6) ショック応答
        assert sol.Q.shape == (5, 6)
        # R/S: 9制御変数への写像
        assert sol.R.shape == (9, 5)
        assert sol.S.shape == (9, 6)

    def test_state_persistence(self) -> None:
        """状態変数の持続性が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)
        sol = model.solution

        # g, a はAR(1)持続性にほぼ一致
        assert sol.P[0, 0] == pytest.approx(params.shocks.rho_g)  # g
        assert sol.P[1, 1] == pytest.approx(params.shocks.rho_a)  # a
        # そのほかの状態遷移は構造解の同時決定なので安定性のみ確認
        assert abs(sol.P[2, 2]) < 1.0
        assert abs(sol.P[3, 3]) < 1.0

    def test_capital_accumulation_identity_in_solution(self) -> None:
        """構造解でも資本蓄積恒等式が行列レベルで成立することを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)
        sol = model.solution

        delta = params.firm.delta
        idx_k = model.vars.state_vars.index("k")
        idx_i = model.vars.state_vars.index("i")

        basis_k = np.zeros(model.vars.n_state)
        basis_k[idx_k] = 1.0

        # k_t = (1-δ)k_{t-1} + δi_t かつ i_t = P_i s_{t-1} + Q_i ε_t より、
        # P_k = (1-δ)e_k' + δP_i, Q_k = δQ_i が成立する
        expected_p_row = (1.0 - delta) * basis_k + delta * sol.P[idx_i, :]
        expected_q_row = delta * sol.Q[idx_i, :]

        np.testing.assert_allclose(sol.P[idx_k, :], expected_p_row, atol=1e-10, rtol=1e-8)
        np.testing.assert_allclose(sol.Q[idx_k, :], expected_q_row, atol=1e-10, rtol=1e-8)

    def test_investment_shock_response(self) -> None:
        """投資ショックへの応答が正しいことを確認"""
        params = DefaultParameters()
        model = NewKeynesianModel(params)

        # 投資ショックのインパルス応答
        irf = model.impulse_response("e_i", size=0.01, periods=40)

        # i が正の応答を示す
        assert irf["i"][0] > 0

        # k は正に反応し、蓄積の効果が続く
        assert irf["k"][0] > 0
        assert irf["k"][10] > 0

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

    def test_invalid_resource_share_raises(self) -> None:
        """資源制約シェアが不正な場合は例外を投げる"""
        params = DefaultParameters()
        bad_government = params.government.__class__(
            tau_c=params.government.tau_c,
            tau_l=params.government.tau_l,
            tau_k=params.government.tau_k,
            g_y_ratio=-0.1,
            b_y_ratio=params.government.b_y_ratio,
            transfer_y_ratio=params.government.transfer_y_ratio,
            rho_g=params.government.rho_g,
            rho_tau=params.government.rho_tau,
            phi_b=params.government.phi_b,
        )
        bad_params = params.with_updates(government=bad_government)
        model = NewKeynesianModel(bad_params)
        with pytest.raises(ValidationError):
            _ = model.solution


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
        assert N_SHOCKS == 7

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
