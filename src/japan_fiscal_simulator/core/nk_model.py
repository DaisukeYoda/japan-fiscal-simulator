"""New Keynesian DSGEモデル

11方程式NKモデル（コア3方程式 + 財政・資本・労働ブロック）

縮約形解法を使用（行列が特異な場合でも解ける）
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from japan_fiscal_simulator.core.equation_system import EquationSystem, SystemMatrices
from japan_fiscal_simulator.core.equations import (
    CapitalAccumulation,
    CapitalAccumulationParameters,
    CapitalRentalRateEquation,
    GovernmentSpendingProcess,
    InvestmentAdjustmentEquation,
    InvestmentAdjustmentParameters,
    ISCurve,
    ISCurveParameters,
    LaborDemand,
    LaborDemandParameters,
    PhillipsCurve,
    PhillipsCurveParameters,
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
from japan_fiscal_simulator.core.exceptions import ValidationError
from japan_fiscal_simulator.core.steady_state import SteadyState, SteadyStateSolver

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@dataclass
class NKSolutionResult:
    """NKモデルの縮約形解"""

    # 状態遷移: s_t = P @ s_{t-1} + Q @ ε_t
    P: np.ndarray  # 状態遷移行列 (n_state x n_state)
    Q: np.ndarray  # ショック応答 (n_state x n_shock)

    # 制御変数: c_t = R @ s_t + S @ ε_t
    R: np.ndarray  # 状態依存 (n_control x n_state)
    S: np.ndarray  # ショック直接効果 (n_control x n_shock)

    # 診断情報
    kappa: float  # Phillips曲線スロープ
    determinacy: str  # 解の性質
    message: str


@dataclass
class ModelVariables:
    """モデル変数の定義"""

    # 状態変数（先決変数）- t-1期に決定
    # Phase 2: w（賃金）を追加（ラグ項あり）
    state_vars: tuple[str, ...] = ("g", "a", "k", "i", "w")

    # 制御変数（ジャンプ変数）- t期に決定
    # Phase 2: n（労働）を追加
    control_vars: tuple[str, ...] = ("y", "pi", "r", "q", "rk", "n")

    # ショック
    # Phase 3: e_p（価格マークアップショック）を追加
    shocks: tuple[str, ...] = ("e_g", "e_a", "e_m", "e_i", "e_w", "e_p")

    @property
    def n_state(self) -> int:
        return len(self.state_vars)

    @property
    def n_control(self) -> int:
        return len(self.control_vars)

    @property
    def n_total(self) -> int:
        return self.n_state + self.n_control

    @property
    def n_shock(self) -> int:
        return len(self.shocks)

    def index(self, var: str) -> int:
        """変数のインデックスを取得（状態変数が先、制御変数が後）"""
        if var in self.state_vars:
            return self.state_vars.index(var)
        if var in self.control_vars:
            return self.n_state + self.control_vars.index(var)
        raise ValidationError(f"無効な変数名です: '{var}'")

    def shock_index(self, shock: str) -> int:
        return self.shocks.index(shock)


class NewKeynesianModel:
    """New Keynesian DSGEモデル

    方程式体系（対数線形化済み）:

    1. IS曲線（動学的IS）:
       y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y * g_t

    2. Phillips曲線（NKPC）:
       π_t = (ι_p/(1+βι_p)) * π_{t-1} + (β/(1+βι_p)) * E[π_{t+1}] + κ * mc_t + e_p,t

    3. Taylor則:
       r_t = φ_π * π_t + φ_y * y_t + e_m,t

    4. 政府支出（AR(1)）:
       g_t = ρ_g * g_{t-1} + e_g,t

    5. 技術（AR(1)）:
       a_t = ρ_a * a_{t-1} + e_a,t

    ここで y_t は産出ギャップ（産出 - 自然産出）
    """

    def __init__(self, params: DefaultParameters) -> None:
        self.params = params
        self.vars = ModelVariables()
        self._steady_state: SteadyState | None = None
        self._solution: NKSolutionResult | None = None

    @property
    def steady_state(self) -> SteadyState:
        if self._steady_state is None:
            solver = SteadyStateSolver(self.params)
            self._steady_state = solver.solve()
        return self._steady_state

    @property
    def solution(self) -> NKSolutionResult:
        if self._solution is None:
            self._solution = self._solve_reduced_form()
        return self._solution

    def _create_equations(
        self,
    ) -> tuple[
        GovernmentSpendingProcess,
        TechnologyProcess,
        CapitalAccumulation,
        InvestmentAdjustmentEquation,
        ISCurve,
        PhillipsCurve,
        TaylorRule,
        TobinsQEquation,
        CapitalRentalRateEquation,
        WagePhillipsCurve,
        LaborDemand,
    ]:
        """方程式オブジェクトを作成

        11方程式を返す:
        - 状態変数: g, a, k, i, w (5)
        - 制御変数: y, π, r, q, rk, n (6)
        """
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        inv = self.params.investment
        labor = self.params.labor
        shocks = self.params.shocks

        g_process = GovernmentSpendingProcess(rho_g=shocks.rho_g)
        a_process = TechnologyProcess(rho_a=shocks.rho_a)
        capital = CapitalAccumulation(CapitalAccumulationParameters(delta=firm.delta))
        investment = InvestmentAdjustmentEquation(
            InvestmentAdjustmentParameters(S_double_prime=inv.S_double_prime)
        )
        is_curve = ISCurve(
            ISCurveParameters(sigma=hh.sigma, g_y=gov.g_y_ratio, habit=hh.habit)
        )
        phillips = PhillipsCurve(
            PhillipsCurveParameters(beta=hh.beta, theta=firm.theta, iota_p=firm.iota_p)
        )
        taylor = TaylorRule(TaylorRuleParameters(phi_pi=cb.phi_pi, phi_y=cb.phi_y))
        tobins_q = TobinsQEquation(TobinsQParameters(beta=hh.beta, delta=firm.delta))
        capital_rental = CapitalRentalRateEquation()
        wage_phillips = WagePhillipsCurve(
            WagePhillipsCurveParameters(
                beta=hh.beta, theta_w=labor.theta_w, sigma=hh.sigma, phi=hh.phi
            )
        )
        labor_demand = LaborDemand(LaborDemandParameters(alpha=firm.alpha))

        return (
            g_process,
            a_process,
            capital,
            investment,
            is_curve,
            phillips,
            taylor,
            tobins_q,
            capital_rental,
            wage_phillips,
            labor_demand,
        )

    def _solve_reduced_form(self) -> NKSolutionResult:
        """縮約形で解く

        11方程式NKモデルの解の形:
            状態変数: g, a, k, i, w (5)
            制御変数: y, π, r, q, rk, n (6)

        コアブロック (y, π, r) は g, a に依存:
            y_t = ψ_yg * g_t + ψ_ya * a_t
            π_t = ψ_πg * g_t + ψ_πa * a_t
            r_t = ψ_rg * g_t + ψ_ra * a_t

        資本ブロック (k, i, q, rk) はコアブロックと投資ショックに依存:
            rk_t = y_t - k_{t-1}  (限界生産物条件)

        労働ブロック (w, n):
            n_t = (1/(1-α))·(y_t - a_t) - (α/(1-α))·k_{t-1}  (労働需要)
            w_t は賃金NKPCで動学的に決定
        """
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        labor = self.params.labor
        shocks = self.params.shocks

        # パラメータ
        beta = hh.beta
        sigma = hh.sigma
        delta = firm.delta
        alpha = firm.alpha
        g_y = gov.g_y_ratio
        phi_pi = cb.phi_pi
        phi_y = cb.phi_y
        rho_g = shocks.rho_g
        rho_a = shocks.rho_a
        rho_i = shocks.rho_i
        rho_w = shocks.rho_w
        theta_w = labor.theta_w

        # Phillips曲線スロープ（方程式モジュールを使用）
        kappa = compute_phillips_slope(beta, firm.theta, firm.iota_p)

        # 賃金調整速度
        lambda_w = compute_wage_adjustment_speed(beta, theta_w)

        # Taylor原則のチェック
        is_determinate, taylor_criterion = check_taylor_principle(phi_pi, phi_y, beta, kappa)
        determinacy = "determinate" if is_determinate else "indeterminate"

        # === コアブロック: 政府支出ショックへの応答係数 ===
        denom_pc_g = 1 - beta * rho_g
        pi_y_ratio_g = kappa / denom_pc_g if abs(denom_pc_g) > 1e-10 else 0.0
        coef_y_g = 1 - rho_g + (1 / sigma) * (phi_pi - rho_g) * pi_y_ratio_g + (1 / sigma) * phi_y

        psi_yg = g_y / coef_y_g if abs(coef_y_g) > 1e-10 else 0.0
        psi_pig = pi_y_ratio_g * psi_yg
        psi_rg = phi_pi * psi_pig + phi_y * psi_yg

        # === コアブロック: 技術ショックへの応答係数 ===
        denom_pc_a = 1 - beta * rho_a
        pi_y_ratio_a = kappa / denom_pc_a if abs(denom_pc_a) > 1e-10 else 0.0
        coef_y_a = 1 - rho_a + (1 / sigma) * (phi_pi - rho_a) * pi_y_ratio_a + (1 / sigma) * phi_y

        psi_ya = 1.0 / coef_y_a if abs(coef_y_a) > 1e-10 else 0.0
        psi_pia = pi_y_ratio_a * psi_ya
        psi_ra = phi_pi * psi_pia + phi_y * psi_ya

        # === コアブロック: 金融政策ショックへの応答係数 ===
        rho_m = 0.0
        denom_pc_m = 1 - beta * rho_m
        pi_y_ratio_m = kappa / denom_pc_m if abs(denom_pc_m) > 1e-10 else 0.0
        coef_y_m = 1 + (1 / sigma) * phi_y + (1 / sigma) * phi_pi * pi_y_ratio_m

        psi_ym = -(1 / sigma) / coef_y_m if abs(coef_y_m) > 1e-10 else 0.0
        psi_pim = pi_y_ratio_m * psi_ym
        psi_rm = phi_pi * psi_pim + phi_y * psi_ym + 1.0

        # === 資本ブロックの解法 ===
        q_r_coefficient = -1.0 / (1 - beta * (1 - delta))

        # === 労働ブロックの解法 ===
        # 労働需要: n = (1/(1-α))·(y - a) - (α/(1-α))·k
        labor_share = 1 - alpha
        n_y_coef = 1.0 / labor_share
        n_a_coef = -1.0 / labor_share
        n_k_coef = -alpha / labor_share

        # 労働の各ショックへの応答
        psi_ng = n_y_coef * psi_yg  # e_g -> n (via y)
        psi_na = n_y_coef * psi_ya + n_a_coef  # e_a -> n (via y and directly)
        psi_nm = n_y_coef * psi_ym  # e_m -> n (via y)

        # === 解行列の構築 ===
        # 状態変数: [g, a, k, i, w] (5)
        # 制御変数: [y, π, r, q, rk, n] (6)
        # ショック: [e_g, e_a, e_m, e_i, e_w, e_p] (6)

        n_state = 5
        n_control = 6
        n_shock = 6

        # P: 状態遷移行列 (n_state x n_state)
        P = np.zeros((n_state, n_state))
        P[0, 0] = rho_g  # g の AR(1) 係数
        P[1, 1] = rho_a  # a の AR(1) 係数
        P[2, 2] = 1 - delta  # k の減耗
        P[2, 3] = delta  # i -> k
        P[3, 3] = rho_i  # 投資の持続性（近似）
        # 賃金の持続性: 賃金NKPCから前期賃金への依存
        # 縮約形では rho_w で近似
        P[4, 4] = rho_w

        # Q: 状態へのショック応答 (n_state x n_shock)
        Q = np.zeros((n_state, n_shock))
        Q[0, 0] = 1.0  # e_g -> g
        Q[1, 1] = 1.0  # e_a -> a
        Q[3, 3] = 1.0  # e_i -> i
        Q[4, 4] = 1.0  # e_w -> w

        # R: 制御変数の状態依存 (n_control x n_state)
        # [y, π, r, q, rk, n] = R @ [g, a, k, i, w]
        R = np.zeros((n_control, n_state))
        # y は g, a に依存
        R[0, 0] = psi_yg
        R[0, 1] = psi_ya
        # π は g, a に依存
        R[1, 0] = psi_pig
        R[1, 1] = psi_pia
        # r は g, a に依存
        R[2, 0] = psi_rg
        R[2, 1] = psi_ra
        # q は r を通じて g, a に依存
        R[3, 0] = q_r_coefficient * psi_rg
        R[3, 1] = q_r_coefficient * psi_ra
        # rk = y - k_{t-1}
        R[4, 0] = psi_yg
        R[4, 1] = psi_ya
        R[4, 2] = -1.0  # rk の k への応答
        # n = (1/(1-α))·(y - a) - (α/(1-α))·k
        R[5, 0] = psi_ng  # n の g への応答
        R[5, 1] = psi_na  # n の a への応答（y経由 + 直接）
        R[5, 2] = n_k_coef  # n の k への応答

        # S: 制御変数へのショック直接効果 (n_control x n_shock)
        S = np.zeros((n_control, n_shock))
        # e_m: 金融政策ショック
        S[0, 2] = psi_ym  # e_m -> y
        S[1, 2] = psi_pim  # e_m -> π
        S[2, 2] = psi_rm  # e_m -> r
        S[3, 2] = q_r_coefficient * psi_rm  # e_m -> q (via r)
        S[4, 2] = psi_ym  # e_m -> rk (y と同じ応答)
        S[5, 2] = psi_nm  # e_m -> n (via y)
        # e_w: 賃金マークアップショック -> インフレに波及
        # 賃金上昇 -> 限界費用上昇 -> インフレ上昇 -> 金利上昇 -> 産出減少
        # 賃金インデクセーション(iota_w)を使用してパススルー係数を計算
        # iota_w=0: 完全後向き（前期インフレに連動）、iota_w=1: 完全前向き
        # 参考: Smets-Wouters (2007), Erceg-Henderson-Levin (2000)
        iota_w = labor.iota_w
        wage_inflation_pass_through = lambda_w * iota_w
        S[1, 4] = wage_inflation_pass_through  # e_w -> π
        S[0, 4] = -wage_inflation_pass_through / kappa if kappa > 0 else 0.0  # e_w -> y (via inflation)
        S[5, 4] = n_y_coef * S[0, 4]  # e_w -> n (via y)

        # e_p: 価格マークアップショック -> インフレに直接波及
        # 価格上昇 -> 金利上昇 -> 産出減少
        S[1, 5] = 1.0  # e_p -> π
        # 価格硬直性が高いほど産出への即時下押しは小さくなる近似
        S[0, 5] = -kappa * S[1, 5]
        S[2, 5] = phi_pi * S[1, 5] + phi_y * S[0, 5]  # e_p -> r (Taylor則)
        S[3, 5] = q_r_coefficient * S[2, 5]  # e_p -> q
        S[4, 5] = S[0, 5]  # e_p -> rk (yと同じ)
        S[5, 5] = n_y_coef * S[0, 5]  # e_p -> n (via y)

        return NKSolutionResult(
            P=P,
            Q=Q,
            R=R,
            S=S,
            kappa=kappa,
            determinacy=determinacy,
            message=f"縮約形解法で解を取得 (Taylor criterion = {taylor_criterion:.3f})",
        )

    def _build_system_matrices(self) -> SystemMatrices:
        """システム行列を構築

        モデル形式: A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0

        y_t = [g, a, k, i, w, y, π, r, q, rk, n]  (状態 + 制御)
        """
        (
            g_process,
            a_process,
            capital,
            investment,
            is_curve,
            phillips,
            taylor,
            tobins_q,
            capital_rental,
            wage_phillips,
            labor_demand,
        ) = self._create_equations()

        equation_system = EquationSystem()
        equations = [
            g_process.coefficients(),
            a_process.coefficients(),
            capital.coefficients(),
            investment.coefficients(),
            is_curve.coefficients(),
            phillips.coefficients(),
            taylor.coefficients(),
            tobins_q.coefficients(),
            capital_rental.coefficients(),
            wage_phillips.coefficients(),
            labor_demand.coefficients(),
        ]

        return equation_system.build_matrices(equations)

    def impulse_response(
        self,
        shock: str,
        size: float = 0.01,
        periods: int = 40,
    ) -> dict[str, np.ndarray]:
        """インパルス応答を計算

        Args:
            shock: ショック名 ('e_g', 'e_a', 'e_m', 'e_i', 'e_w', 'e_p')
            size: ショックサイズ
            periods: 期間数

        Returns:
            変数名 -> 応答時系列 の辞書
        """
        sol = self.solution
        shock_idx = self.vars.shock_index(shock)

        # 状態変数の時系列
        n_s = self.vars.n_state
        state = np.zeros((periods + 1, n_s))

        # 初期ショック
        epsilon = np.zeros(self.vars.n_shock)
        epsilon[shock_idx] = size

        # t=0: 初期インパクト
        state[0] = sol.Q @ epsilon

        # t=1,...: 状態遷移
        for t in range(1, periods + 1):
            state[t] = sol.P @ state[t - 1]

        # 制御変数を計算
        # 制御変数 = R @ 状態。非状態ショックは t=0 で S から直接効果を加える
        control = np.zeros((periods + 1, self.vars.n_control))
        for t in range(periods + 1):
            control[t] = sol.R @ state[t]
            if t == 0:
                control[t] += sol.S[:, shock_idx] * size

        # 結果を辞書に
        result = {}
        for i, var in enumerate(self.vars.state_vars):
            result[var] = state[:, i]
        for i, var in enumerate(self.vars.control_vars):
            result[var] = control[:, i]

        return result

    def fiscal_multiplier(self, horizon: int = 20) -> dict[str, float]:
        """財政乗数を計算"""
        irf = self.impulse_response("e_g", size=0.01, periods=horizon)

        y = irf["y"]
        g = irf["g"]

        g_y = self.params.government.g_y_ratio

        # インパクト乗数: dY/dG at t=0
        impact = y[0] / g[0] / g_y if abs(g[0]) > 1e-10 else 0.0

        # 累積乗数
        def cumulative(h: int) -> float:
            y_cum = np.sum(y[:h])
            g_cum = np.sum(g[:h])
            return y_cum / g_cum / g_y if abs(g_cum) > 1e-10 else 0.0

        return {
            "impact": impact,
            "cumulative_4q": cumulative(4),
            "cumulative_8q": cumulative(8),
            "peak": np.max(np.abs(y)) / np.max(np.abs(g)) / g_y,
        }

    def invalidate_cache(self) -> None:
        self._steady_state = None
        self._solution = None
