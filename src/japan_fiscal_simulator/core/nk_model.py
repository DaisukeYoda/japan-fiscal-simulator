"""New Keynesian DSGEモデル

標準的な3方程式NKモデル + 財政拡張

縮約形解法を使用（行列が特異な場合でも解ける）
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from japan_fiscal_simulator.core.equation_system import EquationSystem, SystemMatrices
from japan_fiscal_simulator.core.equations import (
    GovernmentSpendingProcess,
    ISCurve,
    ISCurveParameters,
    PhillipsCurve,
    PhillipsCurveParameters,
    TaylorRule,
    TaylorRuleParameters,
    TechnologyProcess,
    check_taylor_principle,
    compute_phillips_slope,
)
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
    state_vars: tuple[str, ...] = ("g", "a")  # 政府支出、技術

    # 制御変数（ジャンプ変数）- t期に決定
    control_vars: tuple[str, ...] = ("y", "pi", "r")  # 産出、インフレ、金利

    # ショック
    shocks: tuple[str, ...] = ("e_g", "e_a", "e_m")  # 政府支出、技術、金融政策

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
        raise ValueError(f"Unknown variable: {var}")

    def shock_index(self, shock: str) -> int:
        return self.shocks.index(shock)


class NewKeynesianModel:
    """New Keynesian DSGEモデル

    方程式体系（対数線形化済み）:

    1. IS曲線（動学的IS）:
       y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y * g_t

    2. Phillips曲線（NKPC）:
       π_t = β * E[π_{t+1}] + κ * y_t

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
        ISCurve,
        PhillipsCurve,
        TaylorRule,
    ]:
        """方程式オブジェクトを作成"""
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        shocks = self.params.shocks

        g_process = GovernmentSpendingProcess(rho_g=shocks.rho_g)
        a_process = TechnologyProcess(rho_a=shocks.rho_a)
        is_curve = ISCurve(ISCurveParameters(sigma=hh.sigma, g_y=gov.g_y_ratio))
        phillips = PhillipsCurve(PhillipsCurveParameters(beta=hh.beta, theta=firm.theta))
        taylor = TaylorRule(TaylorRuleParameters(phi_pi=cb.phi_pi, phi_y=cb.phi_y))

        return g_process, a_process, is_curve, phillips, taylor

    def _solve_reduced_form(self) -> NKSolutionResult:
        """縮約形で解く

        NKモデルの解の形:
            y_t = ψ_yg * g_t + ψ_ya * a_t
            π_t = ψ_πg * g_t + ψ_πa * a_t
            r_t = ψ_rg * g_t + ψ_ra * a_t

        状態遷移:
            g_t = ρ_g * g_{t-1} + e_g
            a_t = ρ_a * a_{t-1} + e_a
        """
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        shocks = self.params.shocks

        # パラメータ
        beta = hh.beta
        sigma = hh.sigma
        g_y = gov.g_y_ratio
        phi_pi = cb.phi_pi
        phi_y = cb.phi_y
        rho_g = shocks.rho_g
        rho_a = shocks.rho_a

        # Phillips曲線スロープ（方程式モジュールを使用）
        kappa = compute_phillips_slope(beta, firm.theta)

        # Taylor原則のチェック
        is_determinate, taylor_criterion = check_taylor_principle(phi_pi, phi_y, beta, kappa)
        determinacy = "determinate" if is_determinate else "indeterminate"

        # === 政府支出ショックへの応答係数 ===
        # IS: y = ρ_g * y - σ^{-1}[(φ_π*π + φ_y*y) - ρ_g*π] + g_y * g
        # PC: π = β * ρ_g * π + κ * y
        # これらを解いて ψ_yg, ψ_πg を求める

        # Phillips曲線から: π = κ/(1 - β*ρ_g) * y
        denom_pc_g = 1 - beta * rho_g
        pi_y_ratio_g = kappa / denom_pc_g if abs(denom_pc_g) > 1e-10 else 0.0

        # IS曲線に代入して y を解く
        # y * (1 - ρ_g + σ^{-1} * (φ_π - ρ_g) * pi_y_ratio_g + σ^{-1} * φ_y) = g_y * g
        coef_y_g = 1 - rho_g + (1 / sigma) * (phi_pi - rho_g) * pi_y_ratio_g + (1 / sigma) * phi_y

        psi_yg = g_y / coef_y_g if abs(coef_y_g) > 1e-10 else 0.0
        psi_pig = pi_y_ratio_g * psi_yg
        psi_rg = phi_pi * psi_pig + phi_y * psi_yg

        # === 技術ショックへの応答係数 ===
        denom_pc_a = 1 - beta * rho_a
        pi_y_ratio_a = kappa / denom_pc_a if abs(denom_pc_a) > 1e-10 else 0.0

        coef_y_a = 1 - rho_a + (1 / sigma) * (phi_pi - rho_a) * pi_y_ratio_a + (1 / sigma) * phi_y

        psi_ya = 1.0 / coef_y_a if abs(coef_y_a) > 1e-10 else 0.0
        psi_pia = pi_y_ratio_a * psi_ya
        psi_ra = phi_pi * psi_pia + phi_y * psi_ya

        # === 金融政策ショックへの応答係数 ===
        # e_m は AR(1) でないので、ρ_m = 0 として扱う
        rho_m = 0.0
        denom_pc_m = 1 - beta * rho_m
        pi_y_ratio_m = kappa / denom_pc_m if abs(denom_pc_m) > 1e-10 else 0.0

        # IS曲線: y = -σ^{-1} * e_m + σ^{-1} * E[π'] - σ^{-1} * (φ_π*π + φ_y*y)
        # e_m が持続しないので E[π'] = 0
        # y = -σ^{-1} * e_m - σ^{-1} * φ_π * π - σ^{-1} * φ_y * y
        # y * (1 + σ^{-1} * φ_y + σ^{-1} * φ_π * pi_y_ratio_m) = -σ^{-1} * e_m
        coef_y_m = 1 + (1 / sigma) * phi_y + (1 / sigma) * phi_pi * pi_y_ratio_m

        psi_ym = -(1 / sigma) / coef_y_m if abs(coef_y_m) > 1e-10 else 0.0
        psi_pim = pi_y_ratio_m * psi_ym
        psi_rm = phi_pi * psi_pim + phi_y * psi_ym + 1.0  # +1 for direct e_m effect

        # === 解行列の構築 ===
        # 状態変数: [g, a]
        # 制御変数: [y, π, r]
        # ショック: [e_g, e_a, e_m]

        n_state = 2
        n_control = 3
        n_shock = 3

        # P: 状態遷移 (g, a の AR(1))
        P = np.diag([rho_g, rho_a])

        # Q: 状態へのショック応答 (e_g -> g, e_a -> a)
        Q = np.zeros((n_state, n_shock))
        Q[0, 0] = 1.0  # e_g -> g
        Q[1, 1] = 1.0  # e_a -> a

        # R: 制御変数の状態依存
        R = np.array(
            [
                [psi_yg, psi_ya],  # y = ψ_yg * g + ψ_ya * a
                [psi_pig, psi_pia],  # π = ψ_πg * g + ψ_πa * a
                [psi_rg, psi_ra],  # r = ψ_rg * g + ψ_ra * a
            ]
        )

        # S: 制御変数へのショック直接効果
        S = np.zeros((n_control, n_shock))
        S[0, 0] = psi_yg  # y への e_g 効果
        S[0, 1] = psi_ya  # y への e_a 効果
        S[0, 2] = psi_ym  # y への e_m 効果
        S[1, 0] = psi_pig  # π への e_g 効果
        S[1, 1] = psi_pia  # π への e_a 効果
        S[1, 2] = psi_pim  # π への e_m 効果
        S[2, 0] = psi_rg  # r への e_g 効果
        S[2, 1] = psi_ra  # r への e_a 効果
        S[2, 2] = psi_rm  # r への e_m 効果

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

        y_t = [g_t, a_t, y_t, π_t, r_t]  (状態 + 制御)
        """
        g_process, a_process, is_curve, phillips, taylor = self._create_equations()

        equation_system = EquationSystem()
        equations = [
            g_process.coefficients(),
            a_process.coefficients(),
            is_curve.coefficients(),
            phillips.coefficients(),
            taylor.coefficients(),
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
            shock: ショック名 ('e_g', 'e_a', 'e_m')
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

        # 金融政策ショック(e_m)は状態変数に入らない
        is_monetary_shock = shock == "e_m"

        # t=0: 初期インパクト
        state[0] = sol.Q @ epsilon

        # t=1,...: 状態遷移
        for t in range(1, periods + 1):
            state[t] = sol.P @ state[t - 1]

        # 制御変数を計算
        # 制御変数 = R @ 状態 で、状態にはショックが既に含まれている
        # ただし金融政策ショックは状態に入らないのでt=0のみ直接効果
        control = np.zeros((periods + 1, self.vars.n_control))
        for t in range(periods + 1):
            control[t] = sol.R @ state[t]
            # 金融政策ショックはt=0のみ直接効果
            if is_monetary_shock and t == 0:
                # e_m の効果は S[:, 2] に格納
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
