"""DSGEモデル本体

NewKeynesianModelの構造的解法を使用し、
追加の変数は定常状態関係から導出する。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from japan_fiscal_simulator.core.derived_coefficients import DerivedCoefficients
from japan_fiscal_simulator.core.exceptions import ValidationError
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.core.steady_state import SteadyState, SteadyStateSolver

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import DefaultParameters


# 変数インデックスの定義
VARIABLE_INDICES = {
    "y": 0,  # 産出
    "c": 1,  # 消費
    "i": 2,  # 投資（先決）
    "n": 3,  # 労働
    "k": 4,  # 資本（先決）
    "pi": 5,  # インフレ率
    "r": 6,  # 実質金利
    "R": 7,  # 名目金利
    "w": 8,  # 実質賃金
    "mc": 9,  # 限界費用
    "g": 10,  # 政府支出（先決）
    "b": 11,  # 政府債務（先決）
    "tau_c": 12,  # 消費税率（先決）
    "a": 13,  # 技術ショック（先決）
    "q": 14,  # Tobin's Q
    "rk": 15,  # 資本収益率
}

N_VARIABLES = len(VARIABLE_INDICES)

# 先決変数
PREDETERMINED_VARS = ["k", "i", "g", "b", "tau_c", "a"]
N_PREDETERMINED = len(PREDETERMINED_VARS)

# ショック変数
SHOCK_VARS = ["e_a", "e_g", "e_m", "e_tau", "e_risk", "e_i", "e_p"]
N_SHOCKS = len(SHOCK_VARS)


@dataclass
class LinearizedSystem:
    """対数線形化されたシステム"""

    A0: np.ndarray
    A1: np.ndarray
    A_1: np.ndarray
    B: np.ndarray
    var_names: list[str]
    shock_names: list[str]


@dataclass
class PolicyFunctionResult:
    """政策関数の結果"""

    P: np.ndarray  # 状態遷移: x_t = P @ x_{t-1}
    Q: np.ndarray  # ショック応答: x_0 = Q @ ε
    n_stable: int
    n_unstable: int
    bk_satisfied: bool
    eigenvalues: np.ndarray


class DSGEModel:
    """日本財政政策DSGEモデル

    内部でNewKeynesianModelの構造的解法を使用し、
    追加変数は定常状態関係から導出する。
    """

    def __init__(self, params: "DefaultParameters") -> None:
        self.params = params
        self._steady_state: SteadyState | None = None
        self._policy_result: PolicyFunctionResult | None = None
        self._nk_model: NewKeynesianModel | None = None
        self._derived_coefficients: DerivedCoefficients | None = None

    @property
    def nk_model(self) -> NewKeynesianModel:
        """内部のNKモデル"""
        if self._nk_model is None:
            self._nk_model = NewKeynesianModel(self.params)
        return self._nk_model

    @property
    def derived_coefficients(self) -> DerivedCoefficients:
        """導出係数"""
        if self._derived_coefficients is None:
            self._derived_coefficients = DerivedCoefficients(self.params)
        return self._derived_coefficients

    @property
    def steady_state(self) -> SteadyState:
        if self._steady_state is None:
            self._steady_state = self.compute_steady_state()
        return self._steady_state

    @property
    def policy_function(self) -> PolicyFunctionResult:
        if self._policy_result is None:
            self._policy_result = self._build_policy_function()
        return self._policy_result

    def compute_steady_state(self) -> SteadyState:
        solver = SteadyStateSolver(self.params)
        return solver.solve()

    def _build_policy_function(self) -> PolicyFunctionResult:
        """NKモデルの解を拡張して政策関数を構築

        NKモデルの状態・制御ブロックから
        16変数システムへ拡張する。
        """
        nk_sol = self.nk_model.solution
        gov = self.params.government
        cb = self.params.central_bank
        inv = self.params.investment
        shocks = self.params.shocks

        # 導出係数を取得
        imp = self.derived_coefficients.compute_impulse_coefficients()
        trans = self.derived_coefficients.compute_transition_coefficients()

        n = N_VARIABLES
        idx = VARIABLE_INDICES

        # 状態遷移行列 P（x_t = P @ x_{t-1}）
        P = np.zeros((n, n))

        nk_P = nk_sol.P  # [g, a, k, i, w] x [g, a, k, i, w]
        nk_R = nk_sol.R  # [y, pi, r, q, rk, n, c, mc, mrs] x [g, a, k, i, w]
        nk_Q = nk_sol.Q  # [g, a, k, i, w] x [e_g, e_a, e_m, e_i, e_w, e_p]
        nk_S = nk_sol.S  # [y, pi, r, q, rk, n, c, mc, mrs] x [e_g, e_a, e_m, e_i, e_w, e_p]

        state_cols = [idx["g"], idx["a"], idx["k"], idx["i"], idx["w"]]
        state_rows = [idx["g"], idx["a"], idx["k"], idx["i"], idx["w"]]

        # NK状態ブロック
        for row_i, row in enumerate(state_rows):
            P[row, state_cols] = nk_P[row_i, :]

        # NK制御ブロック（x_t = P x_{t-1} へ写像）
        control_transition = nk_R @ nk_P
        P[idx["y"], state_cols] = control_transition[0, :]
        P[idx["pi"], state_cols] = control_transition[1, :]
        P[idx["R"], state_cols] = control_transition[2, :]  # NKのrは名目政策金利
        P[idx["q"], state_cols] = control_transition[3, :]
        P[idx["rk"], state_cols] = control_transition[4, :]
        P[idx["n"], state_cols] = control_transition[5, :]
        P[idx["c"], state_cols] = control_transition[6, :]
        P[idx["mc"], state_cols] = control_transition[7, :]

        # 実質金利 = 名目金利 - インフレ
        P[idx["r"], :] = P[idx["R"], :] - P[idx["pi"], :]

        # 政府債務: 財政ルール（導出係数を使用）
        P[idx["b"], idx["b"]] = trans.debt_persistence
        P[idx["b"], idx["g"]] = gov.g_y_ratio

        # 消費税率: 外生
        P[idx["tau_c"], idx["tau_c"]] = shocks.rho_tau_c

        # ショック応答行列 Q
        # ショック順序: e_a, e_g, e_m, e_tau, e_risk, e_i, e_p
        Q = np.zeros((n, N_SHOCKS))

        # NKショックをfullショックへマッピング
        # full: [e_a, e_g, e_m, e_tau, e_risk, e_i, e_p]
        # nk  : [e_g, e_a, e_m, e_i, e_w, e_p]
        mapped_shocks = {
            0: 1,  # e_a
            1: 0,  # e_g
            2: 2,  # e_m
            5: 3,  # e_i
            6: 5,  # e_p
        }

        for full_shock_idx, nk_shock_idx in mapped_shocks.items():
            state_impact = nk_Q[:, nk_shock_idx]
            control_impact = nk_R @ state_impact + nk_S[:, nk_shock_idx]

            # NK state vars
            Q[idx["g"], full_shock_idx] = state_impact[0]
            Q[idx["a"], full_shock_idx] = state_impact[1]
            Q[idx["k"], full_shock_idx] = state_impact[2]
            Q[idx["i"], full_shock_idx] = state_impact[3]
            Q[idx["w"], full_shock_idx] = state_impact[4]

            # NK control vars
            Q[idx["y"], full_shock_idx] = control_impact[0]
            Q[idx["pi"], full_shock_idx] = control_impact[1]
            Q[idx["R"], full_shock_idx] = control_impact[2]
            Q[idx["q"], full_shock_idx] = control_impact[3]
            Q[idx["rk"], full_shock_idx] = control_impact[4]
            Q[idx["n"], full_shock_idx] = control_impact[5]
            Q[idx["c"], full_shock_idx] = control_impact[6]
            Q[idx["mc"], full_shock_idx] = control_impact[7]
            Q[idx["r"], full_shock_idx] = control_impact[2] - control_impact[1]

        # e_g は債務へ直接影響
        Q[idx["b"], 1] = gov.g_y_ratio

        # e_tau: 消費税ショック (index 3)
        Q[idx["tau_c"], 3] = 1.0
        Q[idx["c"], 3] = -imp.consumption_tax_elasticity
        Q[idx["y"], 3] = -imp.consumption_tax_elasticity * imp.output_tax_multiplier_factor
        Q[idx["pi"], 3] = imp.inflation_tax_passthrough
        Q[idx["R"], 3] = cb.phi_pi * imp.inflation_tax_passthrough + cb.phi_y * (
            -imp.consumption_tax_elasticity * imp.output_tax_multiplier_factor
        )
        Q[idx["b"], 3] = -imp.debt_tax_effect

        # e_risk: リスクプレミアムショック (index 4)
        Q[idx["r"], 4] = imp.risk_interest_rate_response
        Q[idx["i"], 4] = -imp.risk_investment_response
        Q[idx["y"], 4] = -imp.risk_output_response
        Q[idx["c"], 4] = -imp.risk_consumption_response
        Q[idx["q"], 4] = -imp.risk_investment_response / inv.S_double_prime  # q への影響

        return PolicyFunctionResult(
            P=P,
            Q=Q,
            n_stable=nk_sol.n_stable,
            n_unstable=nk_sol.n_unstable,
            bk_satisfied=nk_sol.bk_satisfied,
            eigenvalues=nk_sol.eigenvalues,
        )

    def get_variable_index(self, name: str) -> int:
        return VARIABLE_INDICES[name]

    def get_variable_name(self, index: int) -> str:
        for name, idx in VARIABLE_INDICES.items():
            if idx == index:
                return name
        raise ValidationError(f"無効な変数インデックスです: {index}")

    def invalidate_cache(self) -> None:
        self._steady_state = None
        self._policy_result = None
        self._nk_model = None
        self._derived_coefficients = None
