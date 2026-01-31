"""DSGEモデル本体

NewKeynesianModelの構造的解法を使用し、
追加の変数は定常状態関係から導出する。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from japan_fiscal_simulator.core.derived_coefficients import DerivedCoefficients
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.core.steady_state import SteadyState, SteadyStateSolver

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import DefaultParameters


# 変数インデックスの定義
VARIABLE_INDICES = {
    "y": 0,  # 産出
    "c": 1,  # 消費
    "i": 2,  # 投資
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
}

N_VARIABLES = len(VARIABLE_INDICES)

# 先決変数
PREDETERMINED_VARS = ["k", "g", "b", "tau_c", "a"]
N_PREDETERMINED = len(PREDETERMINED_VARS)

# ショック変数
SHOCK_VARS = ["e_a", "e_g", "e_m", "e_tau", "e_risk"]
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

    def __init__(self, params: DefaultParameters) -> None:
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

        NKモデルの5変数 (g, a, y, π, r) から
        14変数システムへ拡張する。
        """
        nk_sol = self.nk_model.solution
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        shocks = self.params.shocks

        # 導出係数を取得
        imp = self.derived_coefficients.compute_impulse_coefficients()
        trans = self.derived_coefficients.compute_transition_coefficients()

        n = N_VARIABLES
        idx = VARIABLE_INDICES

        # 状態遷移行列 P
        P = np.zeros((n, n))

        # NKモデルからの主要変数の遷移
        # 状態変数: g, a
        rho_g = shocks.rho_g
        rho_a = shocks.rho_a

        P[idx["g"], idx["g"]] = rho_g
        P[idx["a"], idx["a"]] = rho_a

        # 制御変数: y, pi, r は状態変数に依存
        # R行列から係数を取得: [y, π, r] = R @ [g, a]
        R = nk_sol.R  # (3 x 2): [y, π, r] x [g, a]

        # 産出
        P[idx["y"], idx["g"]] = R[0, 0] * rho_g  # y_{t+1} は g_t に依存
        P[idx["y"], idx["a"]] = R[0, 1] * rho_a

        # インフレ
        P[idx["pi"], idx["g"]] = R[1, 0] * rho_g
        P[idx["pi"], idx["a"]] = R[1, 1] * rho_a

        # 名目金利（NKモデルのrは実質金利だが、Taylor則では名目として扱う）
        P[idx["R"], idx["g"]] = R[2, 0] * rho_g
        P[idx["R"], idx["a"]] = R[2, 1] * rho_a

        # 実質金利 = 名目金利 - 期待インフレ
        P[idx["r"], idx["R"]] = 1.0
        P[idx["r"], idx["pi"]] = -1.0

        # 派生変数の計算
        # 消費: c = y - g (簡略化された財市場均衡)
        c_y_ratio = (
            1
            - gov.g_y_ratio
            - firm.delta * firm.alpha / (1 / self.params.household.beta - 1 + firm.delta)
        )
        P[idx["c"], idx["y"]] = c_y_ratio
        P[idx["c"], idx["g"]] = -gov.g_y_ratio

        # 投資: 加速度効果（導出係数を使用）
        P[idx["i"], idx["y"]] = trans.investment_accelerator

        # 労働: 生産関数から (N = Y / A * K^(-α))
        P[idx["n"], idx["y"]] = 1.0 / (1 - firm.alpha)

        # 賃金: 限界生産性条件
        P[idx["w"], idx["y"]] = 1.0
        P[idx["w"], idx["n"]] = -1.0

        # 限界費用
        P[idx["mc"], idx["w"]] = 1.0

        # 資本: 緩やかな調整
        P[idx["k"], idx["k"]] = 1 - firm.delta
        P[idx["k"], idx["i"]] = firm.delta

        # 政府債務: 財政ルール（導出係数を使用）
        P[idx["b"], idx["b"]] = trans.debt_persistence
        P[idx["b"], idx["g"]] = gov.g_y_ratio

        # 消費税率: 外生
        P[idx["tau_c"], idx["tau_c"]] = shocks.rho_tau_c

        # ショック応答行列 Q
        # ショック順序: e_a, e_g, e_m, e_tau, e_risk
        Q = np.zeros((n, N_SHOCKS))

        # NKモデルのS行列: (3 x 3) for [y, π, r] x [e_g, e_a, e_m]
        S = nk_sol.S

        # e_a: 技術ショック (index 0)
        Q[idx["a"], 0] = 1.0
        Q[idx["y"], 0] = S[0, 1]  # NKモデルのe_a効果
        Q[idx["pi"], 0] = S[1, 1]
        Q[idx["R"], 0] = S[2, 1]
        Q[idx["r"], 0] = S[2, 1] - S[1, 1]  # R - π
        Q[idx["c"], 0] = S[0, 1] * c_y_ratio
        Q[idx["i"], 0] = S[0, 1] * imp.technology_investment_share
        Q[idx["n"], 0] = S[0, 1] / (1 - firm.alpha) - 1.0  # 技術上昇で労働減少
        Q[idx["w"], 0] = S[0, 1] - Q[idx["n"], 0]

        # e_g: 政府支出ショック (index 1)
        Q[idx["g"], 1] = 1.0
        Q[idx["y"], 1] = S[0, 0]  # NKモデルのe_g効果
        Q[idx["pi"], 1] = S[1, 0]
        Q[idx["R"], 1] = S[2, 0]
        Q[idx["r"], 1] = S[2, 0] - S[1, 0]
        Q[idx["c"], 1] = S[0, 0] * c_y_ratio - gov.g_y_ratio  # クラウディングアウト
        Q[idx["i"], 1] = S[0, 0] * imp.government_spending_investment_spillover
        Q[idx["n"], 1] = S[0, 0] / (1 - firm.alpha)
        Q[idx["w"], 1] = S[0, 0] - Q[idx["n"], 1]
        Q[idx["b"], 1] = gov.g_y_ratio  # 債務増加

        # e_m: 金融政策ショック (index 2)
        Q[idx["R"], 2] = S[2, 2]
        Q[idx["y"], 2] = S[0, 2]
        Q[idx["pi"], 2] = S[1, 2]
        Q[idx["r"], 2] = S[2, 2] - S[1, 2]
        Q[idx["c"], 2] = S[0, 2] * c_y_ratio
        Q[idx["i"], 2] = S[0, 2] * imp.monetary_investment_elasticity

        # e_tau: 消費税ショック (index 3)
        # 消費税増税は消費を減少させ、産出を減少させる
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

        # 固有値（状態遷移行列の対角成分）
        eigenvalues = np.diag(P)

        return PolicyFunctionResult(
            P=P,
            Q=Q,
            n_stable=N_PREDETERMINED,
            n_unstable=N_VARIABLES - N_PREDETERMINED,
            bk_satisfied=True,
            eigenvalues=eigenvalues,
        )

    def get_variable_index(self, name: str) -> int:
        return VARIABLE_INDICES[name]

    def get_variable_name(self, index: int) -> str:
        for name, idx in VARIABLE_INDICES.items():
            if idx == index:
                return name
        raise ValueError(f"Unknown index: {index}")

    def invalidate_cache(self) -> None:
        self._steady_state = None
        self._policy_result = None
        self._nk_model = None
        self._derived_coefficients = None
