"""家計部門モデル

消費・労働供給の最適化、習慣形成を含む
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import HouseholdParameters, GovernmentParameters


@dataclass
class HouseholdSteadyState:
    """家計の定常状態"""

    consumption: float
    labor: float
    marginal_utility: float


class HouseholdSector:
    """家計部門

    最適化問題:
    max E_0 Σ β^t [u(C_t - h*C_{t-1}) - χ*N_t^(1+φ)/(1+φ)]
    s.t. (1+τ_c)C_t + B_t = (1-τ_l)W_t*N_t + R_{t-1}*B_{t-1} + Π_t + T_t

    対数線形化された方程式:
    1. オイラー方程式（習慣形成付き）
    2. 労働供給方程式
    """

    def __init__(
        self, params: "HouseholdParameters", gov_params: "GovernmentParameters"
    ) -> None:
        self.beta = params.beta
        self.sigma = params.sigma
        self.phi = params.phi
        self.habit = params.habit
        self.chi = params.chi
        self.tau_c = gov_params.tau_c
        self.tau_l = gov_params.tau_l

    def compute_steady_state(
        self, real_wage: float, real_rate: float
    ) -> HouseholdSteadyState:
        """定常状態を計算

        Args:
            real_wage: 定常状態の実質賃金
            real_rate: 定常状態の実質金利

        Returns:
            HouseholdSteadyState
        """
        # 定常状態の労働供給（first-order condition）
        # W(1-τ_l) = χ * N^φ * λ^(-1) where λ = (1+τ_c) * (C(1-h))^(-σ)
        # 初期推定
        labor = 0.33  # 労働時間の1/3

        # 実質賃金から消費を逆算
        # C = W*N*(1-τ_l)/(1+τ_c) * share (消費の所得比率)
        income = real_wage * labor * (1 - self.tau_l)
        consumption = income * 0.7 / (1 + self.tau_c)  # 消費性向0.7と仮定

        # 限界効用
        marginal_utility = (1 + self.tau_c) * (
            (consumption * (1 - self.habit)) ** (-self.sigma)
        )

        return HouseholdSteadyState(
            consumption=consumption, labor=labor, marginal_utility=marginal_utility
        )

    def euler_equation_coefficients(self) -> dict[str, float]:
        """オイラー方程式の係数を返す

        対数線形化オイラー方程式（習慣形成付き）:
        λ̂_t = (σh/(1-h))ĉ_{t-1} - (σ/(1-h))ĉ_t + β*h*(σ/(1-h))E_t[ĉ_{t+1}]
        λ̂_t = E_t[λ̂_{t+1}] + r̂_t - E_t[π̂_{t+1}]
        """
        h = self.habit
        sigma = self.sigma

        # 習慣形成を含む限界効用の係数
        coef_c_lag = sigma * h / (1 - h)
        coef_c_current = -sigma / (1 - h)
        coef_c_lead = self.beta * h * sigma / (1 - h)

        return {
            "c_lag": coef_c_lag,
            "c_current": coef_c_current,
            "c_lead": coef_c_lead,
            "beta": self.beta,
        }

    def labor_supply_coefficients(self) -> dict[str, float]:
        """労働供給方程式の係数を返す

        対数線形化労働供給:
        ŵ_t = φ*n̂_t + σ/(1-h)*(ĉ_t - h*ĉ_{t-1}) + τ̂_l/(1-τ_l)
        """
        h = self.habit
        sigma = self.sigma

        return {
            "phi": self.phi,
            "c_current": sigma / (1 - h),
            "c_lag": -sigma * h / (1 - h),
            "tau_l_coef": 1 / (1 - self.tau_l),
        }

    def consumption_tax_effect(self) -> dict[str, float]:
        """消費税変更の効果係数

        消費税率変更 τ̂_c は：
        1. オイラー方程式を通じて異時点間消費配分に影響
        2. 労働供給の実質賃金に影響
        """
        return {
            "euler_effect": 1 / (1 + self.tau_c),  # 消費の相対価格効果
            "labor_supply_effect": -1 / (1 + self.tau_c),  # 実質賃金への影響
        }

    def get_linearized_matrices(
        self, n_vars: int, var_indices: dict[str, int]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """対数線形化された方程式の係数行列を構築

        Returns:
            (A, B, C): x_t関連、x_{t+1}期待値関連、x_{t-1}関連の係数行列
        """
        A = np.zeros((2, n_vars))
        B = np.zeros((2, n_vars))
        C = np.zeros((2, n_vars))

        euler = self.euler_equation_coefficients()
        labor = self.labor_supply_coefficients()

        # インデックス取得
        c_idx = var_indices["c"]
        n_idx = var_indices["n"]
        w_idx = var_indices["w"]
        r_idx = var_indices["r"]
        pi_idx = var_indices["pi"]

        # オイラー方程式 (row 0)
        A[0, c_idx] = euler["c_current"]
        A[0, r_idx] = -1.0
        B[0, c_idx] = -euler["c_lead"] - euler["beta"]
        B[0, pi_idx] = euler["beta"]
        C[0, c_idx] = -euler["c_lag"]

        # 労働供給方程式 (row 1)
        A[1, w_idx] = 1.0
        A[1, n_idx] = -labor["phi"]
        A[1, c_idx] = -labor["c_current"]
        C[1, c_idx] = labor["c_lag"]

        return A, B, C
