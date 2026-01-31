"""企業部門モデル

Calvo型価格硬直性、CES生産関数を含む
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import FirmParameters


@dataclass
class FirmSteadyState:
    """企業の定常状態"""

    output: float
    capital: float
    labor_demand: float
    marginal_cost: float
    markup: float
    investment: float


class FirmSector:
    """企業部門

    生産関数: Y_t = A_t * K_{t-1}^α * N_t^(1-α)
    価格設定: Calvo型（θの確率で価格維持）
    """

    def __init__(self, params: "FirmParameters") -> None:
        self.alpha = params.alpha
        self.delta = params.delta
        self.theta = params.theta
        self.epsilon = params.epsilon
        self.psi = params.psi

        # マークアップ（定常状態）
        self.markup_ss = self.epsilon / (self.epsilon - 1)

    def compute_steady_state(
        self, real_rate: float, real_wage: float, technology: float = 1.0
    ) -> FirmSteadyState:
        """定常状態を計算"""
        # 資本の限界生産性 = 実質金利 + 減耗率
        mpk = real_rate + self.delta

        # 資本/労働比率
        # α * Y/K = mpk => α * A * (K/N)^(α-1) = mpk
        k_n_ratio = (self.alpha * technology / mpk) ** (1 / (1 - self.alpha))

        # 実質限界費用（定常状態）
        mc = 1 / self.markup_ss

        # 労働需要の決定には一般均衡が必要
        labor = 0.33

        # 資本
        capital = k_n_ratio * labor

        # 産出
        output = technology * (capital**self.alpha) * (labor ** (1 - self.alpha))

        # 投資（定常状態では減耗を補う）
        investment = self.delta * capital

        return FirmSteadyState(
            output=output,
            capital=capital,
            labor_demand=labor,
            marginal_cost=mc,
            markup=self.markup_ss,
            investment=investment,
        )

    def phillips_curve_coefficients(self, beta: float) -> dict[str, float]:
        """New Keynesian Phillips曲線の係数

        対数線形化:
        π̂_t = β*E_t[π̂_{t+1}] + κ*m̂c_t + ψ*π̂_{t-1}

        where κ = (1-θ)(1-βθ)/θ * (1-α)/(1-α+α*ε)
        """
        # 価格硬直性から派生するスロープ係数
        kappa_base = (1 - self.theta) * (1 - beta * self.theta) / self.theta

        # 戦略的補完性の調整
        strategic_comp = (1 - self.alpha) / (1 - self.alpha + self.alpha * self.epsilon)
        kappa = kappa_base * strategic_comp

        return {
            "beta": beta,
            "kappa": kappa,
            "psi": self.psi,  # インデクセーション
        }

    def marginal_cost_coefficients(self) -> dict[str, float]:
        """実質限界費用の係数

        m̂c_t = ŵ_t - (ŷ_t - n̂_t)
             = ŵ_t - (1-α)(ŷ_t - n̂_t)/((1-α))
             = ŵ_t - mpn_t
        """
        return {
            "wage": 1.0,
            "output": -(1 - self.alpha),
            "labor": 1 - self.alpha,
            "technology": -1.0,
        }

    def production_function_coefficients(self) -> dict[str, float]:
        """生産関数の対数線形化係数

        ŷ_t = â_t + α*k̂_{t-1} + (1-α)*n̂_t
        """
        return {
            "technology": 1.0,
            "capital_lag": self.alpha,
            "labor": 1 - self.alpha,
        }

    def capital_demand_coefficients(self) -> dict[str, float]:
        """資本需要の係数

        実質レンタル率 = α * Y/K
        r̂_k = ŷ_t - k̂_{t-1}
        """
        return {
            "output": 1.0,
            "capital_lag": -1.0,
        }

    def investment_adjustment_coefficients(self, beta: float) -> dict[str, float]:
        """投資調整の係数（Tobin's Q理論）

        î_t = (1/(1+β))*î_{t-1} + (β/(1+β))*E_t[î_{t+1}] + (1/S'')*q̂_t

        S'' は投資調整コストの二次導関数（典型値: 2-4）
        """
        s_double_prime = 2.5

        return {
            "investment_lag": 1 / (1 + beta),
            "investment_lead": beta / (1 + beta),
            "tobin_q": 1 / (s_double_prime * (1 + beta)),
        }

    def get_linearized_matrices(
        self, n_vars: int, var_indices: dict[str, int], beta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """対数線形化された方程式の係数行列を構築"""
        A = np.zeros((4, n_vars))
        B = np.zeros((4, n_vars))
        C = np.zeros((4, n_vars))

        phillips = self.phillips_curve_coefficients(beta)
        production = self.production_function_coefficients()
        mc_coef = self.marginal_cost_coefficients()

        y_idx = var_indices["y"]
        pi_idx = var_indices["pi"]
        n_idx = var_indices["n"]
        k_idx = var_indices["k"]
        w_idx = var_indices["w"]
        mc_idx = var_indices["mc"]
        a_idx = var_indices["a"]

        # Phillips曲線 (row 0)
        A[0, pi_idx] = 1.0
        A[0, mc_idx] = -phillips["kappa"]
        B[0, pi_idx] = -phillips["beta"]
        C[0, pi_idx] = -phillips["psi"]

        # 生産関数 (row 1)
        A[1, y_idx] = 1.0
        A[1, n_idx] = -production["labor"]
        A[1, a_idx] = -production["technology"]
        C[1, k_idx] = -production["capital_lag"]

        # 限界費用 (row 2)
        A[2, mc_idx] = 1.0
        A[2, w_idx] = -mc_coef["wage"]
        A[2, y_idx] = -mc_coef["output"]
        A[2, n_idx] = -mc_coef["labor"]

        # 資本蓄積 (row 3)
        # k̂_t = (1-δ)*k̂_{t-1} + δ*î_t
        i_idx = var_indices["i"]
        A[3, k_idx] = 1.0
        A[3, i_idx] = -self.delta
        C[3, k_idx] = -(1 - self.delta)

        return A, B, C
