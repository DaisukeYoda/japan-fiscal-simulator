"""金融部門モデル

金融加速器（BGG型簡略版）
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from japan_fiscal.parameters.defaults import FinancialParameters


@dataclass
class FinancialSteadyState:
    """金融部門の定常状態"""

    external_finance_premium: float
    leverage: float
    net_worth: float
    credit_spread: float


class FinancialSector:
    """金融部門（BGG型金融加速器簡略版）

    Bernanke-Gertler-Gilchrist (1999) モデルの簡略版:
    - 企業家は外部資金調達に追加コスト（外部資金プレミアム）を支払う
    - プレミアムはレバレッジの関数
    - 金融ショックはプレミアムを通じて実体経済に波及

    外部資金プレミアム:
    S_t = S_ss * (N_t / (Q_t * K_t))^(-χ)

    where:
    - S_t: 外部資金プレミアム
    - N_t: 純資産
    - Q_t * K_t: 総資産価値
    - χ: 弾力性パラメータ
    """

    def __init__(self, params: "FinancialParameters") -> None:
        self.chi_b = params.chi_b
        self.leverage_ss = params.leverage_ss
        self.survival_rate = params.survival_rate

    def compute_steady_state(self, real_rate: float) -> FinancialSteadyState:
        """定常状態を計算"""
        # 定常状態の外部資金プレミアム
        external_finance_premium = 0.02 / 4  # 年率2%、四半期ベース

        # 純資産（企業価値の一定割合）
        net_worth = 1.0  # 正規化

        # クレジットスプレッド
        credit_spread = external_finance_premium

        return FinancialSteadyState(
            external_finance_premium=external_finance_premium,
            leverage=self.leverage_ss,
            net_worth=net_worth,
            credit_spread=credit_spread,
        )

    def external_finance_premium_coefficients(self) -> dict[str, float]:
        """外部資金プレミアムの係数

        対数線形化:
        ŝ_t = -χ * (n̂_t - q̂_t - k̂_t)
            = -χ * n̂_t + χ * q̂_t + χ * k̂_t
        """
        return {
            "net_worth": -self.chi_b,
            "tobin_q": self.chi_b,
            "capital": self.chi_b,
        }

    def net_worth_evolution_coefficients(self, beta: float) -> dict[str, float]:
        """純資産の発展方程式係数

        対数線形化:
        n̂_t = γ * [r̂_k - r̂_{t-1} + (1-γ)/γ * n̂_{t-1}]

        where γ = survival_rate
        """
        gamma = self.survival_rate

        return {
            "return_on_capital": gamma,
            "borrowing_cost": -gamma,
            "net_worth_lag": 1 - gamma,
        }

    def capital_demand_with_premium(self) -> dict[str, float]:
        """外部資金プレミアム考慮した資本需要

        E_t[r̂_k,{t+1}] = r̂_t + ŝ_t

        資本の期待収益率 = リスクフリーレート + 外部資金プレミアム
        """
        return {
            "expected_return": 1.0,
            "risk_free_rate": -1.0,
            "premium": -1.0,
        }

    def tobin_q_coefficients(self, beta: float, delta: float) -> dict[str, float]:
        """Tobin's Q の係数

        q̂_t = β*(1-δ)*E_t[q̂_{t+1}] + [1-β*(1-δ)]*E_t[r̂_k,{t+1}] - r̂_t - ŝ_t
        """
        return {
            "q_lead": beta * (1 - delta),
            "expected_return": 1 - beta * (1 - delta),
            "interest_rate": -1.0,
            "premium": -1.0,
        }

    def financial_accelerator_mechanism(self) -> dict[str, str]:
        """金融加速器のメカニズム説明"""
        return {
            "asset_price_decline": "資産価格下落 → 純資産減少",
            "premium_increase": "純資産減少 → 外部資金プレミアム上昇",
            "investment_decline": "プレミアム上昇 → 投資減少",
            "further_decline": "投資減少 → さらなる資産価格下落（増幅メカニズム）",
        }

    def get_linearized_matrices(
        self, n_vars: int, var_indices: dict[str, int], beta: float, delta: float
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """対数線形化された方程式の係数行列を構築

        Returns:
            (A, B, C, D): x_t, E[x_{t+1}], x_{t-1}, ショック の係数行列
        """
        n_eqs = 3  # 外部資金プレミアム、純資産発展、Tobin's Q
        A = np.zeros((n_eqs, n_vars))
        B = np.zeros((n_eqs, n_vars))
        C = np.zeros((n_eqs, n_vars))
        D = np.zeros((n_eqs, 1))  # 1ショック: リスクプレミアム

        premium = self.external_finance_premium_coefficients()
        net_worth = self.net_worth_evolution_coefficients(beta)
        tobin = self.tobin_q_coefficients(beta, delta)

        s_idx = var_indices["s"]  # 外部資金プレミアム
        nw_idx = var_indices["nw"]  # 純資産
        q_idx = var_indices["q"]  # Tobin's Q
        k_idx = var_indices["k"]
        r_idx = var_indices["r"]
        rk_idx = var_indices["rk"]  # 資本収益率

        # 外部資金プレミアム (row 0): ŝ_t = -χ*(n̂_t - q̂_t - k̂_t)
        A[0, s_idx] = 1.0
        A[0, nw_idx] = -premium["net_worth"]
        A[0, q_idx] = -premium["tobin_q"]
        A[0, k_idx] = -premium["capital"]
        D[0, 0] = 1.0  # リスクプレミアムショック

        # 純資産発展 (row 1): n̂_t = γ*r̂_k - γ*r̂_{t-1} + (1-γ)*n̂_{t-1}
        A[1, nw_idx] = 1.0
        A[1, rk_idx] = -net_worth["return_on_capital"]
        C[1, r_idx] = -net_worth["borrowing_cost"]
        C[1, nw_idx] = -net_worth["net_worth_lag"]

        # Tobin's Q (row 2)
        A[2, q_idx] = 1.0
        A[2, r_idx] = -tobin["interest_rate"]
        A[2, s_idx] = -tobin["premium"]
        B[2, q_idx] = -tobin["q_lead"]
        B[2, rk_idx] = -tobin["expected_return"]

        return A, B, C, D
