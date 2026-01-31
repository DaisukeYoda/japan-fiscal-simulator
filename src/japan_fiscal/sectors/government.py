"""政府部門モデル

財政政策変数（消費税率τ_c、政府支出G、移転支払いT）と財政ルール
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from japan_fiscal.parameters.defaults import GovernmentParameters


@dataclass
class GovernmentSteadyState:
    """政府の定常状態"""

    spending: float
    tax_revenue: float
    transfers: float
    debt: float
    primary_balance: float
    debt_service: float


class GovernmentSector:
    """政府部門

    予算制約:
    B_t = (1+r_{t-1})*B_{t-1} + G_t + T_t - τ_c*C_t - τ_l*W_t*N_t - τ_k*r_k*K

    財政ルール:
    τ_t = τ_ss + φ_b*(B_{t-1}/Y_{t-1} - b_ss)
    """

    def __init__(self, params: "GovernmentParameters") -> None:
        self.tau_c = params.tau_c
        self.tau_l = params.tau_l
        self.tau_k = params.tau_k
        self.g_y_ratio = params.g_y_ratio
        self.b_y_ratio = params.b_y_ratio
        self.transfer_y_ratio = params.transfer_y_ratio
        self.rho_g = params.rho_g
        self.rho_tau = params.rho_tau
        self.phi_b = params.phi_b

    def compute_steady_state(
        self,
        output: float,
        consumption: float,
        wage_bill: float,
        capital_income: float,
        real_rate: float,
    ) -> GovernmentSteadyState:
        """定常状態を計算"""
        # 政府支出
        spending = self.g_y_ratio * output

        # 移転支払い
        transfers = self.transfer_y_ratio * output

        # 税収
        consumption_tax = self.tau_c * consumption
        labor_tax = self.tau_l * wage_bill
        capital_tax = self.tau_k * capital_income
        total_tax = consumption_tax + labor_tax + capital_tax

        # 政府債務
        debt = self.b_y_ratio * output

        # 利払い
        debt_service = real_rate * debt

        # プライマリーバランス
        primary_balance = total_tax - spending - transfers

        return GovernmentSteadyState(
            spending=spending,
            tax_revenue=total_tax,
            transfers=transfers,
            debt=debt,
            primary_balance=primary_balance,
            debt_service=debt_service,
        )

    def fiscal_rule_coefficients(self) -> dict[str, float]:
        """財政ルールの係数

        対数線形化:
        τ̂_c,t = ρ_τ * τ̂_c,{t-1} + φ_b * (b̂_{t-1} - ŷ_{t-1}) + ε^τ_t
        ĝ_t = ρ_g * ĝ_{t-1} + ε^g_t
        """
        return {
            "rho_tau": self.rho_tau,
            "rho_g": self.rho_g,
            "phi_b": self.phi_b,
        }

    def budget_constraint_coefficients(
        self, ss: GovernmentSteadyState, output_ss: float, consumption_ss: float
    ) -> dict[str, float]:
        """予算制約の係数

        b̂_t = (1+r)/β * b̂_{t-1} + g/y * ĝ_t + tr/y * t̂r_t
             - τ_c*c/y * (τ̂_c + ĉ_t) - τ_l*wn/y * (τ̂_l + ŵ_t + n̂_t) - ...
        """
        y = output_ss
        c = consumption_ss
        g = ss.spending
        tr = ss.transfers
        b = ss.debt
        tax = ss.tax_revenue

        # 各項目のGDP比
        return {
            "debt_lag": (1 + 0.01) / 0.999,  # (1+r)/β ≈ 1
            "spending": g / y,
            "transfer": tr / y,
            "consumption_tax_base": self.tau_c * c / y,
            "labor_tax_base": self.tau_l * 0.67 * y / y,  # 労働所得シェア約67%
            "capital_tax_base": self.tau_k * 0.33 * y / y,  # 資本所得シェア約33%
        }

    def consumption_tax_shock_effects(self) -> dict[str, float]:
        """消費税ショックの効果

        消費税率の変化が各変数に与える直接効果の係数
        """
        return {
            # 消費への影響（代替効果と所得効果）
            "consumption_substitution": -1 / (1 + self.tau_c),
            # 労働供給への影響
            "labor_supply": 1 / (1 + self.tau_c) * (1 - self.tau_l),
            # 税収への影響
            "revenue_direct": 1.0,
            "revenue_indirect": -0.5,  # 税基盤縮小効果
        }

    def spending_shock_effects(self) -> dict[str, float]:
        """政府支出ショックの効果"""
        return {
            "output_direct": self.g_y_ratio,
            "crowding_out": -0.3,  # クラウディングアウト効果
        }

    def transfer_shock_effects(self) -> dict[str, float]:
        """移転支払いショックの効果"""
        return {
            "consumption_mpc": 0.6,  # 限界消費性向
            "debt_impact": self.transfer_y_ratio,
        }

    def get_linearized_matrices(
        self,
        n_vars: int,
        var_indices: dict[str, int],
        ss: GovernmentSteadyState,
        output_ss: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """対数線形化された方程式の係数行列を構築

        Returns:
            (A, B, C, D): x_t, E[x_{t+1}], x_{t-1}, ショック の係数行列
        """
        n_eqs = 3  # 予算制約、政府支出ルール、税率ルール
        A = np.zeros((n_eqs, n_vars))
        B = np.zeros((n_eqs, n_vars))
        C = np.zeros((n_eqs, n_vars))
        D = np.zeros((n_eqs, 3))  # 3ショック: g, tau_c, transfer

        fiscal = self.fiscal_rule_coefficients()

        b_idx = var_indices["b"]
        g_idx = var_indices["g"]
        tau_c_idx = var_indices["tau_c"]
        y_idx = var_indices["y"]
        c_idx = var_indices["c"]

        # 政府支出ルール (row 0): ĝ_t = ρ_g * ĝ_{t-1} + ε^g_t
        A[0, g_idx] = 1.0
        C[0, g_idx] = -fiscal["rho_g"]
        D[0, 0] = 1.0  # 政府支出ショック

        # 消費税ルール (row 1): τ̂_c,t = ρ_τ * τ̂_c,{t-1} + φ_b * (b̂_{t-1} - ŷ_{t-1}) + ε^τ_t
        A[1, tau_c_idx] = 1.0
        C[1, tau_c_idx] = -fiscal["rho_tau"]
        C[1, b_idx] = -fiscal["phi_b"]
        C[1, y_idx] = fiscal["phi_b"]
        D[1, 1] = 1.0  # 消費税ショック

        # 予算制約 (row 2): 簡略化版
        # b̂_t = (1/β)*b̂_{t-1} + (g/y)*ĝ_t - (τc*c/y)*(τ̂_c + ĉ_t)
        budget = self.budget_constraint_coefficients(
            ss, output_ss, output_ss * 0.6  # C/Y ≈ 0.6
        )
        A[2, b_idx] = 1.0
        A[2, g_idx] = -budget["spending"]
        A[2, tau_c_idx] = budget["consumption_tax_base"]
        A[2, c_idx] = budget["consumption_tax_base"]
        C[2, b_idx] = -budget["debt_lag"]

        return A, B, C, D
