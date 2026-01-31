"""中央銀行部門モデル

テイラールール（ZLB考慮）
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import CentralBankParameters


@dataclass
class CentralBankSteadyState:
    """中央銀行の定常状態"""

    nominal_rate: float
    real_rate: float
    inflation: float


class CentralBankSector:
    """中央銀行部門

    テイラールール:
    R_t = R_ss + ρ_R*(R_{t-1} - R_ss) + (1-ρ_R)*[φ_π*(π_t - π*) + φ_y*ŷ_t] + ε^m_t

    ZLB制約:
    R_t = max(R_t, R_lower_bound)
    """

    def __init__(self, params: "CentralBankParameters") -> None:
        self.rho_r = params.rho_r
        self.phi_pi = params.phi_pi
        self.phi_y = params.phi_y
        self.pi_target = params.pi_target
        self.r_lower_bound = params.r_lower_bound

    def compute_steady_state(self, beta: float) -> CentralBankSteadyState:
        """定常状態を計算

        定常状態では:
        - インフレ率 = インフレ目標
        - 実質金利 = 1/β - 1
        - 名目金利 = 実質金利 + インフレ率
        """
        real_rate = 1 / beta - 1
        inflation = self.pi_target
        nominal_rate = real_rate + inflation

        return CentralBankSteadyState(
            nominal_rate=nominal_rate, real_rate=real_rate, inflation=inflation
        )

    def taylor_rule_coefficients(self) -> dict[str, float]:
        """テイラールールの係数

        対数線形化:
        R̂_t = ρ_R * R̂_{t-1} + (1-ρ_R) * [φ_π * π̂_t + φ_y * ŷ_t] + ε^m_t
        """
        return {
            "rho_r": self.rho_r,
            "phi_pi": (1 - self.rho_r) * self.phi_pi,
            "phi_y": (1 - self.rho_r) * self.phi_y,
        }

    def is_at_zlb(self, nominal_rate: float) -> bool:
        """ZLB制約に当たっているか判定"""
        return nominal_rate <= self.r_lower_bound

    def apply_zlb(self, nominal_rate: float) -> float:
        """ZLB制約を適用"""
        return max(nominal_rate, self.r_lower_bound)

    def zlb_adjusted_coefficients(self, is_zlb_binding: bool) -> dict[str, float]:
        """ZLB考慮した係数

        ZLBが拘束的な場合、金利は外生的に下限に固定
        """
        if is_zlb_binding:
            return {
                "rho_r": 1.0,  # 金利は下限で固定
                "phi_pi": 0.0,
                "phi_y": 0.0,
            }
        return self.taylor_rule_coefficients()

    def fisher_equation_coefficients(self) -> dict[str, float]:
        """フィッシャー方程式の係数

        実質金利: r̂_t = R̂_t - E_t[π̂_{t+1}]
        """
        return {
            "nominal_rate": 1.0,
            "expected_inflation": -1.0,
        }

    def get_linearized_matrices(
        self, n_vars: int, var_indices: dict[str, int], is_zlb: bool = False
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """対数線形化された方程式の係数行列を構築

        Returns:
            (A, B, C, D): x_t, E[x_{t+1}], x_{t-1}, ショック の係数行列
        """
        n_eqs = 2  # テイラールール、フィッシャー方程式
        A = np.zeros((n_eqs, n_vars))
        B = np.zeros((n_eqs, n_vars))
        C = np.zeros((n_eqs, n_vars))
        D = np.zeros((n_eqs, 1))  # 1ショック: 金融政策

        taylor = self.zlb_adjusted_coefficients(is_zlb)
        fisher = self.fisher_equation_coefficients()

        R_idx = var_indices["R"]
        r_idx = var_indices["r"]
        pi_idx = var_indices["pi"]
        y_idx = var_indices["y"]

        # テイラールール (row 0): R̂_t = ρ_R * R̂_{t-1} + (1-ρ_R)*[φ_π*π̂_t + φ_y*ŷ_t] + ε^m_t
        A[0, R_idx] = 1.0
        A[0, pi_idx] = -taylor["phi_pi"]
        A[0, y_idx] = -taylor["phi_y"]
        C[0, R_idx] = -taylor["rho_r"]
        D[0, 0] = 1.0  # 金融政策ショック

        # フィッシャー方程式 (row 1): r̂_t = R̂_t - E_t[π̂_{t+1}]
        A[1, r_idx] = 1.0
        A[1, R_idx] = -fisher["nominal_rate"]
        B[1, pi_idx] = fisher["expected_inflation"]

        return A, B, C, D

    def monetary_transmission(self) -> dict[str, str]:
        """金融政策の波及経路を説明"""
        return {
            "interest_rate_channel": "金利変化 → 消費・投資決定",
            "inflation_expectations": "期待インフレ変化 → 実質金利",
            "exchange_rate": "金利差 → 為替レート → 純輸出（本モデルでは省略）",
            "asset_prices": "資産価格 → 富効果（金融部門で考慮）",
        }
