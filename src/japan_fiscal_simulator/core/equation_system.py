"""方程式システム

方程式リストからシステム行列を構築する。

モデル形式: A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0
"""

from dataclasses import dataclass

import numpy as np

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class VariableIndices:
    """変数のインデックスマッピング"""

    # 状態変数（先決変数）
    g: int = 0  # 政府支出
    a: int = 1  # 技術
    k: int = 2  # 資本
    i: int = 3  # 投資
    w: int = 4  # 賃金

    # 制御変数（ジャンプ変数）
    y: int = 5  # 産出
    pi: int = 6  # インフレ
    r: int = 7  # 金利
    q: int = 8  # Tobin's Q
    rk: int = 9  # 資本収益率
    n: int = 10  # 労働

    @property
    def n_state(self) -> int:
        return 5

    @property
    def n_control(self) -> int:
        return 6

    @property
    def n_total(self) -> int:
        return 11


@dataclass(frozen=True)
class ShockIndices:
    """ショックのインデックスマッピング"""

    e_g: int = 0  # 政府支出ショック
    e_a: int = 1  # 技術ショック
    e_m: int = 2  # 金融政策ショック
    e_i: int = 3  # 投資固有技術ショック
    e_w: int = 4  # 賃金マークアップショック

    @property
    def n_shocks(self) -> int:
        return 5


@dataclass
class SystemMatrices:
    """システム行列

    モデル形式: A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0
    """

    A: np.ndarray  # E[y_{t+1}] の係数 (n x n)
    B: np.ndarray  # y_t の係数 (n x n)
    C: np.ndarray  # y_{t-1} の係数 (n x n)
    D: np.ndarray  # ε_t の係数 (n x m)


class EquationSystem:
    """方程式からシステム行列を構築するクラス"""

    def __init__(self) -> None:
        self.var_idx = VariableIndices()
        self.shock_idx = ShockIndices()

    def build_matrices(self, equations: list[EquationCoefficients]) -> SystemMatrices:
        """方程式リストからシステム行列を構築

        Args:
            equations: 方程式の係数リスト（順序: g, a, IS, Phillips, Taylor）

        Returns:
            SystemMatrices
        """
        n = self.var_idx.n_total
        m = self.shock_idx.n_shocks

        A = np.zeros((n, n))
        B = np.zeros((n, n))
        C = np.zeros((n, n))
        D = np.zeros((n, m))

        for row, eq in enumerate(equations):
            self._fill_row(A, B, C, D, row, eq)

        return SystemMatrices(A=A, B=B, C=C, D=D)

    def _fill_row(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        row: int,
        eq: EquationCoefficients,
    ) -> None:
        """1つの方程式の係数を行列に書き込む"""
        idx = self.var_idx
        sidx = self.shock_idx

        # 期待値（t+1期）→ A行列
        A[row, idx.y] = eq.y_forward
        A[row, idx.pi] = eq.pi_forward
        A[row, idx.r] = eq.r_forward
        A[row, idx.g] = eq.g_forward
        A[row, idx.a] = eq.a_forward
        A[row, idx.k] = eq.k_forward
        A[row, idx.i] = eq.i_forward
        A[row, idx.q] = eq.q_forward
        A[row, idx.rk] = eq.rk_forward
        A[row, idx.w] = eq.w_forward

        # 当期（t期）→ B行列
        B[row, idx.y] = eq.y_current
        B[row, idx.pi] = eq.pi_current
        B[row, idx.r] = eq.r_current
        B[row, idx.g] = eq.g_current
        B[row, idx.a] = eq.a_current
        B[row, idx.k] = eq.k_current
        B[row, idx.i] = eq.i_current
        B[row, idx.q] = eq.q_current
        B[row, idx.rk] = eq.rk_current
        B[row, idx.w] = eq.w_current
        B[row, idx.n] = eq.n_current

        # 前期（t-1期）→ C行列
        C[row, idx.y] = eq.y_lag
        C[row, idx.pi] = eq.pi_lag
        C[row, idx.r] = eq.r_lag
        C[row, idx.g] = eq.g_lag
        C[row, idx.a] = eq.a_lag
        C[row, idx.k] = eq.k_lag
        C[row, idx.i] = eq.i_lag
        C[row, idx.q] = eq.q_lag
        C[row, idx.rk] = eq.rk_lag
        C[row, idx.w] = eq.w_lag

        # ショック → D行列
        D[row, sidx.e_g] = eq.e_g
        D[row, sidx.e_a] = eq.e_a
        D[row, sidx.e_m] = eq.e_m
        D[row, sidx.e_i] = eq.e_i
        D[row, sidx.e_w] = eq.e_w
