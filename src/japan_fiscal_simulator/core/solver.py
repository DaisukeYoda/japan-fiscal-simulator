"""Blanchard-Kahn解法によるDSGE線形化モデルのソルバー"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.linalg import ordqz

if TYPE_CHECKING:
    pass


@dataclass
class BlanchardKahnResult:
    """Blanchard-Kahn解法の結果"""

    # 政策関数: x_t = P * x_{t-1} + Q * ε_t
    P: np.ndarray  # 状態変数の遷移行列
    Q: np.ndarray  # ショックの影響行列

    # 診断情報
    n_stable: int  # 安定固有値の数
    n_unstable: int  # 不安定固有値の数
    n_predetermined: int  # 先決変数の数
    bk_satisfied: bool  # Blanchard-Kahn条件充足
    eigenvalues: np.ndarray  # 固有値


class BlanchardKahnError(Exception):
    """Blanchard-Kahn条件違反エラー"""

    pass


class BlanchardKahnSolver:
    """Blanchard-Kahn解法

    対数線形化されたDSGEモデル:
    A * E_t[x_{t+1}] = B * x_t + C * x_{t-1} + D * ε_t

    を解いて政策関数:
    x_t = P * x_{t-1} + Q * ε_t

    を導出する。

    Blanchard-Kahn条件:
    - 不安定固有値の数 = 非先決（ジャンプ）変数の数
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        n_predetermined: int,
    ) -> None:
        """
        Args:
            A: E_t[x_{t+1}]の係数行列 (n x n)
            B: x_tの係数行列 (n x n)
            C: x_{t-1}の係数行列 (n x n)
            D: ショックの係数行列 (n x m)
            n_predetermined: 先決変数の数
        """
        self.A = A
        self.B = B
        self.C = C
        self.D = D
        self.n_predetermined = n_predetermined
        self.n_vars = A.shape[0]
        self.n_shocks = D.shape[1]

    def solve(self, tol: float = 1e-10) -> BlanchardKahnResult:
        """Blanchard-Kahn解法を実行

        Returns:
            BlanchardKahnResult: 政策関数と診断情報
        """
        n = self.n_vars

        # 拡大システムの構築
        # [A  0] [x_{t+1}  ]   [B  C] [x_t  ]   [D]
        # [I  0] [x_t      ] = [0  I] [x_{t-1}] + [0] * ε_t

        # 左辺行列 (2n x 2n)
        F = np.zeros((2 * n, 2 * n))
        F[:n, :n] = self.A
        F[n:, :n] = np.eye(n)

        # 右辺行列 (2n x 2n)
        G = np.zeros((2 * n, 2 * n))
        G[:n, :n] = self.B
        G[:n, n:] = self.C
        G[n:, n:] = np.eye(n)

        # 一般化シューア分解 (QZ分解)
        # F * Z = Q * S, G * Z = Q * T
        # where S, T are upper triangular
        try:
            S, T, alpha, beta, Q, Z = ordqz(G, F, sort="ouc")  # outside unit circle
        except Exception as e:
            raise BlanchardKahnError(f"QZ分解に失敗: {e}") from e

        # 固有値の計算
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta) > tol, alpha / beta, np.inf)

        # 安定・不安定固有値のカウント
        n_stable = np.sum(np.abs(eigenvalues) < 1 - tol).item()
        n_unstable = 2 * n - n_stable

        # Blanchard-Kahn条件のチェック
        n_jump = n - self.n_predetermined
        bk_satisfied = n_unstable == n_jump

        if not bk_satisfied:
            if n_unstable > n_jump:
                raise BlanchardKahnError(
                    f"不安定固有値が多すぎます: {n_unstable} > {n_jump} (解なし)"
                )
            else:
                raise BlanchardKahnError(
                    f"不安定固有値が少なすぎます: {n_unstable} < {n_jump} (不定解)"
                )

        # 政策関数の抽出
        P, Q_mat = self._extract_policy_functions(Z, S, T, n, n_stable, tol)

        return BlanchardKahnResult(
            P=P,
            Q=Q_mat,
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_predetermined=self.n_predetermined,
            bk_satisfied=bk_satisfied,
            eigenvalues=eigenvalues,
        )

    def _extract_policy_functions(
        self,
        Z: np.ndarray,
        S: np.ndarray,
        T: np.ndarray,
        n: int,
        n_stable: int,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """政策関数行列を抽出"""

        # Zを分割
        Z11 = Z[:n, :n_stable]
        Z21 = Z[n:, :n_stable]

        # 政策関数 P の計算
        # x_t = Z21 * Z11^{-1} * x_{t-1} (状態変数部分)
        try:
            Z11_inv = np.linalg.solve(Z11.T, Z21.T).T
            P = Z11_inv
        except np.linalg.LinAlgError:
            # 特異行列の場合は擬似逆行列を使用
            P = Z21 @ np.linalg.pinv(Z11)

        # ショック応答行列 Q の計算
        # まずインパクト行列を計算
        # D行列からの直接効果
        try:
            # 単純化: D行列の効果を直接使用
            # より厳密な計算は (A - B*P)^{-1} * D を使用
            impact_matrix = self.A - self.B @ P[:n, :n] if P.shape[0] >= n else self.A
            Q_mat = (
                np.linalg.solve(impact_matrix[:n, :n], self.D)
                if impact_matrix.shape[0] >= n
                else self.D
            )
        except (np.linalg.LinAlgError, ValueError):
            Q_mat = self.D

        return P, Q_mat

    @staticmethod
    def from_model_matrices(
        A0: np.ndarray,  # x_tの係数
        A1: np.ndarray,  # E[x_{t+1}]の係数
        A_1: np.ndarray,  # x_{t-1}の係数
        B: np.ndarray,  # ショック係数
        n_predetermined: int,
    ) -> BlanchardKahnSolver:
        """モデル行列から直接ソルバーを構築

        モデル形式: A0 * x_t = A1 * E[x_{t+1}] + A_1 * x_{t-1} + B * ε_t

        変換: A1 * E[x_{t+1}] = A0 * x_t - A_1 * x_{t-1} - B * ε_t
        """
        return BlanchardKahnSolver(
            A=A1,
            B=A0,
            C=-A_1,
            D=-B,
            n_predetermined=n_predetermined,
        )


def compute_eigenvalues(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """一般化固有値問題 A*x = λ*B*x を解く"""
    try:
        eigenvalues = np.linalg.eigvals(np.linalg.solve(B, A))
    except np.linalg.LinAlgError:
        # 特異行列の場合
        eigenvalues, _ = np.linalg.eig(np.linalg.pinv(B) @ A)
    return eigenvalues


def check_blanchard_kahn(
    eigenvalues: np.ndarray, n_predetermined: int, n_total: int
) -> tuple[bool, str]:
    """Blanchard-Kahn条件をチェック

    Returns:
        (条件充足フラグ, メッセージ)
    """
    n_stable = np.sum(np.abs(eigenvalues) < 1)
    n_unstable = n_total - n_stable
    n_jump = n_total - n_predetermined

    if n_unstable == n_jump:
        return True, f"BK条件充足: 不安定固有値 {n_unstable} = ジャンプ変数 {n_jump}"
    elif n_unstable > n_jump:
        return False, f"解なし: 不安定固有値 {n_unstable} > ジャンプ変数 {n_jump}"
    else:
        return False, f"不定解: 不安定固有値 {n_unstable} < ジャンプ変数 {n_jump}"
