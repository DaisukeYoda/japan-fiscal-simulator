"""線形合理的期待モデルのソルバー

Blanchard-Kahn法による解法を実装。
参考: Blanchard & Kahn (1980), Uhlig (1999)
"""

from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz, solve


@dataclass
class SolutionResult:
    """解法の結果"""

    P: np.ndarray  # 状態遷移: s_t = P @ s_{t-1} + Q @ ε_t
    Q: np.ndarray  # ショック応答
    R: np.ndarray  # 制御変数: c_t = R @ s_{t-1} + S @ ε_t
    S: np.ndarray
    eigenvalues: np.ndarray
    n_stable: int
    n_unstable: int
    n_state: int
    n_control: int
    bk_satisfied: bool
    message: str


class LinearRESolver:
    """線形合理的期待モデルのソルバー

    モデル形式:
        A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0

    ここで y_t = [s_t; c_t] は状態変数と制御変数の結合ベクトル

    解の形式:
        s_t = P @ s_{t-1} + Q @ ε_t
        c_t = R @ s_{t-1} + S @ ε_t
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        n_state: int,
    ) -> None:
        """
        Args:
            A: E[y_{t+1}]の係数 (n x n)
            B: y_tの係数 (n x n)
            C: y_{t-1}の係数 (n x n)
            D: ショックの係数 (n x m)
            n_state: 状態変数（先決変数）の数
        """
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)
        self.n_state = n_state
        self.n_total = A.shape[0]
        self.n_control = self.n_total - n_state
        self.n_shock = D.shape[1]

    def solve(self, tol: float = 1e-10) -> SolutionResult:
        """Blanchard-Kahn法で解く

        拡大システム:
        [B  A] [y_t    ]   [-C  0 ] [y_{t-1}    ]   [-D]
        [I  0] [E[y_{t+1}]] = [0   I] [y_t (dummy)] + [0 ] ε_t

        これを Γ0 @ z_t = Γ1 @ z_{t-1} + Ψ @ ε_t の形に書く
        """
        n = self.n_total

        # 拡大システムの構築
        # z_t = [y_t; E[y_{t+1}]]
        Gamma0 = np.zeros((2 * n, 2 * n))
        Gamma1 = np.zeros((2 * n, 2 * n))
        Psi = np.zeros((2 * n, self.n_shock))

        # 上半分: B @ y_t + A @ E[y_{t+1}] = -C @ y_{t-1} - D @ ε_t
        Gamma0[:n, :n] = self.B
        Gamma0[:n, n:] = self.A
        Gamma1[:n, :n] = -self.C
        Psi[:n, :] = -self.D

        # 下半分: y_t = y_t (恒等式で E[y_{t+1}] を繋ぐ)
        Gamma0[n:, :n] = np.eye(n)
        Gamma1[n:, n:] = np.eye(n)

        # 一般化Schur分解 (QZ分解)
        # Gamma0 @ Z = Q @ S, Gamma1 @ Z = Q @ T
        # 安定な固有値（|λ| < 1）を左上に並べる
        try:
            S, T, alpha, beta, Q, Z = ordqz(
                Gamma0,
                Gamma1,
                sort="iuc",  # inside unit circle
            )
        except Exception as e:
            return self._failure_result(f"QZ分解失敗: {e}")

        # 固有値を計算
        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta) > tol, alpha / beta, np.inf * np.sign(alpha))

        # 安定・不安定固有値のカウント
        n_stable = np.sum(np.abs(eigenvalues) < 1.0 - tol).item()
        n_unstable = 2 * n - n_stable

        # Blanchard-Kahn条件: 不安定固有値の数 = 制御変数の数
        if n_unstable != self.n_control:
            if n_unstable > self.n_control:
                msg = f"BK条件違反: 不安定固有値 {n_unstable} > 制御変数 {self.n_control} (解なし)"
            else:
                msg = f"BK条件違反: 不安定固有値 {n_unstable} < 制御変数 {self.n_control} (不定解)"
            return self._failure_result(msg, eigenvalues, n_stable, n_unstable)

        # 解の抽出
        # Z を分割: Z = [[Z11, Z12], [Z21, Z22]]
        # 安定部分空間に対応する部分を使う
        ns = n_stable

        Z11 = Z[:n, :ns]  # y_t の安定部分

        # さらに状態・制御で分割
        # Z11 = [[Z11_s], [Z11_c]] (状態変数、制御変数)
        Z11_s = Z11[: self.n_state, :]
        Z11_c = Z11[self.n_state :, :]

        # 政策関数の導出
        # 状態変数の発展: s_t は Z11_s の列空間に入る
        # 制御変数: c_t = Z11_c @ inv(Z11_s) @ s_t

        # Z11_s が正則でない場合の処理
        if np.linalg.matrix_rank(Z11_s, tol=tol) < min(Z11_s.shape):
            return self._failure_result(
                "Z11_s が特異: 状態変数の解が一意に定まらない",
                eigenvalues,
                n_stable,
                n_unstable,
            )

        try:
            # R = Z11_c @ inv(Z11_s): 制御変数の状態依存
            R = solve(Z11_s.T, Z11_c.T).T
        except np.linalg.LinAlgError:
            R = Z11_c @ np.linalg.pinv(Z11_s)

        # 状態遷移行列 P の導出
        # S, T の安定ブロック
        S11 = S[:ns, :ns]
        T11 = T[:ns, :ns]

        # P_tilde = inv(S11) @ T11 (安定部分空間での遷移)
        try:
            P_tilde = solve(S11, T11)
        except np.linalg.LinAlgError:
            P_tilde = np.linalg.pinv(S11) @ T11

        # 状態変数への射影: P = Z11_s @ P_tilde @ inv(Z11_s)
        try:
            P = Z11_s @ P_tilde @ np.linalg.inv(Z11_s)
        except np.linalg.LinAlgError:
            P = Z11_s @ P_tilde @ np.linalg.pinv(Z11_s)

        # 状態変数のみを取り出す
        P = P[: self.n_state, : self.n_state]

        # ショック応答行列 Q, S の導出
        # (B + A @ [P; R @ P]) @ [Q; S] = -D
        # ただし簡略化: 状態方程式から直接計算
        PR = np.vstack([P, R @ P]) if self.n_control > 0 else P

        impact_matrix = self.B + self.A @ PR if self.A.any() else self.B

        try:
            QS = solve(impact_matrix[: self.n_state, : self.n_state], -self.D[: self.n_state, :])
        except np.linalg.LinAlgError:
            QS = np.linalg.lstsq(
                impact_matrix[: self.n_state, : self.n_state],
                -self.D[: self.n_state, :],
                rcond=None,
            )[0]

        Q = QS
        S = R @ Q if self.n_control > 0 else np.zeros((0, self.n_shock))

        return SolutionResult(
            P=P,
            Q=Q,
            R=R,
            S=S,
            eigenvalues=eigenvalues,
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_state=self.n_state,
            n_control=self.n_control,
            bk_satisfied=True,
            message="解が見つかりました",
        )

    def _failure_result(
        self,
        message: str,
        eigenvalues: np.ndarray | None = None,
        n_stable: int = 0,
        n_unstable: int = 0,
    ) -> SolutionResult:
        """失敗時の結果を返す"""
        return SolutionResult(
            P=np.zeros((self.n_state, self.n_state)),
            Q=np.zeros((self.n_state, self.n_shock)),
            R=np.zeros((self.n_control, self.n_state)),
            S=np.zeros((self.n_control, self.n_shock)),
            eigenvalues=eigenvalues if eigenvalues is not None else np.array([]),
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_state=self.n_state,
            n_control=self.n_control,
            bk_satisfied=False,
            message=message,
        )
