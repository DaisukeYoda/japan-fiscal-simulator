"""Blanchard-Kahn / Klein系の一般QZソルバー"""

import warnings
from dataclasses import dataclass

import numpy as np
from scipy.linalg import ordqz
from scipy.optimize import least_squares, root

from japan_fiscal_simulator.core.exceptions import (
    BlanchardKahnError,
    SingularMatrixError,
)
from japan_fiscal_simulator.parameters.constants import SOLVER_CONSTANTS

__all__ = ["BlanchardKahnError", "BlanchardKahnResult", "BlanchardKahnSolver"]


@dataclass
class BlanchardKahnResult:
    """Blanchard-Kahn解法の結果"""

    # 状態方程式: s_t = P @ s_{t-1} + Q @ ε_t
    P: np.ndarray
    Q: np.ndarray

    # 制御方程式: c_t = R @ s_t + S @ ε_t
    R: np.ndarray
    S: np.ndarray

    # 診断情報
    n_stable: int
    n_unstable: int
    n_predetermined: int
    n_control: int
    n_forward_looking: int
    bk_satisfied: bool
    eigenvalues: np.ndarray
    message: str
    policy_residual_inf: float = 0.0
    used_fallback: bool = False
    numerically_reliable: bool = True


class BlanchardKahnSolver:
    """一般QZ分解に基づくBKソルバー

    モデル形式:
        A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0

    変数分割:
        y_t = [s_t; c_t]
        s_t: 先決変数 (n_state)
        c_t: ジャンプ変数 (n_control)

    解の形式:
        s_t = P @ s_{t-1} + Q @ ε_t
        c_t = R @ s_t + S @ ε_t
    """

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        n_predetermined: int,
        n_forward_looking: int | None = None,
    ) -> None:
        self.A = np.asarray(A, dtype=float)
        self.B = np.asarray(B, dtype=float)
        self.C = np.asarray(C, dtype=float)
        self.D = np.asarray(D, dtype=float)

        if self.A.shape != self.B.shape or self.A.shape != self.C.shape:
            raise ValueError("A, B, C は同一サイズの正方行列である必要があります")

        if self.A.shape[0] != self.D.shape[0]:
            raise ValueError("D の行数は A と一致する必要があります")

        self.n_total = self.A.shape[0]
        self.n_shocks = self.D.shape[1]
        self.n_state = n_predetermined
        self.n_control = self.n_total - self.n_state
        self.n_forward_looking = (
            n_forward_looking if n_forward_looking is not None else self.n_control
        )

        if not 0 <= self.n_state <= self.n_total:
            raise ValueError("n_predetermined が不正です")
        if not 0 <= self.n_forward_looking <= self.n_total:
            raise ValueError("n_forward_looking が不正です")

    def solve(self, tol: float = 1e-8, *, emit_warnings: bool = False) -> BlanchardKahnResult:
        """Blanchard-Kahn解法を実行"""
        residual_reliability_tol = SOLVER_CONSTANTS.policy_residual_warning
        eigenvalues, n_stable, n_unstable = self._qz_diagnostics(tol)

        if n_unstable != self.n_forward_looking:
            if n_unstable > self.n_forward_looking:
                raise BlanchardKahnError(
                    "不安定固有値が多すぎます: "
                    f"{n_unstable} > {self.n_forward_looking} (解なし)"
                )
            raise BlanchardKahnError(
                "不安定固有値が少なすぎます: "
                f"{n_unstable} < {self.n_forward_looking} (不定解)"
            )

        P, R, policy_residual_inf, used_fallback = self._solve_policy_matrices(
            tol,
            emit_warnings=emit_warnings,
        )
        Q, S = self._solve_shock_matrices(P, R, tol)
        numerically_reliable = policy_residual_inf <= residual_reliability_tol
        message = "QZ分解と係数一致条件により解を取得しました"
        if not numerically_reliable:
            message += f" (警告: 政策残差={policy_residual_inf:.2e})"

        return BlanchardKahnResult(
            P=P,
            Q=Q,
            R=R,
            S=S,
            n_stable=n_stable,
            n_unstable=n_unstable,
            n_predetermined=self.n_state,
            n_control=self.n_control,
            n_forward_looking=self.n_forward_looking,
            bk_satisfied=True,
            eigenvalues=eigenvalues,
            message=message,
            policy_residual_inf=policy_residual_inf,
            used_fallback=used_fallback,
            numerically_reliable=numerically_reliable,
        )

    def _qz_diagnostics(self, tol: float) -> tuple[np.ndarray, int, int]:
        """companion形式のQZ診断を返す"""
        n = self.n_total
        companion_left = np.zeros((2 * n, 2 * n))
        companion_right = np.zeros((2 * n, 2 * n))

        # F @ E[x_{t+1}] = G @ x_t
        # x_t = [y_t, y_{t-1}]'
        companion_left[:n, :n] = self.A
        companion_left[n:, n:] = np.eye(n)

        companion_right[:n, :n] = -self.B
        companion_right[:n, n:] = -self.C
        companion_right[n:, :n] = np.eye(n)

        try:
            _, _, alpha, beta, _, _ = ordqz(companion_right, companion_left, sort="ouc")
        except Exception as e:  # noqa: BLE001
            raise BlanchardKahnError(f"QZ分解に失敗しました: {e}") from e

        with np.errstate(divide="ignore", invalid="ignore"):
            eigenvalues = np.where(np.abs(beta) > tol, alpha / beta, np.inf)

        finite = np.abs(beta) > tol
        boundary = finite & (np.abs(np.abs(eigenvalues) - 1.0) <= tol)
        if np.any(boundary):
            raise BlanchardKahnError("単位円境界上の固有値が存在し、解の一意性が判定できません")

        stable = finite & (np.abs(eigenvalues) < 1.0 - tol)
        unstable = finite & (np.abs(eigenvalues) > 1.0 + tol)

        n_stable = int(np.sum(stable))
        n_unstable = int(np.sum(unstable))

        return eigenvalues, n_stable, n_unstable

    def _solve_policy_matrices(
        self,
        tol: float,
        *,
        emit_warnings: bool,
    ) -> tuple[np.ndarray, np.ndarray, float, bool]:
        """P, R を係数一致条件から求める

        QZでBK条件と安定/不安定の本数判定を行った後、政策関数は係数一致の非線形系
        を解いて求める。Z分割法に比べて、分割ブロックがほぼ特異なケースでも同じ
        実装で扱えるため、この手法を採用している。
        """
        ns = self.n_state
        nc = self.n_control
        n = self.n_total
        max_policy_residual = SOLVER_CONSTANTS.policy_residual_max
        warning_residual = SOLVER_CONSTANTS.policy_residual_warning
        stability_tol = SOLVER_CONSTANTS.verification_tolerance

        identity_state = np.eye(ns)

        def unpack(vec: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            p_size = ns * ns
            P = vec[:p_size].reshape(ns, ns)
            R = vec[p_size:].reshape(nc, ns)
            return P, R

        def residual(vec: np.ndarray) -> np.ndarray:
            P, R = unpack(vec)
            F = np.vstack([identity_state, R])  # y_t = F @ s_t
            # A F P^2 + B F P + C F = 0
            det_residual = self.A @ F @ (P @ P) + self.B @ F @ P + self.C @ F
            return np.asarray(det_residual, dtype=float).reshape(-1)

        x0 = self._initial_guess()
        solution = root(residual, x0=x0, method="hybr", tol=tol)
        x_candidate = np.asarray(solution.x, dtype=float)
        err = float(np.linalg.norm(residual(x_candidate), ord=np.inf))
        used_fallback = False

        if (not solution.success) or err > max_policy_residual:
            # local minima 回避のため、dogbox + 決定的な微小摂動初期値で再探索
            candidates: list[tuple[np.ndarray, float, bool]] = [(x_candidate, err, bool(solution.success))]
            for guess in (
                x0,
                x0 + np.random.default_rng(0).normal(scale=0.2, size=x0.shape),
            ):
                lsq = least_squares(
                    residual,
                    x0=guess,
                    method="dogbox",
                    ftol=tol,
                    xtol=tol,
                    gtol=tol,
                    max_nfev=SOLVER_CONSTANTS.nonlinear_solver_max_nfev,
                )
                x_lsq = np.asarray(lsq.x, dtype=float)
                err_lsq = float(np.linalg.norm(residual(x_lsq), ord=np.inf))
                candidates.append((x_lsq, err_lsq, bool(lsq.success)))

            # 安定な候補を優先し、なければ残差最小を採用
            stable_candidates: list[tuple[np.ndarray, float, bool]] = []
            for cand_x, cand_err, cand_success in candidates:
                cand_P, _ = unpack(cand_x)
                cand_sr = float(np.max(np.abs(np.linalg.eigvals(cand_P)))) if cand_P.size > 0 else 0.0
                if cand_sr < 1.0 + stability_tol:
                    stable_candidates.append((cand_x, cand_err, cand_success))

            selected_pool = stable_candidates if stable_candidates else candidates
            best_x, best_err, best_success = min(selected_pool, key=lambda v: v[1])
            x_candidate = np.asarray(best_x, dtype=float)
            err = float(best_err)
            used_fallback = True
            if (not best_success) and err > max_policy_residual:
                raise BlanchardKahnError(
                    "政策関数の非線形方程式が収束しません: "
                    f"{solution.message} / fallback best residual={err:.2e}"
                )

        P, R = unpack(x_candidate)
        if err > max_policy_residual:
            raise BlanchardKahnError(f"政策関数残差が大きすぎます: {err:.2e}")
        if emit_warnings and (used_fallback or err > warning_residual):
            warnings.warn(
                f"政策関数の残差がしきい値付近です (||res||_inf={err:.2e}, fallback={used_fallback})",
                RuntimeWarning,
                stacklevel=2,
            )

        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(P)))) if P.size > 0 else 0.0
        if spectral_radius >= 1.0 + stability_tol:
            raise BlanchardKahnError(
                f"状態遷移行列Pが不安定です（最大固有値絶対値={spectral_radius:.4f}）"
            )

        if n != ns + nc:
            raise BlanchardKahnError("内部次元計算が不正です")

        return P, R, err, used_fallback

    def _initial_guess(self) -> np.ndarray:
        """非線形解法の初期値"""
        ns = self.n_state
        nc = self.n_control

        B_ss = self.B[:ns, :ns]
        B_sc = self.B[:ns, ns:]
        C_ss = self.C[:ns, :ns]
        C_sc = self.C[:ns, ns:]
        B_cs = self.B[ns:, :ns]
        B_cc = self.B[ns:, ns:]

        R0 = np.zeros((nc, ns))
        if nc > 0 and np.linalg.matrix_rank(B_cc) == nc:
            R0 = np.linalg.solve(B_cc, -B_cs)

        P0 = np.zeros((ns, ns))
        state_matrix = B_ss + B_sc @ R0
        lag_matrix = C_ss + C_sc @ R0
        if np.linalg.matrix_rank(state_matrix) == ns:
            P0 = np.linalg.solve(state_matrix, -lag_matrix)

        # 数値発散を避けるため初期値を緩やかにクリップ
        p_clip = SOLVER_CONSTANTS.initial_guess_p_clip
        r_clip = SOLVER_CONSTANTS.initial_guess_r_clip
        P0 = np.clip(P0, -p_clip, p_clip)
        R0 = np.clip(R0, -r_clip, r_clip)

        return np.concatenate([P0.ravel(), R0.ravel()])

    def _solve_shock_matrices(
        self,
        P: np.ndarray,
        R: np.ndarray,
        tol: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Q, S を係数一致の線形方程式から求める"""
        ns = self.n_state

        identity_state = np.eye(ns)
        F = np.vstack([identity_state, R])

        # (A F P + B F) Q + B_c S + D = 0
        lhs_q = self.A @ F @ P + self.B @ F
        lhs_s = self.B[:, ns:]
        system_matrix = np.hstack([lhs_q, lhs_s])  # n x n
        rhs = -self.D

        try:
            solution = np.linalg.solve(system_matrix, rhs)
        except np.linalg.LinAlgError:
            solution, residuals, _, _ = np.linalg.lstsq(system_matrix, rhs, rcond=None)
            if residuals.size > 0 and float(np.max(residuals)) > 1e-8:
                raise SingularMatrixError("ショック応答行列の連立方程式が特異で解けません") from None

        Q = solution[:ns, :]
        S = solution[ns:, :]

        check = lhs_q @ Q + lhs_s @ S + self.D
        if float(np.linalg.norm(check, ord=np.inf)) > SOLVER_CONSTANTS.verification_tolerance + tol:
            raise SingularMatrixError("ショック応答行列の整合性検証に失敗しました")

        return Q, S

    @staticmethod
    def from_model_matrices(
        A0: np.ndarray,  # y_t の係数
        A1: np.ndarray,  # E[y_{t+1}] の係数
        A_1: np.ndarray,  # y_{t-1} の係数
        B: np.ndarray,  # ショック係数
        n_predetermined: int,
    ) -> BlanchardKahnSolver:
        """モデル行列から直接ソルバーを構築"""
        return BlanchardKahnSolver(
            A=A1,
            B=-A0,
            C=A_1,
            D=B,
            n_predetermined=n_predetermined,
        )


def compute_eigenvalues(A: np.ndarray, B: np.ndarray, tol: float = 1e-10) -> np.ndarray:
    """一般化固有値問題 A*x = λ*B*x を解く"""
    A_arr = np.asarray(A, dtype=float)
    B_arr = np.asarray(B, dtype=float)
    try:
        _, _, alpha, beta, _, _ = ordqz(A_arr, B_arr)
    except Exception:  # noqa: BLE001
        values = np.linalg.eigvals(np.linalg.pinv(B_arr) @ A_arr)
        return np.asarray(values, dtype=complex)

    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(np.abs(beta) > tol, alpha / beta, np.inf)


def check_blanchard_kahn(
    eigenvalues: np.ndarray,
    n_predetermined: int,
    n_total: int,
    tol: float = 1e-8,
) -> tuple[bool, str]:
    """Blanchard-Kahn条件をチェック"""
    eig = np.asarray(eigenvalues)
    finite = np.isfinite(eig)
    boundary = finite & (np.abs(np.abs(eig) - 1.0) <= tol)
    if np.any(boundary):
        return False, "境界根が存在するためBK条件を判定できません"

    n_unstable = int(np.sum(finite & (np.abs(eig) > 1.0 + tol)))
    n_jump = n_total - n_predetermined

    if n_unstable == n_jump:
        return True, f"BK条件充足: 不安定固有値 {n_unstable} = ジャンプ変数 {n_jump}"
    if n_unstable > n_jump:
        return False, f"解なし: 不安定固有値 {n_unstable} > ジャンプ変数 {n_jump}"
    return False, f"不定解: 不安定固有値 {n_unstable} < ジャンプ変数 {n_jump}"
