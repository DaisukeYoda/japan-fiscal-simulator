"""方程式システム

方程式リストからシステム行列を構築する。

モデル形式: A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0
"""

from dataclasses import dataclass

import numpy as np

from japan_fiscal_simulator.core.equations.base import EquationCoefficients

DEFAULT_STATE_VARS: tuple[str, ...] = ("g", "a", "k", "i", "w")
DEFAULT_CONTROL_VARS: tuple[str, ...] = (
    "y",
    "pi",
    "r",
    "q",
    "rk",
    "n",
    "c",
    "mc",
    "mrs",
)
DEFAULT_SHOCKS: tuple[str, ...] = ("e_g", "e_a", "e_m", "e_i", "e_w", "e_p")


@dataclass(frozen=True)
class SystemMatrices:
    """システム行列"""

    A: np.ndarray  # E[y_{t+1}] の係数 (n x n)
    B: np.ndarray  # y_t の係数 (n x n)
    C: np.ndarray  # y_{t-1} の係数 (n x n)
    D: np.ndarray  # ε_t の係数 (n x m)


class EquationSystem:
    """方程式からシステム行列を構築するクラス"""

    def __init__(
        self,
        state_vars: tuple[str, ...] = DEFAULT_STATE_VARS,
        control_vars: tuple[str, ...] = DEFAULT_CONTROL_VARS,
        shocks: tuple[str, ...] = DEFAULT_SHOCKS,
    ) -> None:
        self.state_vars = state_vars
        self.control_vars = control_vars
        self.var_order = state_vars + control_vars
        self.shocks = shocks
        self.var_index = {name: i for i, name in enumerate(self.var_order)}
        self.shock_index = {name: i for i, name in enumerate(self.shocks)}

    @property
    def n_state(self) -> int:
        return len(self.state_vars)

    @property
    def n_control(self) -> int:
        return len(self.control_vars)

    @property
    def n_total(self) -> int:
        return len(self.var_order)

    @property
    def n_shocks(self) -> int:
        return len(self.shocks)

    def build_matrices(self, equations: list[EquationCoefficients]) -> SystemMatrices:
        """方程式リストからシステム行列を構築"""
        n = self.n_total
        m = self.n_shocks

        if len(equations) != n:
            raise ValueError(f"方程式数({len(equations)})と変数数({n})が一致しません")

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
        for var in self.var_order:
            col = self.var_index[var]
            A[row, col] = getattr(eq, f"{var}_forward", 0.0)
            B[row, col] = getattr(eq, f"{var}_current", 0.0)
            C[row, col] = getattr(eq, f"{var}_lag", 0.0)

        for shock in self.shocks:
            col = self.shock_index[shock]
            D[row, col] = getattr(eq, shock, 0.0)
