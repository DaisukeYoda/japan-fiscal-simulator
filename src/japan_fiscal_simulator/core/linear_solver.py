"""線形合理的期待モデルの互換ラッパー

`core.solver.BlanchardKahnSolver` へ集約した実装を提供する。
"""

from dataclasses import dataclass

import numpy as np

from japan_fiscal_simulator.core.solver import BlanchardKahnSolver


@dataclass
class SolutionResult:
    """互換結果型"""

    P: np.ndarray
    Q: np.ndarray
    R: np.ndarray
    S: np.ndarray
    eigenvalues: np.ndarray
    n_stable: int
    n_unstable: int
    n_state: int
    n_control: int
    bk_satisfied: bool
    message: str


class LinearRESolver:
    """旧インターフェース互換のREソルバー"""

    def __init__(
        self,
        A: np.ndarray,
        B: np.ndarray,
        C: np.ndarray,
        D: np.ndarray,
        n_state: int,
    ) -> None:
        self._solver = BlanchardKahnSolver(
            A=A,
            B=B,
            C=C,
            D=D,
            n_predetermined=n_state,
        )

    def solve(self, tol: float = 1e-8) -> SolutionResult:
        result = self._solver.solve(tol=tol)
        return SolutionResult(
            P=result.P,
            Q=result.Q,
            R=result.R,
            S=result.S,
            eigenvalues=result.eigenvalues,
            n_stable=result.n_stable,
            n_unstable=result.n_unstable,
            n_state=result.n_predetermined,
            n_control=result.n_control,
            bk_satisfied=result.bk_satisfied,
            message=result.message,
        )
