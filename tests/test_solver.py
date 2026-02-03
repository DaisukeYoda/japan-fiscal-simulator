"""Blanchard-Kahnソルバーのテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.core.solver import (
    BlanchardKahnError,
    BlanchardKahnResult,
    BlanchardKahnSolver,
    check_blanchard_kahn,
)
from japan_fiscal_simulator.parameters.defaults import CentralBankParameters, DefaultParameters


class TestBlanchardKahnSolver:
    """BlanchardKahnSolverのテスト"""

    def test_solve_nk_system(self) -> None:
        """14方程式システムを解けることを確認"""
        model = NewKeynesianModel(DefaultParameters())
        matrices = model._build_system_matrices()

        solver = BlanchardKahnSolver(
            A=matrices.A,
            B=matrices.B,
            C=matrices.C,
            D=matrices.D,
            n_predetermined=model.vars.n_state,
            n_forward_looking=int(np.linalg.matrix_rank(matrices.A)),
        )

        result = solver.solve()
        assert result.bk_satisfied
        assert result.P.shape == (model.vars.n_state, model.vars.n_state)
        assert result.Q.shape == (model.vars.n_state, model.vars.n_shock)
        assert result.R.shape == (model.vars.n_control, model.vars.n_state)
        assert result.S.shape == (model.vars.n_control, model.vars.n_shock)
        assert result.n_control == model.vars.n_control
        assert result.n_forward_looking == int(np.linalg.matrix_rank(matrices.A))

    def test_indeterminate_when_phi_pi_below_one(self) -> None:
        """phi_pi<1近傍で不定解が発生することを確認"""
        params = DefaultParameters().with_updates(
            central_bank=CentralBankParameters(phi_pi=0.8)
        )
        model = NewKeynesianModel(params)
        matrices = model._build_system_matrices()

        solver = BlanchardKahnSolver(
            A=matrices.A,
            B=matrices.B,
            C=matrices.C,
            D=matrices.D,
            n_predetermined=model.vars.n_state,
            n_forward_looking=int(np.linalg.matrix_rank(matrices.A)),
        )

        with pytest.raises(BlanchardKahnError, match="不定解|少なすぎ"):
            solver.solve()

    def test_boundary_root_detection(self) -> None:
        """単位円境界の固有値を検知することを確認"""
        eigenvalues = np.array([1.0, 0.5, 1.2])
        satisfied, message = check_blanchard_kahn(
            eigenvalues=eigenvalues,
            n_predetermined=1,
            n_total=3,
            tol=1e-8,
        )
        assert not satisfied
        assert "境界根" in message

    def test_bk_condition_check(self) -> None:
        """BK条件チェック関数のテスト"""
        eigenvalues = np.array([0.5, 0.8, 1.5])
        satisfied, message = check_blanchard_kahn(
            eigenvalues=eigenvalues,
            n_predetermined=2,
            n_total=3,
        )
        assert satisfied
        assert "充足" in message

    def test_from_model_matrices_sign_conversion(self) -> None:
        """from_model_matrices の符号変換が正しいことを確認"""
        # x_t = 0.5 x_{t-1}
        # A0 x_t = A1 E[x_{t+1}] + A_1 x_{t-1}
        solver = BlanchardKahnSolver.from_model_matrices(
            A0=np.array([[1.0]]),
            A1=np.array([[0.0]]),
            A_1=np.array([[0.5]]),
            B=np.array([[0.0]]),
            n_predetermined=1,
        )

        result = solver.solve()
        assert result.P.shape == (1, 1)
        assert result.P[0, 0] == pytest.approx(0.5, abs=1e-8)


class TestBlanchardKahnResult:
    """BlanchardKahnResultのテスト"""

    def test_result_attributes(self) -> None:
        """結果オブジェクトの属性を確認"""
        n_state = 2
        n_control = 1
        n_shock = 1
        result = BlanchardKahnResult(
            P=np.eye(n_state) * 0.9,
            Q=np.ones((n_state, n_shock)),
            R=np.ones((n_control, n_state)),
            S=np.ones((n_control, n_shock)),
            n_stable=2,
            n_unstable=1,
            n_predetermined=2,
            n_control=1,
            n_forward_looking=1,
            bk_satisfied=True,
            eigenvalues=np.array([0.9, 0.8, 1.1]),
            message="ok",
        )

        assert result.P.shape == (n_state, n_state)
        assert result.Q.shape == (n_state, n_shock)
        assert result.R.shape == (n_control, n_state)
        assert result.S.shape == (n_control, n_shock)
        assert result.bk_satisfied
