"""Blanchard-Kahnソルバーのテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.core.solver import (
    BlanchardKahnError,
    BlanchardKahnResult,
    BlanchardKahnSolver,
    check_blanchard_kahn,
    compute_eigenvalues,
)
from japan_fiscal_simulator.parameters.constants import SOLVER_CONSTANTS
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
        assert result.policy_residual_inf >= 0.0
        assert isinstance(result.used_fallback, bool)
        assert isinstance(result.numerically_reliable, bool)

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


class TestSolverNumericalStability:
    """数値安定性のテスト"""

    def test_low_rate_japan_stability(self) -> None:
        """beta=0.999の日本低金利パラメータでの安定性"""
        # DefaultParameters already has beta=0.999
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
        spectral_radius = float(np.max(np.abs(np.linalg.eigvals(result.P))))
        assert spectral_radius < 1.0

    def test_policy_residual_within_threshold(self) -> None:
        """残差が閾値内"""
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
        assert result.policy_residual_inf < SOLVER_CONSTANTS.policy_residual_warning

    def test_spectral_radius_all_eigenvalues_inside_unit_circle(self) -> None:
        """P行列の全固有値が単位円内"""
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
        eigvals = np.linalg.eigvals(result.P)
        for ev in eigvals:
            assert abs(ev) < 1.0 + 1e-6  # small tolerance


class TestSolverInputValidation:
    """入力バリデーションのテスト"""

    def test_matrix_size_mismatch_ab(self) -> None:
        """A,Bのサイズ不整合"""
        with pytest.raises(ValueError, match="同一サイズ"):
            BlanchardKahnSolver(
                A=np.eye(2),
                B=np.eye(3),
                C=np.eye(2),
                D=np.zeros((2, 1)),
                n_predetermined=1,
            )

    def test_matrix_size_mismatch_ac(self) -> None:
        """A,Cのサイズ不整合"""
        with pytest.raises(ValueError, match="同一サイズ"):
            BlanchardKahnSolver(
                A=np.eye(2),
                B=np.eye(2),
                C=np.eye(3),
                D=np.zeros((2, 1)),
                n_predetermined=1,
            )

    def test_d_matrix_row_mismatch(self) -> None:
        """D行列の行数不整合"""
        with pytest.raises(ValueError, match="D の行数"):
            BlanchardKahnSolver(
                A=np.eye(2),
                B=np.eye(2),
                C=np.eye(2),
                D=np.zeros((3, 1)),
                n_predetermined=1,
            )

    def test_n_predetermined_negative(self) -> None:
        """n_predetermined負値"""
        with pytest.raises(ValueError, match="n_predetermined"):
            BlanchardKahnSolver(
                A=np.eye(2),
                B=np.eye(2),
                C=np.eye(2),
                D=np.zeros((2, 1)),
                n_predetermined=-1,
            )

    def test_n_predetermined_too_large(self) -> None:
        """n_predetermined過大"""
        with pytest.raises(ValueError, match="n_predetermined"):
            BlanchardKahnSolver(
                A=np.eye(2),
                B=np.eye(2),
                C=np.eye(2),
                D=np.zeros((2, 1)),
                n_predetermined=100,
            )

    def test_n_forward_looking_too_large(self) -> None:
        """n_forward_looking過大"""
        with pytest.raises(ValueError, match="n_forward_looking"):
            BlanchardKahnSolver(
                A=np.eye(2),
                B=np.eye(2),
                C=np.eye(2),
                D=np.zeros((2, 1)),
                n_predetermined=1,
                n_forward_looking=100,
            )


class TestComputeEigenvalues:
    """compute_eigenvaluesユーティリティのテスト"""

    def test_known_eigenvalues(self) -> None:
        """既知の固有値を持つ行列"""
        A = np.array([[2.0, 0.0], [0.0, 3.0]])
        B = np.eye(2)
        eigvals = compute_eigenvalues(A, B)
        eigvals_sorted = np.sort(np.abs(eigvals))
        assert eigvals_sorted[0] == pytest.approx(2.0, abs=1e-8)
        assert eigvals_sorted[1] == pytest.approx(3.0, abs=1e-8)


class TestCheckBlanchardKahnPatterns:
    """check_blanchard_kahnの3パターンテスト"""

    def test_no_solution(self) -> None:
        """解なし: 不安定固有値過多"""
        eigvals = np.array([1.5, 1.8, 2.0])
        satisfied, msg = check_blanchard_kahn(eigvals, n_predetermined=2, n_total=3)
        assert not satisfied
        assert "解なし" in msg

    def test_indeterminate(self) -> None:
        """不定解: 不安定固有値不足"""
        eigvals = np.array([0.2, 0.3, 0.5])
        satisfied, msg = check_blanchard_kahn(eigvals, n_predetermined=2, n_total=3)
        assert not satisfied
        assert "不定解" in msg
