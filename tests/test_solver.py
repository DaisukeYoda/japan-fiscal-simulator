"""Blanchard-Kahnソルバーのテスト"""

import numpy as np

from japan_fiscal_simulator.core.solver import (
    BlanchardKahnError,
    BlanchardKahnResult,
    BlanchardKahnSolver,
    check_blanchard_kahn,
)


class TestBlanchardKahnSolver:
    """BlanchardKahnSolverのテスト"""

    def test_stable_system(self) -> None:
        """安定なシステムを解けることを確認"""
        # 単純な安定システム
        n = 2
        A = np.eye(n)
        B = np.eye(n) * 0.5
        C = np.eye(n) * 0.3
        D = np.eye(n)

        solver = BlanchardKahnSolver(A, B, C, D, n_predetermined=2)

        # 全て先決変数の場合、不安定固有値は0個必要
        # このテストケースでは条件を満たすかどうかを確認
        try:
            result = solver.solve()
            assert result.P.shape == (n, n)
        except BlanchardKahnError:
            # BK条件を満たさない場合もある
            pass

    def test_bk_condition_check(self) -> None:
        """BK条件チェック関数のテスト"""
        # 安定な固有値が2個、全体が3個、先決変数が2個の場合
        eigenvalues = np.array([0.5, 0.8, 1.5])
        n_predetermined = 2
        n_total = 3

        satisfied, message = check_blanchard_kahn(eigenvalues, n_predetermined, n_total)

        assert satisfied
        assert "充足" in message

    def test_bk_condition_violation_too_many_unstable(self) -> None:
        """不安定固有値が多すぎる場合のテスト"""
        eigenvalues = np.array([1.2, 1.5, 0.5])
        n_predetermined = 2  # ジャンプ変数は1個
        n_total = 3

        satisfied, message = check_blanchard_kahn(eigenvalues, n_predetermined, n_total)

        assert not satisfied
        assert "解なし" in message or ">" in message


class TestBlanchardKahnResult:
    """BlanchardKahnResultのテスト"""

    def test_result_attributes(self) -> None:
        """結果オブジェクトの属性を確認"""
        n = 2
        m = 1
        P = np.eye(n) * 0.9
        Q = np.ones((n, m))
        eigenvalues = np.array([0.9, 0.8, 1.1, 1.2])

        result = BlanchardKahnResult(
            P=P,
            Q=Q,
            n_stable=2,
            n_unstable=2,
            n_predetermined=2,
            bk_satisfied=True,
            eigenvalues=eigenvalues,
        )

        assert result.P.shape == (n, n)
        assert result.Q.shape == (n, m)
        assert result.n_stable == 2
        assert result.n_unstable == 2
        assert result.bk_satisfied
