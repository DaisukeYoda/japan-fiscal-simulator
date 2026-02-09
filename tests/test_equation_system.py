"""EquationSystemの行列構築テスト"""

from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from japan_fiscal_simulator.core.equation_system import (
    EquationSystem,
    SystemMatrices,
)
from japan_fiscal_simulator.core.equations.base import EquationCoefficients


class TestEquationSystemBuildMatrices:
    """build_matrices の基本テスト"""

    def test_build_matrices_equation_count_mismatch(self) -> None:
        """方程式数と変数数が不一致のとき ValueError が発生する"""
        system = EquationSystem()  # デフォルト14変数
        equations = [EquationCoefficients() for _ in range(5)]

        with pytest.raises(ValueError, match="方程式数"):
            system.build_matrices(equations)

    def test_build_matrices_correct_sizes(self) -> None:
        """14方程式で行列サイズが正しいことを確認"""
        system = EquationSystem()
        equations = [EquationCoefficients() for _ in range(14)]

        matrices = system.build_matrices(equations)

        assert matrices.A.shape == (14, 14)
        assert matrices.B.shape == (14, 14)
        assert matrices.C.shape == (14, 14)
        assert matrices.D.shape == (14, 6)

    def test_coefficient_zero_defaults(self) -> None:
        """全てゼロの EquationCoefficients では全行列がゼロになる"""
        system = EquationSystem()
        equations = [EquationCoefficients() for _ in range(14)]

        matrices = system.build_matrices(equations)

        np.testing.assert_array_equal(matrices.A, np.zeros((14, 14)))
        np.testing.assert_array_equal(matrices.B, np.zeros((14, 14)))
        np.testing.assert_array_equal(matrices.C, np.zeros((14, 14)))
        np.testing.assert_array_equal(matrices.D, np.zeros((14, 6)))


class TestSystemMatricesFrozen:
    """SystemMatrices の不変性テスト"""

    def test_system_matrices_frozen(self) -> None:
        """SystemMatrices は frozen=True のため属性設定で FrozenInstanceError が発生する"""
        matrices = SystemMatrices(
            A=np.zeros((2, 2)),
            B=np.zeros((2, 2)),
            C=np.zeros((2, 2)),
            D=np.zeros((2, 1)),
        )

        with pytest.raises(FrozenInstanceError):
            matrices.A = np.ones((2, 2))  # type: ignore[misc]


class TestVariableOrdering:
    """変数順序のテスト"""

    def test_variable_ordering(self) -> None:
        """状態変数→制御変数の順序で var_index が構成される"""
        system = EquationSystem()

        # 状態変数 (g, a, k, i, w) がインデックス 0-4
        assert system.var_index["g"] == 0
        assert system.var_index["a"] == 1
        assert system.var_index["k"] == 2
        assert system.var_index["i"] == 3
        assert system.var_index["w"] == 4

        # 制御変数 (y, pi, r, q, rk, n, c, mc, mrs) がインデックス 5-13
        assert system.var_index["y"] == 5
        assert system.var_index["mrs"] == 13


class TestSpecificCoefficientPlacement:
    """係数が正しい行列位置に配置されるテスト"""

    def test_specific_coefficient_placement(self) -> None:
        """y_forward, pi_current, e_g が正しい行列セルに入る"""
        system = EquationSystem()

        eq_with_values = EquationCoefficients(
            y_forward=2.0,
            pi_current=-3.0,
            e_g=0.5,
        )
        equations: list[EquationCoefficients] = [eq_with_values] + [
            EquationCoefficients() for _ in range(13)
        ]

        matrices = system.build_matrices(equations)

        # y_forward=2.0 → A[0, var_index["y"]]
        assert matrices.A[0, system.var_index["y"]] == 2.0

        # pi_current=-3.0 → B[0, var_index["pi"]]
        assert matrices.B[0, system.var_index["pi"]] == -3.0

        # e_g=0.5 → D[0, shock_index["e_g"]]
        assert matrices.D[0, system.shock_index["e_g"]] == 0.5


class TestEquationSystemProperties:
    """プロパティのテスト"""

    def test_properties(self) -> None:
        """n_state, n_control, n_total, n_shocks が正しい"""
        system = EquationSystem()

        assert system.n_state == 5
        assert system.n_control == 9
        assert system.n_total == 14
        assert system.n_shocks == 6


class TestCustomVariableSets:
    """カスタム変数セットのテスト"""

    def test_custom_variable_sets(self) -> None:
        """カスタム変数セットで EquationSystem を構築できる"""
        system = EquationSystem(
            state_vars=("x",),
            control_vars=("y",),
            shocks=("e",),
        )

        assert system.n_total == 2
        assert system.n_shocks == 1

        equations = [EquationCoefficients() for _ in range(2)]
        matrices = system.build_matrices(equations)

        assert matrices.A.shape == (2, 2)
        assert matrices.D.shape == (2, 1)
