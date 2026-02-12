"""状態空間変換のテスト

拡張状態ベクトル (dim=14):
    [s_t(5), y_t, c_t, π_t, n_t, r_t, y_{t-1}, c_{t-1}, i_{t-1}, w_{t-1}]
"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.nk_model import NewKeynesianModel, NKSolutionResult
from japan_fiscal_simulator.estimation.state_space import StateSpaceBuilder, StateSpaceMatrices
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@pytest.fixture
def default_model() -> NewKeynesianModel:
    """デフォルトパラメータのNKモデル"""
    return NewKeynesianModel(DefaultParameters())


@pytest.fixture
def solution(default_model: NewKeynesianModel) -> NKSolutionResult:
    """モデルの構造解"""
    return default_model.solution


@pytest.fixture
def shock_stds() -> np.ndarray:
    """デフォルトショック標準偏差 [σ_g, σ_a, σ_m, σ_i, σ_w, σ_p]"""
    return np.array([0.01, 0.01, 0.0025, 0.01, 0.01, 0.01])


@pytest.fixture
def measurement_errors() -> np.ndarray:
    """デフォルト測定誤差標準偏差 [me_y, me_c, me_i, me_pi, me_w, me_n, me_r]"""
    return np.array([0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01])


@pytest.fixture
def ss_matrices(
    solution: NKSolutionResult,
    shock_stds: np.ndarray,
    measurement_errors: np.ndarray,
) -> StateSpaceMatrices:
    """状態空間行列"""
    return StateSpaceBuilder.build(solution, shock_stds, measurement_errors)


N_AUG = 14
N_OBS = 7
N_SHOCK = 6
N_STATE = 5


class TestMatrixDimensions:
    """行列の次元が正しいことを検証する"""

    def test_T_shape(self, ss_matrices: StateSpaceMatrices) -> None:
        assert ss_matrices.T.shape == (N_AUG, N_AUG)

    def test_R_aug_shape(self, ss_matrices: StateSpaceMatrices) -> None:
        assert ss_matrices.R_aug.shape == (N_AUG, N_SHOCK)

    def test_Q_cov_shape(self, ss_matrices: StateSpaceMatrices) -> None:
        assert ss_matrices.Q_cov.shape == (N_SHOCK, N_SHOCK)

    def test_Z_shape(self, ss_matrices: StateSpaceMatrices) -> None:
        assert ss_matrices.Z.shape == (N_OBS, N_AUG)

    def test_H_shape(self, ss_matrices: StateSpaceMatrices) -> None:
        assert ss_matrices.H.shape == (N_OBS, N_OBS)

    def test_dimension_attributes(self, ss_matrices: StateSpaceMatrices) -> None:
        assert ss_matrices.n_aug == N_AUG
        assert ss_matrices.n_obs == N_OBS
        assert ss_matrices.n_shock == N_SHOCK


class TestTMatrixStructure:
    """T行列の構造を検証する"""

    def test_top_left_is_P(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """T行列の左上5x5ブロックがPと一致する"""
        np.testing.assert_allclose(ss_matrices.T[:5, :5], solution.P, atol=1e-12)

    def test_state_rows_no_aux_dependency(self, ss_matrices: StateSpaceMatrices) -> None:
        """s_t 行 (0-4) は補助・ラグ列 (5-13) に依存しない"""
        np.testing.assert_allclose(ss_matrices.T[:5, 5:], 0.0, atol=1e-15)

    def test_y_current_row(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """y_t 行 (row 5) が R[0,:] @ P と一致する"""
        expected = solution.R[0, :] @ solution.P
        np.testing.assert_allclose(ss_matrices.T[5, :5], expected, atol=1e-12)
        np.testing.assert_allclose(ss_matrices.T[5, 5:], 0.0, atol=1e-15)

    def test_c_current_row(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """c_t 行 (row 6) が R[6,:] @ P と一致する"""
        expected = solution.R[6, :] @ solution.P
        np.testing.assert_allclose(ss_matrices.T[6, :5], expected, atol=1e-12)
        np.testing.assert_allclose(ss_matrices.T[6, 5:], 0.0, atol=1e-15)

    def test_pi_current_row(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """π_t 行 (row 7) が R[1,:] @ P と一致する"""
        expected = solution.R[1, :] @ solution.P
        np.testing.assert_allclose(ss_matrices.T[7, :5], expected, atol=1e-12)
        np.testing.assert_allclose(ss_matrices.T[7, 5:], 0.0, atol=1e-15)

    def test_n_current_row(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """n_t 行 (row 8) が R[5,:] @ P と一致する"""
        expected = solution.R[5, :] @ solution.P
        np.testing.assert_allclose(ss_matrices.T[8, :5], expected, atol=1e-12)
        np.testing.assert_allclose(ss_matrices.T[8, 5:], 0.0, atol=1e-15)

    def test_r_current_row(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """r_t 行 (row 9) が R[2,:] @ P と一致する"""
        expected = solution.R[2, :] @ solution.P
        np.testing.assert_allclose(ss_matrices.T[9, :5], expected, atol=1e-12)
        np.testing.assert_allclose(ss_matrices.T[9, 5:], 0.0, atol=1e-15)

    def test_y_lag_copies_y_current(self, ss_matrices: StateSpaceMatrices) -> None:
        """y_{t-1} 行 (row 10) は前期の y_t (col 5) のみ 1.0"""
        expected = np.zeros(N_AUG)
        expected[5] = 1.0
        np.testing.assert_allclose(ss_matrices.T[10, :], expected, atol=1e-15)

    def test_c_lag_copies_c_current(self, ss_matrices: StateSpaceMatrices) -> None:
        """c_{t-1} 行 (row 11) は前期の c_t (col 6) のみ 1.0"""
        expected = np.zeros(N_AUG)
        expected[6] = 1.0
        np.testing.assert_allclose(ss_matrices.T[11, :], expected, atol=1e-15)

    def test_i_lag_copies_state_i(self, ss_matrices: StateSpaceMatrices) -> None:
        """i_{t-1} 行 (row 12) は前期の s_t[3] (col 3) のみ 1.0"""
        expected = np.zeros(N_AUG)
        expected[3] = 1.0
        np.testing.assert_allclose(ss_matrices.T[12, :], expected, atol=1e-15)

    def test_w_lag_copies_state_w(self, ss_matrices: StateSpaceMatrices) -> None:
        """w_{t-1} 行 (row 13) は前期の s_t[4] (col 4) のみ 1.0"""
        expected = np.zeros(N_AUG)
        expected[4] = 1.0
        np.testing.assert_allclose(ss_matrices.T[13, :], expected, atol=1e-15)


class TestRAugStructure:
    """R_aug行列の構造を検証する"""

    def test_top_block_is_Q(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """R_aug の上部5行がQと一致する"""
        np.testing.assert_allclose(ss_matrices.R_aug[:5, :], solution.Q, atol=1e-12)

    def test_y_shock_response(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """y_t のショック応答が R[0,:] @ Q + S[0,:] と一致する"""
        expected = solution.R[0, :] @ solution.Q + solution.S[0, :]
        np.testing.assert_allclose(ss_matrices.R_aug[5, :], expected, atol=1e-12)

    def test_c_shock_response(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """c_t のショック応答が R[6,:] @ Q + S[6,:] と一致する"""
        expected = solution.R[6, :] @ solution.Q + solution.S[6, :]
        np.testing.assert_allclose(ss_matrices.R_aug[6, :], expected, atol=1e-12)

    def test_pi_shock_response(
        self, ss_matrices: StateSpaceMatrices, solution: NKSolutionResult
    ) -> None:
        """π_t のショック応答が R[1,:] @ Q + S[1,:] と一致する"""
        expected = solution.R[1, :] @ solution.Q + solution.S[1, :]
        np.testing.assert_allclose(ss_matrices.R_aug[7, :], expected, atol=1e-12)

    def test_lag_rows_zero(self, ss_matrices: StateSpaceMatrices) -> None:
        """ラグ変数のショック応答がゼロ"""
        np.testing.assert_allclose(ss_matrices.R_aug[10:, :], 0.0, atol=1e-15)


class TestZMatrixStructure:
    """Z行列（観測行列）の構造を検証する"""

    def test_output_growth_signs(self, ss_matrices: StateSpaceMatrices) -> None:
        """Δy_t: y_t (col 5) に +1、y_{t-1} (col 10) に -1"""
        Z = ss_matrices.Z
        assert Z[0, 5] == 1.0
        assert Z[0, 10] == -1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[[5, 10]] = False
        np.testing.assert_allclose(Z[0, mask], 0.0, atol=1e-15)

    def test_consumption_growth_signs(self, ss_matrices: StateSpaceMatrices) -> None:
        """Δc_t: c_t (col 6) に +1、c_{t-1} (col 11) に -1"""
        Z = ss_matrices.Z
        assert Z[1, 6] == 1.0
        assert Z[1, 11] == -1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[[6, 11]] = False
        np.testing.assert_allclose(Z[1, mask], 0.0, atol=1e-15)

    def test_investment_growth_signs(self, ss_matrices: StateSpaceMatrices) -> None:
        """Δi_t: s_t[3] (col 3) に +1、i_{t-1} (col 12) に -1"""
        Z = ss_matrices.Z
        assert Z[2, 3] == 1.0
        assert Z[2, 12] == -1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[[3, 12]] = False
        np.testing.assert_allclose(Z[2, mask], 0.0, atol=1e-15)

    def test_inflation_from_aux(self, ss_matrices: StateSpaceMatrices) -> None:
        """π_t は補助状態 α_t[7] から直接観測"""
        Z = ss_matrices.Z
        assert Z[3, 7] == 1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[7] = False
        np.testing.assert_allclose(Z[3, mask], 0.0, atol=1e-15)

    def test_wage_growth_signs(self, ss_matrices: StateSpaceMatrices) -> None:
        """Δw_t: s_t[4] (col 4) に +1、w_{t-1} (col 13) に -1"""
        Z = ss_matrices.Z
        assert Z[4, 4] == 1.0
        assert Z[4, 13] == -1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[[4, 13]] = False
        np.testing.assert_allclose(Z[4, mask], 0.0, atol=1e-15)

    def test_hours_from_aux(self, ss_matrices: StateSpaceMatrices) -> None:
        """n_t は補助状態 α_t[8] から直接観測"""
        Z = ss_matrices.Z
        assert Z[5, 8] == 1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[8] = False
        np.testing.assert_allclose(Z[5, mask], 0.0, atol=1e-15)

    def test_interest_rate_from_aux(self, ss_matrices: StateSpaceMatrices) -> None:
        """r_t は補助状態 α_t[9] から直接観測"""
        Z = ss_matrices.Z
        assert Z[6, 9] == 1.0
        mask = np.ones(N_AUG, dtype=bool)
        mask[9] = False
        np.testing.assert_allclose(Z[6, mask], 0.0, atol=1e-15)


class TestCovarianceMatrices:
    """共分散行列の検証"""

    def test_H_is_diagonal(self, ss_matrices: StateSpaceMatrices) -> None:
        off_diag = ss_matrices.H - np.diag(np.diag(ss_matrices.H))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-15)

    def test_H_positive_diagonal(self, ss_matrices: StateSpaceMatrices) -> None:
        assert np.all(np.diag(ss_matrices.H) > 0)

    def test_H_values_match_input(
        self, ss_matrices: StateSpaceMatrices, measurement_errors: np.ndarray
    ) -> None:
        np.testing.assert_allclose(np.diag(ss_matrices.H), measurement_errors**2, atol=1e-15)

    def test_Q_cov_is_diagonal(self, ss_matrices: StateSpaceMatrices) -> None:
        off_diag = ss_matrices.Q_cov - np.diag(np.diag(ss_matrices.Q_cov))
        np.testing.assert_allclose(off_diag, 0.0, atol=1e-15)

    def test_Q_cov_positive_diagonal(self, ss_matrices: StateSpaceMatrices) -> None:
        assert np.all(np.diag(ss_matrices.Q_cov) > 0)

    def test_Q_cov_values_match_input(
        self, ss_matrices: StateSpaceMatrices, shock_stds: np.ndarray
    ) -> None:
        np.testing.assert_allclose(np.diag(ss_matrices.Q_cov), shock_stds**2, atol=1e-15)


class TestNumericalProperties:
    """数値的性質の検証"""

    def test_matrices_are_finite(self, ss_matrices: StateSpaceMatrices) -> None:
        assert np.all(np.isfinite(ss_matrices.T))
        assert np.all(np.isfinite(ss_matrices.R_aug))
        assert np.all(np.isfinite(ss_matrices.Q_cov))
        assert np.all(np.isfinite(ss_matrices.Z))
        assert np.all(np.isfinite(ss_matrices.H))

    def test_T_eigenvalues_stable(self, ss_matrices: StateSpaceMatrices) -> None:
        """T行列の固有値が単位円内"""
        eigvals = np.linalg.eigvals(ss_matrices.T)
        max_abs_eigval = np.max(np.abs(eigvals))
        assert max_abs_eigval < 1.0 + 1e-8, (
            f"T行列の最大固有値絶対値が1を超えています: {max_abs_eigval}"
        )

    def test_P_block_well_conditioned(self, ss_matrices: StateSpaceMatrices) -> None:
        """状態遷移のP部分ブロックは条件数が適切"""
        P_block = ss_matrices.T[:5, :5]
        cond = np.linalg.cond(P_block)
        assert cond < 1e10, f"P部分ブロックの条件数が大きすぎます: {cond}"


class TestIRFConsistency:
    """NKモデルのIRFと状態空間シミュレーションの整合性を検証する"""

    @pytest.mark.parametrize(
        "shock_name,shock_idx",
        [
            ("e_g", 0),
            ("e_a", 1),
            ("e_m", 2),
        ],
    )
    def test_irf_matches_nk_model(
        self,
        default_model: NewKeynesianModel,
        ss_matrices: StateSpaceMatrices,
        shock_name: str,
        shock_idx: int,
    ) -> None:
        """状態空間シミュレーションがNKモデルのIRFと一致する"""
        periods = 20
        size = 0.01

        irf = default_model.impulse_response(shock_name, size=size, periods=periods)

        T = ss_matrices.T
        R_aug = ss_matrices.R_aug
        Z = ss_matrices.Z

        alpha = np.zeros(ss_matrices.n_aug)
        epsilon = np.zeros(ss_matrices.n_shock)
        epsilon[shock_idx] = size

        rho = default_model._shock_persistence(shock_name)

        obs_history = []
        for t in range(periods + 1):
            if t == 0:
                alpha = R_aug @ epsilon
            else:
                eps_t = np.zeros(ss_matrices.n_shock)
                if rho is not None:
                    eps_t[shock_idx] = size * (rho**t)
                alpha = T @ alpha + R_aug @ eps_t

            obs = Z @ alpha
            obs_history.append(obs)

        obs_array = np.array(obs_history)

        # π_t, n_t, r_t はレベル変数として直接比較
        for t in range(periods + 1):
            np.testing.assert_allclose(
                obs_array[t, 3],
                irf["pi"][t],
                atol=1e-8,
                err_msg=f"π不一致 at t={t} for {shock_name}",
            )
            np.testing.assert_allclose(
                obs_array[t, 5],
                irf["n"][t],
                atol=1e-8,
                err_msg=f"n不一致 at t={t} for {shock_name}",
            )
            np.testing.assert_allclose(
                obs_array[t, 6],
                irf["r"][t],
                atol=1e-8,
                err_msg=f"r不一致 at t={t} for {shock_name}",
            )

        # 成長率変数: Δy, Δc, Δi, Δw
        for t in range(periods + 1):
            y_prev = irf["y"][t - 1] if t > 0 else 0.0
            c_prev = irf["c"][t - 1] if t > 0 else 0.0
            i_prev = irf["i"][t - 1] if t > 0 else 0.0
            w_prev = irf["w"][t - 1] if t > 0 else 0.0

            np.testing.assert_allclose(
                obs_array[t, 0],
                irf["y"][t] - y_prev,
                atol=1e-8,
                err_msg=f"Δy不一致 at t={t} for {shock_name}",
            )
            np.testing.assert_allclose(
                obs_array[t, 1],
                irf["c"][t] - c_prev,
                atol=1e-8,
                err_msg=f"Δc不一致 at t={t} for {shock_name}",
            )
            np.testing.assert_allclose(
                obs_array[t, 2],
                irf["i"][t] - i_prev,
                atol=1e-8,
                err_msg=f"Δi不一致 at t={t} for {shock_name}",
            )
            np.testing.assert_allclose(
                obs_array[t, 4],
                irf["w"][t] - w_prev,
                atol=1e-8,
                err_msg=f"Δw不一致 at t={t} for {shock_name}",
            )


class TestInputValidation:
    """入力バリデーションのテスト"""

    def test_wrong_shock_stds_size(
        self, solution: NKSolutionResult, measurement_errors: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="shock_stds"):
            StateSpaceBuilder.build(solution, np.ones(3), measurement_errors)

    def test_wrong_measurement_errors_size(
        self, solution: NKSolutionResult, shock_stds: np.ndarray
    ) -> None:
        with pytest.raises(ValueError, match="measurement_errors"):
            StateSpaceBuilder.build(solution, shock_stds, np.ones(3))
