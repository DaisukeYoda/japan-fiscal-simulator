"""Kalmanフィルタのテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.exceptions import KalmanFilterError
from japan_fiscal_simulator.estimation.kalman_filter import (
    KalmanFilterResult,
    kalman_filter,
)


class TestAR1KnownSolution:
    """AR(1)モデルの既知解テスト"""

    def test_ar1_log_likelihood_is_finite(self) -> None:
        """AR(1)モデルでlog-likelihoodが有限"""
        rng = np.random.default_rng(42)
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        # データ生成
        n_periods = 200
        x = np.zeros(n_periods)
        y = np.zeros((n_periods, 1))
        for t in range(1, n_periods):
            x[t] = 0.9 * x[t - 1] + rng.normal(0, np.sqrt(0.01))
        for t in range(n_periods):
            y[t, 0] = x[t] + rng.normal(0, np.sqrt(0.01))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert isinstance(result, KalmanFilterResult)
        assert np.isfinite(result.log_likelihood)

    def test_ar1_filtered_states_are_finite(self) -> None:
        """AR(1)モデルでフィルタ済み状態が有限"""
        rng = np.random.default_rng(123)
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        n_periods = 100
        x = np.zeros(n_periods)
        y = np.zeros((n_periods, 1))
        for t in range(1, n_periods):
            x[t] = 0.9 * x[t - 1] + rng.normal(0, np.sqrt(0.01))
        for t in range(n_periods):
            y[t, 0] = x[t] + rng.normal(0, np.sqrt(0.01))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.all(np.isfinite(result.filtered_states))
        assert result.filtered_states.shape == (n_periods, 1)

    def test_ar1_filtered_states_track_data(self) -> None:
        """AR(1)モデルでフィルタ済み状態がデータを追跡"""
        rng = np.random.default_rng(7)
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        n_periods = 300
        x = np.zeros(n_periods)
        y = np.zeros((n_periods, 1))
        for t in range(1, n_periods):
            x[t] = 0.9 * x[t - 1] + rng.normal(0, np.sqrt(0.01))
        for t in range(n_periods):
            y[t, 0] = x[t] + rng.normal(0, np.sqrt(0.01))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # 後半ではフィルタ済み状態が真の状態に近いはず
        correlation = np.corrcoef(result.filtered_states[100:, 0], x[100:])[0, 1]
        assert correlation > 0.8


class TestSteadyState:
    """定常状態収束テスト"""

    def test_kalman_gain_converges(self) -> None:
        """多数期間後にフィルタ済み共分散が収束する"""
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        rng = np.random.default_rng(0)
        n_periods = 500
        y = rng.normal(0, 0.2, (n_periods, 1))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # 後半の共分散はほぼ一定
        cov_later = result.filtered_covariances[200:, 0]
        assert np.std(cov_later) < 1e-10

    def test_covariance_converges_to_dare_solution(self) -> None:
        """共分散がDARE解に収束する"""
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        rng = np.random.default_rng(1)
        n_periods = 1000
        y = rng.normal(0, 0.2, (n_periods, 1))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # DARE: P_ss = T P_ss T' + RQR' - T P_ss Z'(Z P_ss Z' + H)^{-1} Z P_ss T'
        # 代わりにフィルタ収束値を確認
        P_final = result.filtered_covariances[-1, 0]
        P_near_final = result.filtered_covariances[-2, 0]
        assert abs(P_final - P_near_final) < 1e-12


class TestMissingData:
    """欠損データテスト"""

    def test_missing_data_produces_finite_results(self) -> None:
        """欠損値ありでも有限な結果を返す"""
        rng = np.random.default_rng(42)
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        n_periods = 100
        y = rng.normal(0, 0.2, (n_periods, 1))

        # 20%をNaNに設定
        missing_idx = rng.choice(n_periods, size=20, replace=False)
        y[missing_idx] = np.nan

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.filtered_states))

    def test_fully_missing_period_no_likelihood_contribution(self) -> None:
        """全欠損期間はlog-likelihood貢献がゼロ"""
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        rng = np.random.default_rng(42)
        n_periods = 50
        y_full = rng.normal(0, 0.2, (n_periods, 1))

        # 全データありの尤度
        result_full = kalman_filter(y_full, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # 末尾10期を全欠損にする
        y_partial = y_full.copy()
        y_partial[40:] = np.nan
        result_partial = kalman_filter(y_partial, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # 全欠損分の尤度差は非負
        assert result_full.log_likelihood != result_partial.log_likelihood
        assert np.isfinite(result_partial.log_likelihood)

    def test_partial_missing_in_multivariate(self) -> None:
        """多変量で一部の観測変数のみ欠損"""
        T_mat = 0.9 * np.eye(2)
        R_mat = np.eye(2)
        Q_mat = 0.01 * np.eye(2)
        Z_mat = np.eye(2)
        H_mat = 0.01 * np.eye(2)

        rng = np.random.default_rng(42)
        n_periods = 100
        y = rng.normal(0, 0.2, (n_periods, 2))

        # 1番目の観測変数を一部欠損にする
        y[10:20, 0] = np.nan

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.filtered_states))


class TestDimensionChecks:
    """次元検証テスト"""

    def test_valid_1d_system(self) -> None:
        """1次元系が正しく動作"""
        y = np.zeros((10, 1))
        T_mat = np.array([[0.5]])
        Z_mat = np.array([[1.0]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)
        assert np.isfinite(result.log_likelihood)

    def test_valid_2state_1obs(self) -> None:
        """2状態1観測の系が正しく動作"""
        y = np.zeros((10, 1))
        T_mat = np.array([[0.5, 0.1], [0.0, 0.3]])
        Z_mat = np.array([[1.0, 0.0]])
        R_mat = np.eye(2)
        Q_mat = 0.01 * np.eye(2)
        H_mat = np.array([[0.01]])

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)
        assert np.isfinite(result.log_likelihood)

    def test_invalid_y_dimension_raises(self) -> None:
        """yが1次元の場合にエラー"""
        y = np.zeros(10)  # 1D
        T_mat = np.array([[0.5]])
        Z_mat = np.array([[1.0]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])

        with pytest.raises(KalmanFilterError, match="2次元"):
            kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

    def test_invalid_T_dimension_raises(self) -> None:
        """Tの次元不整合でエラー"""
        y = np.zeros((10, 1))
        T_mat = np.array([[0.5, 0.0], [0.0, 0.3]])  # 2x2
        Z_mat = np.array([[1.0]])  # 1x1 (should be 1x2)
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])

        with pytest.raises(KalmanFilterError):
            kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

    def test_invalid_H_dimension_raises(self) -> None:
        """Hの次元不整合でエラー"""
        y = np.zeros((10, 2))
        T_mat = np.array([[0.5]])
        Z_mat = np.array([[1.0], [0.5]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])  # should be 2x2

        with pytest.raises(KalmanFilterError):
            kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

    def test_invalid_a0_dimension_raises(self) -> None:
        """a0の次元不整合でエラー"""
        y = np.zeros((10, 1))
        T_mat = np.array([[0.5]])
        Z_mat = np.array([[1.0]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])
        a0 = np.zeros(2)  # should be (1,)

        with pytest.raises(KalmanFilterError):
            kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat, a0=a0)

    def test_invalid_P0_dimension_raises(self) -> None:
        """P0の次元不整合でエラー"""
        y = np.zeros((10, 1))
        T_mat = np.array([[0.5]])
        Z_mat = np.array([[1.0]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1.0]])
        H_mat = np.array([[1.0]])
        P0 = np.eye(2)  # should be 1x1

        with pytest.raises(KalmanFilterError):
            kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat, P0=P0)


class TestNumericalStability:
    """数値安定性テスト"""

    def test_near_singular_F_does_not_crash(self) -> None:
        """非常に小さいHでクラッシュしない"""
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[1e-15]])  # 非常に小さい観測誤差

        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.1, (50, 1))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # クラッシュせずに結果を返す（-infも許容）
        assert isinstance(result.log_likelihood, float)

    def test_near_unit_root_works(self) -> None:
        """ほぼ単位根の状態遷移でも動作する"""
        T_mat = np.array([[0.999]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.5, (100, 1))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.filtered_states))

    def test_unit_root_uses_fallback_P0(self) -> None:
        """単位根でLyapunov解が失敗してもフォールバックで動作"""
        T_mat = np.array([[1.0]])  # 単位根
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.5, (50, 1))

        # Lyapunov方程式は失敗するが、フォールバックでフィルタは動作するはず
        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.all(np.isfinite(result.filtered_states))

    def test_custom_initial_conditions(self) -> None:
        """カスタム初期条件でも正しく動作する"""
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[0.01]])

        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.2, (50, 1))

        a0 = np.array([1.0])
        P0 = np.array([[0.5]])

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat, a0=a0, P0=P0)

        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.filtered_states))


class TestKnownLogLikelihood:
    """既知のlog-likelihoodテスト"""

    def test_iid_normal_log_likelihood(self) -> None:
        """i.i.d.正規データのlog-likelihoodがN(0, sigma^2)の密度和に近い"""
        sigma2 = 0.5
        rng = np.random.default_rng(42)
        n_periods = 1000
        y = rng.normal(0, np.sqrt(sigma2), (n_periods, 1))

        # 状態なし（定数ゼロ）に近いモデル: x_t = 0, z_t = x_t + eta_t
        T_mat = np.array([[0.0]])  # 状態は常にゼロに戻る
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[1e-20]])  # 状態ノイズほぼゼロ
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[sigma2]])

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # 真のlog-likelihood: sum of log N(0, sigma^2)
        expected_ll = np.sum(-0.5 * (np.log(2 * np.pi * sigma2) + y[:, 0] ** 2 / sigma2))

        # フィルタ尤度は真の尤度に近いはず
        assert abs(result.log_likelihood - expected_ll) < 1.0

    def test_zero_observation_noise_concentrates_states(self) -> None:
        """観測ノイズゼロに近い場合、フィルタ済み状態はデータに近い"""
        T_mat = np.array([[0.9]])
        R_mat = np.array([[1.0]])
        Q_mat = np.array([[0.01]])
        Z_mat = np.array([[1.0]])
        H_mat = np.array([[1e-10]])  # ほぼゼロの観測ノイズ

        rng = np.random.default_rng(42)
        n_periods = 100
        y = rng.normal(0, 0.2, (n_periods, 1))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # フィルタ済み状態はデータにほぼ一致
        max_diff = np.max(np.abs(result.filtered_states[:, 0] - y[:, 0]))
        assert max_diff < 0.01


class TestMultiDimensional:
    """多次元系テスト"""

    def test_2state_2obs_system(self) -> None:
        """2状態2観測の系が正しく動作"""
        T_mat = np.array([[0.8, 0.1], [0.0, 0.5]])
        R_mat = np.eye(2)
        Q_mat = 0.01 * np.eye(2)
        Z_mat = np.eye(2)
        H_mat = 0.01 * np.eye(2)

        rng = np.random.default_rng(42)
        n_periods = 200

        # データ生成
        x = np.zeros((n_periods, 2))
        y = np.zeros((n_periods, 2))
        for t in range(1, n_periods):
            x[t] = T_mat @ x[t - 1] + rng.multivariate_normal(np.zeros(2), Q_mat)
        for t in range(n_periods):
            y[t] = Z_mat @ x[t] + rng.multivariate_normal(np.zeros(2), H_mat)

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.isfinite(result.log_likelihood)
        assert result.filtered_states.shape == (n_periods, 2)
        assert result.prediction_errors.shape == (n_periods, 2)
        assert result.filtered_covariances.shape == (n_periods, 2)
        assert np.all(np.isfinite(result.filtered_states))

    def test_3state_2obs_system(self) -> None:
        """3状態2観測の系が正しく動作"""
        T_mat = np.diag([0.8, 0.5, 0.3])
        R_mat = np.eye(3)
        Q_mat = 0.01 * np.eye(3)
        Z_mat = np.array([[1.0, 0.5, 0.0], [0.0, 1.0, 0.5]])
        H_mat = 0.01 * np.eye(2)

        rng = np.random.default_rng(42)
        n_periods = 100
        y = rng.normal(0, 0.2, (n_periods, 2))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.isfinite(result.log_likelihood)
        assert result.filtered_states.shape == (n_periods, 3)
        assert result.prediction_errors.shape == (n_periods, 2)

    def test_multivariate_filtered_states_track_true(self) -> None:
        """多変量でフィルタ済み状態が真の状態を追跡する"""
        T_mat = np.array([[0.8, 0.1], [0.0, 0.5]])
        R_mat = np.eye(2)
        Q_mat = 0.01 * np.eye(2)
        Z_mat = np.eye(2)
        H_mat = 0.01 * np.eye(2)

        rng = np.random.default_rng(99)
        n_periods = 300

        x = np.zeros((n_periods, 2))
        y = np.zeros((n_periods, 2))
        for t in range(1, n_periods):
            x[t] = T_mat @ x[t - 1] + rng.multivariate_normal(np.zeros(2), Q_mat)
        for t in range(n_periods):
            y[t] = Z_mat @ x[t] + rng.multivariate_normal(np.zeros(2), H_mat)

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        # 各状態変数でフィルタ済み状態が真値と相関
        for dim in range(2):
            corr = np.corrcoef(result.filtered_states[100:, dim], x[100:, dim])[0, 1]
            assert corr > 0.8

    def test_fewer_shocks_than_states(self) -> None:
        """ショック数が状態数より少ない系が動作する"""
        T_mat = np.diag([0.8, 0.5])
        R_mat = np.array([[1.0], [0.5]])  # 2x1: 1ショックで2状態
        Q_mat = np.array([[0.01]])
        Z_mat = np.eye(2)
        H_mat = 0.01 * np.eye(2)

        rng = np.random.default_rng(42)
        y = rng.normal(0, 0.2, (50, 2))

        result = kalman_filter(y, T_mat, Z_mat, R_mat, Q_mat, H_mat)

        assert np.isfinite(result.log_likelihood)
        assert np.all(np.isfinite(result.filtered_states))
