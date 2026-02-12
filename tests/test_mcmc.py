"""MCMCサンプラーのテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.exceptions import EstimationError
from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator
from japan_fiscal_simulator.estimation.mcmc import (
    MCMCConfig,
    MCMCResult,
    MetropolisHastings,
    make_log_posterior,
)
from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping
from japan_fiscal_simulator.estimation.priors import PriorConfig
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


def _make_gaussian_log_posterior(mean: np.ndarray, cov: np.ndarray) -> callable:
    """テスト用のガウス対数事後関数を生成"""
    inv_cov = np.linalg.inv(cov)
    log_norm = -0.5 * len(mean) * np.log(2 * np.pi) - 0.5 * np.linalg.slogdet(cov)[1]

    def log_posterior(theta: np.ndarray) -> float:
        diff = theta - mean
        return log_norm - 0.5 * float(diff @ inv_cov @ diff)

    return log_posterior


class TestMCMCConfig:
    """MCMCConfig のテスト"""

    def test_default_values(self) -> None:
        """デフォルト値が正しい"""
        cfg = MCMCConfig()
        assert cfg.n_chains == 4
        assert cfg.n_draws == 100_000
        assert cfg.n_burnin == 50_000
        assert cfg.thinning == 10
        assert cfg.target_acceptance == pytest.approx(0.234)

    def test_custom_values(self) -> None:
        """カスタム値が設定可能"""
        cfg = MCMCConfig(n_chains=2, n_draws=500, n_burnin=200, thinning=5)
        assert cfg.n_chains == 2
        assert cfg.n_draws == 500


class TestMetropolisHastingsGaussian:
    """ガウスターゲットでのMHサンプラーテスト"""

    def test_find_mode_converges(self) -> None:
        """find_modeがガウス分布のモードに収束する"""
        n = 3
        true_mean = np.array([1.0, 2.0, 3.0])
        cov = np.eye(n) * 0.5
        log_post = _make_gaussian_log_posterior(true_mean, cov)

        mh = MetropolisHastings(
            log_posterior_fn=log_post,
            n_params=n,
            config=MCMCConfig(mode_search_max_iter=200),
        )

        theta0 = np.zeros(n)
        mode, hessian_inv = mh.find_mode(theta0)

        np.testing.assert_allclose(mode, true_mean, atol=0.1)
        assert hessian_inv.shape == (n, n)

    def test_find_mode_hessian_inverse_shape(self) -> None:
        """ヘシアン逆行列の形状が正しい"""
        n = 2
        log_post = _make_gaussian_log_posterior(np.zeros(n), np.eye(n))
        mh = MetropolisHastings(log_posterior_fn=log_post, n_params=n)
        _, hessian_inv = mh.find_mode(np.ones(n))
        assert hessian_inv.shape == (n, n)

    def test_short_chain_produces_finite_samples(self) -> None:
        """短いチェーンで有限なサンプルが得られる"""
        n = 2
        true_mean = np.array([1.0, 2.0])
        cov = np.eye(n) * 0.5
        log_post = _make_gaussian_log_posterior(true_mean, cov)

        cfg = MCMCConfig(n_chains=2, n_draws=200, n_burnin=100, thinning=1)
        mh = MetropolisHastings(
            log_posterior_fn=log_post,
            n_params=n,
            config=cfg,
            parameter_names=["x", "y"],
        )

        result = mh.run(theta0=np.zeros(n))

        assert isinstance(result, MCMCResult)
        assert np.all(np.isfinite(result.chains))
        assert np.all(np.isfinite(result.log_posteriors))

    def test_chain_dimensions(self) -> None:
        """出力配列の次元が正しい"""
        n = 3
        log_post = _make_gaussian_log_posterior(np.zeros(n), np.eye(n))

        cfg = MCMCConfig(n_chains=2, n_draws=300, n_burnin=100, thinning=2)
        mh = MetropolisHastings(log_posterior_fn=log_post, n_params=n, config=cfg)

        result = mh.run(theta0=np.zeros(n))

        n_kept = (cfg.n_draws - cfg.n_burnin) // cfg.thinning
        assert result.chains.shape == (cfg.n_chains, n_kept, n)
        assert result.log_posteriors.shape == (cfg.n_chains, n_kept)
        assert result.acceptance_rates.shape == (cfg.n_chains,)
        assert result.mode.shape == (n,)
        assert result.mode_hessian.shape == (n, n)

    def test_acceptance_rate_reasonable(self) -> None:
        """受容率が妥当な範囲内"""
        n = 2
        log_post = _make_gaussian_log_posterior(np.zeros(n), np.eye(n))

        cfg = MCMCConfig(n_chains=2, n_draws=500, n_burnin=200, thinning=1)
        mh = MetropolisHastings(log_posterior_fn=log_post, n_params=n, config=cfg)

        result = mh.run(theta0=np.zeros(n))

        # 受容率が0〜1の範囲
        assert np.all(result.acceptance_rates >= 0.0)
        assert np.all(result.acceptance_rates <= 1.0)
        # 合理的な範囲（完全拒否・完全受容でない）
        assert np.all(result.acceptance_rates > 0.01)
        assert np.all(result.acceptance_rates < 0.99)

    def test_parameter_names_stored(self) -> None:
        """パラメータ名が結果に保存される"""
        n = 2
        log_post = _make_gaussian_log_posterior(np.zeros(n), np.eye(n))
        names = ["alpha", "beta"]

        cfg = MCMCConfig(n_chains=1, n_draws=100, n_burnin=50, thinning=1)
        mh = MetropolisHastings(
            log_posterior_fn=log_post,
            n_params=n,
            config=cfg,
            parameter_names=names,
        )

        result = mh.run(theta0=np.zeros(n))
        assert result.parameter_names == names

    def test_bounds_enforcement(self) -> None:
        """境界制約が適用される"""
        n = 2
        log_post = _make_gaussian_log_posterior(np.array([0.5, 0.5]), 0.1 * np.eye(n))

        bounds = [(0.0, 1.0), (0.0, 1.0)]
        cfg = MCMCConfig(n_chains=1, n_draws=200, n_burnin=100, thinning=1)
        mh = MetropolisHastings(
            log_posterior_fn=log_post,
            n_params=n,
            config=cfg,
            bounds=bounds,
        )

        result = mh.run(theta0=np.array([0.5, 0.5]))

        # 全サンプルが境界内
        assert np.all(result.chains >= 0.0)
        assert np.all(result.chains <= 1.0)


class TestMakeLogPosterior:
    """make_log_posterior のテスト"""

    @pytest.fixture
    def synthetic_data(self) -> np.ndarray:
        """合成データを生成"""
        gen = SyntheticDataGenerator()
        params = DefaultParameters()
        data = gen.generate(params, n_periods=100, rng=np.random.default_rng(42))
        return data.data

    @pytest.fixture
    def mapping(self) -> ParameterMapping:
        return ParameterMapping()

    @pytest.fixture
    def prior_config(self) -> PriorConfig:
        return PriorConfig.smets_wouters_japan()

    def test_log_posterior_at_default_is_finite(
        self,
        mapping: ParameterMapping,
        prior_config: PriorConfig,
        synthetic_data: np.ndarray,
    ) -> None:
        """デフォルトパラメータでlog_posteriorが有限"""
        log_post = make_log_posterior(mapping, prior_config, synthetic_data)
        theta = mapping.defaults()
        lp = log_post(theta)
        assert np.isfinite(lp)

    def test_invalid_theta_returns_neg_inf(
        self,
        mapping: ParameterMapping,
        prior_config: PriorConfig,
        synthetic_data: np.ndarray,
    ) -> None:
        """範囲外のθで-infが返る"""
        log_post = make_log_posterior(mapping, prior_config, synthetic_data)
        # 全て0にすると事前分布で範囲外になるはず
        theta_bad = np.zeros(mapping.n_params)
        lp = log_post(theta_bad)
        assert lp == -np.inf

    def test_log_posterior_type(
        self,
        mapping: ParameterMapping,
        prior_config: PriorConfig,
        synthetic_data: np.ndarray,
    ) -> None:
        """log_posteriorの戻り値がfloat"""
        log_post = make_log_posterior(mapping, prior_config, synthetic_data)
        theta = mapping.defaults()
        lp = log_post(theta)
        assert isinstance(lp, float)


class TestMCMCWithDSGE:
    """DSGEモデルを使ったMCMC統合テスト（軽量）"""

    def test_short_run_produces_result(self) -> None:
        """短い実行で有効な結果が得られる"""
        mapping = ParameterMapping()
        prior_config = PriorConfig.smets_wouters_japan()

        gen = SyntheticDataGenerator()
        params = DefaultParameters()
        data = gen.generate(params, n_periods=50, rng=np.random.default_rng(42))

        log_post = make_log_posterior(mapping, prior_config, data.data)
        theta0 = mapping.defaults()

        cfg = MCMCConfig(
            n_chains=1,
            n_draws=50,
            n_burnin=20,
            thinning=1,
            mode_search_max_iter=10,
        )
        mh = MetropolisHastings(
            log_posterior_fn=log_post,
            n_params=mapping.n_params,
            config=cfg,
            parameter_names=mapping.names,
            bounds=mapping.bounds(),
        )

        result = mh.run(theta0=theta0)

        assert isinstance(result, MCMCResult)
        n_kept = (cfg.n_draws - cfg.n_burnin) // cfg.thinning
        assert result.chains.shape == (1, n_kept, mapping.n_params)
        assert np.isfinite(result.mode_log_posterior)


class TestMCMCValidation:
    """MCMC設定バリデーションのテスト"""

    def test_burnin_exceeds_draws_raises(self) -> None:
        """n_burnin >= n_draws でEstimationError"""
        n = 2
        log_post = _make_gaussian_log_posterior(np.zeros(n), np.eye(n))
        cfg = MCMCConfig(n_chains=1, n_draws=100, n_burnin=100, thinning=1)
        mh = MetropolisHastings(log_posterior_fn=log_post, n_params=n, config=cfg)
        with pytest.raises(EstimationError, match="n_burnin"):
            mh.run(theta0=np.zeros(n))

    def test_thinning_too_large_raises(self) -> None:
        """thinningが大きすぎてn_kept=0になる場合にEstimationError"""
        n = 2
        log_post = _make_gaussian_log_posterior(np.zeros(n), np.eye(n))
        cfg = MCMCConfig(n_chains=1, n_draws=100, n_burnin=50, thinning=100)
        mh = MetropolisHastings(log_posterior_fn=log_post, n_params=n, config=cfg)
        with pytest.raises(EstimationError, match="n_kept=0"):
            mh.run(theta0=np.zeros(n))
