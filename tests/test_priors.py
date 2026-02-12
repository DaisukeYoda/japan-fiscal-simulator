"""事前分布のテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.estimation.priors import (
    DistributionType,
    ParameterPrior,
    PriorConfig,
)


class TestParameterPrior:
    """ParameterPriorのテスト"""

    def test_beta_log_pdf_finite_at_mean(self) -> None:
        """Beta分布が平均値で有限のlog_pdfを返す"""
        prior = ParameterPrior(
            name="habit",
            dist_type=DistributionType.BETA,
            mean=0.7,
            std=0.1,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        lp = prior.log_pdf(0.7)
        assert np.isfinite(lp)

    def test_gamma_log_pdf_finite_at_mean(self) -> None:
        """Gamma分布が平均値で有限のlog_pdfを返す"""
        prior = ParameterPrior(
            name="sigma",
            dist_type=DistributionType.GAMMA,
            mean=1.5,
            std=0.37,
            lower_bound=0.0,
        )
        lp = prior.log_pdf(1.5)
        assert np.isfinite(lp)

    def test_normal_log_pdf_finite_at_mean(self) -> None:
        """Normal分布が平均値で有限のlog_pdfを返す"""
        prior = ParameterPrior(
            name="phi_pi",
            dist_type=DistributionType.NORMAL,
            mean=1.5,
            std=0.25,
            lower_bound=-np.inf,
            upper_bound=np.inf,
        )
        lp = prior.log_pdf(1.5)
        assert np.isfinite(lp)

    def test_inv_gamma_log_pdf_finite_at_mean(self) -> None:
        """逆Gamma分布が平均値で有限のlog_pdfを返す"""
        prior = ParameterPrior(
            name="sigma_a",
            dist_type=DistributionType.INV_GAMMA,
            mean=0.01,
            std=0.01,
            lower_bound=0.0,
        )
        lp = prior.log_pdf(0.01)
        assert np.isfinite(lp)

    def test_beta_log_pdf_negative_inf_outside_bounds(self) -> None:
        """Beta分布が範囲外で-infを返す"""
        prior = ParameterPrior(
            name="habit",
            dist_type=DistributionType.BETA,
            mean=0.7,
            std=0.1,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        assert prior.log_pdf(-0.1) == -np.inf
        assert prior.log_pdf(1.1) == -np.inf
        assert prior.log_pdf(0.0) == -np.inf
        assert prior.log_pdf(1.0) == -np.inf

    def test_gamma_log_pdf_negative_inf_at_zero(self) -> None:
        """Gamma分布が0以下で-infを返す"""
        prior = ParameterPrior(
            name="sigma",
            dist_type=DistributionType.GAMMA,
            mean=1.5,
            std=0.37,
            lower_bound=0.0,
        )
        assert prior.log_pdf(0.0) == -np.inf
        assert prior.log_pdf(-1.0) == -np.inf

    def test_inv_gamma_log_pdf_negative_inf_at_zero(self) -> None:
        """逆Gamma分布が0以下で-infを返す"""
        prior = ParameterPrior(
            name="sigma_a",
            dist_type=DistributionType.INV_GAMMA,
            mean=0.01,
            std=0.01,
            lower_bound=0.0,
        )
        assert prior.log_pdf(0.0) == -np.inf
        assert prior.log_pdf(-0.01) == -np.inf

    def test_beta_sample_within_bounds(self) -> None:
        """Beta分布のサンプルが範囲内"""
        prior = ParameterPrior(
            name="habit",
            dist_type=DistributionType.BETA,
            mean=0.7,
            std=0.1,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, size=1000)
        assert np.all(samples > 0.0)
        assert np.all(samples < 1.0)

    def test_gamma_sample_within_bounds(self) -> None:
        """Gamma分布のサンプルが正の値"""
        prior = ParameterPrior(
            name="sigma",
            dist_type=DistributionType.GAMMA,
            mean=1.5,
            std=0.37,
            lower_bound=0.0,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, size=1000)
        assert np.all(samples > 0.0)

    def test_normal_sample_returns_values(self) -> None:
        """Normal分布がサンプルを返す"""
        prior = ParameterPrior(
            name="phi_pi",
            dist_type=DistributionType.NORMAL,
            mean=1.5,
            std=0.25,
            lower_bound=-np.inf,
            upper_bound=np.inf,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, size=100)
        assert len(samples) == 100
        assert np.all(np.isfinite(samples))

    def test_inv_gamma_sample_within_bounds(self) -> None:
        """逆Gamma分布のサンプルが正の値"""
        prior = ParameterPrior(
            name="sigma_a",
            dist_type=DistributionType.INV_GAMMA,
            mean=0.01,
            std=0.01,
            lower_bound=0.0,
        )
        rng = np.random.default_rng(42)
        samples = prior.sample(rng, size=1000)
        assert np.all(samples > 0.0)

    def test_beta_shape_parameters_correct(self) -> None:
        """Beta分布の形状パラメータが正しく変換される"""
        prior = ParameterPrior(
            name="test",
            dist_type=DistributionType.BETA,
            mean=0.5,
            std=0.1,
            lower_bound=0.0,
            upper_bound=1.0,
        )
        dist = prior._get_scipy_dist()
        # Beta(a, b)のmean = a/(a+b)
        a, b = dist.args
        expected_mean = a / (a + b)
        assert abs(expected_mean - 0.5) < 1e-10

    def test_gamma_shape_parameters_correct(self) -> None:
        """Gamma分布の形状パラメータが正しく変換される"""
        prior = ParameterPrior(
            name="test",
            dist_type=DistributionType.GAMMA,
            mean=2.0,
            std=0.5,
            lower_bound=0.0,
        )
        dist = prior._get_scipy_dist()
        # Gamma(a, scale)のmean = a * scale
        a = dist.args[0]
        scale = dist.kwds["scale"]
        expected_mean = a * scale
        assert abs(expected_mean - 2.0) < 1e-10


class TestPriorConfig:
    """PriorConfigのテスト"""

    def test_smets_wouters_japan_parameter_count(self) -> None:
        """SW日本版事前分布のパラメータ数が正しい"""
        config = PriorConfig.smets_wouters_japan()
        # 構造8 + 金融政策2 + rho_r 1 + ショック持続性3 + ショックσ6 + 観測誤差7 = 27
        assert config.n_params == 27

    def test_smets_wouters_japan_names(self) -> None:
        """SW日本版事前分布のパラメータ名が正しい"""
        config = PriorConfig.smets_wouters_japan()
        names = config.names
        # 構造パラメータ
        assert "sigma" in names
        assert "phi" in names
        assert "habit" in names
        assert "theta" in names
        assert "theta_w" in names
        assert "psi" in names
        assert "iota_w" in names
        assert "S_double_prime" in names
        # 金融政策
        assert "phi_pi" in names
        assert "phi_y" in names
        assert "rho_r" in names
        # ショック持続性
        assert "rho_a" in names
        assert "rho_g" in names
        assert "rho_p" in names
        # ショック標準偏差
        assert "sigma_a" in names
        assert "sigma_m" in names
        # 観測誤差
        assert "me_y" in names
        assert "me_r" in names

    def test_log_prior_at_means_is_finite(self) -> None:
        """事前分布の平均値で対数事前確率が有限"""
        config = PriorConfig.smets_wouters_japan()
        theta = config.means()
        lp = config.log_prior(theta)
        assert np.isfinite(lp)

    def test_log_prior_sums_correctly(self) -> None:
        """log_priorが各log_pdfの合計と一致する"""
        config = PriorConfig.smets_wouters_japan()
        theta = config.means()
        expected = sum(p.log_pdf(theta[i]) for i, p in enumerate(config.priors))
        actual = config.log_prior(theta)
        assert abs(actual - expected) < 1e-10

    def test_log_prior_returns_neg_inf_for_invalid(self) -> None:
        """範囲外のパラメータで-infを返す"""
        config = PriorConfig.smets_wouters_japan()
        theta = config.means()
        # habitを範囲外に設定
        habit_idx = config.names.index("habit")
        theta_bad = theta.copy()
        theta_bad[habit_idx] = -1.0
        assert config.log_prior(theta_bad) == -np.inf

    def test_sample_returns_correct_shape(self) -> None:
        """sampleが正しい形状のベクトルを返す"""
        config = PriorConfig.smets_wouters_japan()
        rng = np.random.default_rng(42)
        sample = config.sample(rng)
        assert sample.shape == (config.n_params,)

    def test_sample_produces_finite_log_prior(self) -> None:
        """サンプルに対してlog_priorが有限値を返す"""
        config = PriorConfig.smets_wouters_japan()
        rng = np.random.default_rng(42)
        for _ in range(10):
            sample = config.sample(rng)
            lp = config.log_prior(sample)
            assert np.isfinite(lp)

    def test_get_prior_by_name(self) -> None:
        """名前でパラメータ事前分布を取得できる"""
        config = PriorConfig.smets_wouters_japan()
        sigma_prior = config.get_prior("sigma")
        assert sigma_prior.name == "sigma"
        assert sigma_prior.dist_type == DistributionType.GAMMA
        assert sigma_prior.mean == 1.5

    def test_get_prior_unknown_name_raises(self) -> None:
        """存在しないパラメータ名でKeyErrorが発生する"""
        config = PriorConfig.smets_wouters_japan()
        with pytest.raises(KeyError):
            config.get_prior("nonexistent")

    def test_no_duplicate_names(self) -> None:
        """パラメータ名に重複がないことを確認"""
        config = PriorConfig.smets_wouters_japan()
        names = config.names
        assert len(names) == len(set(names))

    def test_means_returns_correct_shape(self) -> None:
        """meansが正しい形状のベクトルを返す"""
        config = PriorConfig.smets_wouters_japan()
        m = config.means()
        assert m.shape == (config.n_params,)
        assert np.all(np.isfinite(m))
