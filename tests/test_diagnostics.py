"""収束診断と推定結果のテスト"""

import numpy as np

from japan_fiscal_simulator.estimation.diagnostics import (
    ConvergenceDiagnostics,
    compute_ess,
    compute_rhat,
    geweke_test,
    run_diagnostics,
)
from japan_fiscal_simulator.estimation.results import (
    EstimationResult,
    PosteriorSummary,
    compute_hpd,
    compute_marginal_likelihood_laplace,
)


class TestRhat:
    """Gelman-Rubin R-hat統計量のテスト"""

    def test_identical_chains_rhat_near_one(self) -> None:
        """同一分布からのチェーンは R-hat ≈ 1.0"""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (4, 1000, 5))

        r_hat = compute_rhat(chains)

        assert r_hat.shape == (5,)
        np.testing.assert_allclose(r_hat, 1.0, atol=0.05)

    def test_divergent_chains_rhat_above_threshold(self) -> None:
        """異なる平均を持つチェーンは R-hat > 1.1"""
        rng = np.random.default_rng(42)
        # 各チェーンの平均が 0, 1, 2, 3 と大きく異なる
        div_chains = np.stack([rng.normal(i * 3.0, 1, (1000, 5)) for i in range(4)])

        r_hat = compute_rhat(div_chains)

        assert np.all(r_hat > 1.1)


class TestESS:
    """有効サンプルサイズのテスト"""

    def test_ess_less_than_total(self) -> None:
        """ESSは常に総サンプル数以下"""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (4, 1000, 5))
        n_total = 4 * 1000

        ess = compute_ess(chains)

        assert ess.shape == (5,)
        assert np.all(ess <= n_total + 1)  # 数値誤差許容
        assert np.all(ess > 0)

    def test_ess_for_iid_samples(self) -> None:
        """独立サンプルの場合 ESS ≈ n_total"""
        rng = np.random.default_rng(42)
        chains = rng.normal(0, 1, (4, 2000, 3))
        n_total = 4 * 2000

        ess = compute_ess(chains)

        # i.i.d.なので ESS は n_total に近い（±50%以内）
        assert np.all(ess > n_total * 0.5)


class TestGeweke:
    """Geweke収束診断のテスト"""

    def test_converged_chain_small_z_score(self) -> None:
        """収束したチェーンでは |z| < 2 程度（概ね）"""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (5000, 5))

        z_scores, p_values = geweke_test(chain)

        assert z_scores.shape == (5,)
        assert p_values.shape == (5,)
        # 大部分のパラメータで |z| < 3（95% 信頼区間よりも広く取る）
        assert np.sum(np.abs(z_scores) < 3) >= 4

    def test_geweke_p_values_range(self) -> None:
        """p値は [0, 1] の範囲"""
        rng = np.random.default_rng(42)
        chain = rng.normal(0, 1, (3000, 3))

        _, p_values = geweke_test(chain)

        assert np.all(p_values >= 0.0)
        assert np.all(p_values <= 1.0)


class TestHPD:
    """HPD区間のテスト"""

    def test_hpd_contains_mode(self) -> None:
        """単峰分布のHPD区間はモードを含む"""
        rng = np.random.default_rng(42)
        samples = rng.normal(5.0, 1.0, 10000)

        lower, upper = compute_hpd(samples, alpha=0.1)

        # モード ≈ 平均 ≈ 5.0 がHPD区間内にある
        assert lower < 5.0 < upper

    def test_hpd_coverage(self) -> None:
        """90% HPD区間は約90%のサンプルを含む"""
        rng = np.random.default_rng(42)
        samples = rng.normal(0, 1, 50000)

        lower, upper = compute_hpd(samples, alpha=0.1)

        coverage = np.mean((samples >= lower) & (samples <= upper))
        # 90% ± 2% の範囲内
        assert abs(coverage - 0.90) < 0.02


class TestLaplaceApproximation:
    """Laplace近似のテスト"""

    def test_gaussian_posterior_exact(self) -> None:
        """既知のガウス事後分布では Laplace 近似が正確"""
        # 2次元正規分布: p(θ|y) = N(μ, Σ)
        d = 2
        mu = np.array([1.0, 2.0])
        sigma = np.array([[1.0, 0.3], [0.3, 0.5]])

        # ガウス分布の log normalizing constant
        # log Z = (d/2) log(2π) + 0.5 log|Σ|
        _, log_det_sigma = np.linalg.slogdet(sigma)
        log_z_true = 0.5 * d * np.log(2 * np.pi) + 0.5 * log_det_sigma

        # モードでの対数事後確率（正規化されていない）
        # p(θ*|y) ∝ exp(0) = 1 → log p(θ*|y)_unnorm = 0
        # 正規化定数を含めると log p(θ*|y) = -log_z_true
        mode_log_posterior = -log_z_true

        # ヘッセ行列 = Σ^{-1}（負の対数事後確率の2次微分）
        hessian = np.linalg.inv(sigma)

        # Laplace 近似
        log_ml = compute_marginal_likelihood_laplace(
            mode=mu,
            mode_log_posterior=mode_log_posterior,
            hessian=hessian,
        )

        # ガウス分布の場合、正規化定数の log が一致
        # log p(y) = log p(θ*|y) + (d/2)log(2π) - 0.5 log|H|
        # = -log_z_true + (d/2)log(2π) - 0.5 log|Σ^{-1}|
        # = -log_z_true + (d/2)log(2π) + 0.5 log|Σ|
        # = -log_z_true + log_z_true = 0
        np.testing.assert_allclose(log_ml, 0.0, atol=1e-10)


class TestSummaryTable:
    """サマリーテーブルのテスト"""

    def test_summary_table_has_headers(self) -> None:
        """テーブルに期待されるカラムヘッダーが含まれる"""
        diagnostics = ConvergenceDiagnostics(
            r_hat=np.array([1.01]),
            ess=np.array([500.0]),
            acceptance_rates=np.array([0.25]),
            converged=True,
            geweke_z=np.array([0.5]),
            geweke_p=np.array([0.6]),
            parameter_names=["sigma"],
        )

        summaries = [
            PosteriorSummary(
                name="sigma",
                mean=1.5,
                median=1.49,
                std=0.3,
                hpd_lower=1.0,
                hpd_upper=2.0,
                prior_mean=1.5,
                prior_std=0.37,
            )
        ]

        result = EstimationResult(
            posterior_samples=np.zeros((100, 1)),
            parameter_names=["sigma"],
            log_marginal_likelihood=-100.0,
            diagnostics=diagnostics,
            summaries=summaries,
            mode=np.array([1.5]),
            mode_log_posterior=-50.0,
            n_chains=4,
            n_draws=1000,
            n_burnin=500,
        )

        table = result.summary_table()

        assert "Parameter" in table
        assert "Prior Mean" in table
        assert "Post. Mean" in table
        assert "90% HPD Lower" in table
        assert "90% HPD Upper" in table
        assert "sigma" in table


class TestRunDiagnostics:
    """run_diagnostics のテスト"""

    def test_returns_complete_result(self) -> None:
        """全フィールドが正しく設定される"""
        rng = np.random.default_rng(42)
        n_chains, n_draws, n_params = 4, 1000, 5
        chains = rng.normal(0, 1, (n_chains, n_draws, n_params))
        acceptance_rates = np.array([0.25, 0.26, 0.24, 0.27])
        parameter_names = [f"param_{i}" for i in range(n_params)]

        diag = run_diagnostics(chains, acceptance_rates, parameter_names)

        assert isinstance(diag, ConvergenceDiagnostics)
        assert diag.r_hat.shape == (n_params,)
        assert diag.ess.shape == (n_params,)
        assert diag.geweke_z.shape == (n_params,)
        assert diag.geweke_p.shape == (n_params,)
        assert diag.acceptance_rates.shape == (n_chains,)
        assert diag.parameter_names == parameter_names
        assert isinstance(diag.converged, bool)
        # 同一分布からのチェーンなので収束判定
        assert diag.converged is True
