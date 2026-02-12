"""推定パイプラインの統合テスト

合成データ → MCMC短鎖推定 → 結果集計の全フローを検証する。
"""

import numpy as np
import pytest

from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator
from japan_fiscal_simulator.estimation.kalman_filter import kalman_filter
from japan_fiscal_simulator.estimation.mcmc import (
    MCMCConfig,
    MetropolisHastings,
    make_log_posterior,
)
from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping
from japan_fiscal_simulator.estimation.priors import PriorConfig
from japan_fiscal_simulator.estimation.results import (
    EstimationResult,
    build_estimation_result,
)
from japan_fiscal_simulator.estimation.state_space import StateSpaceBuilder
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestEstimationPipeline:
    """推定パイプライン全体の統合テスト"""

    @pytest.fixture(scope="class")
    def default_params(self) -> DefaultParameters:
        return DefaultParameters()

    @pytest.fixture(scope="class")
    def mapping(self) -> ParameterMapping:
        return ParameterMapping()

    @pytest.fixture(scope="class")
    def prior_config(self) -> PriorConfig:
        return PriorConfig.smets_wouters_japan()

    @pytest.fixture(scope="class")
    def synthetic_data(self, default_params: DefaultParameters) -> np.ndarray:
        """合成データを生成"""
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        est_data = gen.generate(default_params, n_periods=80, rng=rng)
        return est_data.data

    def test_parameter_mapping_prior_consistency(
        self, mapping: ParameterMapping, prior_config: PriorConfig
    ) -> None:
        """ParameterMappingとPriorConfigのパラメータ名が一致する"""
        assert mapping.names == prior_config.names
        assert mapping.n_params == prior_config.n_params

    def test_defaults_produce_finite_log_prior(
        self, mapping: ParameterMapping, prior_config: PriorConfig
    ) -> None:
        """デフォルトパラメータで事前確率が有限"""
        theta = mapping.defaults()
        lp = prior_config.log_prior(theta)
        assert np.isfinite(lp)

    def test_defaults_produce_valid_model(self, mapping: ParameterMapping) -> None:
        """デフォルトθからモデル解が得られる"""
        theta = mapping.defaults()
        params = mapping.theta_to_params(theta)
        model = NewKeynesianModel(params)
        solution = model.solution
        assert solution.P.shape == (5, 5)
        assert solution.Q.shape == (5, 6)

    def test_state_space_from_defaults(self, mapping: ParameterMapping) -> None:
        """デフォルトθから状態空間行列が構築できる"""
        theta = mapping.defaults()
        params = mapping.theta_to_params(theta)
        model = NewKeynesianModel(params)
        solution = model.solution

        shock_std_names = ["sigma_g", "sigma_a", "sigma_m", "sigma_i", "sigma_w", "sigma_p"]
        shock_stds = np.array([theta[mapping._name_to_index[n]] for n in shock_std_names])
        measurement_errors = mapping.theta_to_measurement_errors(theta)

        ss = StateSpaceBuilder.build(solution, shock_stds, measurement_errors)
        assert ss.T.shape == (14, 14)
        assert ss.Z.shape == (7, 14)
        assert ss.R_aug.shape == (14, 6)
        assert ss.Q_cov.shape == (6, 6)
        assert ss.H.shape == (7, 7)

    def test_kalman_filter_with_synthetic_data(
        self, mapping: ParameterMapping, synthetic_data: np.ndarray
    ) -> None:
        """合成データでKalmanフィルタが有限な尤度を返す"""
        theta = mapping.defaults()
        params = mapping.theta_to_params(theta)
        model = NewKeynesianModel(params)
        solution = model.solution

        shock_std_names = ["sigma_g", "sigma_a", "sigma_m", "sigma_i", "sigma_w", "sigma_p"]
        shock_stds = np.array([theta[mapping._name_to_index[n]] for n in shock_std_names])
        measurement_errors = mapping.theta_to_measurement_errors(theta)

        ss = StateSpaceBuilder.build(solution, shock_stds, measurement_errors)
        result = kalman_filter(synthetic_data, ss.T, ss.Z, ss.R_aug, ss.Q_cov, ss.H)

        assert np.isfinite(result.log_likelihood)

    def test_log_posterior_finite_at_defaults(
        self,
        mapping: ParameterMapping,
        prior_config: PriorConfig,
        synthetic_data: np.ndarray,
    ) -> None:
        """デフォルトパラメータで対数事後確率が有限"""
        log_post = make_log_posterior(mapping, prior_config, synthetic_data)
        theta = mapping.defaults()
        lp = log_post(theta)
        assert np.isfinite(lp)

    def test_log_posterior_neg_inf_for_invalid(
        self,
        mapping: ParameterMapping,
        prior_config: PriorConfig,
        synthetic_data: np.ndarray,
    ) -> None:
        """範囲外パラメータで-inf"""
        log_post = make_log_posterior(mapping, prior_config, synthetic_data)
        theta_bad = np.zeros(mapping.n_params)
        assert log_post(theta_bad) == -np.inf


class TestShortChainEstimation:
    """短鎖MCMCによる統合テスト"""

    @pytest.fixture(scope="class")
    def estimation_result(self) -> EstimationResult:
        """短鎖MCMCを実行して結果を返す"""
        mapping = ParameterMapping()
        prior_config = PriorConfig.smets_wouters_japan()

        gen = SyntheticDataGenerator()
        params = DefaultParameters()
        rng = np.random.default_rng(42)
        est_data = gen.generate(params, n_periods=50, rng=rng)

        log_post = make_log_posterior(mapping, prior_config, est_data.data)
        theta0 = mapping.defaults()

        cfg = MCMCConfig(
            n_chains=2,
            n_draws=40,
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

        mcmc_result = mh.run(theta0=theta0)

        return build_estimation_result(
            chains=mcmc_result.chains,
            acceptance_rates=mcmc_result.acceptance_rates,
            mode=mcmc_result.mode,
            mode_log_posterior=mcmc_result.mode_log_posterior,
            hessian=mcmc_result.mode_hessian,
            prior_config=prior_config,
            mapping=mapping,
            n_burnin=20,
        )

    def test_result_has_correct_types(self, estimation_result: EstimationResult) -> None:
        """EstimationResultの各フィールドが正しい型"""
        r = estimation_result
        assert isinstance(r.posterior_samples, np.ndarray)
        assert isinstance(r.parameter_names, list)
        assert isinstance(r.log_marginal_likelihood, float)
        assert isinstance(r.diagnostics.converged, bool)
        assert len(r.summaries) == 27

    def test_posterior_samples_shape(self, estimation_result: EstimationResult) -> None:
        """事後サンプルの形状が正しい"""
        r = estimation_result
        n_kept = 40 - 20  # n_draws - n_burnin
        assert r.posterior_samples.shape == (2 * n_kept, 27)

    def test_summaries_have_valid_statistics(self, estimation_result: EstimationResult) -> None:
        """サマリー統計量が有限値"""
        for s in estimation_result.summaries:
            assert np.isfinite(s.mean)
            assert np.isfinite(s.median)
            assert np.isfinite(s.std)
            assert s.std >= 0
            assert s.hpd_lower <= s.hpd_upper

    def test_mode_is_finite(self, estimation_result: EstimationResult) -> None:
        """モードが有限値"""
        assert np.all(np.isfinite(estimation_result.mode))
        assert np.isfinite(estimation_result.mode_log_posterior)

    def test_summary_table_format(self, estimation_result: EstimationResult) -> None:
        """サマリーテーブルが正しい形式"""
        table = estimation_result.summary_table()
        assert "Parameter" in table
        assert "sigma" in table
        assert "habit" in table

    def test_get_summary_by_name(self, estimation_result: EstimationResult) -> None:
        """名前でサマリーを取得できる"""
        s = estimation_result.get_summary("sigma")
        assert s.name == "sigma"
        assert np.isfinite(s.mean)

    def test_get_summary_unknown_raises(self, estimation_result: EstimationResult) -> None:
        """不存在パラメータでKeyError"""
        with pytest.raises(KeyError):
            estimation_result.get_summary("nonexistent")

    def test_to_default_params_posterior_mean(self, estimation_result: EstimationResult) -> None:
        """posterior_meanからDefaultParametersを生成できる"""
        params = estimation_result.to_default_params(use="posterior_mean")
        assert isinstance(params, DefaultParameters)

    def test_to_default_params_mode(self, estimation_result: EstimationResult) -> None:
        """modeからDefaultParametersを生成できる"""
        params = estimation_result.to_default_params(use="mode")
        assert isinstance(params, DefaultParameters)

    def test_log_marginal_likelihood_finite(self, estimation_result: EstimationResult) -> None:
        """周辺尤度が有限"""
        assert np.isfinite(estimation_result.log_marginal_likelihood)

    def test_diagnostics_complete(self, estimation_result: EstimationResult) -> None:
        """診断結果が完全"""
        d = estimation_result.diagnostics
        assert d.r_hat.shape == (27,)
        assert d.ess.shape == (27,)
        assert d.geweke_z.shape == (27,)
        assert d.geweke_p.shape == (27,)
        assert len(d.parameter_names) == 27
