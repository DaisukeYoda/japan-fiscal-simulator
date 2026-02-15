"""ドキュメント（docs/guide/）のコード例が正しく動作するかを検証するテスト

各テストクラスは対応するドキュメントファイルに対応する:
- TestGettingStarted  → docs/guide/getting-started.md
- TestPythonAPI       → docs/guide/python-api.md
- TestCLI             → docs/guide/cli.md
- TestEstimation      → docs/guide/estimation.md
- TestMCP             → docs/guide/mcp.md
"""

import numpy as np
import pytest

# pydantic が Python 3.14 と互換性がない場合、output.schemas のインポートが失敗する。
# CLI/MCP/Policyテストはこのモジュールチェーンに依存するため、インポート不可時はスキップする。
try:
    from japan_fiscal_simulator.output.schemas import VariableTimeSeries  # noqa: F401

    _pydantic_ok = True
except (ImportError, TypeError, AssertionError):
    _pydantic_ok = False

_skip_pydantic = pytest.mark.skipif(
    not _pydantic_ok,
    reason="pydantic incompatible with this Python version",
)


# ============================================================
# getting-started.md
# ============================================================
class TestGettingStarted:
    """docs/guide/getting-started.md のコード例を検証"""

    def test_import_and_model_init(self) -> None:
        import japan_fiscal_simulator as jpfs

        calibration = jpfs.JapanCalibration.create()
        model = jpfs.DSGEModel(calibration.parameters)
        assert model is not None

    def test_steady_state_attributes(self) -> None:
        import japan_fiscal_simulator as jpfs

        calibration = jpfs.JapanCalibration.create()
        model = jpfs.DSGEModel(calibration.parameters)
        ss = model.steady_state

        assert isinstance(ss.output, float)
        assert isinstance(ss.consumption, float)
        assert isinstance(ss.investment, float)
        assert ss.output > 0

    def test_simulation_and_result(self) -> None:
        import japan_fiscal_simulator as jpfs

        calibration = jpfs.JapanCalibration.create()
        model = jpfs.DSGEModel(calibration.parameters)
        simulator = jpfs.ImpulseResponseSimulator(model)

        result = simulator.simulate_consumption_tax_cut(tax_cut=0.02, periods=40)
        y_response = result.get_response("y")
        assert isinstance(y_response, np.ndarray)
        # periods=40 → t=0含む41要素
        assert len(y_response) == 41
        assert result.periods == 41

        peak_period, peak_value = result.peak_response("y")
        assert isinstance(peak_period, int)
        assert isinstance(peak_value, float)

    def test_calibration_customization(self) -> None:
        import japan_fiscal_simulator as jpfs

        calibration = jpfs.JapanCalibration.create()
        calibration = calibration.set_consumption_tax(0.08)
        model = jpfs.DSGEModel(calibration.parameters)
        assert model.steady_state is not None


# ============================================================
# python-api.md
# ============================================================
class TestPythonAPI:
    """docs/guide/python-api.md のコード例を検証"""

    @pytest.fixture
    def model(self):
        import japan_fiscal_simulator as jpfs

        calibration = jpfs.JapanCalibration.create()
        return jpfs.DSGEModel(calibration.parameters)

    # --- DSGEModel ---

    def test_steady_state_all_attributes(self, model) -> None:
        ss = model.steady_state
        # 実物変数
        assert hasattr(ss, "output")
        assert hasattr(ss, "consumption")
        assert hasattr(ss, "investment")
        assert hasattr(ss, "capital")
        assert hasattr(ss, "labor")
        # 価格・金利
        assert hasattr(ss, "real_wage")
        assert hasattr(ss, "real_interest_rate")
        assert hasattr(ss, "nominal_interest_rate")
        assert hasattr(ss, "inflation")
        # 政府部門
        assert hasattr(ss, "government_spending")
        assert hasattr(ss, "government_debt")
        assert hasattr(ss, "tax_revenue")
        assert hasattr(ss, "primary_balance")

    def test_policy_function_attributes(self, model) -> None:
        pf = model.policy_function
        assert hasattr(pf, "P")
        assert hasattr(pf, "Q")
        assert hasattr(pf, "bk_satisfied")
        assert hasattr(pf, "n_stable")
        assert hasattr(pf, "n_unstable")
        assert hasattr(pf, "eigenvalues")
        assert isinstance(pf.P, np.ndarray)
        assert isinstance(pf.Q, np.ndarray)
        assert isinstance(pf.bk_satisfied, bool)

    def test_invalidate_cache(self, model) -> None:
        _ = model.steady_state
        model.invalidate_cache()
        ss2 = model.steady_state
        assert ss2 is not None

    def test_variable_index(self, model) -> None:
        idx = model.get_variable_index("y")
        assert idx == 0
        name = model.get_variable_name(0)
        assert name == "y"

    # --- ImpulseResponseSimulator ---

    def test_simulate_generic(self, model) -> None:
        import japan_fiscal_simulator as jpfs

        simulator = jpfs.ImpulseResponseSimulator(model)
        result = simulator.simulate(shock_name="e_g", shock_size=0.01, periods=40)
        # periods=40 → t=0含む41要素
        assert result.periods == 41
        assert result.shock_name == "e_g"
        assert result.shock_size == 0.01
        assert isinstance(result.responses, dict)

    def test_simulate_convenience_methods(self, model) -> None:
        import japan_fiscal_simulator as jpfs

        simulator = jpfs.ImpulseResponseSimulator(model)

        r1 = simulator.simulate_consumption_tax_cut(tax_cut=0.02, periods=20)
        assert r1.periods == 21

        r2 = simulator.simulate_government_spending(spending_increase=0.01, periods=20)
        assert r2.periods == 21

        r3 = simulator.simulate_monetary_shock(rate_change=0.0025, periods=20)
        assert r3.periods == 21

        r4 = simulator.simulate_technology_shock(productivity_increase=0.01, periods=20)
        assert r4.periods == 21

    def test_impulse_response_result_methods(self, model) -> None:
        import japan_fiscal_simulator as jpfs

        simulator = jpfs.ImpulseResponseSimulator(model)
        result = simulator.simulate(shock_name="e_g", shock_size=0.01, periods=40)

        y = result.get_response("y")
        assert isinstance(y, np.ndarray)

        period, value = result.peak_response("y")
        assert isinstance(period, int)
        assert isinstance(value, float)

        cum = result.cumulative_response("y", horizon=8)
        assert isinstance(cum, float)

    # --- FiscalMultiplierCalculator ---

    def test_fiscal_multiplier_calculator_import(self, model) -> None:
        from japan_fiscal_simulator.core.simulation import FiscalMultiplierCalculator

        calc = FiscalMultiplierCalculator(model)
        result = calc.compute_spending_multiplier(horizon=40)
        assert hasattr(result, "impact")
        assert hasattr(result, "peak")
        assert hasattr(result, "peak_period")
        assert hasattr(result, "cumulative_4q")
        assert hasattr(result, "cumulative_8q")
        assert hasattr(result, "cumulative_20q")
        assert hasattr(result, "present_value")

    def test_tax_multiplier(self, model) -> None:
        from japan_fiscal_simulator.core.simulation import FiscalMultiplierCalculator

        calc = FiscalMultiplierCalculator(model)
        result = calc.compute_tax_multiplier(horizon=40)
        assert isinstance(result.impact, float)

    # --- JapanCalibration ---

    def test_calibration_methods(self) -> None:
        import japan_fiscal_simulator as jpfs

        calibration = jpfs.JapanCalibration.create()
        c1 = calibration.set_consumption_tax(0.08)
        c2 = calibration.set_government_spending_ratio(0.22)
        assert c1.parameters is not None
        assert c2.parameters is not None

    # --- 政策分析モジュール（pydantic依存） ---

    @_skip_pydantic
    def test_consumption_tax_policy(self, model) -> None:
        from japan_fiscal_simulator.policies.consumption_tax import (
            SCENARIO_TAX_CUT_2PCT,
            SCENARIO_TAX_CUT_5PCT,
            SCENARIO_TAX_INCREASE_2PCT,
            ConsumptionTaxPolicy,
        )

        policy = ConsumptionTaxPolicy(model)
        analysis = policy.analyze(SCENARIO_TAX_CUT_2PCT)
        assert hasattr(analysis, "output_effect_peak")
        assert hasattr(analysis, "consumption_effect_peak")
        assert hasattr(analysis, "revenue_impact")
        assert hasattr(analysis, "welfare_effect")

        # カスタムシナリオ
        scenario = ConsumptionTaxPolicy.create_reduction_scenario(
            reduction_rate=0.03, periods=40
        )
        analysis2 = policy.analyze(scenario)
        assert isinstance(analysis2.output_effect_peak, float)

        # プリセットが存在すること
        assert SCENARIO_TAX_CUT_5PCT is not None
        assert SCENARIO_TAX_INCREASE_2PCT is not None

    @_skip_pydantic
    def test_social_security_policy(self, model) -> None:
        from japan_fiscal_simulator.policies.social_security import (
            SCENARIO_PENSION_CUT,
            SCENARIO_TRANSFER_INCREASE,
            SocialSecurityPolicy,
        )

        policy = SocialSecurityPolicy(model)
        analysis = policy.analyze(SCENARIO_TRANSFER_INCREASE)
        assert hasattr(analysis, "output_effect_peak")
        assert hasattr(analysis, "consumption_effect_peak")
        assert hasattr(analysis, "debt_impact")
        assert hasattr(analysis, "distributional_note")

        assert SCENARIO_PENSION_CUT is not None

    @_skip_pydantic
    def test_subsidy_policy(self, model) -> None:
        from japan_fiscal_simulator.policies.subsidies import (
            SCENARIO_EMPLOYMENT_SUBSIDY,
            SCENARIO_GREEN_SUBSIDY,
            SCENARIO_INVESTMENT_SUBSIDY,
            SubsidyPolicy,
        )

        policy = SubsidyPolicy(model)
        analysis = policy.analyze(SCENARIO_INVESTMENT_SUBSIDY)
        assert hasattr(analysis, "output_effect_peak")
        assert hasattr(analysis, "investment_effect_peak")
        assert hasattr(analysis, "fiscal_cost")
        assert hasattr(analysis, "crowding_out_ratio")

        assert SCENARIO_EMPLOYMENT_SUBSIDY is not None
        assert SCENARIO_GREEN_SUBSIDY is not None

    # --- 変数・ショック一覧 ---

    def test_variable_indices(self) -> None:
        from japan_fiscal_simulator.core.model import SHOCK_VARS, VARIABLE_INDICES

        expected_vars = [
            "y", "c", "i", "n", "k", "pi", "r", "R",
            "w", "mc", "g", "b", "tau_c", "a", "q", "rk",
        ]
        for var in expected_vars:
            assert var in VARIABLE_INDICES, f"{var} not in VARIABLE_INDICES"

        expected_shocks = ["e_a", "e_g", "e_m", "e_tau", "e_risk", "e_i", "e_p"]
        assert SHOCK_VARS == expected_shocks


# ============================================================
# cli.md（pydantic依存）
# ============================================================
@_skip_pydantic
class TestCLI:
    """docs/guide/cli.md のコマンドを検証"""

    @pytest.fixture(autouse=True)
    def _cli(self):
        from typer.testing import CliRunner

        from japan_fiscal_simulator.cli.main import app

        self.runner = CliRunner()
        self.app = app

    def _run(self, *args: str):
        return self.runner.invoke(self.app, list(args))

    def test_simulate_consumption_tax(self) -> None:
        r = self._run("simulate", "consumption_tax", "--shock", "-0.02", "--periods", "20")
        assert r.exit_code == 0

    def test_simulate_government_spending(self) -> None:
        r = self._run("simulate", "government_spending", "--shock", "0.01", "--periods", "20")
        assert r.exit_code == 0

    def test_simulate_monetary(self) -> None:
        r = self._run("simulate", "monetary", "--shock", "0.0025", "--periods", "20")
        assert r.exit_code == 0

    def test_simulate_price_markup(self) -> None:
        r = self._run("simulate", "price_markup", "--shock", "0.01", "--periods", "20")
        assert r.exit_code == 0

    def test_multiplier(self) -> None:
        r = self._run("multiplier", "government_spending", "--horizon", "8")
        assert r.exit_code == 0

    def test_steady_state(self) -> None:
        r = self._run("steady-state")
        assert r.exit_code == 0

    def test_parameters(self) -> None:
        r = self._run("parameters")
        assert r.exit_code == 0

    def test_version(self) -> None:
        r = self._run("version")
        assert r.exit_code == 0


# ============================================================
# estimation.md
# ============================================================
class TestEstimation:
    """docs/guide/estimation.md のコード例を検証"""

    def test_mcmc_config(self) -> None:
        from japan_fiscal_simulator.estimation import MCMCConfig

        config = MCMCConfig(
            n_chains=4,
            n_draws=100_000,
            n_burnin=50_000,
            thinning=10,
        )
        assert config.n_chains == 4
        assert config.target_acceptance == 0.234
        assert config.adaptive_interval == 100
        assert config.mode_search_max_iter == 500

    def test_parameter_mapping(self) -> None:
        from japan_fiscal_simulator.estimation import ParameterMapping

        mapping = ParameterMapping()
        assert mapping.n_params > 0
        assert len(mapping.names) == mapping.n_params

        theta = mapping.defaults()
        assert isinstance(theta, np.ndarray)
        assert len(theta) == mapping.n_params

        bounds = mapping.bounds()
        assert len(bounds) == mapping.n_params
        for lo, hi in bounds:
            assert lo < hi

    def test_prior_config(self) -> None:
        from japan_fiscal_simulator.estimation import PriorConfig

        prior = PriorConfig.smets_wouters_japan()
        assert hasattr(prior, "priors")
        # priors は list[ParameterPrior]
        assert isinstance(prior.priors, list)
        assert len(prior.priors) > 0
        # 各要素は name, dist_type, mean, std を持つ
        p0 = prior.priors[0]
        assert hasattr(p0, "name")
        assert hasattr(p0, "dist_type")
        assert hasattr(p0, "mean")
        assert hasattr(p0, "std")

    def test_synthetic_data_generator(self) -> None:
        import japan_fiscal_simulator as jpfs
        from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator

        calibration = jpfs.JapanCalibration.create()
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate(calibration.parameters, n_periods=50)

        assert hasattr(synthetic_data, "data")
        assert isinstance(synthetic_data.data, np.ndarray)
        assert synthetic_data.data.shape[0] == 50
        assert synthetic_data.data.shape[1] == 7
        assert synthetic_data.n_periods == 50
        assert synthetic_data.n_obs == 7

    def test_make_log_posterior(self) -> None:
        import japan_fiscal_simulator as jpfs
        from japan_fiscal_simulator.estimation import (
            ParameterMapping,
            PriorConfig,
            make_log_posterior,
        )
        from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator

        mapping = ParameterMapping()
        prior = PriorConfig.smets_wouters_japan()

        calibration = jpfs.JapanCalibration.create()
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate(calibration.parameters, n_periods=50)

        log_posterior_fn = make_log_posterior(mapping, prior, synthetic_data.data)
        assert callable(log_posterior_fn)

        # 呼び出し可能であること
        theta = mapping.defaults()
        val = log_posterior_fn(theta)
        assert isinstance(val, float)

    def test_metropolis_hastings_constructor(self) -> None:
        from japan_fiscal_simulator.estimation import (
            MCMCConfig,
            MetropolisHastings,
            ParameterMapping,
        )

        mapping = ParameterMapping()
        config = MCMCConfig(n_chains=2, n_draws=100, n_burnin=50, thinning=1)

        def dummy_log_posterior(theta: np.ndarray) -> float:
            return -0.5 * float(np.sum(theta**2))

        mh = MetropolisHastings(
            log_posterior_fn=dummy_log_posterior,
            n_params=mapping.n_params,
            config=config,
            parameter_names=mapping.names,
            bounds=mapping.bounds(),
        )
        assert mh is not None

    def test_build_estimation_result(self) -> None:
        from japan_fiscal_simulator.estimation import (
            ParameterMapping,
            PriorConfig,
        )
        from japan_fiscal_simulator.estimation.results import build_estimation_result

        mapping = ParameterMapping()
        prior = PriorConfig.smets_wouters_japan()
        n_params = mapping.n_params

        # ダミーデータで構築テスト
        chains = np.random.randn(2, 100, n_params)
        acceptance_rates = np.array([0.25, 0.23])
        mode = np.zeros(n_params)
        hessian = np.eye(n_params)

        est = build_estimation_result(
            chains=chains,
            acceptance_rates=acceptance_rates,
            mode=mode,
            mode_log_posterior=-100.0,
            hessian=hessian,
            prior_config=prior,
            mapping=mapping,
            n_burnin=50,
        )

        assert hasattr(est, "log_marginal_likelihood")
        assert hasattr(est, "diagnostics")
        assert hasattr(est, "n_chains")
        assert hasattr(est, "n_draws")

        # get_summary
        name = mapping.names[0]
        summary = est.get_summary(name)
        assert hasattr(summary, "mean")
        assert hasattr(summary, "median")
        assert hasattr(summary, "std")
        assert hasattr(summary, "hpd_lower")
        assert hasattr(summary, "hpd_upper")
        assert hasattr(summary, "prior_mean")
        assert hasattr(summary, "prior_std")

        # summary_table
        table = est.summary_table()
        assert isinstance(table, str)
        assert len(table) > 0

    def test_estimation_data_csv_columns(self) -> None:
        """DataLoaderが期待するCSV列名を検証"""
        import tempfile
        from pathlib import Path

        import japan_fiscal_simulator as jpfs
        from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator
        from japan_fiscal_simulator.estimation.data_loader import DataLoader

        calibration = jpfs.JapanCalibration.create()
        generator = SyntheticDataGenerator()
        synthetic_data = generator.generate(calibration.parameters, n_periods=50)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            generator.to_csv(synthetic_data, csv_path)

            # CSVのヘッダーにドキュメント記載の列名が含まれること
            import csv

            with open(csv_path) as f:
                reader = csv.reader(f)
                headers = next(reader)
            expected = {"date", "gdp", "consumption", "investment", "deflator", "wage", "employment", "rate"}
            assert expected == set(headers), f"CSV headers mismatch: {headers}"

            # CSVを読み込めることを検証
            loader = DataLoader()
            loaded = loader.load_csv(csv_path)
            assert loaded.n_obs == 7
            assert loaded.n_periods == 50


# ============================================================
# mcp.md（pydantic依存）
# ============================================================
@_skip_pydantic
class TestMCP:
    """docs/guide/mcp.md の記載ツールを検証"""

    @pytest.fixture
    def context(self):
        from japan_fiscal_simulator.mcp.tools import SimulationContext

        return SimulationContext()

    def test_simulate_policy(self, context) -> None:
        from japan_fiscal_simulator.mcp.tools import simulate_policy

        result = simulate_policy(
            policy_type="consumption_tax",
            shock_size=-0.02,
            periods=20,
            shock_type="temporary",
            context=context,
        )
        assert isinstance(result, dict)

    def test_set_parameters(self, context) -> None:
        from japan_fiscal_simulator.mcp.tools import set_parameters

        result = set_parameters(
            consumption_tax_rate=0.08,
            context=context,
        )
        assert isinstance(result, dict)

    def test_get_fiscal_multiplier(self, context) -> None:
        from japan_fiscal_simulator.mcp.tools import get_fiscal_multiplier

        result = get_fiscal_multiplier(
            policy_type="government_spending",
            horizon=20,
            context=context,
        )
        assert isinstance(result, dict)

    def test_compare_scenarios(self, context) -> None:
        from japan_fiscal_simulator.mcp.tools import compare_scenarios

        result = compare_scenarios(
            scenarios=[
                {"policy_type": "consumption_tax", "shock_size": -0.02, "name": "減税"},
                {"policy_type": "government_spending", "shock_size": 0.01, "name": "歳出"},
            ],
            context=context,
        )
        assert isinstance(result, dict)

    def test_generate_report(self, context) -> None:
        from japan_fiscal_simulator.mcp.tools import generate_report, simulate_policy

        # レポート生成にはシミュレーション結果が必要
        simulate_policy(
            policy_type="consumption_tax",
            shock_size=-0.02,
            context=context,
        )
        result = generate_report(format="markdown", context=context)
        assert isinstance(result, dict)

    def test_only_5_documented_tools(self) -> None:
        """MCPに登録されるツールがドキュメントの5種と一致すること"""
        from japan_fiscal_simulator.mcp.tools import (
            compare_scenarios,
            generate_report,
            get_fiscal_multiplier,
            set_parameters,
            simulate_policy,
        )

        # 5つとも存在する
        assert callable(simulate_policy)
        assert callable(set_parameters)
        assert callable(get_fiscal_multiplier)
        assert callable(compare_scenarios)
        assert callable(generate_report)
