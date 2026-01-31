"""MCPツール定義"""

import time
from datetime import datetime
from typing import Any

from japan_fiscal_simulator.core.model import DSGEModel
from japan_fiscal_simulator.core.simulation import (
    FiscalMultiplierCalculator,
    ImpulseResponseSimulator,
)
from japan_fiscal_simulator.output.reports import ReportGenerator
from japan_fiscal_simulator.output.schemas import (
    ComparisonResult,
    FiscalMultiplier,
    ImpulseResponse,
    ParameterSet,
    PolicyScenario,
    PolicyType,
    ScenarioComparison,
    ShockType,
    SimulationResult,
    SteadyStateValues,
    VariableTimeSeries,
)
from japan_fiscal_simulator.parameters.calibration import JapanCalibration


class SimulationContext:
    """シミュレーションコンテキスト（状態管理）"""

    def __init__(
        self,
        calibration: JapanCalibration | None = None,
        model: DSGEModel | None = None,
    ) -> None:
        self.calibration = calibration or JapanCalibration.create()
        self.model = model or DSGEModel(self.calibration.parameters)
        self.latest_result: SimulationResult | None = None
        self.results_history: list[SimulationResult] = []

    def reset_model(self) -> None:
        """モデルをリセット"""
        self.model = DSGEModel(self.calibration.parameters)

    def update_calibration(self, calibration: JapanCalibration) -> None:
        """キャリブレーションを更新"""
        self.calibration = calibration
        self.reset_model()


class ContextManager:
    """コンテキスト管理（DI対応）"""

    _instance: SimulationContext | None = None

    @classmethod
    def get(cls) -> SimulationContext:
        """コンテキストを取得（遅延初期化）"""
        if cls._instance is None:
            cls._instance = SimulationContext()
        return cls._instance

    @classmethod
    def set(cls, context: SimulationContext) -> None:
        """コンテキストを設定（テスト用）"""
        cls._instance = context

    @classmethod
    def reset(cls) -> None:
        """コンテキストをリセット（テスト用）"""
        cls._instance = None


def get_context() -> SimulationContext:
    """コンテキストを取得（後方互換性のため維持）"""
    return ContextManager.get()


def simulate_policy(
    policy_type: str,
    shock_size: float,
    periods: int = 40,
    shock_type: str = "temporary",
    scenario_name: str | None = None,
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """財政政策のインパルス応答シミュレーションを実行

    Args:
        policy_type: 政策タイプ（consumption_tax, government_spending, transfer, monetary）
        shock_size: ショックサイズ（例: -0.02 = 2%減税）
        periods: シミュレーション期間（四半期）
        shock_type: ショックタイプ（temporary, permanent, gradual）
        scenario_name: シナリオ名
        context: シミュレーションコンテキスト（DI用、Noneの場合はグローバルを使用）

    Returns:
        シミュレーション結果のJSON
    """
    start_time = time.time()
    ctx = context or get_context()

    # ポリシータイプからショック名へのマッピング
    shock_mapping = {
        "consumption_tax": "e_tau",
        "government_spending": "e_g",
        "transfer": "e_g",  # 移転は政府支出として近似
        "monetary": "e_m",
        "subsidy": "e_g",
    }

    shock_name = shock_mapping.get(policy_type)
    if shock_name is None:
        raise ValueError(f"Unknown policy type: {policy_type}")

    # シミュレーション実行
    simulator = ImpulseResponseSimulator(ctx.model)
    irf_result = simulator.simulate(shock_name, shock_size, periods)

    # 財政乗数計算（該当する場合）
    fiscal_mult = None
    if policy_type in ["government_spending", "subsidy"]:
        calc = FiscalMultiplierCalculator(ctx.model)
        mult_result = calc.compute_spending_multiplier(periods)
        fiscal_mult = FiscalMultiplier(
            impact_multiplier=mult_result.impact,
            peak_multiplier=mult_result.peak,
            peak_period=mult_result.peak_period,
            cumulative_multiplier_4q=mult_result.cumulative_4q,
            cumulative_multiplier_8q=mult_result.cumulative_8q,
            cumulative_multiplier_20q=mult_result.cumulative_20q,
            present_value_multiplier=mult_result.present_value,
        )
    elif policy_type == "consumption_tax":
        calc = FiscalMultiplierCalculator(ctx.model)
        mult_result = calc.compute_tax_multiplier(periods)
        fiscal_mult = FiscalMultiplier(
            impact_multiplier=mult_result.impact,
            peak_multiplier=mult_result.peak,
            peak_period=mult_result.peak_period,
            cumulative_multiplier_4q=mult_result.cumulative_4q,
            cumulative_multiplier_8q=mult_result.cumulative_8q,
            cumulative_multiplier_20q=mult_result.cumulative_20q,
            present_value_multiplier=mult_result.present_value,
        )

    # シナリオ作成
    scenario = PolicyScenario(
        name=scenario_name or f"{policy_type}_{shock_size}",
        description=f"{policy_type}政策シミュレーション（ショック: {shock_size * 100:.1f}%）",
        policy_type=PolicyType(policy_type),
        shock_type=ShockType(shock_type),
        shock_size=shock_size,
        periods=periods,
    )

    # パラメータセット
    params = ctx.calibration.parameters
    param_set = ParameterSet(
        beta=params.household.beta,
        sigma=params.household.sigma,
        phi=params.household.phi,
        habit=params.household.habit,
        alpha=params.firm.alpha,
        delta=params.firm.delta,
        theta=params.firm.theta,
        tau_c=params.government.tau_c,
        tau_l=params.government.tau_l,
        tau_k=params.government.tau_k,
        g_y_ratio=params.government.g_y_ratio,
        b_y_ratio=params.government.b_y_ratio,
        rho_r=params.central_bank.rho_r,
        phi_pi=params.central_bank.phi_pi,
        phi_y=params.central_bank.phi_y,
    )

    # 定常状態
    ss = ctx.model.steady_state
    steady_state = SteadyStateValues(
        output=ss.output,
        consumption=ss.consumption,
        investment=ss.investment,
        capital=ss.capital,
        labor=ss.labor,
        real_wage=ss.real_wage,
        real_interest_rate=ss.real_interest_rate,
        inflation=ss.inflation,
        government_spending=ss.government_spending,
        government_debt=ss.government_debt,
        tax_revenue=ss.tax_revenue,
        primary_balance=ss.primary_balance,
    )

    # インパルス応答
    variable_descriptions = {
        "y": "産出（GDP）",
        "c": "消費",
        "i": "投資",
        "n": "労働",
        "k": "資本",
        "w": "実質賃金",
        "pi": "インフレ率",
        "r": "実質金利",
        "R": "名目金利",
        "g": "政府支出",
        "b": "政府債務",
        "tau_c": "消費税率",
    }

    variables = {}
    for var_name in ["y", "c", "i", "n", "pi", "r", "g", "b", "tau_c"]:
        response = irf_result.get_response(var_name)
        variables[var_name] = VariableTimeSeries(
            name=var_name,
            description=variable_descriptions.get(var_name, var_name),
            values=response.tolist(),
        )

    impulse_response = ImpulseResponse(
        periods=periods,
        time_axis=list(range(periods)),
        variables=variables,
    )

    # 結果の構築
    policy = ctx.model.policy_function
    result = SimulationResult(
        scenario=scenario,
        parameters=param_set,
        steady_state=steady_state,
        impulse_response=impulse_response,
        fiscal_multiplier=fiscal_mult,
        blanchard_kahn_satisfied=policy.bk_satisfied,
        num_stable_eigenvalues=policy.n_stable,
        num_unstable_eigenvalues=policy.n_unstable,
        computation_time_ms=(time.time() - start_time) * 1000,
    )

    ctx.latest_result = result
    ctx.results_history.append(result)

    return result.model_dump()


def set_parameters(
    consumption_tax_rate: float | None = None,
    government_spending_ratio: float | None = None,
    debt_ratio: float | None = None,
    interest_rate_smoothing: float | None = None,
    inflation_response: float | None = None,
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """モデルパラメータを設定

    Args:
        consumption_tax_rate: 消費税率（例: 0.10 = 10%）
        government_spending_ratio: 政府支出/GDP比率
        debt_ratio: 政府債務/GDP比率
        interest_rate_smoothing: 金利平滑化パラメータ
        inflation_response: インフレ反応係数
        context: シミュレーションコンテキスト（DI用）

    Returns:
        更新後のパラメータ
    """
    ctx = context or get_context()

    if consumption_tax_rate is not None:
        ctx.calibration = ctx.calibration.set_consumption_tax(consumption_tax_rate)

    if government_spending_ratio is not None:
        ctx.calibration = ctx.calibration.set_government_spending_ratio(government_spending_ratio)

    ctx.reset_model()

    params = ctx.calibration.parameters
    return {
        "consumption_tax_rate": params.government.tau_c,
        "government_spending_ratio": params.government.g_y_ratio,
        "debt_ratio": params.government.b_y_ratio,
        "interest_rate_smoothing": params.central_bank.rho_r,
        "inflation_response": params.central_bank.phi_pi,
    }


def get_fiscal_multiplier(
    policy_type: str = "government_spending",
    horizon: int = 40,
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """財政乗数を計算

    Args:
        policy_type: 政策タイプ（government_spending, consumption_tax）
        horizon: 計算期間
        context: シミュレーションコンテキスト（DI用）

    Returns:
        財政乗数の詳細
    """
    ctx = context or get_context()
    calc = FiscalMultiplierCalculator(ctx.model)

    if policy_type == "government_spending":
        result = calc.compute_spending_multiplier(horizon)
    elif policy_type == "consumption_tax":
        result = calc.compute_tax_multiplier(horizon)
    else:
        raise ValueError(f"Unknown policy type: {policy_type}")

    return {
        "policy_type": policy_type,
        "impact_multiplier": result.impact,
        "peak_multiplier": result.peak,
        "peak_period": result.peak_period,
        "cumulative_multiplier_4q": result.cumulative_4q,
        "cumulative_multiplier_8q": result.cumulative_8q,
        "cumulative_multiplier_20q": result.cumulative_20q,
        "present_value_multiplier": result.present_value,
    }


def compare_scenarios(
    scenarios: list[dict[str, Any]],
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """複数シナリオを比較

    Args:
        scenarios: シナリオ定義のリスト
            各シナリオ: {"policy_type": str, "shock_size": float, "name": str}
        context: シミュレーションコンテキスト（DI用）

    Returns:
        比較結果
    """
    ctx = context or get_context()
    comparisons = []

    for scenario_def in scenarios:
        result = simulate_policy(
            policy_type=scenario_def["policy_type"],
            shock_size=scenario_def["shock_size"],
            scenario_name=scenario_def.get("name"),
            context=ctx,
        )

        irf = result["impulse_response"]["variables"]
        y_peak = max(irf["y"]["values"], key=abs) if irf["y"]["values"] else 0
        c_peak = max(irf["c"]["values"], key=abs) if irf["c"]["values"] else 0
        i_peak = max(irf["i"]["values"], key=abs) if irf["i"]["values"] else 0
        pi_peak = max(irf["pi"]["values"], key=abs) if irf["pi"]["values"] else 0
        b_final = irf["b"]["values"][-1] if irf["b"]["values"] else 0

        fiscal_mult = (
            result["fiscal_multiplier"]["impact_multiplier"] if result["fiscal_multiplier"] else 0
        )

        comparisons.append(
            ScenarioComparison(
                scenario_name=scenario_def.get("name", scenario_def["policy_type"]),
                impact_on_output=y_peak * 100,
                impact_on_consumption=c_peak * 100,
                impact_on_investment=i_peak * 100,
                impact_on_inflation=pi_peak * 100,
                impact_on_debt=b_final * 100,
                fiscal_multiplier=fiscal_mult,
            )
        )

    # サマリー生成
    best_output = max(comparisons, key=lambda x: x.impact_on_output)
    summary = f"産出効果が最も大きいのは「{best_output.scenario_name}」（{best_output.impact_on_output:.2f}%）です。"

    comparison_result = ComparisonResult(
        baseline_scenario=scenarios[0].get("name", "baseline"),
        comparisons=comparisons,
        summary=summary,
    )

    return comparison_result.model_dump()


def generate_report(
    format: str = "markdown",
    include_graphs: bool = False,
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """最新のシミュレーション結果からレポートを生成

    Args:
        format: 出力形式（markdown, json）
        include_graphs: グラフを含めるか
        context: シミュレーションコンテキスト（DI用）

    Returns:
        レポート内容
    """
    ctx = context or get_context()

    if ctx.latest_result is None:
        return {"error": "No simulation results available. Run simulate_policy first."}

    generator = ReportGenerator()
    report = generator.generate_simulation_report(ctx.latest_result)

    return {
        "format": format,
        "content": report,
        "scenario": ctx.latest_result.scenario.name,
        "generated_at": datetime.now().isoformat(),
    }
