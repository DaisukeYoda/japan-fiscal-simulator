"""MCPリソース定義"""

from typing import Any

from japan_fiscal_simulator.mcp.tools import SimulationContext, get_context
from japan_fiscal_simulator.policies.consumption_tax import (
    SCENARIO_TAX_CUT_2PCT,
    SCENARIO_TAX_CUT_5PCT,
    SCENARIO_TAX_INCREASE_2PCT,
)
from japan_fiscal_simulator.policies.social_security import (
    SCENARIO_PENSION_CUT,
    SCENARIO_TRANSFER_INCREASE,
)
from japan_fiscal_simulator.policies.subsidies import (
    SCENARIO_EMPLOYMENT_SUBSIDY,
    SCENARIO_GREEN_SUBSIDY,
    SCENARIO_INVESTMENT_SUBSIDY,
)


def get_current_parameters(
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """現在のパラメータを取得

    Resource URI: fiscal://parameters/current

    Args:
        context: シミュレーションコンテキスト（DI用）
    """
    ctx = context or get_context()
    params = ctx.calibration.parameters

    return {
        "household": {
            "beta": params.household.beta,
            "sigma": params.household.sigma,
            "phi": params.household.phi,
            "habit": params.household.habit,
            "chi": params.household.chi,
        },
        "firm": {
            "alpha": params.firm.alpha,
            "delta": params.firm.delta,
            "theta": params.firm.theta,
            "epsilon": params.firm.epsilon,
            "psi": params.firm.psi,
        },
        "government": {
            "tau_c": params.government.tau_c,
            "tau_l": params.government.tau_l,
            "tau_k": params.government.tau_k,
            "g_y_ratio": params.government.g_y_ratio,
            "b_y_ratio": params.government.b_y_ratio,
            "transfer_y_ratio": params.government.transfer_y_ratio,
        },
        "central_bank": {
            "rho_r": params.central_bank.rho_r,
            "phi_pi": params.central_bank.phi_pi,
            "phi_y": params.central_bank.phi_y,
            "pi_target": params.central_bank.pi_target,
            "r_lower_bound": params.central_bank.r_lower_bound,
        },
        "financial": {
            "chi_b": params.financial.chi_b,
            "leverage_ss": params.financial.leverage_ss,
            "survival_rate": params.financial.survival_rate,
        },
    }


def get_steady_state(
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """定常状態値を取得

    Resource URI: fiscal://steady-state/current

    Args:
        context: シミュレーションコンテキスト（DI用）
    """
    ctx = context or get_context()
    ss = ctx.model.steady_state

    return {
        "real_variables": {
            "output": ss.output,
            "consumption": ss.consumption,
            "investment": ss.investment,
            "capital": ss.capital,
            "labor": ss.labor,
        },
        "prices": {
            "real_wage": ss.real_wage,
            "real_interest_rate": ss.real_interest_rate,
            "nominal_interest_rate": ss.nominal_interest_rate,
            "inflation": ss.inflation,
            "marginal_cost": ss.marginal_cost,
            "tobin_q": ss.tobin_q,
        },
        "government": {
            "government_spending": ss.government_spending,
            "government_debt": ss.government_debt,
            "tax_revenue": ss.tax_revenue,
            "transfers": ss.transfers,
            "consumption_tax_rate": ss.consumption_tax_rate,
            "primary_balance": ss.primary_balance,
        },
        "financial": {
            "external_finance_premium": ss.external_finance_premium,
            "net_worth": ss.net_worth,
            "capital_return": ss.capital_return,
        },
    }


def get_scenarios_list() -> dict[str, Any]:
    """定義済みシナリオ一覧を取得

    Resource URI: fiscal://scenarios/list
    """
    scenarios = [
        # 消費税シナリオ
        {
            "id": "tax_cut_2pct",
            "name": SCENARIO_TAX_CUT_2PCT.name,
            "description": SCENARIO_TAX_CUT_2PCT.description,
            "policy_type": SCENARIO_TAX_CUT_2PCT.policy_type.value,
            "shock_size": SCENARIO_TAX_CUT_2PCT.shock_size,
        },
        {
            "id": "tax_cut_5pct",
            "name": SCENARIO_TAX_CUT_5PCT.name,
            "description": SCENARIO_TAX_CUT_5PCT.description,
            "policy_type": SCENARIO_TAX_CUT_5PCT.policy_type.value,
            "shock_size": SCENARIO_TAX_CUT_5PCT.shock_size,
        },
        {
            "id": "tax_increase_2pct",
            "name": SCENARIO_TAX_INCREASE_2PCT.name,
            "description": SCENARIO_TAX_INCREASE_2PCT.description,
            "policy_type": SCENARIO_TAX_INCREASE_2PCT.policy_type.value,
            "shock_size": SCENARIO_TAX_INCREASE_2PCT.shock_size,
        },
        # 社会保障シナリオ
        {
            "id": "transfer_increase",
            "name": SCENARIO_TRANSFER_INCREASE.name,
            "description": SCENARIO_TRANSFER_INCREASE.description,
            "policy_type": SCENARIO_TRANSFER_INCREASE.policy_type.value,
            "shock_size": SCENARIO_TRANSFER_INCREASE.shock_size,
        },
        {
            "id": "pension_cut",
            "name": SCENARIO_PENSION_CUT.name,
            "description": SCENARIO_PENSION_CUT.description,
            "policy_type": SCENARIO_PENSION_CUT.policy_type.value,
            "shock_size": SCENARIO_PENSION_CUT.shock_size,
        },
        # 補助金シナリオ
        {
            "id": "investment_subsidy",
            "name": SCENARIO_INVESTMENT_SUBSIDY.name,
            "description": SCENARIO_INVESTMENT_SUBSIDY.description,
            "policy_type": SCENARIO_INVESTMENT_SUBSIDY.policy_type.value,
            "shock_size": SCENARIO_INVESTMENT_SUBSIDY.shock_size,
        },
        {
            "id": "employment_subsidy",
            "name": SCENARIO_EMPLOYMENT_SUBSIDY.name,
            "description": SCENARIO_EMPLOYMENT_SUBSIDY.description,
            "policy_type": SCENARIO_EMPLOYMENT_SUBSIDY.policy_type.value,
            "shock_size": SCENARIO_EMPLOYMENT_SUBSIDY.shock_size,
        },
        {
            "id": "green_subsidy",
            "name": SCENARIO_GREEN_SUBSIDY.name,
            "description": SCENARIO_GREEN_SUBSIDY.description,
            "policy_type": SCENARIO_GREEN_SUBSIDY.policy_type.value,
            "shock_size": SCENARIO_GREEN_SUBSIDY.shock_size,
        },
    ]

    return {"scenarios": scenarios, "count": len(scenarios)}


def get_latest_results(
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """最新シミュレーション結果を取得

    Resource URI: fiscal://results/latest

    Args:
        context: シミュレーションコンテキスト（DI用）
    """
    ctx = context or get_context()

    if ctx.latest_result is None:
        return {"available": False, "message": "No simulation results available yet."}

    result = ctx.latest_result
    return {
        "available": True,
        "scenario": result.scenario.model_dump(),
        "summary": {
            "output_peak_effect": max(result.impulse_response.variables["y"].values, key=abs) * 100
            if "y" in result.impulse_response.variables
            else None,
            "fiscal_multiplier": result.fiscal_multiplier.impact_multiplier
            if result.fiscal_multiplier
            else None,
            "bk_satisfied": result.blanchard_kahn_satisfied,
        },
        "computation_time_ms": result.computation_time_ms,
        "timestamp": result.timestamp.isoformat(),
    }


def get_results_history(
    *,
    context: SimulationContext | None = None,
) -> dict[str, Any]:
    """シミュレーション履歴を取得

    Resource URI: fiscal://results/history

    Args:
        context: シミュレーションコンテキスト（DI用）
    """
    ctx = context or get_context()

    history = []
    for result in ctx.results_history[-10:]:  # 最新10件
        history.append(
            {
                "scenario_name": result.scenario.name,
                "policy_type": result.scenario.policy_type.value,
                "shock_size": result.scenario.shock_size,
                "timestamp": result.timestamp.isoformat(),
            }
        )

    return {"results": history, "count": len(history)}
