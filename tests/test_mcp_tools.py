"""MCPツール関数のテスト"""

import pytest

# pydantic が Python 3.14 と互換性がない場合、output.schemas のインポートが失敗する。
# MCP toolsはこのモジュールチェーンに依存するため、インポート不可時はスキップする。
try:
    from japan_fiscal_simulator.output.schemas import VariableTimeSeries  # noqa: F401

    _pydantic_ok = True
except (ImportError, TypeError, AssertionError):
    _pydantic_ok = False

pytestmark = pytest.mark.skipif(not _pydantic_ok, reason="pydantic incompatible with this Python version")

if _pydantic_ok:
    from japan_fiscal_simulator.core.exceptions import ShockValidationError
    from japan_fiscal_simulator.mcp.tools import (
        ContextManager,
        SimulationContext,
        compare_scenarios,
        generate_report,
        get_fiscal_multiplier,
        set_parameters,
        simulate_policy,
    )


@pytest.fixture
def ctx() -> SimulationContext:
    """テスト用のシミュレーションコンテキスト"""
    return SimulationContext()


@pytest.fixture(autouse=True)
def reset_context_manager() -> None:
    """各テスト後にContextManagerをリセット"""
    yield
    if _pydantic_ok:
        ContextManager.reset()


class TestSimulatePolicy:
    """simulate_policyのテスト"""

    def test_government_spending(self, ctx: SimulationContext) -> None:
        result = simulate_policy("government_spending", 0.01, context=ctx)
        assert "scenario" in result
        assert "impulse_response" in result
        assert result["blanchard_kahn_satisfied"] is True

    def test_consumption_tax_has_multiplier(self, ctx: SimulationContext) -> None:
        result = simulate_policy("consumption_tax", -0.02, context=ctx)
        assert result["fiscal_multiplier"] is not None

    def test_invalid_policy_type(self, ctx: SimulationContext) -> None:
        with pytest.raises(ShockValidationError):
            simulate_policy("invalid_policy", 0.01, context=ctx)


class TestSetParameters:
    """set_parametersのテスト"""

    def test_set_consumption_tax(self, ctx: SimulationContext) -> None:
        result = set_parameters(consumption_tax_rate=0.08, context=ctx)
        assert result["consumption_tax_rate"] == pytest.approx(0.08)

    def test_set_multiple_parameters(self, ctx: SimulationContext) -> None:
        result = set_parameters(
            consumption_tax_rate=0.15,
            government_spending_ratio=0.25,
            context=ctx,
        )
        assert result["consumption_tax_rate"] == pytest.approx(0.15)
        assert result["government_spending_ratio"] == pytest.approx(0.25)


class TestGetFiscalMultiplier:
    """get_fiscal_multiplierのテスト"""

    def test_government_spending(self, ctx: SimulationContext) -> None:
        result = get_fiscal_multiplier("government_spending", context=ctx)
        assert "impact_multiplier" in result
        assert "peak_multiplier" in result

    def test_invalid_policy_type(self, ctx: SimulationContext) -> None:
        with pytest.raises(ShockValidationError):
            get_fiscal_multiplier("invalid", context=ctx)


class TestCompareScenarios:
    """compare_scenariosのテスト"""

    def test_two_scenarios(self, ctx: SimulationContext) -> None:
        scenarios = [
            {"policy_type": "government_spending", "shock_size": 0.01, "name": "Scenario A"},
            {"policy_type": "consumption_tax", "shock_size": -0.02, "name": "Scenario B"},
        ]
        result = compare_scenarios(scenarios, context=ctx)
        assert len(result["comparisons"]) == 2
        assert "summary" in result


class TestGenerateReport:
    """generate_reportのテスト"""

    def test_no_results(self, ctx: SimulationContext) -> None:
        result = generate_report(context=ctx)
        assert "error" in result

    def test_after_simulation(self, ctx: SimulationContext) -> None:
        simulate_policy("government_spending", 0.01, context=ctx)
        result = generate_report(context=ctx)
        assert "content" in result


class TestSimulationContext:
    """SimulationContextのテスト"""

    def test_reset_model(self, ctx: SimulationContext) -> None:
        old_model = ctx.model
        ctx.reset_model()
        assert ctx.model is not old_model


class TestContextManager:
    """ContextManagerのテスト"""

    def test_get_creates_instance(self) -> None:
        ContextManager.reset()
        ctx = ContextManager.get()
        assert ctx is not None

    def test_set_and_get(self) -> None:
        custom_ctx = SimulationContext()
        ContextManager.set(custom_ctx)
        assert ContextManager.get() is custom_ctx

    def test_reset_clears(self) -> None:
        ContextManager.set(SimulationContext())
        ContextManager.reset()
        ctx1 = ContextManager.get()
        ContextManager.reset()
        ctx2 = ContextManager.get()
        assert ctx1 is not ctx2
