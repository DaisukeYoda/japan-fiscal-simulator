"""CLIコマンドのテスト"""

import pytest

# pydantic が現在のPython実行環境と互換性がない場合、output.schemas のインポートが失敗する。
# CLI/MCPテストはこのモジュールチェーンに依存するため、インポート不可時はスキップする。
try:
    from japan_fiscal_simulator.output.schemas import VariableTimeSeries  # noqa: F401

    _pydantic_ok = True
except (ImportError, TypeError, AssertionError):
    _pydantic_ok = False

pytestmark = pytest.mark.skipif(not _pydantic_ok, reason="pydantic incompatible with this Python version")

if _pydantic_ok:
    from typer.testing import CliRunner

    from japan_fiscal_simulator.cli.main import app

    runner = CliRunner()


class TestSimulateCommand:
    """simulateコマンドのテスト"""

    def test_simulate_government_spending_default(self) -> None:
        result = runner.invoke(app, ["simulate", "government_spending"])
        assert result.exit_code == 0

    def test_simulate_invalid_policy_type(self) -> None:
        result = runner.invoke(app, ["simulate", "invalid_type"])
        assert result.exit_code != 0

    def test_simulate_consumption_tax(self) -> None:
        result = runner.invoke(app, ["simulate", "consumption_tax", "--shock", "-0.02"])
        assert result.exit_code == 0

    def test_simulate_monetary(self) -> None:
        result = runner.invoke(app, ["simulate", "monetary", "--shock", "0.01"])
        assert result.exit_code == 0

    def test_simulate_price_markup(self) -> None:
        result = runner.invoke(app, ["simulate", "price_markup", "--shock", "0.01"])
        assert result.exit_code == 0


class TestMultiplierCommand:
    """multiplierコマンドのテスト"""

    def test_multiplier_government_spending(self) -> None:
        result = runner.invoke(app, ["multiplier", "government_spending"])
        assert result.exit_code == 0

    def test_multiplier_consumption_tax(self) -> None:
        result = runner.invoke(app, ["multiplier", "consumption_tax"])
        assert result.exit_code == 0

    def test_multiplier_invalid_type(self) -> None:
        result = runner.invoke(app, ["multiplier", "invalid_type"])
        assert result.exit_code != 0


class TestOtherCommands:
    """その他のコマンドのテスト"""

    def test_steady_state(self) -> None:
        result = runner.invoke(app, ["steady-state"])
        assert result.exit_code == 0

    def test_parameters(self) -> None:
        result = runner.invoke(app, ["parameters"])
        assert result.exit_code == 0

    def test_version(self) -> None:
        result = runner.invoke(app, ["version"])
        assert result.exit_code == 0
        assert "version" in result.output.lower()

    def test_help(self) -> None:
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
