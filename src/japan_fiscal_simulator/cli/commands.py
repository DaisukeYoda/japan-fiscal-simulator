"""CLIコマンド実装"""

from pathlib import Path
from typing import Annotated, Protocol

import typer
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from japan_fiscal_simulator.core.model import DSGEModel
from japan_fiscal_simulator.core.simulation import (
    FiscalMultiplierCalculator,
    ImpulseResponseSimulator,
)
from japan_fiscal_simulator.mcp.tools import ContextManager
from japan_fiscal_simulator.output.graphs import GraphGenerator
from japan_fiscal_simulator.output.reports import ReportGenerator
from japan_fiscal_simulator.parameters.calibration import JapanCalibration

console = Console()


class ModelFactory(Protocol):
    """モデル生成のプロトコル（DI用）"""

    def create_calibration(self) -> JapanCalibration: ...
    def create_model(self, calibration: JapanCalibration) -> DSGEModel: ...


class DefaultModelFactory:
    """デフォルトのモデルファクトリ"""

    def create_calibration(self) -> JapanCalibration:
        return JapanCalibration.create()

    def create_model(self, calibration: JapanCalibration) -> DSGEModel:
        return DSGEModel(calibration.parameters)


class ModelFactoryManager:
    """モデルファクトリ管理（DI用）"""

    _instance: ModelFactory | None = None

    @classmethod
    def get(cls) -> ModelFactory:
        if cls._instance is None:
            cls._instance = DefaultModelFactory()
        return cls._instance

    @classmethod
    def set(cls, factory: ModelFactory) -> None:
        """テスト用にファクトリを設定"""
        cls._instance = factory

    @classmethod
    def reset(cls) -> None:
        """テスト用にリセット"""
        cls._instance = None


def simulate_command(
    policy_type: Annotated[
        str,
        typer.Argument(help="政策タイプ: consumption_tax, government_spending, transfer, monetary"),
    ],
    shock: Annotated[
        float,
        typer.Option("--shock", "-s", help="ショックサイズ（例: -0.02 = 2%減税）"),
    ] = 0.01,
    periods: Annotated[
        int,
        typer.Option("--periods", "-p", help="シミュレーション期間（四半期）"),
    ] = 40,
    graph: Annotated[
        bool,
        typer.Option("--graph", "-g", help="グラフを生成"),
    ] = False,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="出力ディレクトリ"),
    ] = None,
) -> None:
    """財政政策シミュレーションを実行"""
    factory = ModelFactoryManager.get()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("モデルを初期化中...", total=None)

        # モデル初期化（ファクトリ経由）
        calibration = factory.create_calibration()
        model = factory.create_model(calibration)

        progress.add_task("シミュレーション実行中...", total=None)

        # ショック名へのマッピング
        shock_mapping = {
            "consumption_tax": "e_tau",
            "government_spending": "e_g",
            "transfer": "e_g",
            "monetary": "e_m",
        }

        shock_name = shock_mapping.get(policy_type)
        if shock_name is None:
            console.print(f"[red]エラー: 不明な政策タイプ '{policy_type}'[/red]")
            raise typer.Exit(1)

        # シミュレーション実行
        simulator = ImpulseResponseSimulator(model)
        result = simulator.simulate(shock_name, shock, periods)

    # 結果表示
    console.print()
    console.print(
        Panel(
            f"[bold]シミュレーション結果[/bold]\n"
            f"政策タイプ: {policy_type}\n"
            f"ショック: {shock * 100:.1f}%\n"
            f"期間: {periods}四半期",
            title="Japan Fiscal DSGE Simulator",
        )
    )

    # 主要変数の応答をテーブル表示
    table = Table(title="主要変数のピーク応答")
    table.add_column("変数", style="cyan")
    table.add_column("ピーク値（%）", style="green")
    table.add_column("ピーク時期", style="yellow")

    for var in ["y", "c", "i", "pi", "r", "b"]:
        peak_period, peak_value = result.peak_response(var)
        var_names = {
            "y": "産出（Y）",
            "c": "消費（C）",
            "i": "投資（I）",
            "pi": "インフレ率（π）",
            "r": "実質金利（r）",
            "b": "政府債務（B）",
        }
        table.add_row(
            var_names.get(var, var),
            f"{peak_value * 100:+.3f}",
            f"{peak_period}Q",
        )

    console.print(table)

    # グラフ生成
    if graph:
        out_dir = output_dir or Path("./output")
        graph_gen = GraphGenerator(out_dir / "graphs")
        graph_gen.plot_impulse_response(result, show=False)
        save_path = out_dir / "graphs" / f"irf_{policy_type}.png"
        console.print(f"\n[green]グラフを保存しました: {save_path}[/green]")


def multiplier_command(
    policy_type: Annotated[
        str,
        typer.Argument(help="政策タイプ: government_spending, consumption_tax"),
    ],
    horizon: Annotated[
        int,
        typer.Option("--horizon", "-h", help="計算期間（四半期）"),
    ] = 40,
) -> None:
    """財政乗数を計算"""
    factory = ModelFactoryManager.get()

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("財政乗数を計算中...", total=None)

        calibration = factory.create_calibration()
        model = factory.create_model(calibration)
        calc = FiscalMultiplierCalculator(model)

        if policy_type == "government_spending":
            result = calc.compute_spending_multiplier(horizon)
            title = "政府支出乗数"
        elif policy_type == "consumption_tax":
            result = calc.compute_tax_multiplier(horizon)
            title = "消費税乗数（減税の効果）"
        else:
            console.print(f"[red]エラー: 不明な政策タイプ '{policy_type}'[/red]")
            raise typer.Exit(1)

    # 結果表示
    table = Table(title=title)
    table.add_column("乗数タイプ", style="cyan")
    table.add_column("値", style="green")

    table.add_row("インパクト乗数", f"{result.impact:.3f}")
    table.add_row("ピーク乗数", f"{result.peak:.3f}（{result.peak_period}Q）")
    table.add_row("累積乗数（1年）", f"{result.cumulative_4q:.3f}")
    table.add_row("累積乗数（2年）", f"{result.cumulative_8q:.3f}")
    table.add_row("累積乗数（5年）", f"{result.cumulative_20q:.3f}")
    table.add_row("現在価値乗数", f"{result.present_value:.3f}")

    console.print()
    console.print(table)


def steady_state_command() -> None:
    """定常状態を表示"""
    factory = ModelFactoryManager.get()
    calibration = factory.create_calibration()
    model = factory.create_model(calibration)
    ss = model.steady_state

    console.print()
    console.print(Panel("[bold]定常状態[/bold]", title="Japan Fiscal DSGE"))

    # 実体変数
    table1 = Table(title="実体経済変数")
    table1.add_column("変数", style="cyan")
    table1.add_column("値", style="green")

    table1.add_row("産出（Y）", f"{ss.output:.4f}")
    table1.add_row("消費（C）", f"{ss.consumption:.4f}")
    table1.add_row("投資（I）", f"{ss.investment:.4f}")
    table1.add_row("資本（K）", f"{ss.capital:.4f}")
    table1.add_row("労働（N）", f"{ss.labor:.4f}")

    console.print(table1)

    # 価格変数
    table2 = Table(title="価格・金利")
    table2.add_column("変数", style="cyan")
    table2.add_column("値", style="green")

    table2.add_row("実質賃金（W）", f"{ss.real_wage:.4f}")
    table2.add_row("実質金利（r）", f"{ss.real_interest_rate * 100:.2f}%")
    table2.add_row("名目金利（R）", f"{ss.nominal_interest_rate * 100:.2f}%")
    table2.add_row("インフレ率（π）", f"{ss.inflation * 100:.2f}%")

    console.print(table2)

    # 政府変数
    table3 = Table(title="政府部門")
    table3.add_column("変数", style="cyan")
    table3.add_column("値", style="green")

    table3.add_row("政府支出（G）", f"{ss.government_spending:.4f}")
    table3.add_row("政府債務（B）", f"{ss.government_debt:.4f}")
    table3.add_row("税収", f"{ss.tax_revenue:.4f}")
    table3.add_row("プライマリーバランス", f"{ss.primary_balance:.4f}")

    console.print(table3)


def parameters_command() -> None:
    """パラメータを表示"""
    factory = ModelFactoryManager.get()
    calibration = factory.create_calibration()
    params = calibration.parameters

    console.print()
    console.print(Panel("[bold]モデルパラメータ[/bold]", title="Japan Calibration"))

    # 家計
    table1 = Table(title="家計部門")
    table1.add_column("パラメータ", style="cyan")
    table1.add_column("値", style="green")
    table1.add_column("説明", style="yellow")

    table1.add_row("β", f"{params.household.beta}", "割引率")
    table1.add_row("σ", f"{params.household.sigma}", "相対的リスク回避度")
    table1.add_row("φ", f"{params.household.phi}", "労働供給弾力性逆数")
    table1.add_row("h", f"{params.household.habit}", "習慣形成")

    console.print(table1)

    # 企業
    table2 = Table(title="企業部門")
    table2.add_column("パラメータ", style="cyan")
    table2.add_column("値", style="green")
    table2.add_column("説明", style="yellow")

    table2.add_row("α", f"{params.firm.alpha}", "資本分配率")
    table2.add_row("δ", f"{params.firm.delta}", "資本減耗率")
    table2.add_row("θ", f"{params.firm.theta}", "Calvo価格硬直性")

    console.print(table2)

    # 政府
    table3 = Table(title="政府部門")
    table3.add_column("パラメータ", style="cyan")
    table3.add_column("値", style="green")
    table3.add_column("説明", style="yellow")

    table3.add_row("τ_c", f"{params.government.tau_c * 100:.0f}%", "消費税率")
    table3.add_row("G/Y", f"{params.government.g_y_ratio * 100:.0f}%", "政府支出比率")
    table3.add_row("B/Y", f"{params.government.b_y_ratio * 100:.0f}%", "政府債務比率")

    console.print(table3)

    # 中央銀行
    table4 = Table(title="中央銀行")
    table4.add_column("パラメータ", style="cyan")
    table4.add_column("値", style="green")
    table4.add_column("説明", style="yellow")

    table4.add_row("ρ_R", f"{params.central_bank.rho_r}", "金利平滑化")
    table4.add_row("φ_π", f"{params.central_bank.phi_pi}", "インフレ反応")
    table4.add_row("φ_y", f"{params.central_bank.phi_y}", "産出ギャップ反応")

    console.print(table4)


def report_command(
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="出力ファイル"),
    ] = None,
) -> None:
    """レポートを生成"""
    ctx = ContextManager.get()

    if ctx.latest_result is None:
        console.print(
            "[yellow]シミュレーション結果がありません。先にsimulateコマンドを実行してください。[/yellow]"
        )
        raise typer.Exit(1)

    generator = ReportGenerator()
    report = generator.generate_simulation_report(ctx.latest_result)

    if output_file:
        output_file.write_text(report, encoding="utf-8")
        console.print(f"[green]レポートを保存しました: {output_file}[/green]")
    else:
        console.print(report)
