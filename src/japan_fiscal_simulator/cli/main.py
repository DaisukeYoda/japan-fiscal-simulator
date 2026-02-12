"""CLIメインエントリーポイント"""

import asyncio
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console

from japan_fiscal_simulator import __version__
from japan_fiscal_simulator.cli.commands import (
    estimate_command,
    fetch_data_command,
    multiplier_command,
    parameters_command,
    report_command,
    simulate_command,
    steady_state_command,
)
from japan_fiscal_simulator.mcp.server import run_server

app = typer.Typer(
    name="japan-fiscal",
    help="日本財政政策DSGEシミュレーター",
    no_args_is_help=True,
)
console = Console()


@app.command("simulate")
def simulate(
    policy_type: Annotated[
        str,
        typer.Argument(
            help="政策タイプ: consumption_tax, government_spending, transfer, monetary, price_markup"
        ),
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
    """財政政策シミュレーションを実行

    例:
        japan-fiscal simulate consumption_tax --shock -0.02 --periods 40 --graph
        japan-fiscal simulate government_spending --shock 0.01
        japan-fiscal simulate price_markup --shock 0.01
    """
    simulate_command(policy_type, shock, periods, graph, output_dir)


@app.command("multiplier")
def multiplier(
    policy_type: Annotated[
        str,
        typer.Argument(help="政策タイプ: government_spending, consumption_tax"),
    ],
    horizon: Annotated[
        int,
        typer.Option("--horizon", "-h", help="計算期間（四半期）"),
    ] = 40,
) -> None:
    """財政乗数を計算

    例:
        japan-fiscal multiplier government_spending --horizon 8
        japan-fiscal multiplier consumption_tax
    """
    multiplier_command(policy_type, horizon)


@app.command("steady-state")
def steady_state() -> None:
    """定常状態を表示"""
    steady_state_command()


@app.command("parameters")
def parameters() -> None:
    """モデルパラメータを表示"""
    parameters_command()


@app.command("report")
def report(
    output_file: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="出力ファイル"),
    ] = None,
) -> None:
    """レポートを生成"""
    report_command(output_file)


@app.command("estimate")
def estimate(
    data_file: Annotated[
        Path | None,
        typer.Argument(help="観測データCSVファイル"),
    ] = None,
    draws: Annotated[
        int,
        typer.Option("--draws", "-d", help="MCMCドロー数"),
    ] = 100_000,
    chains: Annotated[
        int,
        typer.Option("--chains", help="チェーン数"),
    ] = 4,
    burnin: Annotated[
        int,
        typer.Option("--burnin", help="バーンイン数"),
    ] = 50_000,
    thinning: Annotated[
        int,
        typer.Option("--thinning", help="間引き間隔"),
    ] = 10,
    output_dir: Annotated[
        Path | None,
        typer.Option("--output", "-o", help="結果出力ディレクトリ"),
    ] = None,
    synthetic: Annotated[
        bool,
        typer.Option("--synthetic", help="合成データを使用"),
    ] = False,
    synthetic_periods: Annotated[
        int,
        typer.Option("--synthetic-periods", help="合成データの期間数"),
    ] = 200,
) -> None:
    """ベイズ推定（Metropolis-Hastings MCMC）を実行

    例:
        jpfs estimate data.csv --draws 100000 --chains 4 --output results/
        jpfs estimate --synthetic --draws 1000 --burnin 500 --chains 2
    """
    estimate_command(
        data_file,
        draws,
        chains,
        burnin,
        thinning,
        output_dir,
        synthetic,
        synthetic_periods,
    )


@app.command("fetch-data")
def fetch_data(
    output_file: Annotated[
        Path,
        typer.Option("--output", "-o", help="出力CSVファイルパス"),
    ] = Path("data/japan_quarterly.csv"),
    periods: Annotated[
        int,
        typer.Option("--periods", "-p", help="生成期間数（合成データ）"),
    ] = 200,
) -> None:
    """合成データを生成してCSV出力

    例:
        jpfs fetch-data --output data/japan_quarterly.csv
        jpfs fetch-data --periods 100 -o data/test_data.csv
    """
    fetch_data_command(output_file, periods)


@app.command("mcp")
def mcp_server() -> None:
    """MCPサーバーを起動"""
    console.print("[bold]Japan Fiscal DSGE MCP Server[/bold]")
    console.print("MCPサーバーを起動しています...")

    asyncio.run(run_server())


@app.command("version")
def version() -> None:
    """バージョン情報を表示"""
    console.print(f"japan-fiscal version {__version__}")


if __name__ == "__main__":
    app()
