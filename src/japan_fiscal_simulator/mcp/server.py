"""MCPサーバー本体"""

import json
from collections.abc import Callable
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import (
    Resource,
    TextContent,
    Tool,
)
from pydantic import AnyUrl

from japan_fiscal_simulator.mcp.resources import (
    get_current_parameters,
    get_latest_results,
    get_results_history,
    get_scenarios_list,
    get_steady_state,
)
from japan_fiscal_simulator.mcp.tools import (
    compare_scenarios,
    generate_report,
    get_fiscal_multiplier,
    set_parameters,
    simulate_policy,
)


def create_server() -> Server:
    """MCPサーバーを作成"""
    server = Server("japan-fiscal-dsge")

    @server.list_tools()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_tools() -> list[Tool]:
        """利用可能なツールを列挙"""
        return [
            Tool(
                name="simulate_policy",
                description="財政政策のインパルス応答シミュレーションを実行します。消費税減税、政府支出増加、移転支払い増額などの効果を分析できます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "policy_type": {
                            "type": "string",
                            "description": "政策タイプ: consumption_tax（消費税）, government_spending（政府支出）, transfer（移転支払い）, monetary（金融政策）",
                            "enum": [
                                "consumption_tax",
                                "government_spending",
                                "transfer",
                                "monetary",
                                "subsidy",
                            ],
                        },
                        "shock_size": {
                            "type": "number",
                            "description": "ショックサイズ（例: -0.02 = 2%pt減税、0.01 = GDP比1%の支出増）",
                        },
                        "periods": {
                            "type": "integer",
                            "description": "シミュレーション期間（四半期）",
                            "default": 40,
                        },
                        "shock_type": {
                            "type": "string",
                            "description": "ショックタイプ",
                            "enum": ["temporary", "permanent", "gradual"],
                            "default": "temporary",
                        },
                        "scenario_name": {
                            "type": "string",
                            "description": "シナリオ名（オプション）",
                        },
                    },
                    "required": ["policy_type", "shock_size"],
                },
            ),
            Tool(
                name="set_parameters",
                description="モデルパラメータを設定します。消費税率、政府支出比率などを変更できます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "consumption_tax_rate": {
                            "type": "number",
                            "description": "消費税率（例: 0.10 = 10%）",
                        },
                        "government_spending_ratio": {
                            "type": "number",
                            "description": "政府支出/GDP比率",
                        },
                        "debt_ratio": {
                            "type": "number",
                            "description": "政府債務/GDP比率",
                        },
                        "interest_rate_smoothing": {
                            "type": "number",
                            "description": "金利平滑化パラメータ",
                        },
                        "inflation_response": {
                            "type": "number",
                            "description": "インフレ反応係数",
                        },
                    },
                },
            ),
            Tool(
                name="get_fiscal_multiplier",
                description="財政乗数を計算します。政府支出乗数または消費税乗数を求められます。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "policy_type": {
                            "type": "string",
                            "description": "政策タイプ",
                            "enum": ["government_spending", "consumption_tax"],
                            "default": "government_spending",
                        },
                        "horizon": {
                            "type": "integer",
                            "description": "計算期間（四半期）",
                            "default": 40,
                        },
                    },
                },
            ),
            Tool(
                name="compare_scenarios",
                description="複数の政策シナリオを比較します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "scenarios": {
                            "type": "array",
                            "description": "比較するシナリオのリスト",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "policy_type": {"type": "string"},
                                    "shock_size": {"type": "number"},
                                    "name": {"type": "string"},
                                },
                                "required": ["policy_type", "shock_size"],
                            },
                        }
                    },
                    "required": ["scenarios"],
                },
            ),
            Tool(
                name="generate_report",
                description="最新のシミュレーション結果からレポートを生成します。",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "format": {
                            "type": "string",
                            "description": "出力形式",
                            "enum": ["markdown", "json"],
                            "default": "markdown",
                        },
                        "include_graphs": {
                            "type": "boolean",
                            "description": "グラフを含めるか",
                            "default": False,
                        },
                    },
                },
            ),
        ]

    @server.call_tool()  # type: ignore[untyped-decorator]
    async def call_tool(name: str, arguments: dict[str, Any]) -> list[TextContent]:
        """ツールを実行"""
        try:
            if name == "simulate_policy":
                result = simulate_policy(**arguments)
            elif name == "set_parameters":
                result = set_parameters(**arguments)
            elif name == "get_fiscal_multiplier":
                result = get_fiscal_multiplier(**arguments)
            elif name == "compare_scenarios":
                result = compare_scenarios(**arguments)
            elif name == "generate_report":
                result = generate_report(**arguments)
            else:
                result = {"error": f"Unknown tool: {name}"}

            return [
                TextContent(
                    type="text", text=json.dumps(result, ensure_ascii=False, indent=2, default=str)
                )
            ]
        except Exception as e:
            return [TextContent(type="text", text=json.dumps({"error": str(e)}))]

    @server.list_resources()  # type: ignore[no-untyped-call, untyped-decorator]
    async def list_resources() -> list[Resource]:
        """リソース一覧"""
        return [
            Resource(
                uri=AnyUrl("fiscal://parameters/current"),
                name="Current Parameters",
                description="現在のモデルパラメータ",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("fiscal://steady-state/current"),
                name="Steady State",
                description="現在の定常状態値",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("fiscal://scenarios/list"),
                name="Scenarios List",
                description="定義済みシナリオ一覧",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("fiscal://results/latest"),
                name="Latest Results",
                description="最新のシミュレーション結果",
                mimeType="application/json",
            ),
            Resource(
                uri=AnyUrl("fiscal://results/history"),
                name="Results History",
                description="シミュレーション履歴",
                mimeType="application/json",
            ),
        ]

    @server.read_resource()  # type: ignore[no-untyped-call, untyped-decorator]
    async def read_resource(uri: str) -> str:
        """リソースを読み取り"""
        resource_handlers: dict[str, Callable[[], dict[str, Any]]] = {
            "fiscal://parameters/current": get_current_parameters,
            "fiscal://steady-state/current": get_steady_state,
            "fiscal://scenarios/list": get_scenarios_list,
            "fiscal://results/latest": get_latest_results,
            "fiscal://results/history": get_results_history,
        }

        handler = resource_handlers.get(uri)
        if handler is not None:
            return json.dumps(handler(), ensure_ascii=False, indent=2)

        return json.dumps({"error": f"Unknown resource: {uri}"})

    return server


async def run_server() -> None:
    """サーバーを実行"""
    server = create_server()
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())
