# MCP連携（Claude Desktop）

MCPサーバーを通じて、Claude Desktopから直接DSGEモデルを操作できる。

## セットアップ

Claude Desktopの設定ファイル（`claude_desktop_config.json`）にサーバーを追加する:

```json
{
  "mcpServers": {
    "jpfs": {
      "command": "jpfs",
      "args": ["mcp"]
    }
  }
}
```

設定ファイルの場所:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

uvで実行する場合:

```json
{
  "mcpServers": {
    "jpfs": {
      "command": "uv",
      "args": ["run", "--directory", "/path/to/japan-fiscal-simulator", "jpfs", "mcp"]
    }
  }
}
```

## 利用可能なツール（5種類）

### simulate_policy

政策ショックのシミュレーションを実行する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `policy_type` | string | (必須) | `consumption_tax`, `government_spending`, `transfer`, `monetary`, `subsidy`, `price_markup` |
| `shock_size` | float | (必須) | ショックサイズ |
| `periods` | int | 40 | シミュレーション期間 |
| `shock_type` | string | `temporary` | `temporary`, `permanent`, `gradual` |
| `scenario_name` | string | null | シナリオ名 |

### set_parameters

モデルのキャリブレーションを変更する。

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `consumption_tax_rate` | float | 消費税率 |
| `government_spending_ratio` | float | 政府支出/GDP比 |
| `debt_ratio` | float | 政府債務/GDP比 |
| `interest_rate_smoothing` | float | 金利平滑化パラメータ |
| `inflation_response` | float | Taylor則インフレ反応係数 |

全パラメータはオプション。指定したもののみ更新される。

### get_fiscal_multiplier

財政乗数を計算する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `policy_type` | string | `government_spending` | `government_spending` または `consumption_tax` |
| `horizon` | int | 40 | 計算期間 |

### compare_scenarios

複数の政策シナリオを同時に比較する。

| パラメータ | 型 | 説明 |
|-----------|-----|------|
| `scenarios` | list | シナリオのリスト。各要素は `{"policy_type", "shock_size", "name"}` |

### generate_report

最新のシミュレーション結果からレポートを生成する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `format` | string | `markdown` | 出力形式 |
| `include_graphs` | bool | false | グラフを含める |

---

## 使用例

Claude Desktop上での対話例:

> **ユーザー**: 消費税を2%下げたときの経済への影響を教えて

Claudeが `simulate_policy` ツールを呼び出し、結果を解説してくれる。

> **ユーザー**: 政府支出1%増加のケースと比較して

Claudeが `compare_scenarios` ツールで2つのシナリオを同時に分析する。

> **ユーザー**: 政府支出の財政乗数はどのくらい？

Claudeが `get_fiscal_multiplier` ツールでインパクト乗数・累積乗数を計算する。

> **ユーザー**: 金利平滑化パラメータを0.9に変更して、再度シミュレーションして

Claudeが `set_parameters` でパラメータを変更し、`simulate_policy` で再計算する。

> **ユーザー**: レポートを作って

Claudeが `generate_report` で分析結果をMarkdownレポートにまとめる。
