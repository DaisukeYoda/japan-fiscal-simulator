# MCP連携（Claude Desktop）

MCPサーバーを通じて、Claude Desktopから直接DSGEモデルを操作できる。

## セットアップ

Claude Desktopの設定ファイル（`claude_desktop_config.json`）にサーバーを追加する。`uvx` を使えば事前インストール不要で、初回起動時に自動でダウンロード・隔離環境の構築・実行まで行われる（`uv` のインストールのみ必要）:

```json
{
  "mcpServers": {
    "jpfs": {
      "command": "uvx",
      "args": ["jpfs", "mcp"]
    }
  }
}
```

設定ファイルの場所:
- macOS: `~/Library/Application Support/Claude/claude_desktop_config.json`
- Windows: `%APPDATA%\Claude\claude_desktop_config.json`

すでに `jpfs` をインストール済み（`pip install jpfs` や `uv tool install jpfs`）でPATHが通っている場合:

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

開発者向け（ローカルのリポジトリを直接実行する場合）:

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
| `shock_type` | string | `temporary` | `temporary`, `permanent`, `gradual`。詳細は下記「ショックタイプの定義」を参照 |
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
| `scenarios` | list | シナリオのリスト。各要素は `{"policy_type", "shock_size", "name", "shock_type"}`。`shock_type` は省略時 `temporary` |

### generate_report

最新のシミュレーション結果からレポートを生成する。

| パラメータ | 型 | デフォルト | 説明 |
|-----------|-----|-----------|------|
| `format` | string | `markdown` | 出力形式 |
| `include_graphs` | bool | false | グラフを含める |

---

## ショックタイプの定義

`simulate_policy` と `compare_scenarios` の `shock_type` は、政策ショックの時間的な発生パターンを指定する。

| タイプ | 挙動 |
|--------|------|
| `temporary` | 期初（t=0）にのみショックが発生。その後はモデル内生的な持続性（状態変数の AR(1) 係数）に従って減衰する。非状態ショック（`e_p` など）は `ρ^t` で減衰。 |
| `permanent` | 全期間にわたり同一サイズのショックが継続する（`ε_t = shock_size`）。線形化 DSGE の IRF としては「定常状態からの外生的撹動が永続する」ことを表す。大きなショックでは線形化の仮定（定常状態周りの摂動）から逸脱する恐れがあるため、小さなショックでの利用を想定する。 |
| `gradual` | ランプ期間（デフォルト 4 四半期 = 1 年）をかけてショックサイズに線形に到達し、その後は維持される。t=0 はショックゼロから開始し、`t = ramp_periods` で目標サイズに到達する。 |

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
