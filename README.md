# jpfs - Japan Fiscal Simulator

[![PyPI version](https://badge.fury.io/py/jpfs.svg)](https://pypi.org/project/jpfs/)
[![Python 3.14+](https://img.shields.io/badge/python-3.14+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

消費税減税・社会保障費増額・補助金政策などの財政政策が日本経済に与える影響をシミュレートするツール。Smets-Wouters級の中規模New Keynesian DSGEモデルをPythonでフルスクラッチ実装。

## 特徴

- **14方程式の構造NKモデル**: 16変数（state 5 + control 9 + 財政・税ブロック）、7構造ショック
- **5部門経済**: 家計（習慣形成）、企業（Calvo価格・賃金硬直性）、政府、中央銀行、金融部門
- **QZ分解ベースBK/Klein解法**: `scipy.linalg.ordqz` による合理的期待均衡の一般解法
- **ベイズ推定**: Metropolis-Hastings MCMCサンプラー + カルマンフィルタ尤度計算
- **日本経済向けキャリブレーション**: 低金利環境、高債務水準、消費税10%
- **MCPサーバー**: Claude Desktopとの連携
- **CLI**: コマンドラインからのシミュレーション実行

## インストール

```bash
pip install jpfs
```

または

```bash
uvx jpfs --help
```

## 使用方法

### CLI

```bash
# シミュレーション実行
jpfs simulate consumption_tax --shock -0.02 --periods 40 --graph
jpfs simulate price_markup --shock 0.01 --periods 40

# 財政乗数計算
jpfs multiplier government_spending --horizon 8

# 定常状態表示
jpfs steady-state

# パラメータ表示
jpfs parameters

# MCPサーバー起動
jpfs mcp
```

### Pythonからの使用

```python
import japan_fiscal_simulator as jpfs

# モデル初期化
calibration = jpfs.JapanCalibration.create()
model = jpfs.DSGEModel(calibration.parameters)

# 定常状態
ss = model.steady_state
print(f"産出: {ss.output:.4f}")
print(f"消費: {ss.consumption:.4f}")

# シミュレーション
simulator = jpfs.ImpulseResponseSimulator(model)
result = simulator.simulate_consumption_tax_cut(tax_cut=0.02, periods=40)

# 結果
y_response = result.get_response("y")
print(f"産出ピーク効果: {max(y_response) * 100:.2f}%")
```

### MCP連携

Claude Desktopの設定ファイル（`claude_desktop_config.json`）に追加:

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

## モデル概要

### 家計部門

- 異時点間効用最大化（消費のオイラー方程式）
- 習慣形成（外部習慣）
- 労働供給の内生化（限界代替率）

### 企業部門

- Calvo型価格硬直性 + 価格インデクセーション
- Calvo型賃金硬直性 + 賃金マークアップショック
- Cobb-Douglas生産関数、実質限界費用の明示化
- 資本蓄積（投資調整コスト、Tobin's Q）

### 政府部門

- 消費税・所得税・資本所得税
- 政府支出・移転支払い
- 財政ルール（債務安定化）

### 中央銀行

- テイラールール
- 金利平滑化
- ゼロ金利下限（ZLB）考慮

### 金融部門

- 金融加速器（BGG型簡略版）
- 外部資金プレミアム
- リスクプレミアムショック

### 構造ショック（7種類）

技術(TFP)、リスクプレミアム、投資固有技術、賃金マークアップ、価格マークアップ、政府支出、金融政策

## パラメータ

主要パラメータ（日本キャリブレーション）:

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| β | 0.999 | 割引率（低金利環境） |
| τ_c | 0.10 | 消費税率（10%） |
| B/Y | 2.00 | 政府債務/GDP比率 |
| ρ_R | 0.85 | 金利平滑化 |
| θ | 0.75 | Calvo価格硬直性 |

## 出力形式

### シミュレーション結果（JSON）

```json
{
  "scenario": {
    "name": "消費税2%pt減税",
    "policy_type": "consumption_tax",
    "shock_size": -0.02
  },
  "impulse_response": {
    "y": {"values": [...]},
    "c": {"values": [...]},
    "pi": {"values": [...]}
  },
  "fiscal_multiplier": {
    "impact_multiplier": 0.85,
    "cumulative_multiplier_4q": 1.12
  }
}
```

## 開発

```bash
# 依存関係インストール
uv sync

# テスト実行
uv run pytest

# 型チェック（strictモード）
uv run mypy src/japan_fiscal_simulator

# Lint & フォーマット
uv run ruff check src tests
uv run ruff format src tests
```

## 今後の拡張候補（Phase 6: 日本固有の拡張）

### モデル拡張

- **ZLB制約の明示的モデル化**: ゼロ金利下限での非線形ダイナミクス
- **高債務経済**: リカーディアン/非リカーディアン家計の混合、財政持続可能性条件
- **人口動態**: OLG要素の導入、労働力人口減少のトレンド
- **開放経済拡張**: 為替レート、輸出入、海外金利の導入
- **金融加速器の本格実装**: BGG型の完全版（現在は簡略化）

### インターフェース

- **Web UI**: Streamlit/Gradioによるインタラクティブダッシュボード
- **API サーバー**: FastAPIによるREST API提供

## ライセンス

MIT License

## 参考文献

- Smets, F., & Wouters, R. (2007). Shocks and frictions in US business cycles: A Bayesian DSGE approach.
- Bernanke, B. S., Gertler, M., & Gilchrist, S. (1999). The financial accelerator in a quantitative business cycle framework.
- Blanchard, O. J., & Kahn, C. M. (1980). The solution of linear difference models under rational expectations.
