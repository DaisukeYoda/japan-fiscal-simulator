# jpfs - Japan Fiscal Simulator

[![PyPI version](https://badge.fury.io/py/jpfs.svg)](https://pypi.org/project/jpfs/)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

消費税減税・社会保障費増額・補助金政策などの財政政策が日本経済に与える影響をシミュレートするツール。中規模New Keynesian DSGEモデルをPythonでフルスクラッチ実装。

## 特徴

- **5部門DSGEモデル**: 家計、企業、政府、中央銀行、金融部門
- **日本経済向けキャリブレーション**: 低金利環境、高債務水準、消費税10%
- **Blanchard-Kahn解法**: 合理的期待均衡の数値解法
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

- 異時点間効用最大化
- 習慣形成（外部習慣）
- 労働供給の内生化

### 企業部門

- Calvo型価格硬直性
- CES生産関数
- New Keynesian Phillips曲線

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
- 純資産発展方程式

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

### テスト実行

```bash
pytest tests/
```

### 型チェック

```bash
mypy src/japan_fiscal
```

## 今後の拡張候補

### モデル拡張

- **消費税ショックの持続性**: 現在は一時的ショックのみ。AR(1)プロセスで持続的な効果をモデル化
- **金融加速器の本格実装**: BGG型の完全版（現在は簡略化）
- **ZLB制約の明示的モデル化**: ゼロ金利下限での非線形ダイナミクス
- **開放経済拡張**: 為替レート、輸出入、海外金利の導入
- **異質的家計**: 流動性制約を持つ家計（Hand-to-Mouth）の追加

### 分析機能

- **グラフ出力**: `--graph`オプションでMatplotlibによる可視化
- **レポート生成**: Jinja2テンプレートによるHTML/PDFレポート
- **感度分析**: パラメータの変化に対する結果の感応度
- **シナリオ比較**: 複数政策シナリオの同時比較機能
- **ベイズ推定**: 実データを用いたパラメータ推定

### インターフェース

- **MCPサーバーの拡充**: 追加ツール・リソースの実装
- **Web UI**: Streamlit/Gradioによるインタラクティブダッシュボード
- **API サーバー**: FastAPIによるREST API提供

### コード品質

- **型チェック強化**: mypy strict モードへの対応
- **ドキュメント**: Sphinxによる API ドキュメント自動生成
- **CI/CD**: GitHub Actionsによる自動テスト・デプロイ

## ライセンス

MIT License

## 参考文献

- Smets, F., & Wouters, R. (2007). Shocks and frictions in US business cycles: A Bayesian DSGE approach.
- Bernanke, B. S., Gertler, M., & Gilchrist, S. (1999). The financial accelerator in a quantitative business cycle framework.
- Blanchard, O. J., & Kahn, C. M. (1980). The solution of linear difference models under rational expectations.
