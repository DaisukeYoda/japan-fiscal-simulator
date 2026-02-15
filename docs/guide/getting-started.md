# クイックスタート

## インストール

### pip

```bash
pip install jpfs
```

### uvx（一時実行）

```bash
uvx jpfs --help
```

### 開発環境

```bash
git clone https://github.com/DaisukeYoda/japan-fiscal-simulator.git
cd japan-fiscal-simulator
uv sync
```

## 最初のシミュレーション（CLI）

消費税2%pt減税のインパルス応答を40期間シミュレーションする:

```bash
jpfs simulate consumption_tax --shock -0.02 --periods 40
```

グラフ付きで出力する場合:

```bash
jpfs simulate consumption_tax --shock -0.02 --periods 40 --graph
```

財政乗数を計算する:

```bash
jpfs multiplier government_spending --horizon 8
```

定常状態を確認する:

```bash
jpfs steady-state
```

## 最初のシミュレーション（Python）

```python
import japan_fiscal_simulator as jpfs

# モデル初期化（日本経済向けキャリブレーション）
calibration = jpfs.JapanCalibration.create()
model = jpfs.DSGEModel(calibration.parameters)

# 定常状態の確認
ss = model.steady_state
print(f"産出: {ss.output:.4f}")
print(f"消費: {ss.consumption:.4f}")
print(f"投資: {ss.investment:.4f}")

# 消費税2%pt減税のシミュレーション
simulator = jpfs.ImpulseResponseSimulator(model)
result = simulator.simulate_consumption_tax_cut(tax_cut=0.02, periods=40)

# 産出の応答を取得
y_response = result.get_response("y")
peak_period, peak_value = result.peak_response("y")
print(f"産出ピーク: {peak_value * 100:.2f}%（第{peak_period}期）")
```

## パラメータのカスタマイズ

```python
# 消費税率を変更
calibration = jpfs.JapanCalibration.create()
calibration = calibration.set_consumption_tax(0.08)  # 8%に変更

model = jpfs.DSGEModel(calibration.parameters)
```

## 次に読むドキュメント

- [CLIリファレンス](cli.md) — 全コマンドの詳細
- [Python API](python-api.md) — モデル・シミュレーション・政策分析のAPI
- [ベイズ推定ガイド](estimation.md) — MCMCによるパラメータ推定
- [MCP連携](mcp.md) — Claude Desktopとの統合
