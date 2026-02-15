# CLIリファレンス

全コマンドは `jpfs <command>` または `uv run jpfs <command>` で実行できる。

## simulate

政策ショックのインパルス応答をシミュレーションする。

```bash
jpfs simulate <policy_type> [OPTIONS]
```

### 政策タイプ

| タイプ | 説明 |
|--------|------|
| `consumption_tax` | 消費税ショック |
| `government_spending` | 政府支出ショック |
| `transfer` | 移転支出ショック |
| `monetary` | 金融政策ショック |
| `price_markup` | 価格マークアップショック |

### オプション

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--shock` | `-s` | 0.01 | ショックサイズ（例: -0.02 = 2%pt減税） |
| `--periods` | `-p` | 40 | シミュレーション期間（四半期） |
| `--graph` | `-g` | false | グラフを生成する |
| `--output` | `-o` | なし | 出力ディレクトリ |

### 実行例

```bash
# 消費税2%pt減税
jpfs simulate consumption_tax --shock -0.02 --periods 40

# 政府支出1%増加（グラフ付き）
jpfs simulate government_spending --shock 0.01 --periods 40 --graph

# 金融引き締め（25bp利上げ）
jpfs simulate monetary --shock 0.0025 --periods 40

# 価格マークアップショック
jpfs simulate price_markup --shock 0.01 --periods 40
```

出力は主要6変数（Y, C, I, π, r, B）のピーク応答をテーブル形式で表示する。

---

## multiplier

財政乗数を計算する。

```bash
jpfs multiplier <policy_type> [OPTIONS]
```

### 政策タイプ

`government_spending`、`consumption_tax`

### オプション

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--horizon` | `-h` | 40 | 計算期間（四半期） |

### 実行例

```bash
jpfs multiplier government_spending --horizon 8
```

出力される乗数:

| 乗数 | 説明 |
|------|------|
| impact | インパクト乗数（t=0） |
| peak | ピーク乗数 |
| cumulative_4q | 累積乗数（1年） |
| cumulative_8q | 累積乗数（2年） |
| cumulative_20q | 累積乗数（5年） |
| present_value | 現在価値乗数（β割引） |

---

## steady-state

定常状態の値を表示する。

```bash
jpfs steady-state
```

3つのテーブルを出力する:

1. **実物変数**: Y, C, I, K, N
2. **価格・金利**: W, r, R, π
3. **政府部門**: G, B, 税収, プライマリーバランス

---

## parameters

現在のキャリブレーションパラメータを表示する。

```bash
jpfs parameters
```

4部門のパラメータをテーブル形式で出力する:

1. **家計**: β, σ, φ, habit
2. **企業**: α, δ, θ
3. **政府**: τ_c, G/Y, B/Y
4. **中央銀行**: ρ_R, φ_π, φ_y

---

## estimate

ベイズ推定（Metropolis-Hastings MCMC）を実行する。

```bash
jpfs estimate [DATA_FILE] [OPTIONS]
```

### 引数

| 引数 | 必須 | 説明 |
|------|------|------|
| `DATA_FILE` | いいえ | 観測データCSVファイルのパス |

### オプション

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--draws` | `-d` | 100,000 | MCMCドロー数 |
| `--chains` | | 4 | チェーン数 |
| `--burnin` | | 50,000 | バーンイン期間 |
| `--thinning` | | 10 | シンニング間隔 |
| `--output` | `-o` | なし | 結果の出力ディレクトリ |
| `--synthetic` | | false | 合成データを使用する |
| `--synthetic-periods` | | 200 | 合成データの期間数 |

### 実行例

```bash
# 合成データで推定（動作確認用）
jpfs estimate --synthetic --draws 10000 --burnin 5000

# CSVデータで推定
jpfs estimate data/japan_quarterly.csv --output results/

# 高精度推定
jpfs estimate data/japan_quarterly.csv --draws 200000 --chains 4 --burnin 100000
```

出力:
- 収束状態と対数周辺尤度
- チェーン別の受容率
- パラメータ事後分布のサマリーテーブル（平均、中央値、標準偏差、90% HPD区間）
- `--output` 指定時: `posterior_samples.npy`, `mode.npy`, `summary.txt` を保存

詳細は [ベイズ推定ガイド](estimation.md) を参照。

---

## fetch-data

観測データを取得する（現在は合成データ生成）。

```bash
jpfs fetch-data [OPTIONS]
```

### オプション

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--output` | `-o` | `data/japan_quarterly.csv` | 出力CSVパス |
| `--periods` | `-p` | 200 | 合成データの期間数 |

### 実行例

```bash
jpfs fetch-data --output data/sample.csv --periods 100
```

生成される7変数: 実質GDP、消費、投資、インフレ率、実質賃金、雇用、名目短期金利

---

## report

最新のシミュレーション結果からMarkdownレポートを生成する。

```bash
jpfs report [OPTIONS]
```

### オプション

| オプション | 短縮 | デフォルト | 説明 |
|-----------|------|-----------|------|
| `--output` | `-o` | なし | 出力ファイルパス |

---

## mcp

MCPサーバーを起動する（Claude Desktop連携用）。

```bash
jpfs mcp
```

詳細は [MCP連携ガイド](mcp.md) を参照。

---

## version

バージョン情報を表示する。

```bash
jpfs version
```
