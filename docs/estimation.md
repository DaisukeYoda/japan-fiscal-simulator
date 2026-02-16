# ベイズ推定ガイド

本モジュールは、DSGEモデルの構造パラメータをMetropolis-Hastings MCMCで推定する。
カルマンフィルタにより状態空間モデルの尤度を計算し、事前分布と組み合わせて事後分布をサンプリングする。

## 概要

推定のフロー:

```
観測データ → 状態空間表現 → カルマンフィルタ（尤度計算）
                                      ↓
事前分布 + 尤度 → 事後分布 → MH-MCMCサンプリング
                                      ↓
                             収束診断 → 推定結果
```

## CLIで推定を実行する

### 合成データでの動作確認

実データがなくても、モデルから生成した合成データで推定を試せる:

```bash
# 少ないドロー数で素早く確認
jpfs estimate --synthetic --draws 10000 --burnin 5000

# デフォルト設定（100,000ドロー）
jpfs estimate --synthetic
```

### 観測データでの推定

CSVファイルを用意し、7つの観測変数を列として含める:

| 列名 | 変数 |
|------|------|
| gdp | 実質GDP成長率（対数偏差） |
| consumption | 消費成長率（対数偏差） |
| investment | 投資成長率（対数偏差） |
| deflator | GDPデフレータ（インフレ率） |
| wage | 実質賃金成長率（対数偏差） |
| employment | 雇用（対数偏差） |
| rate | 名目短期金利 |

先頭に `date` 列を含めること。

```bash
jpfs estimate data/japan_quarterly.csv --output results/
```

### 推定オプション

```bash
jpfs estimate data/japan_quarterly.csv \
  --draws 200000 \      # MCMCドロー数
  --chains 4 \          # チェーン数
  --burnin 100000 \     # バーンイン
  --thinning 10 \       # シンニング間隔
  --output results/     # 出力先
```

出力ファイル（`--output` 指定時）:

| ファイル | 内容 |
|---------|------|
| `posterior_samples.npy` | 事後分布サンプル |
| `mode.npy` | 事後モード |
| `summary.txt` | パラメータサマリー |

---

## Python APIで推定を実行する

### 基本的な流れ

```python
import numpy as np
from japan_fiscal_simulator.estimation import (
    MCMCConfig,
    MetropolisHastings,
    ParameterMapping,
    PriorConfig,
    EstimationResult,
    make_log_posterior,
)

# 1. パラメータ写像と事前分布
mapping = ParameterMapping()
prior = PriorConfig.smets_wouters_japan()

# 2. 観測データ（T期間 × 7変数）
data_y = np.loadtxt("data/japan_quarterly.csv", delimiter=",", skiprows=1)

# 3. 事後分布関数の構築
log_posterior_fn = make_log_posterior(mapping, prior, data_y)

# 4. MCMC設定
config = MCMCConfig(
    n_chains=4,
    n_draws=100_000,
    n_burnin=50_000,
    thinning=10,
)

# 5. サンプラーの作成と実行
mh = MetropolisHastings(
    log_posterior_fn=log_posterior_fn,
    n_params=mapping.n_params,
    config=config,
    parameter_names=mapping.names,
    bounds=mapping.bounds(),
)

mcmc_result = mh.run(theta0=mapping.defaults())
```

### MCMCConfig

| パラメータ | デフォルト | 説明 |
|-----------|-----------|------|
| `n_chains` | 4 | 並列チェーン数 |
| `n_draws` | 100,000 | 総ドロー数 |
| `n_burnin` | 50,000 | バーンイン（破棄）期間 |
| `thinning` | 10 | シンニング間隔 |
| `target_acceptance` | 0.234 | 目標受容率 |
| `adaptive_interval` | 100 | 適応的スケーリングの更新間隔 |
| `mode_search_max_iter` | 500 | 事後モード探索の最大反復回数 |

### MCMCResult

```python
result = mh.run(theta0=mapping.defaults())

result.chains              # (n_chains, n_kept, n_params) 事後サンプル
result.acceptance_rates    # (n_chains,) 各チェーンの受容率
result.log_posteriors      # (n_chains, n_kept) 対数事後確率
result.mode                # (n_params,) 事後モード
result.mode_hessian        # (n_params, n_params) モードでのヘッセ行列逆行列
result.mode_log_posterior  # モードでの対数事後確率
result.parameter_names     # パラメータ名リスト
```

### ParameterMapping

モデルパラメータ（dataclass）と推定パラメータ（ベクトル）の相互変換を担う。

```python
mapping = ParameterMapping()

mapping.n_params    # 推定パラメータ数
mapping.names       # パラメータ名リスト

theta = mapping.defaults()    # デフォルト値ベクトル
bounds = mapping.bounds()     # [(下限, 上限), ...] パラメータ境界
```

### PriorConfig

構造パラメータの事前分布を定義する。

```python
# Smets-Wouters型の日本向けプリセット
prior = PriorConfig.smets_wouters_japan()

# 事前分布の確認
prior.priors  # list[ParameterPrior]
# 各 ParameterPrior は name, dist_type, mean, std, lower_bound, upper_bound を持つ
# 例: ParameterPrior(name='sigma', dist_type='gamma', mean=1.5, std=0.37, ...)
```

### EstimationResult

推定結果の要約を提供する。

```python
from japan_fiscal_simulator.estimation.results import build_estimation_result

est = build_estimation_result(
    chains=mcmc_result.chains,
    acceptance_rates=mcmc_result.acceptance_rates,
    mode=mcmc_result.mode,
    mode_log_posterior=mcmc_result.mode_log_posterior,
    hessian=mcmc_result.mode_hessian,
    prior_config=prior,
    mapping=mapping,
    n_burnin=config.n_burnin,
)

# 全体
est.log_marginal_likelihood    # 対数周辺尤度（Laplace近似）
est.diagnostics                # 収束診断
est.n_chains                   # チェーン数
est.n_draws                    # ドロー数

# パラメータ別サマリー
summary = est.get_summary("sigma")
summary.mean         # 事後平均
summary.median       # 事後中央値
summary.std          # 事後標準偏差
summary.hpd_lower    # 90% HPD下限
summary.hpd_upper    # 90% HPD上限
summary.prior_mean   # 事前分布の平均
summary.prior_std    # 事前分布の標準偏差

# テーブル出力
print(est.summary_table())
```

---

## 収束診断

### Gelman-Rubin統計量（R-hat）

複数チェーンの分散比からMCMCの収束を判定する。

| R-hat | 判定 |
|-------|------|
| < 1.1 | 収束 |
| 1.1 - 1.2 | やや不十分 |
| > 1.2 | 未収束 — ドロー数を増やすか事前分布を見直す |

### 受容率

MHアルゴリズムの受容率は、概ね20〜30%が適正とされる（デフォルト目標: 23.4%）。

| 受容率 | 状態 |
|--------|------|
| < 10% | 提案分布が広すぎる |
| 20-30% | 適正 |
| > 50% | 提案分布が狭すぎる |

適応的スケーリングにより自動調整されるが、収束が不十分な場合は `n_draws` を増やすか `adaptive_interval` を調整する。

---

## 合成データでの試行

推定パイプラインの動作確認には合成データが便利:

```python
from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator

# モデルから合成データを生成
calibration = jpfs.JapanCalibration.create()
generator = SyntheticDataGenerator()
synthetic_data = generator.generate(calibration.parameters, n_periods=200)

# あとは通常の推定フローと同じ
log_posterior_fn = make_log_posterior(mapping, prior, synthetic_data.observations)
mh = MetropolisHastings(log_posterior_fn, mapping.n_params, config)
result = mh.run(theta0=mapping.defaults())
```

CLIでは `--synthetic` フラグで同等のことができる:

```bash
jpfs estimate --synthetic --synthetic-periods 200
```
