# Python API

```python
import japan_fiscal_simulator as jpfs
```

## 目次

1. [モデル（DSGEModel）](#モデルdsgemodel)
2. [シミュレーション](#シミュレーション)
3. [財政乗数](#財政乗数)
4. [キャリブレーション](#キャリブレーション)
5. [政策分析](#政策分析)
6. [変数・ショック一覧](#変数ショック一覧)

---

## モデル（DSGEModel）

### 初期化

```python
calibration = jpfs.JapanCalibration.create()
model = jpfs.DSGEModel(calibration.parameters)
```

### 定常状態

`steady_state` プロパティで定常状態を取得する（初回アクセス時に計算し、キャッシュする）。

```python
ss = model.steady_state

# 実物変数
ss.output          # 産出
ss.consumption     # 消費
ss.investment      # 投資
ss.capital         # 資本ストック
ss.labor           # 雇用

# 価格・金利
ss.real_wage              # 実質賃金
ss.real_interest_rate     # 実質金利
ss.nominal_interest_rate  # 名目金利
ss.inflation              # インフレ率

# 政府部門
ss.government_spending  # 政府支出
ss.government_debt      # 政府債務
ss.tax_revenue          # 税収
ss.primary_balance      # プライマリーバランス
```

### 政策関数

BK解法の結果を取得する。

```python
pf = model.policy_function

pf.P               # 状態遷移行列（16×16）
pf.Q               # ショック応答行列（16×7）
pf.bk_satisfied    # BK条件の成否
pf.n_stable        # 安定固有値の数
pf.n_unstable      # 不安定固有値の数
pf.eigenvalues     # 固有値
```

### キャッシュ

パラメータ変更時にはキャッシュを無効化する。

```python
model.invalidate_cache()
```

### 変数インデックス

```python
idx = model.get_variable_index("y")    # 0
name = model.get_variable_name(0)      # "y"
```

---

## シミュレーション

### ImpulseResponseSimulator

```python
simulator = jpfs.ImpulseResponseSimulator(model)
```

#### 汎用メソッド

```python
result = simulator.simulate(
    shock_name="e_g",    # ショック名（e_a, e_g, e_m, e_tau, e_risk, e_i, e_p）
    shock_size=0.01,     # ショックサイズ
    periods=40           # 期間（四半期）
)
```

#### 便利メソッド

```python
# 消費税減税
result = simulator.simulate_consumption_tax_cut(tax_cut=0.02, periods=40)

# 政府支出増加
result = simulator.simulate_government_spending(spending_increase=0.01, periods=40)

# 金融政策ショック
result = simulator.simulate_monetary_shock(rate_change=0.0025, periods=40)

# 技術ショック
result = simulator.simulate_technology_shock(productivity_increase=0.01, periods=40)
```

### ImpulseResponseResult

```python
# 特定変数の時系列を取得
y_response = result.get_response("y")     # np.ndarray

# ピーク応答（期、値）
period, value = result.peak_response("y")

# 累積応答
cum = result.cumulative_response("y", horizon=8)  # 8四半期の累積

# メタ情報
result.periods       # データ長（t=0を含むため periods引数 + 1）
result.shock_name    # ショック名
result.shock_size    # ショックサイズ
result.responses     # dict[str, np.ndarray] 全変数の応答
```

### 使用例

```python
import japan_fiscal_simulator as jpfs

calibration = jpfs.JapanCalibration.create()
model = jpfs.DSGEModel(calibration.parameters)
simulator = jpfs.ImpulseResponseSimulator(model)

# 政府支出ショック
result = simulator.simulate_government_spending(spending_increase=0.01, periods=40)

# 各変数の応答を取得
for var in ["y", "c", "i", "pi", "r"]:
    period, peak = result.peak_response(var)
    print(f"{var}: ピーク {peak*100:.3f}%（第{period}期）")
```

---

## 財政乗数

### FiscalMultiplierCalculator

```python
from japan_fiscal_simulator.core.simulation import FiscalMultiplierCalculator

calc = FiscalMultiplierCalculator(model)

# 政府支出乗数
result = calc.compute_spending_multiplier(horizon=40)

# 消費税乗数
result = calc.compute_tax_multiplier(horizon=40)
```

### FiscalMultiplierResult

```python
result.impact           # インパクト乗数（t=0）
result.peak             # ピーク乗数
result.peak_period      # ピーク期
result.cumulative_4q    # 累積乗数（1年）
result.cumulative_8q    # 累積乗数（2年）
result.cumulative_20q   # 累積乗数（5年）
result.present_value    # 現在価値乗数（β割引）
```

---

## キャリブレーション

### JapanCalibration

日本経済向けのプリセットパラメータを提供する。

```python
calibration = jpfs.JapanCalibration.create()

# パラメータの調整
calibration = calibration.set_consumption_tax(0.08)              # 消費税率を8%に
calibration = calibration.set_government_spending_ratio(0.22)    # G/Y比を22%に

# DefaultParameters を取得
params = calibration.parameters
model = jpfs.DSGEModel(params)
```

### デフォルト値（主要パラメータ）

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| β | 0.999 | 割引率（低金利環境） |
| σ | 1.5 | 相対的リスク回避度 |
| φ | 2.0 | 労働供給の逆Frisch弾力性 |
| α | 0.33 | 資本分配率 |
| δ | 0.025 | 資本減耗率 |
| θ | 0.75 | Calvo価格硬直性 |
| τ_c | 0.10 | 消費税率（10%） |
| B/Y | 2.00 | 政府債務/GDP比率 |
| ρ_R | 0.85 | 金利平滑化 |
| φ_π | 1.5 | Taylor則インフレ反応 |
| φ_y | 0.125 | Taylor則産出反応 |

---

## 政策分析

政策分析モジュールは高レベルのシナリオベース分析を提供する。

### 消費税政策

```python
from japan_fiscal_simulator.policies.consumption_tax import (
    ConsumptionTaxPolicy,
    SCENARIO_TAX_CUT_2PCT,
    SCENARIO_TAX_CUT_5PCT,
    SCENARIO_TAX_INCREASE_2PCT,
)

policy = ConsumptionTaxPolicy(model)

# プリセットシナリオ
analysis = policy.analyze(SCENARIO_TAX_CUT_2PCT)

# カスタムシナリオ
scenario = ConsumptionTaxPolicy.create_reduction_scenario(
    reduction_rate=0.03,
    periods=40
)
analysis = policy.analyze(scenario)

analysis.output_effect_peak       # 産出ピーク効果
analysis.consumption_effect_peak  # 消費ピーク効果
analysis.revenue_impact           # 税収への影響
analysis.welfare_effect           # 厚生効果（消費等価変分、近似）
```

### 社会保障政策

```python
from japan_fiscal_simulator.policies.social_security import (
    SocialSecurityPolicy,
    SCENARIO_TRANSFER_INCREASE,
    SCENARIO_PENSION_CUT,
)

policy = SocialSecurityPolicy(model)
analysis = policy.analyze(SCENARIO_TRANSFER_INCREASE)

analysis.output_effect_peak       # 産出ピーク効果
analysis.consumption_effect_peak  # 消費ピーク効果
analysis.debt_impact              # 長期債務への影響
analysis.distributional_note      # 分布効果に関する注記
```

### 補助金政策

```python
from japan_fiscal_simulator.policies.subsidies import (
    SubsidyPolicy,
    SCENARIO_INVESTMENT_SUBSIDY,
    SCENARIO_EMPLOYMENT_SUBSIDY,
    SCENARIO_GREEN_SUBSIDY,
)

policy = SubsidyPolicy(model)
analysis = policy.analyze(SCENARIO_INVESTMENT_SUBSIDY)

analysis.output_effect_peak       # 産出ピーク効果
analysis.investment_effect_peak   # 投資ピーク効果
analysis.fiscal_cost              # 財政コスト（累積政府支出）
analysis.crowding_out_ratio       # クラウディングアウト率
```

---

## 変数・ショック一覧

### 状態・制御変数（16変数）

| インデックス | 名前 | 説明 | 分類 |
|-------------|------|------|------|
| 0 | `y` | 産出 | 制御 |
| 1 | `c` | 消費 | 制御 |
| 2 | `i` | 投資 | 先決 |
| 3 | `n` | 雇用 | 制御 |
| 4 | `k` | 資本ストック | 先決 |
| 5 | `pi` | インフレ率 | 制御 |
| 6 | `r` | 実質金利 | 制御 |
| 7 | `R` | 名目金利 | 制御 |
| 8 | `w` | 実質賃金 | 制御 |
| 9 | `mc` | 限界費用 | 制御 |
| 10 | `g` | 政府支出 | 先決 |
| 11 | `b` | 政府債務 | 先決 |
| 12 | `tau_c` | 消費税率 | 先決 |
| 13 | `a` | 技術 | 先決 |
| 14 | `q` | Tobin's Q | 制御 |
| 15 | `rk` | 資本収益率 | 制御 |

### 構造ショック（7種類）

| ショック | 説明 |
|---------|------|
| `e_a` | 技術ショック（TFP） |
| `e_g` | 政府支出ショック |
| `e_m` | 金融政策ショック |
| `e_tau` | 消費税ショック |
| `e_risk` | リスクプレミアムショック |
| `e_i` | 投資固有技術ショック |
| `e_p` | 価格マークアップショック |
