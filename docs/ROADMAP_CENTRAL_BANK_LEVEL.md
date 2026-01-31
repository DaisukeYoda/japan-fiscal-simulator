# 中央銀行レベルDSGEモデルへのロードマップ

本ドキュメントでは、現在の5方程式New Keynesianモデルを中央銀行レベル（Smets-Wouters級）のDSGEモデルへ拡張するための段階的ロードマップを示す。

## 目次

1. [現状分析](#現状分析)
2. [目標モデル](#目標モデル)
3. [フェーズ別実装計画](#フェーズ別実装計画)
4. [解法の選択](#解法の選択)
5. [検証とテスト](#検証とテスト)
6. [参考文献](#参考文献)

---

## 現状分析

### 現在のモデル構成

| 項目 | 現状 |
|------|------|
| コア方程式数 | 5 |
| 変数数 | 14（5コア + 9派生） |
| ショック数 | 5 |
| 解法 | 縮約形解法 |

### コア方程式

```
1. IS曲線:      y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y × g_t
2. Phillips曲線: π_t = β × E[π_{t+1}] + κ × y_t
3. Taylor則:    r_t = φ_π × π_t + φ_y × y_t + e_m,t
4. 政府支出:    g_t = ρ_g × g_{t-1} + e_g,t
5. 技術:        a_t = ρ_a × a_{t-1} + e_a,t
```

### 現在のアーキテクチャ

```
src/japan_fiscal_simulator/
├── core/
│   ├── equations/          # 方程式モジュール（拡張ポイント）
│   │   ├── base.py
│   │   ├── is_curve.py
│   │   ├── phillips_curve.py
│   │   ├── taylor_rule.py
│   │   └── fiscal_rule.py
│   ├── nk_model.py         # 3方程式NKモデル
│   ├── model.py            # 14変数拡張モデル
│   ├── solver.py           # Blanchard-Kahn解法（未使用）
│   └── simulation.py       # インパルス応答
└── parameters/
    ├── defaults.py         # パラメータ定義
    └── calibration.py      # 日本向けキャリブレーション
```

---

## 目標モデル

### Smets-Wouters (2007) 相当

中央銀行DSGEの事実上の標準モデルを目標とする。

| 項目 | 目標 |
|------|------|
| 方程式数 | 15-20 |
| 観測変数 | 7 |
| 構造ショック | 7 |
| 解法 | Blanchard-Kahn / Klein |

### 目標とする7つの構造ショック

1. **技術ショック** (Total Factor Productivity)
2. **リスクプレミアムショック** (Risk Premium)
3. **投資固有技術ショック** (Investment-Specific Technology)
4. **賃金マークアップショック** (Wage Markup)
5. **価格マークアップショック** (Price Markup)
6. **政府支出ショック** (Government Spending)
7. **金融政策ショック** (Monetary Policy)

### 目標とする7つの観測変数

1. 実質GDP
2. 消費
3. 投資
4. GDPデフレータ（インフレ率）
5. 実質賃金
6. 雇用
7. 名目短期金利

---

## フェーズ別実装計画

### Phase 1: 資本蓄積の内生化

**目標**: 投資と資本ストックを明示的にモデル化

**追加方程式**:

```
# 資本蓄積
k_t = (1-δ) × k_{t-1} + δ × i_t

# 投資のオイラー方程式（Tobin's Q）
q_t = β × E[(1-δ)q_{t+1} + r^k_{t+1}] - r_t

# 投資調整コスト
i_t = i_{t-1} + (1/S'') × q_t + e_i,t
```

**実装タスク**:

- [ ] `equations/capital.py` の作成
- [ ] `equations/investment.py` の作成
- [ ] 投資固有技術ショックの追加
- [ ] `model.py` への統合

**パラメータ追加**:

```python
@dataclass(frozen=True)
class InvestmentParameters:
    S_double_prime: float = 5.0  # 投資調整コスト曲率
    delta: float = 0.025         # 資本減耗率（既存）
```

---

### Phase 2: 労働市場の精緻化

**目標**: 賃金硬直性と労働供給を明示的にモデル化

**追加方程式**:

```
# 賃金のNKPC（Calvo型賃金硬直性）
w_t = β × E[w_{t+1}] + (1-θ_w)(1-β×θ_w)/θ_w × (mrs_t - w_t) + e_w,t

# 限界代替率
mrs_t = σ × c_t + φ × n_t

# 労働需要
n_t = y_t - w_t + (1-α) × k_t
```

**実装タスク**:

- [ ] `equations/wage_phillips.py` の作成
- [ ] `equations/labor_supply.py` の作成
- [ ] 賃金マークアップショックの追加
- [ ] 習慣形成の実装

**パラメータ追加**:

```python
@dataclass(frozen=True)
class LaborParameters:
    theta_w: float = 0.75    # Calvo賃金硬直性
    epsilon_w: float = 10.0  # 労働の代替弾力性
    iota_w: float = 0.5      # 賃金インデクセーション
```

---

### Phase 3: 価格設定の精緻化

**目標**: 価格インデクセーションと限界費用の明示化

**追加方程式**:

```
# 拡張Phillips曲線（インデクセーション付き）
π_t = (ι_p/(1+β×ι_p)) × π_{t-1} + (β/(1+β×ι_p)) × E[π_{t+1}]
    + κ × mc_t + e_p,t

# 実質限界費用
mc_t = α × r^k_t + (1-α) × w_t - a_t

# 生産関数
y_t = a_t + α × k_{t-1} + (1-α) × n_t
```

**実装タスク**:

- [ ] `equations/marginal_cost.py` の作成
- [ ] Phillips曲線のインデクセーション対応
- [ ] 価格マークアップショックの追加

---

### Phase 4: 完全なBlanchard-Kahn解法

**目標**: 行列ベースの一般的な解法への移行

**システム形式**:

```
A × E[x_{t+1}] = B × x_t + C × ε_t
```

ここで `x_t` は状態変数と制御変数のベクトル。

**実装タスク**:

- [ ] `solver.py` の完全実装
- [ ] 一般化固有値分解
- [ ] Schur分解による数値安定化
- [ ] BK条件の自動チェック

**解法の核心コード（擬似コード）**:

```python
def solve_blanchard_kahn(A: np.ndarray, B: np.ndarray) -> PolicyFunction:
    # 一般化Schur分解
    T, S, alpha, beta, Q, Z = scipy.linalg.ordqz(A, B, sort='ouc')

    # 安定/不安定部分空間の分離
    n_stable = np.sum(np.abs(beta) > np.abs(alpha))
    n_unstable = len(alpha) - n_stable

    # BK条件チェック
    if n_unstable != n_forward_looking:
        raise BlanchardKahnError(...)

    # 政策関数の構築
    Z_11 = Z[:n_state, :n_state]
    Z_21 = Z[n_state:, :n_state]
    P = Z_21 @ np.linalg.inv(Z_11)

    return PolicyFunction(P=P, ...)
```

---

### Phase 5: ベイズ推定

**目標**: 日本経済データによるパラメータ推定

**手法**: メトロポリス・ヘイスティングス法

**実装タスク**:

- [ ] 状態空間表現への変換
- [ ] カルマンフィルタによる尤度計算
- [ ] MCMCサンプラーの実装
- [ ] 事前分布の設定

**データ要件**:

| 変数 | データソース |
|------|-------------|
| 実質GDP | 内閣府SNA |
| 消費 | 内閣府SNA |
| 投資 | 内閣府SNA |
| インフレ率 | 総務省CPI |
| 賃金 | 厚労省毎月勤労統計 |
| 雇用 | 総務省労働力調査 |
| 金利 | 日本銀行 |

---

### Phase 6: 日本固有の拡張

**目標**: 日本経済の特殊性を反映

**追加機能**:

1. **ゼロ金利制約（ZLB）**
   ```
   R_t = max(R_lower_bound, Taylor則)
   ```

2. **高債務経済**
   - 財政持続可能性条件
   - リカーディアン/非リカーディアン家計の混合

3. **人口動態**
   - OLG要素の導入
   - 労働力人口減少のトレンド

4. **開放経済**
   - 為替レート
   - 輸出入

---

## 解法の選択

### 選択肢の比較

| 手法 | 利点 | 欠点 | 推奨用途 |
|------|------|------|----------|
| **縮約形（現状）** | シンプル、解析解 | 小規模モデルのみ | Phase 1まで |
| **Blanchard-Kahn** | 標準的、理解しやすい | 大規模で不安定 | Phase 2-3 |
| **Klein法** | 数値安定性高い | 実装複雑 | Phase 4以降 |
| **Sims法** | 特異行列対応 | 計算コスト高 | 特殊ケース |

### 推奨戦略

```
Phase 1-2: 縮約形解法を維持しつつ方程式を追加
     ↓
Phase 3:   Blanchard-Kahn解法への移行を検討
     ↓
Phase 4:   Klein法（QZ分解）を本格実装
     ↓
Phase 5-6: Dynare連携またはDSGE.jl活用
```

### 外部ツール連携

**Dynare連携のメリット**:
- 解法が検証済み
- ベイズ推定が組み込み
- IRF、分散分解が自動

**連携方法**:

```python
# Python から Dynare を呼び出す
def run_dynare_model(mod_file: str, params: dict) -> dict:
    # .mod ファイルを生成
    generate_mod_file(mod_file, params)

    # Dynare実行（MATLAB/Octave経由）
    subprocess.run(['octave', '--eval', f"dynare {mod_file}"])

    # 結果を読み込み
    return load_dynare_results(mod_file)
```

---

## 検証とテスト

### 各フェーズの検証項目

| フェーズ | 検証項目 |
|----------|----------|
| Phase 1 | 定常状態の収束、資本の長期水準 |
| Phase 2 | 賃金IRFの形状、労働市場の調整速度 |
| Phase 3 | BK条件の成立、固有値の分布 |
| Phase 4 | Dynareとの結果比較 |
| Phase 5 | 事後分布の収束診断、尤度の改善 |

### ベンチマークテスト

```python
def test_against_smets_wouters():
    """Smets-Wouters (2007) の再現テスト"""
    # 公開されているパラメータで初期化
    params = SmetsWoutersParameters()
    model = DSGEModel(params)

    # 政府支出ショックのIRF
    irf = model.impulse_response("e_g", periods=40)

    # 論文の Figure 2 と比較
    assert_irf_shape(irf["y"], expected_peak=1, expected_decay=0.9)
```

### 数値安定性テスト

```python
def test_numerical_stability():
    """特異点近傍での安定性"""
    # Taylor原則境界
    params = DefaultParameters()
    params = params.with_updates(
        central_bank=CentralBankParameters(phi_pi=1.0)  # 境界値
    )

    model = DSGEModel(params)

    # 解が存在することを確認
    assert model.policy_function.bk_satisfied
```

---

## 参考文献

### 必読論文

1. **Smets, F., & Wouters, R. (2007)**. "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach." *American Economic Review*, 97(3), 586-606.

2. **Blanchard, O. J., & Kahn, C. M. (1980)**. "The Solution of Linear Difference Models under Rational Expectations." *Econometrica*, 48(5), 1305-1311.

3. **Klein, P. (2000)**. "Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model." *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.

### 日本経済向け

4. **Sugo, T., & Ueda, K. (2008)**. "Estimating a Dynamic Stochastic General Equilibrium Model for Japan." *Journal of the Japanese and International Economies*, 22(4), 476-502.

5. **日本銀行 (2019)**. "ハイブリッド型日本経済モデル Q-JEM: 2019年バージョン." Working Paper Series No.19-E-7.

### 実装ガイド

6. **Fernández-Villaverde, J., et al. (2016)**. "Solution and Estimation Methods for DSGE Models." *Handbook of Macroeconomics*, Vol. 2, 527-724.

7. **Dynare Team**. *Dynare Reference Manual*. https://www.dynare.org/manual/

---

## マイルストーン

| フェーズ | 方程式数 | 主要成果物 |
|----------|----------|-----------|
| 現状 | 5 | 基本的なIRF分析 |
| Phase 1 | 8 | 資本・投資ダイナミクス |
| Phase 2 | 11 | 労働市場分析 |
| Phase 3 | 14 | 価格・賃金の相互作用 |
| Phase 4 | 14 | 一般的BK解法 |
| Phase 5 | 14 | ベイズ推定済みモデル |
| Phase 6 | 18+ | 日本特化型モデル |

---

*最終更新: 2026-02-01*
