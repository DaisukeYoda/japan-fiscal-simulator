# 中央銀行レベルDSGEモデルへのロードマップ

本ドキュメントでは、5方程式New Keynesianモデルを中央銀行レベル（Smets-Wouters級）のDSGEモデルへ拡張するためのロードマップを示す。Phase 1〜5は完了済みであり、残るはPhase 6（日本固有の拡張）のみとなっている。

## 目次

1. [初期状態](#初期状態phase-1-開始前)
2. [現在の到達点](#現在の到達点phase-5-完了)
3. [目標モデル](#目標モデル)
4. [フェーズ別実装計画](#フェーズ別実装計画)
5. [解法の選択](#解法の選択)
6. [検証とテスト](#検証とテスト)
7. [参考文献](#参考文献)

---

## 初期状態（Phase 1 開始前）

### 当初のモデル構成

| 項目 | 初期状態 |
|------|----------|
| コア方程式数 | 5 |
| 変数数 | 14（5コア + 9派生） |
| ショック数 | 5 |
| 解法 | 縮約形解法 |

### 当初のコア方程式

```
1. IS曲線:      y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y × g_t
2. Phillips曲線: π_t = β × E[π_{t+1}] + κ × y_t
3. Taylor則:    r_t = φ_π × π_t + φ_y × y_t + e_m,t
4. 政府支出:    g_t = ρ_g × g_{t-1} + e_g,t
5. 技術:        a_t = ρ_a × a_{t-1} + e_a,t
```

---

## 現在の到達点（Phase 5 完了）

### モデル構成

| 項目 | 現状 |
|------|------|
| コア方程式数 | 14 |
| 変数数 | 16（y, c, i, n, k, π, r, R, w, mc, g, b, τ_c, a, q, r^k） |
| 構造ショック数 | 7（e_a, e_g, e_m, e_tau, e_risk, e_i, e_p） |
| 状態変数 | 5 |
| 制御変数 | 9 |
| 解法 | QZ分解ベースのBlanchard-Kahn/Klein法 |
| 推定手法 | Metropolis-Hastings MCMC（カルマンフィルタ尤度） |

### 現在のアーキテクチャ

```
src/japan_fiscal_simulator/
├── core/
│   ├── equations/              # 14方程式モジュール
│   │   ├── base.py
│   │   ├── is_curve.py         # IS曲線
│   │   ├── phillips_curve.py   # Phillips曲線（インデクセーション付き）
│   │   ├── wage_phillips.py    # 賃金NKPC（Calvo型）
│   │   ├── taylor_rule.py      # Taylor則
│   │   ├── marginal_cost.py    # 実質限界費用
│   │   ├── capital.py          # 資本蓄積
│   │   ├── investment.py       # 投資オイラー方程式
│   │   ├── labor_demand.py     # 労働需要
│   │   ├── mrs.py              # 限界代替率
│   │   ├── resource_constraint.py  # 資源制約
│   │   └── fiscal_rule.py      # 財政ルール
│   ├── nk_model.py             # 14方程式NKモデル（BK解法）
│   ├── model.py                # 16変数DSGEモデル
│   ├── equation_system.py      # 方程式システム（A/B/C/D行列構築）
│   ├── solver.py               # QZ分解BKソルバー
│   ├── linear_solver.py        # 互換ラッパー
│   ├── steady_state.py         # 定常状態ソルバー
│   ├── simulation.py           # IRF・乗数計算
│   ├── derived_coefficients.py # 係数計算ヘルパー
│   └── exceptions.py           # カスタム例外
├── estimation/                 # ベイズ推定モジュール
│   ├── mcmc.py                 # Random Walk MH-MCMCサンプラー
│   ├── state_space.py          # 状態空間表現
│   ├── kalman_filter.py        # カルマンフィルタ（尤度計算）
│   ├── priors.py               # 事前分布定義
│   ├── parameter_mapping.py    # パラメータ写像
│   ├── data_loader.py          # データローダー
│   ├── data_fetcher.py         # データ取得ユーティリティ
│   ├── diagnostics.py          # 収束診断（Gelman-Rubin等）
│   └── results.py              # 推定結果管理
├── parameters/
│   ├── defaults.py             # パラメータ定義
│   ├── calibration.py          # 日本向けキャリブレーション
│   └── constants.py            # 定数定義
├── policies/                   # 政策シミュレーション
├── output/                     # グラフ・レポート出力
├── cli/                        # CLIインターフェース
└── mcp/                        # MCP（Claude Desktop連携）
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

### Phase 1: 資本蓄積の内生化 [完了]

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

- [x] `equations/capital.py` の作成
- [x] `equations/investment.py` の作成
- [x] 投資固有技術ショックの追加
- [x] `model.py` への統合

---

### Phase 2: 労働市場の精緻化 [完了]

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

- [x] `equations/wage_phillips.py` の作成
- [x] `equations/labor_demand.py` / `equations/mrs.py` の作成
- [x] 賃金マークアップショックの追加
- [x] 習慣形成の実装

---

### Phase 3: 価格設定の精緻化 [完了]

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

- [x] `equations/marginal_cost.py` の作成
- [x] Phillips曲線のインデクセーション対応
- [x] 価格マークアップショックの追加

---

### Phase 4: 完全なBlanchard-Kahn解法 [完了]

**目標**: 行列ベースの一般的な解法への移行

**システム形式**:

```
A × E[x_{t+1}] = B × x_t + C × ε_t
```

ここで `x_t` は状態変数と制御変数のベクトル。

**実装タスク**:

- [x] `solver.py` の完全実装
- [x] 一般化固有値分解（QZ分解）
- [x] Schur分解による数値安定化
- [x] BK条件の自動チェック
- [x] 14方程式（state=5, control=9）への拡張
- [x] `linear_solver.py` の互換ラッパー化

**実装メモ（2026-02-03）**:

- `EquationSystem` を可変次元化し、`A/B/C/D` を14x14/14x6で構築
- `NewKeynesianModel` を縮約形から構造解へ移行
- `MRS` / `Resource Constraint` を追加し、賃金NKPCは `mrs_t` 参照へ統一
- `Q,S` は係数一致の連立方程式から算出（近似式を廃止）

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

### Phase 5: ベイズ推定 [完了]

**目標**: 日本経済データによるパラメータ推定

**手法**: メトロポリス・ヘイスティングス法

**実装タスク**:

- [x] 状態空間表現への変換（`state_space.py`）
- [x] カルマンフィルタによる尤度計算（`kalman_filter.py`）
- [x] MCMCサンプラーの実装（`mcmc.py` — Random Walk Metropolis-Hastings）
- [x] 事前分布の設定（`priors.py`）
- [x] パラメータ写像（`parameter_mapping.py`）
- [x] データローダー（`data_loader.py` / `data_fetcher.py`）
- [x] 収束診断（`diagnostics.py` — Gelman-Rubin統計量等）
- [x] 推定結果管理（`results.py`）

**実装メモ（2026-02-12）**:

- `estimation/` パッケージとして独立モジュール化（10ファイル）
- MCMCは4チェーン並列、100,000ドロー、50,000バーンイン、シンニング10がデフォルト
- 事後モード探索 → 適応的サンプリングの2段階推定
- Dynare非依存のPure Python実装

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

### 実際の採用戦略

```
Phase 1-3: 方程式を段階的に追加（5 → 14方程式）
     ↓
Phase 4:   QZ分解ベースのBlanchard-Kahn/Klein法を実装（scipy.linalg.ordqz）
     ↓
Phase 5:   Pure PythonでMH-MCMC + カルマンフィルタを実装（Dynare非依存）
     ↓
Phase 6:   日本固有の拡張（今後）
```

Dynare連携は採用せず、Pure Pythonによる自己完結型の実装を選択した。
これによりMCP/CLIとの統合が容易になり、依存関係（MATLAB/Octave）を排除できた。

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

| フェーズ | 方程式数 | 主要成果物 | 状態 |
|----------|----------|-----------|------|
| 初期 | 5 | 基本的なIRF分析 | 完了 |
| Phase 1 | 8 | 資本蓄積・投資ダイナミクス | 完了 |
| Phase 2 | 11 | 賃金硬直性・労働市場 | 完了 |
| Phase 3 | 14 | 価格インデクセーション・限界費用 | 完了 |
| Phase 4 | 14 | QZ分解BK解法（14x14システム） | 完了 |
| Phase 5 | 14 | MH-MCMCベイズ推定 | 完了 |
| Phase 6 | 18+ | 日本固有拡張（ZLB, 高債務, 人口動態, 開放経済） | 未着手 |

---

*最終更新: 2026-02-12*
