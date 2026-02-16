# 数理モデル仕様書

Japan Fiscal Simulator (jpfs) — New Keynesian DSGEモデル

---

## 目次

1. [モデル概要](#1-モデル概要)
2. [変数体系と記法](#2-変数体系と記法)
3. [定常状態](#3-定常状態)
4. [対数線形化方程式](#4-対数線形化方程式)
5. [システム行列表現](#5-システム行列表現)
6. [Blanchard-Kahn解法](#6-blanchard-kahn解法)
7. [ベイズ推定](#7-ベイズ推定)
8. [パラメータ一覧](#8-パラメータ一覧)
9. [参考文献](#9-参考文献)

---

## 1. モデル概要

本モデルは Smets & Wouters (2007) 型の中規模 New Keynesian DSGE モデルであり、日本経済の特徴（低金利・高債務・消費税）を反映したキャリブレーションを持つ。

| 項目 | 仕様 |
|------|------|
| コア方程式数 | 14 |
| 内生変数数 | 16 |
| 先決（状態）変数 | 5 |
| 制御（ジャンプ）変数 | 9 |
| 構造ショック数 | 6（コアモデル）/ 8（拡張モデル） |
| 観測変数数 | 7 |
| 解法 | QZ分解ベース Blanchard-Kahn法 |
| 推定手法 | Random Walk Metropolis-Hastings MCMC |

### 経済主体

- **家計**: 習慣形成付き消費、Calvo型賃金設定
- **企業**: Cobb-Douglas生産、Calvo型価格設定（インデクセーション付き）
- **政府**: 財政ルール、消費税・所得税・資本税
- **中央銀行**: Taylor則
- **金融**: 外部資金プレミアム（BGG簡略型）

---

## 2. 変数体系と記法

### 2.1 内生変数

ハット記法 $\hat{x}_t$ は定常状態からの対数偏差を表す。以下では簡潔のため $x_t$ と記す。

#### コアモデル変数（14方程式体系）

| 記号 | 変数名 | 区分 |
|------|--------|------|
| $g_t$ | 政府支出 | 先決変数 |
| $a_t$ | 技術水準 | 先決変数 |
| $k_t$ | 資本ストック | 先決変数 |
| $i_t$ | 投資 | 先決変数 |
| $w_t$ | 実質賃金 | 先決変数 |
| $y_t$ | 産出 | 制御変数 |
| $\pi_t$ | インフレ率 | 制御変数 |
| $r_t$ | 実質金利 | 制御変数 |
| $q_t$ | Tobin の Q | 制御変数 |
| $r^k_t$ | 資本レンタル率 | 制御変数 |
| $n_t$ | 労働投入 | 制御変数 |
| $c_t$ | 消費 | 制御変数 |
| $mc_t$ | 実質限界費用 | 制御変数 |
| $mrs_t$ | 限界代替率 | 制御変数 |

#### 拡張モデル変数（16変数体系、`model.py`）

上記に加え、以下の変数がモデルラッパーで追加管理される。

| 記号 | 変数名 | インデックス |
|------|--------|------------|
| $R_t$ | 名目金利 | 7 |
| $b_t$ | 政府債務 | 11 |
| $\tau^c_t$ | 消費税率 | 12 |

### 2.2 構造ショック

```math
\varepsilon_t = (e_{g,t},\ e_{a,t},\ e_{m,t},\ e_{i,t},\ e_{w,t},\ e_{p,t})^\top
```

| 記号 | ショック名 | 持続性 $\rho$ | 標準偏差 $\sigma$ |
|------|-----------|-------------|-----------------|
| $e_{g,t}$ | 政府支出 | 0.90 | 0.01 |
| $e_{a,t}$ | 技術（TFP） | 0.90 | 0.01 |
| $e_{m,t}$ | 金融政策 | 0.50 | 0.0025 |
| $e_{i,t}$ | 投資固有技術 | 0.70 | 0.01 |
| $e_{w,t}$ | 賃金マークアップ | 0.90 | 0.01 |
| $e_{p,t}$ | 価格マークアップ | 0.90 | 0.01 |

追加ショック（拡張モデルのみ）：

| 記号 | ショック名 | 持続性 | 標準偏差 |
|------|-----------|--------|---------|
| $e_{\tau,t}$ | 消費税 | 0.95 | 0.005 |
| $e_{risk,t}$ | リスクプレミアム | 0.75 | 0.01 |

---

## 3. 定常状態

定常状態は閉形式解により求める。$\bar{x}$ は変数 $x$ の定常状態値を表す。

### 3.1 解析解

**(1) 実質金利**（オイラー方程式 $\beta(1+\bar{r})=1$ より）

```math
\bar{r} = \frac{1}{\beta} - 1
```

**(2) インフレ率**

```math
\bar{\pi} = \pi^{*} \quad (\text{中央銀行目標})
```

**(3) 名目金利**（Fisher方程式）

```math
\bar{R} = \bar{r} + \bar{\pi}
```

**(4) マークアップと限界費用**

```math
\mu = \frac{\epsilon}{\epsilon - 1}, \qquad \overline{mc} = \frac{1}{\mu} = \frac{\epsilon - 1}{\epsilon}
```

**(5) 資本の限界生産性と資本・産出比率**

```math
MPK = \bar{r} + \delta, \qquad \frac{\bar{K}}{\bar{Y}} = \frac{\alpha}{MPK}
```

**(6) 投資・消費の産出比率**（資源制約 $\bar{Y} = \bar{C} + \bar{I} + \bar{G}$ より）

```math
\frac{\bar{I}}{\bar{Y}} = \delta \cdot \frac{\bar{K}}{\bar{Y}}, \qquad \frac{\bar{C}}{\bar{Y}} = 1 - \frac{\bar{I}}{\bar{Y}} - \frac{\bar{G}}{\bar{Y}}
```

**(7) 水準値**（$\bar{Y}=1$ で正規化）

```math
\bar{Y} = 1, \quad \bar{C} = \frac{\bar{C}}{\bar{Y}}, \quad \bar{I} = \frac{\bar{I}}{\bar{Y}}, \quad \bar{K} = \frac{\bar{K}}{\bar{Y}}, \quad \bar{G} = g_y \cdot \bar{Y}
```

**(8) 労働**（Cobb-Douglas生産関数 $Y = K^\alpha N^{1-\alpha}$ より）

```math
\bar{N} = \left(\frac{\bar{Y}}{\bar{K}^\alpha}\right)^{1/(1-\alpha)}
```

**(9) 実質賃金**（労働の限界生産性条件）

```math
\bar{w} = \overline{mc} \cdot (1-\alpha) \cdot \frac{\bar{Y}}{\bar{N}}
```

**(10) 財政変数**

```math
\bar{T} = \tau_c \bar{C} + \tau_l \bar{w}\bar{N} + \tau_k \bar{r}\bar{K}, \qquad \bar{B} = b_y \cdot \bar{Y}
```

**(11) 金融変数**

```math
\bar{q} = 1, \qquad \bar{s} = 0.005, \qquad \bar{NW} = \frac{\bar{K}}{L_{ss}}, \qquad \bar{r}^k = \bar{r} + \bar{s}
```

### 3.2 デフォルトキャリブレーションでの定常状態値

| 変数 | 値 | 導出 |
|------|-----|------|
| $\bar{r}$ | $\approx 0.001$ | $1/0.999 - 1$ |
| $\bar{\pi}$ | $0.005$ | 年率2%の四半期換算 |
| $\bar{R}$ | $\approx 0.006$ | $\bar{r}+\bar{\pi}$ |
| $\overline{mc}$ | $\approx 0.833$ | $5/6$ |
| $\bar{K}/\bar{Y}$ | $\approx 12.69$ | $0.33/0.026$ |
| $\bar{I}/\bar{Y}$ | $\approx 0.317$ | $0.025 \times 12.69$ |
| $\bar{C}/\bar{Y}$ | $\approx 0.483$ | $1 - 0.317 - 0.20$ |

---

## 4. 対数線形化方程式

全14方程式を以下に記述する。添字の慣例: $x_t$ は当期、$x_{t-1}$ はラグ、$\mathbb{E}_t[\cdot]$ は条件付き期待値。

---

### 方程式 1: IS曲線（習慣形成付き）

```math
y_t = h \cdot y_{t-1} + (1-h) \cdot \mathbb{E}_t[y_{t+1}] - \tilde{\sigma}^{-1}\bigl(r_t - \mathbb{E}_t[\pi_{t+1}]\bigr) + g_y \cdot g_t + a_t
```

ここで $\tilde{\sigma}$ は習慣形成調整済み異時点間代替弾力性：

```math
\tilde{\sigma} = \frac{\sigma(1-h)}{1+h}
```

| 係数 | 式 | デフォルト値 |
|------|-----|-----------|
| $h$ | 習慣形成 | 0.7 |
| $1-h$ | 前方期待ウェイト | 0.3 |
| $\tilde{\sigma}^{-1}$ | 金利感応度 | $\approx 3.78$ |
| $g_y$ | 政府支出乗数 | 0.20 |

---

### 方程式 2: 価格 Phillips 曲線（インデクセーション付き）

```math
\pi_t = \frac{\iota_p}{1+\beta\iota_p}\pi_{t-1} + \frac{\beta}{1+\beta\iota_p}\mathbb{E}_t[\pi_{t+1}] + \kappa \cdot mc_t + \frac{1}{1 - \frac{\beta}{1+\beta\iota_p}\rho_p} \cdot e_{p,t}
```

```math
\kappa = \frac{(1-\theta)(1-\beta\theta)}{\theta(1+\beta\iota_p)}
```

ショック $e_{p,t}$ の係数はAR(1)持続性 $\rho_p$ を通じた期待項の影響をスケーリングしている。

| 係数 | 式 | デフォルト値 |
|------|-----|-----------|
| $\iota_p/(1+\beta\iota_p)$ | 後方項 | $\approx 0.334$ |
| $\beta/(1+\beta\iota_p)$ | 前方項 | $\approx 0.666$ |
| $\kappa$ | Phillips曲線の傾き | $\approx 0.056$ |
| ショックスケール | $1/(1 - 0.666 \times 0.9)$ | $\approx 2.498$ |

---

### 方程式 3: 賃金 Phillips 曲線（Calvo型賃金硬直性）

```math
w_t = \frac{\beta}{1+\beta}\mathbb{E}_t[w_{t+1}] + \frac{1}{1+\beta}w_{t-1} + \frac{\lambda_w}{1+\beta}(mrs_t - w_t) + e_{w,t}
```

```math
\lambda_w = \frac{(1-\theta_w)(1-\beta\theta_w)}{\theta_w}
```

| 係数 | 式 | デフォルト値 |
|------|-----|-----------|
| $\beta/(1+\beta)$ | 前方項 | $\approx 0.500$ |
| $1/(1+\beta)$ | 後方項 | $\approx 0.500$ |
| $\lambda_w$ | 賃金調整速度 | $\approx 0.084$ |

---

### 方程式 4: Taylor則（金融政策）

```math
r_t = \phi_\pi \cdot \pi_t + \phi_y \cdot y_t + e_{m,t}
```

| 係数 | デフォルト値 |
|------|-----------|
| $\phi_\pi$ | 1.5 |
| $\phi_y$ | 0.125 |

---

### 方程式 5: 限界費用

```math
mc_t = \alpha \cdot r^k_t + (1-\alpha) \cdot w_t - a_t
```

---

### 方程式 6: 資本蓄積

```math
k_t = (1-\delta) \cdot k_{t-1} + \delta \cdot i_t
```

---

### 方程式 7: 投資調整コスト

```math
i_t = i_{t-1} + \frac{1}{S''} \cdot q_t + e_{i,t}
```

ここで $S''$ は調整コスト関数の曲率。$S'' \to \infty$ で投資は完全に非弾力的。

---

### 方程式 8: Tobin の Q（資産価格）

```math
q_t = \beta(1-\delta)\mathbb{E}_t[q_{t+1}] + \beta \cdot \mathbb{E}_t[r^k_{t+1}] - r_t
```

---

### 方程式 9: 資本レンタル率（資本の限界生産性）

```math
r^k_t = y_t - k_{t-1}
```

---

### 方程式 10: 労働需要（生産関数の逆関数）

```math
n_t = \frac{1}{1-\alpha}(y_t - a_t) - \frac{\alpha}{1-\alpha}k_{t-1}
```

---

### 方程式 11: 資源制約

```math
y_t = s_c \cdot c_t + s_i \cdot i_t + s_g \cdot g_t
```

ここで $s_c, s_i, s_g$ は定常状態の支出シェア：

```math
s_c = \frac{\bar{C}}{\bar{Y}}, \quad s_i = \frac{\bar{I}}{\bar{Y}}, \quad s_g = \frac{\bar{G}}{\bar{Y}}
```

---

### 方程式 12: 限界代替率（家計の最適条件）

```math
mrs_t = \sigma \cdot c_t + \phi \cdot n_t
```

---

### 方程式 13: 政府支出過程（AR(1)）

```math
g_t = \rho_g \cdot g_{t-1} + e_{g,t}
```

---

### 方程式 14: 技術過程（AR(1)）

```math
a_t = \rho_a \cdot a_{t-1} + e_{a,t}
```

---

## 5. システム行列表現

### 5.1 一般形式

14方程式をベクトル形式で記述する：

```math
A \cdot \mathbb{E}_t[y_{t+1}] + B \cdot y_t + C \cdot y_{t-1} + D \cdot \varepsilon_t = 0
```

ここで：

- $y_t \in \mathbb{R}^{14}$: 内生変数ベクトル
- $\varepsilon_t \in \mathbb{R}^{6}$: ショックベクトル
- $A \in \mathbb{R}^{14 \times 14}$: 期待値の係数行列
- $B \in \mathbb{R}^{14 \times 14}$: 当期変数の係数行列
- $C \in \mathbb{R}^{14 \times 14}$: ラグ変数の係数行列
- $D \in \mathbb{R}^{14 \times 6}$: ショック係数行列

### 5.2 変数の順序

```math
y_t = \underbrace{(g_t, a_t, k_t, i_t, w_t}_{先決変数\ (n_s=5)},\ \underbrace{y_t, \pi_t, r_t, q_t, r^k_t, n_t, c_t, mc_t, mrs_t)}_{制御変数\ (n_c=9)}
```

### 5.3 行列の構造

各方程式が行列の1行に対応する。以下に行列の非ゼロ要素パターンを示す。

**行列 $A$（期待値係数）**: 非ゼロ要素は行 4, 5, 6, 8 のみに現れる（$\text{rank}(A) = 4$）。

| 行 | 方程式 | 非ゼロ列 |
|----|--------|---------|
| 4 | 賃金Phillips | $w$: $-\beta/(1+\beta)$ |
| 5 | IS曲線 | $y$: $-(1-h)$,  $\pi$: $-\tilde{\sigma}^{-1}$ |
| 6 | 価格Phillips | $\pi$: $-\beta/(1+\beta\iota_p)$ |
| 8 | Tobin's Q | $q$: $-\beta(1-\delta)$,  $r^k$: $-\beta$ |

**行列 $B$（当期変数係数）**: 全方程式に非ゼロ要素を持つ。対角要素が支配的。

**行列 $C$（ラグ変数係数）**: 非ゼロ要素は行 0, 1, 2, 3, 4, 5, 6, 9, 10 に現れる。

**行列 $D$（ショック係数）**: 各ショックは対応する方程式行にのみ影響。

| ショック | $D$ 列 | 影響する行 | 方程式 |
|---------|--------|----------|--------|
| $e_g$ | 0 | 行 0 | 政府支出 AR(1) |
| $e_a$ | 1 | 行 1 | 技術 AR(1) |
| $e_m$ | 2 | 行 7 | Taylor則 |
| $e_i$ | 3 | 行 3 | 投資調整コスト |
| $e_w$ | 4 | 行 4 | 賃金Phillips |
| $e_p$ | 5 | 行 6 | 価格Phillips |

#### 方程式の行配置（実装順序）

| 行 | 方程式 | ブロック |
|----|--------|---------|
| 0 | 政府支出 AR(1) | 先決 |
| 1 | 技術 AR(1) | 先決 |
| 2 | 資本蓄積 | 先決 |
| 3 | 投資調整コスト | 先決 |
| 4 | 賃金Phillips曲線 | 先決 |
| 5 | IS曲線 | 制御 |
| 6 | 価格Phillips曲線 | 制御 |
| 7 | Taylor則 | 制御 |
| 8 | Tobin's Q | 制御 |
| 9 | 資本レンタル率 | 制御 |
| 10 | 労働需要 | 制御 |
| 11 | 資源制約 | 制御 |
| 12 | 限界費用 | 制御 |
| 13 | 限界代替率 (MRS) | 制御 |

---

## 6. Blanchard-Kahn解法

### 6.1 Companion形式への変換

$2n$ 次元のcompanion形式を構築する：

```math
\underbrace{\begin{pmatrix} A & 0 \\ I_n & 0 \end{pmatrix}}_{F}
\begin{pmatrix} \mathbb{E}_t[y_{t+1}] \\ y_t \end{pmatrix}
=
\underbrace{\begin{pmatrix} -B & -C \\ I_n & 0 \end{pmatrix}}_{G}
\begin{pmatrix} y_t \\ y_{t-1} \end{pmatrix}
```

### 6.2 QZ分解

一般化Schur分解（`scipy.linalg.ordqz`）を適用する：

```math
G = Q \cdot S \cdot Z^H, \qquad F = Q \cdot T \cdot Z^H
```

固有値は $\lambda_i = S_{ii} / T_{ii}$ として計算され、$|\lambda_i| < 1$（安定）と $|\lambda_i| > 1$（不安定）に分類される。

### 6.3 Blanchard-Kahn条件

一意の有界解が存在する必要十分条件：

```math
n_{unstable} = n_{forward}
```

本実装では $n_{forward} = \text{rank}(A)$ として計算する。行列 $A$ の非ゼロ行は4行（賃金Phillips・IS曲線・価格Phillips・Tobin's Q）であり、$\text{rank}(A) = 4$ となる。これは14個の方程式のうち前方期待項を含む方程式が4本であることに対応する。

- $n_{unstable} > n_{forward}$: 解なし（爆発パスのみ）
- $n_{unstable} < n_{forward}$: 不定解（複数のsunspot均衡）

### 6.4 政策関数の計算

BK条件が満たされたのち、解の形式を求める：

```math
s_t = P \cdot s_{t-1} + Q \cdot \varepsilon_t
```
```math
c_t = R \cdot s_t + S \cdot \varepsilon_t
```

ここで $P \in \mathbb{R}^{5 \times 5}$, $Q \in \mathbb{R}^{5 \times 6}$, $R \in \mathbb{R}^{9 \times 5}$, $S \in \mathbb{R}^{9 \times 6}$。

#### $P, R$ の決定（非線形係数一致条件）

$F = (I_{n_s}^\top, R^\top)^\top$ と定義し、以下を満たす $P, R$ を求める：

```math
A \cdot F \cdot P^2 + B \cdot F \cdot P + C \cdot F = 0
```

この $14 \times 5$ の行列方程式は $n_s^2 + n_c \cdot n_s = 70$ 個の未知数に対する非線形方程式系であり、`scipy.optimize.root`（hybr法）で数値解を得る。収束しない場合は `least_squares`（dogbox法）にフォールバックする。

安定性条件: $\rho(P) < 1$ （$P$ のスペクトル半径が1未満）

#### $Q, S$ の決定（線形方程式系）

$P, R$ が求まったのち、$Q, S$ は以下の線形方程式から得られる：

```math
(A \cdot F \cdot P + B \cdot F) \cdot Q + B_c \cdot S + D = 0
```

ここで $B_c = B[:, n_s:]$ は $B$ 行列の制御変数に対応する列ブロック。$14 \times 14$ の連立方程式を `np.linalg.solve` で解く。

---

## 7. ベイズ推定

### 7.1 状態空間表現

Blanchard-Kahn解を拡張状態空間形式に変換する。

#### 拡張状態ベクトル

```math
\alpha_t = \underbrace{(s_t}_{5},\ \underbrace{y_t, c_t, \pi_t, n_t, r_t}_{補助制御5},\ \underbrace{y_{t-1}, c_{t-1}, i_{t-1}, w_{t-1}}_{ラグ4})^\top \in \mathbb{R}^{14}
```

#### 状態遷移方程式

```math
\alpha_t = \mathcal{T} \cdot \alpha_{t-1} + \mathcal{R} \cdot \varepsilon_t, \qquad \varepsilon_t \sim \mathcal{N}(0, \Sigma_\varepsilon)
```

- $\mathcal{T} \in \mathbb{R}^{14 \times 14}$: 遷移行列
- $\mathcal{R} \in \mathbb{R}^{14 \times 6}$: ショック負荷行列
- $\Sigma_\varepsilon = \text{diag}(\sigma_{g}^2, \sigma_{a}^2, \sigma_{m}^2, \sigma_{i}^2, \sigma_{w}^2, \sigma_{p}^2)$

遷移行列の構造：

```math
\mathcal{T} = \begin{pmatrix}
P & 0 & 0 \\
R_{ctrl} P & 0 & 0 \\
L_{shift} & 0 & 0
\end{pmatrix}
```

ここで $R_{ctrl}$ は $R$ 行列から対応する制御変数行を抽出したもの、$L_{shift}$ はラグ変数のシフト演算子。

ショック負荷行列：

```math
\mathcal{R} = \begin{pmatrix}
Q \\
R_{ctrl} Q + S_{ctrl} \\
0
\end{pmatrix}
```

$S_{ctrl}$ は $S$ 行列から同時ショック効果を捕捉する。

#### 観測方程式

```math
z_t = \mathcal{Z} \cdot \alpha_t + \eta_t, \qquad \eta_t \sim \mathcal{N}(0, H)
```

- $z_t \in \mathbb{R}^7$: 観測ベクトル
- $\mathcal{Z} \in \mathbb{R}^{7 \times 14}$: 観測行列
- $H = \text{diag}(\sigma^2_{me,y}, \sigma^2_{me,c}, \sigma^2_{me,i}, \sigma^2_{me,\pi}, \sigma^2_{me,w}, \sigma^2_{me,n}, \sigma^2_{me,r})$

7つの観測変数と変換：

| # | 観測変数 | 変換 | 式 |
|---|---------|------|-----|
| 1 | 実質GDP成長率 | $\Delta\log$ | $z_{1,t} = y_t - y_{t-1}$ |
| 2 | 消費成長率 | $\Delta\log$ | $z_{2,t} = c_t - c_{t-1}$ |
| 3 | 投資成長率 | $\Delta\log$ | $z_{3,t} = i_t - i_{t-1}$ |
| 4 | インフレ率 | 水準 | $z_{4,t} = \pi_t$ |
| 5 | 実質賃金成長率 | $\Delta\log$ | $z_{5,t} = w_t - w_{t-1}$ |
| 6 | 雇用 | 水準 | $z_{6,t} = n_t$ |
| 7 | 名目短期金利 | 水準 | $z_{7,t} = r_t$ |

### 7.2 カルマンフィルタ

線形ガウス状態空間モデルに対するカルマンフィルタで対数尤度を計算する。

#### 予測ステップ

```math
\hat{\alpha}_{t|t-1} = \mathcal{T} \cdot \hat{\alpha}_{t-1|t-1}
```
```math
P_{t|t-1} = \mathcal{T} \cdot P_{t-1|t-1} \cdot \mathcal{T}^\top + \mathcal{R} \cdot \Sigma_\varepsilon \cdot \mathcal{R}^\top
```

#### 更新ステップ

```math
v_t = z_t - \mathcal{Z} \cdot \hat{\alpha}_{t|t-1} \qquad (\text{イノベーション})
```
```math
F_t = \mathcal{Z} \cdot P_{t|t-1} \cdot \mathcal{Z}^\top + H \qquad (\text{イノベーション共分散})
```
```math
K_t = P_{t|t-1} \cdot \mathcal{Z}^\top \cdot F_t^{-1} \qquad (\text{カルマンゲイン})
```
```math
\hat{\alpha}_{t|t} = \hat{\alpha}_{t|t-1} + K_t \cdot v_t
```
```math
P_{t|t} = P_{t|t-1} - K_t \cdot \mathcal{Z} \cdot P_{t|t-1}
```

#### 対数尤度

```math
\ln \mathcal{L}(\theta) = \sum_{t=1}^{T} \ln p(z_t \mid z_{1:t-1}, \theta)
```

```math
\ln p(z_t \mid z_{1:t-1}, \theta) = -\frac{n_t}{2}\ln(2\pi) - \frac{1}{2}\ln|F_t| - \frac{1}{2}v_t^\top F_t^{-1} v_t
```

ここで $n_t$ は時点 $t$ における有効な観測数（欠損値を除外）。

#### 初期化

状態共分散は離散Lyapunov方程式の解で初期化する：

```math
P_0 = \mathcal{T} \cdot P_0 \cdot \mathcal{T}^\top + \mathcal{R} \cdot \Sigma_\varepsilon \cdot \mathcal{R}^\top
```

#### 数値安定化

- $F_t$ の逆行列はCholesky分解（`scipy.linalg.cho_factor`）で計算
- 共分散行列は各ステップで対称化: $P \leftarrow \frac{1}{2}(P + P^\top)$
- 欠損値は自動的に除外され、有効な観測のみでフィルタリング

### 7.3 事後分布とMCMC

#### ベイズの定理

```math
p(\theta \mid z_{1:T}) \propto \mathcal{L}(z_{1:T} \mid \theta) \cdot p(\theta)
```

対数形式：

```math
\ln p(\theta \mid z_{1:T}) = \ln \mathcal{L}(\theta) + \ln p(\theta) + \text{const.}
```

#### Random Walk Metropolis-Hastings

**第1段階: モード探索**

```math
\theta^{*} = \arg\min_\theta \left[-\ln p(\theta \mid z_{1:T})\right]
```

L-BFGS-B法で解き、モードにおけるHessian $\mathcal{H}$ を有限差分で計算する。

**第2段階: MCMCサンプリング**

提案分布：

```math
\theta^{prop} = \theta^{cur} + \xi, \qquad \xi \sim \mathcal{N}\left(0, \frac{2.38^2}{d} \cdot \mathcal{H}^{-1}\right)
```

ここで $d$ は推定パラメータ数。スケーリング定数 $2.38^2/d$ はRoberts et al. (1997) の最適値。

受容確率：

```math
\alpha = \min\left(1, \frac{p(\theta^{prop} \mid z)}{p(\theta^{cur} \mid z)}\right)
```

対数尤度域で:

```math
\ln\alpha = \ln p(\theta^{prop} \mid z) - \ln p(\theta^{cur} \mid z)
```

$\ln U < \ln\alpha$ ($U \sim \text{Uniform}(0,1)$) なら $\theta^{prop}$ を受容。

**適応的サンプリング**: バーンイン期間中、一定間隔でサンプルの経験的共分散を提案分布に反映する。

#### MCMCの設定

| パラメータ | デフォルト値 | 説明 |
|-----------|-----------|------|
| チェーン数 | 4 | 並列チェーン |
| 総ドロー数 | 100,000 | チェーンあたり |
| バーンイン | 50,000 | 棄却するサンプル数 |
| シンニング | 10 | 保存する間隔 |
| 目標受容率 | 0.234 | Roberts et al. (1997) |
| 適応間隔 | 100 | バーンイン中の更新頻度 |

有効サンプル数: $(100{,}000 - 50{,}000)/10 \times 4 = 20{,}000$

### 7.4 事前分布

Smets & Wouters (2007) および Sugo & Ueda (2008) に基づく。

#### 構造パラメータ（8個）

| パラメータ | 分布 | 平均 | 標準偏差 | 台 |
|-----------|------|------|---------|-----|
| $\sigma$ (危険回避度) | Gamma | 1.5 | 0.37 | $(0, \infty)$ |
| $\phi$ (Frisch弾力性逆数) | Gamma | 2.0 | 0.75 | $(0, \infty)$ |
| $h$ (習慣形成) | Beta | 0.7 | 0.10 | $[0, 1]$ |
| $\theta$ (Calvo価格) | Beta | 0.75 | 0.05 | $[0, 1]$ |
| $\psi$ (価格インデクセーション) | Beta | 0.5 | 0.15 | $[0, 1]$ |
| $S''$ (投資調整コスト) | Gamma | 5.0 | 1.50 | $(0, \infty)$ |
| $\theta_w$ (Calvo賃金) | Beta | 0.75 | 0.05 | $[0, 1]$ |
| $\iota_w$ (賃金インデクセーション) | Beta | 0.5 | 0.15 | $[0, 1]$ |

#### 金融政策パラメータ（3個）

| パラメータ | 分布 | 平均 | 標準偏差 |
|-----------|------|------|---------|
| $\phi_\pi$ (インフレ反応) | Normal | 1.5 | 0.25 |
| $\phi_y$ (産出反応) | Normal | 0.125 | 0.05 |
| $\rho_r$ (金利平滑化) | Beta | 0.85 | 0.10 |

#### ショック持続性（3個）

| パラメータ | 分布 | 平均 | 標準偏差 |
|-----------|------|------|---------|
| $\rho_a$ | Beta | 0.9 | 0.05 |
| $\rho_g$ | Beta | 0.9 | 0.05 |
| $\rho_p$ | Beta | 0.9 | 0.05 |

#### ショック標準偏差（6個）

$\sigma_a, \sigma_g, \sigma_i, \sigma_w, \sigma_p, \sigma_m$ はすべて InvGamma(mean=0.01, std=0.01)。

#### 観測誤差標準偏差（7個）

$\sigma_{me,y}, \sigma_{me,c}, \sigma_{me,i}, \sigma_{me,\pi}, \sigma_{me,w}, \sigma_{me,n}, \sigma_{me,r}$ はすべて InvGamma(mean=0.01, std=0.01)。

#### 事前分布のパラメータ化

各分布の形状パラメータは平均 $\mu$ と標準偏差 $\sigma$ から以下のように決定される：

**Beta分布** ($x \in [0,1]$):

```math
a = \mu \cdot \kappa, \quad b = (1-\mu) \cdot \kappa, \quad \kappa = \frac{\mu(1-\mu)}{\sigma^2} - 1
```

**Gamma分布** ($x > 0$):

```math
k = \left(\frac{\mu}{\sigma}\right)^2, \quad \theta_{scale} = \frac{\sigma^2}{\mu}
```

**逆Gamma分布** ($x > 0$):

```math
\alpha = \left(\frac{\mu}{\sigma}\right)^2 + 2, \quad \beta_{scale} = \mu(\alpha - 1)
```

### 7.5 収束診断

#### Gelman-Rubin $\hat{R}$ 統計量

$m$ チェーン、各 $n$ サンプルに対し：

```math
W = \frac{1}{m}\sum_{j=1}^{m} s_j^2, \qquad \frac{B}{n} = \frac{1}{m-1}\sum_{j=1}^{m}(\bar{\theta}_j - \bar{\theta})^2
```

```math
\hat{V} = \frac{n-1}{n}W + \frac{1}{n}B, \qquad \hat{R} = \sqrt{\frac{\hat{V}}{W}}
```

判定基準: $\hat{R} < 1.1$ で収束と判定。

#### 有効サンプルサイズ (ESS)

```math
ESS = \frac{n_{total}}{1 + 2\sum_{k=1}^{\infty}\hat{\rho}(k)}
```

ここで $\hat{\rho}(k)$ はラグ $k$ の自己相関。

#### Geweke検定

チェーンの最初10%と最後50%の平均を比較：

```math
z = \frac{\bar{\theta}_{first} - \bar{\theta}_{last}}{\sqrt{SE_{first}^2 + SE_{last}^2}} \sim \mathcal{N}(0,1)
```

$p > 0.05$ で収束と判定。

### 7.6 周辺尤度（Laplace近似）

```math
\ln p(z_{1:T}) \approx \ln p(z_{1:T} \mid \theta^{*}) + \ln p(\theta^{*}) + \frac{d}{2}\ln(2\pi) - \frac{1}{2}\ln|\mathcal{H}|
```

モデル比較のためのベイズファクター: $BF_{12} = p(z \mid M_1) / p(z \mid M_2)$

---

## 8. パラメータ一覧

### 8.1 家計部門

| 記号 | パラメータ名 | デフォルト値 | 説明 |
|------|------------|-----------|------|
| $\beta$ | 割引率 | 0.999 | 低金利環境向け |
| $\sigma$ | 異時点間代替弾力性の逆数 | 1.5 | CRRA効用のリスク回避度 |
| $\phi$ | Frisch弾力性の逆数 | 2.0 | 労働供給弾力性 |
| $h$ | 習慣形成 | 0.7 | 日本向け高め設定 |
| $\chi$ | 労働の不効用 | 1.0 | 効用関数のスケール |

### 8.2 企業部門

| 記号 | パラメータ名 | デフォルト値 | 説明 |
|------|------------|-----------|------|
| $\alpha$ | 資本分配率 | 0.33 | Cobb-Douglas |
| $\delta$ | 資本減耗率 | 0.025 | 四半期（年率10%） |
| $\theta$ | Calvo価格硬直性 | 0.75 | 平均4四半期で価格改定 |
| $\epsilon$ | 財の代替弾力性 | 6.0 | マークアップ20% |
| $\iota_p$ | 価格インデクセーション | 0.5 | 部分的後方参照 |

### 8.3 投資

| 記号 | パラメータ名 | デフォルト値 | 参考 |
|------|------------|-----------|------|
| $S''$ | 調整コスト曲率 | 5.0 | SW2007: 5.48 |

### 8.4 労働市場

| 記号 | パラメータ名 | デフォルト値 | 説明 |
|------|------------|-----------|------|
| $\theta_w$ | Calvo賃金硬直性 | 0.75 | 平均4四半期で賃金改定 |
| $\epsilon_w$ | 労働の代替弾力性 | 10.0 | 賃金マークアップ11% |
| $\iota_w$ | 賃金インデクセーション | 0.5 | 部分的後方参照 |

### 8.5 政府部門

| 記号 | パラメータ名 | デフォルト値 | 説明 |
|------|------------|-----------|------|
| $\tau_c$ | 消費税率 | 0.10 | 日本の現行10% |
| $\tau_l$ | 労働所得税率 | 0.25 | 実効税率 |
| $\tau_k$ | 資本所得税率 | 0.30 | 法人税相当 |
| $g_y$ | 政府支出/GDP比 | 0.20 | |
| $b_y$ | 政府債務/GDP比 | 2.00 | 日本の高債務 |
| $tr_y$ | 移転支払い/GDP比 | 0.15 | |
| $\rho_g$ | 政府支出持続性 | 0.90 | AR(1)係数 |
| $\rho_\tau$ | 税率持続性 | 0.90 | AR(1)係数 |
| $\phi_b$ | 債務安定化係数 | 0.02 | 財政ルール |

### 8.6 中央銀行

| 記号 | パラメータ名 | デフォルト値 | 説明 |
|------|------------|-----------|------|
| $\rho_r$ | 金利平滑化 | 0.85 | 慣性パラメータ |
| $\phi_\pi$ | インフレ反応 | 1.5 | Taylor原則 ($>1$) |
| $\phi_y$ | 産出反応 | 0.125 | |
| $\pi^{*}$ | インフレ目標 | 0.005 | 四半期（年率2%） |
| $\underline{R}$ | 名目金利下限 | -0.001 | マイナス金利 |

### 8.7 金融部門

| 記号 | パラメータ名 | デフォルト値 | 説明 |
|------|------------|-----------|------|
| $\chi_b$ | 外部資金プレミアム弾力性 | 0.05 | BGG型 |
| $L_{ss}$ | 定常状態レバレッジ | 2.0 | |
| $\gamma$ | 企業家生存率 | 0.975 | |

---

## 9. 参考文献

1. Smets, F., & Wouters, R. (2007). "Shocks and Frictions in US Business Cycles: A Bayesian DSGE Approach." *American Economic Review*, 97(3), 586-606.

2. Blanchard, O.J., & Kahn, C.M. (1980). "The Solution of Linear Difference Models under Rational Expectations." *Econometrica*, 48(5), 1305-1311.

3. Klein, P. (2000). "Using the Generalized Schur Form to Solve a Multivariate Linear Rational Expectations Model." *Journal of Economic Dynamics and Control*, 24(10), 1405-1423.

4. Sugo, T., & Ueda, K. (2008). "Estimating a Dynamic Stochastic General Equilibrium Model for Japan." *Journal of the Japanese and International Economies*, 22(4), 476-502.

5. Roberts, G.O., Gelman, A., & Gilks, W.R. (1997). "Weak Convergence and Optimal Scaling of Random Walk Metropolis Algorithms." *Annals of Applied Probability*, 7(1), 110-120.

6. Fernandez-Villaverde, J., et al. (2016). "Solution and Estimation Methods for DSGE Models." *Handbook of Macroeconomics*, Vol. 2, 527-724.

7. 日本銀行 (2019). "ハイブリッド型日本経済モデル Q-JEM: 2019年バージョン." Working Paper Series No.19-E-7.

---

*本仕様書はソースコード実装に基づく。最終更新: 2026-02-16*
