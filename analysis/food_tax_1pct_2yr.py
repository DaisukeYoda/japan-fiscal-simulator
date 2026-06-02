"""食料品消費税1%・2年限定の時限減税シミュレーション

シナリオ:
    食料品 8% → 1%（非食料品は10%据え置き）を「2年間（8四半期）限定」で実施し、
    3年目（t=8）に元の制度へ復帰する。

比較基準:
    現行制度（食料品8%・非食料品10%、実効9.5%）からの変化。
    食料品1%案の実効税率 = 0.25*1% + 0.75*10% = 7.75%
    → 実効税率の変化 = 7.75% - 9.5% = -1.75%pt

モデル化:
    tau_c の遷移式は純粋AR(1):  tau_c_t = rho * tau_c_{t-1} + e_tau_t  (rho=0.95)
    これを使い、税率を「t=0..7 で -1.75%pt 一定 → t=8 で 0 に復帰」と
    なるようなショック列 e_tau_t を構築し、線形システムを重ね合わせで伝播させる:
        x_0 = Q e_0
        x_t = P x_{t-1} + Q e_t

    比較用に、既存エンジン相当の「単発インパルス→AR(1)減衰」も並走させる。

    注意（重要な限界）:
        この重ね合わせは「各期ごとに政策の継続/終了を新しいショックとして
        受け取る」非予見（unanticipated）型の経路。事前公表された終了時点を
        家計が織り込む完全予見（perfect-foresight / news shock）型の前倒し効果は
        この線形IRFエンジンでは厳密には表現されない。終了時点の反動（snap-back）は
        t=8 の復帰ショックとして明示的に表現される。
"""

import numpy as np

from japan_fiscal_simulator.core.model import N_SHOCKS, N_VARIABLES, VARIABLE_INDICES, DSGEModel
from japan_fiscal_simulator.parameters.calibration import JapanCalibration

E_TAU = 3  # ショックベクトルにおける消費税ショックのインデックス

# 制度パラメータ
FOOD_SHARE = 0.25
CURRENT_FOOD, CURRENT_NONFOOD = 0.08, 0.10
NEW_FOOD, NEW_NONFOOD = 0.01, 0.10

CURRENT_EFFECTIVE = FOOD_SHARE * CURRENT_FOOD + (1 - FOOD_SHARE) * CURRENT_NONFOOD  # 9.5%
NEW_EFFECTIVE = FOOD_SHARE * NEW_FOOD + (1 - FOOD_SHARE) * NEW_NONFOOD  # 7.75%
SHOCK = NEW_EFFECTIVE - CURRENT_EFFECTIVE  # -1.75%pt

HOLD_QUARTERS = 8  # 2年間維持
PERIODS = 40


def build_innovation_sequence(target: float, rho: float, hold: int, periods: int) -> np.ndarray:
    """tau_c を t=0..hold-1 で target に保ち、t=hold で 0 に戻すための e_tau 列。

    tau_c_t = rho * tau_c_{t-1} + e_tau_t
      t=0          : e = target
      t=1..hold-1  : e = (1-rho)*target   （減衰分を補填して水準維持）
      t=hold       : e = -rho*target      （rho*target から 0 へ復帰）
      t>hold       : e = 0                （0*rho=0 なので 0 を維持）
    """
    eps = np.zeros((periods + 1, N_SHOCKS))
    eps[0, E_TAU] = target
    for t in range(1, hold):
        eps[t, E_TAU] = (1 - rho) * target
    if hold <= periods:
        eps[hold, E_TAU] = -rho * target
    return eps


def propagate(P: np.ndarray, Q: np.ndarray, eps: np.ndarray, periods: int) -> np.ndarray:
    """ショック列 eps を線形システムに通して状態経路を返す。"""
    x = np.zeros((periods + 1, N_VARIABLES))
    x[0] = Q[:, :N_SHOCKS] @ eps[0]
    for t in range(1, periods + 1):
        x[t] = P @ x[t - 1] + Q[:, :N_SHOCKS] @ eps[t]
    return x


def single_impulse(P: np.ndarray, Q: np.ndarray, size: float, periods: int) -> np.ndarray:
    """既存エンジン相当: t=0 に単発ショック→AR(1)減衰。"""
    eps = np.zeros((periods + 1, N_SHOCKS))
    eps[0, E_TAU] = size
    return propagate(P, Q, eps, periods)


def build_model() -> DSGEModel:
    """現行制度の実効税率を定常状態とするモデルを構築する。"""
    calibration = JapanCalibration.create().set_consumption_tax(CURRENT_EFFECTIVE)
    return DSGEModel(calibration.parameters)


def estimate_revenue_loss(model: DSGEModel, tau_path: np.ndarray, hold: int) -> float:
    """実効税率の低下による税収減をGDP比で概算する。"""
    ss = model.steady_state
    c_y = ss.consumption / ss.output
    elasticity = (
        model.derived_coefficients.compute_impulse_coefficients().consumption_tax_elasticity
    )
    behavioral_adjustment = 1.0 - model.params.government.tau_c * elasticity
    return -np.sum(tau_path[:hold]) * c_y * behavioral_adjustment


def main() -> None:
    model = build_model()
    pf = model.policy_function
    P, Q = pf.P, pf.Q
    rho = model.params.shocks.rho_tau_c
    idx = VARIABLE_INDICES

    print("=" * 70)
    print("食料品消費税 1%・2年限定 時限減税シミュレーション")
    print("=" * 70)
    print(f"現行実効税率 : {CURRENT_EFFECTIVE * 100:.2f}%  (食料品8% / 非食料品10%)")
    print(f"新制度実効税率: {NEW_EFFECTIVE * 100:.2f}%  (食料品1% / 非食料品10%)")
    print(f"実効税率変化 : {SHOCK * 100:+.2f}%pt   維持期間: {HOLD_QUARTERS}四半期")
    print(f"AR(1)持続性 rho_tau_c = {rho}  /  BK条件: {pf.bk_satisfied}")
    print()

    eps_timed = build_innovation_sequence(SHOCK, rho, HOLD_QUARTERS, PERIODS)
    x_timed = propagate(P, Q, eps_timed, PERIODS)
    x_ar1 = single_impulse(P, Q, SHOCK, PERIODS)

    def col(x, name):
        return x[:, idx[name]]

    # 経路確認: tau_c がきちんと維持・復帰しているか
    print("税率(tau_c)経路の確認 [%pt 乖離]:")
    tc = col(x_timed, "tau_c")
    print("  四半期 :", " ".join(f"{q:>6d}" for q in [0, 1, 4, 7, 8, 9, 12]))
    print("  時限版 :", " ".join(f"{tc[q] * 100:>6.2f}" for q in [0, 1, 4, 7, 8, 9, 12]))
    tca = col(x_ar1, "tau_c")
    print("  AR1版  :", " ".join(f"{tca[q] * 100:>6.2f}" for q in [0, 1, 4, 7, 8, 9, 12]))
    print()

    # 主要変数の経路
    def summarize(label, x):
        y, c, b = col(x, "y"), col(x, "c"), col(x, "b")
        peak_y = y[np.argmax(np.abs(y))]
        peak_c = c[np.argmax(np.abs(c))]
        cum8_y = np.sum(y[:8])
        print(f"[{label}]")
        print(f"  GDP    ピーク: {peak_y * 100:+.3f}%   2年累積(8Q): {cum8_y * 100:+.3f}%pt·Q")
        print(f"  消費   ピーク: {peak_c * 100:+.3f}%")
        print("  GDP    t=0..9: " + " ".join(f"{y[q] * 100:+.2f}" for q in range(10)))
        print("  消費   t=0..9: " + " ".join(f"{c[q] * 100:+.2f}" for q in range(10)))
        print("  政府債務 t=0..9: " + " ".join(f"{b[q] * 100:+.2f}" for q in range(10)))
        # 反動: 終了直後(t=8,9)で符号が反転しているか
        print(
            f"  反動チェック GDP t=7→8→9: {y[7] * 100:+.3f} → {y[8] * 100:+.3f} → {y[9] * 100:+.3f}"
        )
        print()

    summarize("時限措置 (8Q維持→復帰)", x_timed)
    summarize("AR(1)減衰 (参考)", x_ar1)

    # 財政コストの目安: 実効税率低下 × 消費/GDP（消費の行動反応を補正）
    tau_path = col(x_timed, "tau_c")
    revenue_loss = estimate_revenue_loss(model, tau_path, HOLD_QUARTERS)
    print(
        f"参考: 減税期間中の食料品税収減の目安 ≈ {revenue_loss * 100:.2f}%pt·Q (GDP比, 行動反応補正後)"
    )

    # 図の保存
    try:
        import matplotlib  # noqa: PLC0415

        matplotlib.use("Agg")
        from pathlib import Path  # noqa: PLC0415

        import matplotlib.pyplot as plt  # noqa: PLC0415

        outdir = Path(__file__).resolve().parent.parent / "output" / "food_tax_1pct"
        outdir.mkdir(parents=True, exist_ok=True)

        t = np.arange(PERIODS + 1)
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        panels = [
            ("tau_c", "Effective consumption tax (%pt dev.)", 100),
            ("y", "GDP (% dev.)", 100),
            ("c", "Consumption (% dev.)", 100),
            ("b", "Govt debt (% dev.)", 100),
        ]
        for ax, (var, title, scale) in zip(axes.flat, panels, strict=True):
            ax.plot(t, col(x_timed, var) * scale, label="Time-limited (8Q hold -> revert)", lw=2)
            ax.plot(t, col(x_ar1, var) * scale, label="AR(1) decay (ref.)", ls="--", lw=1.5)
            ax.axvline(HOLD_QUARTERS, color="gray", ls=":", alpha=0.7)
            ax.axhline(0, color="k", lw=0.5)
            ax.set_title(title)
            ax.set_xlabel("Quarter")
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
        fig.suptitle("Food consumption tax cut to 1%, 2-year time-limited", fontsize=13)
        fig.tight_layout()
        path = outdir / "food_tax_1pct_2yr.png"
        fig.savefig(path, dpi=120)
        print(f"\n図を保存: {path}")
    except Exception as e:  # noqa: BLE001
        print(f"\n図の生成をスキップ: {e}")


if __name__ == "__main__":
    main()
