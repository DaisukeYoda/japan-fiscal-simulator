"""DSGEモデルのデフォルトパラメータ定義"""

import math
from dataclasses import dataclass, field, replace
from typing import Any

from japan_fiscal_simulator.core.exceptions import ParameterValidationError


def _validate(
    owner: str,
    name: str,
    value: float,
    low: float | None = None,
    high: float | None = None,
    *,
    low_open: bool = False,
    high_open: bool = False,
) -> None:
    """パラメータが有限かつ許容範囲内であることを検証する

    検証境界はMCMCの探索範囲（ParameterMapping.ESTIMATED_PARAMS）を
    すべて内包するように設計されている。経済学的に珍しいだけの値
    （例: テイラー原理違反のphi_pi < 1）は拒否しない。

    Raises:
        ParameterValidationError: 値が非有限または範囲外の場合
    """
    try:
        is_finite = math.isfinite(value)
    except TypeError:
        is_finite = False
    if not is_finite:
        raise ParameterValidationError(f"{owner}.{name}={value!r} は有限の数値である必要があります")
    in_range = True
    if low is not None:
        in_range = value > low if low_open else value >= low
    if in_range and high is not None:
        in_range = value < high if high_open else value <= high
    if not in_range:
        lower = f"{low} {'<' if low_open else '<='} " if low is not None else ""
        upper = f" {'<' if high_open else '<='} {high}" if high is not None else ""
        raise ParameterValidationError(
            f"{owner}.{name}={value} が有効範囲外です（{lower}{name}{upper}）"
        )


@dataclass(frozen=True)
class HouseholdParameters:
    """家計部門パラメータ"""

    beta: float = 0.999  # 割引率（低金利環境向け）
    sigma: float = 1.5  # 異時点間代替弾力性の逆数
    phi: float = 2.0  # 労働供給弾力性の逆数（Frisch弾力性）
    habit: float = 0.7  # 習慣形成パラメータ
    chi: float = 1.0  # 労働の不効用パラメータ

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "beta", self.beta, 0.0, 1.0, low_open=True, high_open=True)
        _validate(owner, "sigma", self.sigma, 0.0, low_open=True)
        _validate(owner, "phi", self.phi, 0.0, low_open=True)
        _validate(owner, "habit", self.habit, 0.0, 1.0, high_open=True)
        _validate(owner, "chi", self.chi, 0.0, low_open=True)


@dataclass(frozen=True)
class FirmParameters:
    """企業部門パラメータ"""

    alpha: float = 0.33  # 資本分配率
    delta: float = 0.025  # 資本減耗率（四半期）
    theta: float = 0.75  # Calvo価格硬直性（75%が価格維持）
    epsilon: float = 6.0  # 財の代替弾力性
    psi: float = 0.5  # 価格インデクセーション

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "alpha", self.alpha, 0.0, 1.0, low_open=True, high_open=True)
        _validate(owner, "delta", self.delta, 0.0, 1.0, low_open=True)
        # theta=0はPhillips曲線スロープ (1-θ)(1-βθ)/θ でゼロ除算になる
        _validate(owner, "theta", self.theta, 0.0, 1.0, low_open=True, high_open=True)
        # マークアップ ε/(ε-1) が定義できるのは ε > 1 のみ
        _validate(owner, "epsilon", self.epsilon, 1.0, low_open=True)
        _validate(owner, "psi", self.psi, 0.0, 1.0)

    @property
    def iota_p(self) -> float:
        """価格インデクセーション係数（psi の互換別名）"""
        return self.psi


@dataclass(frozen=True)
class GovernmentParameters:
    """政府部門パラメータ"""

    tau_c: float = 0.10  # 消費税率（10%）
    tau_l: float = 0.25  # 労働所得税率
    tau_k: float = 0.30  # 資本所得税率
    g_y_ratio: float = 0.20  # 政府支出/GDP比率
    b_y_ratio: float = 2.00  # 政府債務/GDP比率（日本の高債務状況）
    transfer_y_ratio: float = 0.15  # 移転支払い/GDP比率
    rho_g: float = 0.90  # 政府支出の持続性
    rho_tau: float = 0.90  # 税率の持続性
    phi_b: float = 0.02  # 債務安定化係数

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "tau_c", self.tau_c, 0.0, 1.0, high_open=True)
        _validate(owner, "tau_l", self.tau_l, 0.0, 1.0, high_open=True)
        _validate(owner, "tau_k", self.tau_k, 0.0, 1.0, high_open=True)
        _validate(owner, "g_y_ratio", self.g_y_ratio, 0.0, 1.0, high_open=True)
        _validate(owner, "b_y_ratio", self.b_y_ratio, 0.0)
        _validate(owner, "transfer_y_ratio", self.transfer_y_ratio, 0.0, 1.0, high_open=True)
        _validate(owner, "rho_g", self.rho_g, 0.0, 1.0, high_open=True)
        _validate(owner, "rho_tau", self.rho_tau, 0.0, 1.0, high_open=True)
        _validate(owner, "phi_b", self.phi_b, 0.0)


@dataclass(frozen=True)
class CentralBankParameters:
    """中央銀行パラメータ"""

    rho_r: float = 0.85  # 金利平滑化
    phi_pi: float = 1.5  # インフレ反応係数
    phi_y: float = 0.125  # 産出ギャップ反応係数
    pi_target: float = 0.005  # インフレ目標（四半期、年率2%）
    r_lower_bound: float = -0.001  # 名目金利下限（ZLB、若干のマイナス金利許容）

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "rho_r", self.rho_r, 0.0, 1.0, high_open=True)
        # phi_pi < 1（テイラー原理違反）は不決定性の検証に使うため有限性のみ確認する
        _validate(owner, "phi_pi", self.phi_pi)
        _validate(owner, "phi_y", self.phi_y)
        _validate(owner, "pi_target", self.pi_target)
        _validate(owner, "r_lower_bound", self.r_lower_bound)


@dataclass(frozen=True)
class FinancialParameters:
    """金融部門パラメータ（BGG型簡略版）"""

    chi_b: float = 0.05  # 外部資金プレミアム弾力性
    leverage_ss: float = 2.0  # 定常状態レバレッジ
    survival_rate: float = 0.975  # 企業家生存率

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "chi_b", self.chi_b, 0.0)
        # 定常状態で nw = k / leverage_ss を計算するためゼロは不可
        _validate(owner, "leverage_ss", self.leverage_ss, 0.0, low_open=True)
        _validate(owner, "survival_rate", self.survival_rate, 0.0, 1.0, low_open=True)


@dataclass(frozen=True)
class OpenEconomyParameters:
    """開放経済liteパラメータ"""

    import_share: float = 0.20  # 輸入財シェア
    exchange_rate_passthrough: float = 0.30  # 円安からCPIへの部分転嫁率
    # 実質所得ドラッグの倍率: CPI転嫁分(psi_m)に対する乗数。
    # 1.0 で「物価上昇分＝実質所得の目減り」を意味する（インフレ転嫁チャネルと整合）。
    # 交易条件・輸入数量効果を上乗せしたい場合のみ 1.0 超に設定する。
    eta: float = 1.0

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "import_share", self.import_share, 0.0, 1.0, high_open=True)
        _validate(owner, "exchange_rate_passthrough", self.exchange_rate_passthrough, 0.0, 1.0)
        _validate(owner, "eta", self.eta, 0.0)

    @property
    def psi_m(self) -> float:
        """円安からインフレへのコストプッシュ係数（輸入シェア×転嫁率）"""
        return self.import_share * self.exchange_rate_passthrough

    @property
    def consumption_drag(self) -> float:
        """円安による実質所得ドラッグ＝CPI転嫁分(psi_m)×eta"""
        return self.psi_m * self.eta


@dataclass(frozen=True)
class InvestmentParameters:
    """投資パラメータ

    参考文献:
    - Smets & Wouters (2007): S'' ≈ 5.48
    - Christiano, Eichenbaum & Evans (2005): S'' ≈ 2.5
    """

    # 投資調整コスト曲率: Smets-Wouters (2007) の推定値 5.48 を参考に設定
    S_double_prime: float = 5.0

    def __post_init__(self) -> None:
        # 投資方程式の係数 1/S'' を計算するためゼロは不可
        _validate(type(self).__name__, "S_double_prime", self.S_double_prime, 0.0, low_open=True)


@dataclass(frozen=True)
class LaborParameters:
    """労働市場パラメータ

    参考文献:
    - Smets & Wouters (2007): θ_w ≈ 0.73, ε_w ≈ 10
    - Erceg, Henderson & Levin (2000): 賃金Phillips曲線
    - Sugo & Ueda (2008, 日本): θ_w ≈ 0.70
    """

    theta_w: float = 0.75  # Calvo賃金硬直性（75%が賃金維持）
    epsilon_w: float = 10.0  # 労働の代替弾力性
    iota_w: float = 0.5  # 賃金インデクセーション（0=後向き、1=前向き）

    def __post_init__(self) -> None:
        owner = type(self).__name__
        _validate(owner, "theta_w", self.theta_w, 0.0, 1.0, low_open=True, high_open=True)
        # 賃金マークアップ ε_w/(ε_w-1) が定義できるのは ε_w > 1 のみ
        _validate(owner, "epsilon_w", self.epsilon_w, 1.0, low_open=True)
        _validate(owner, "iota_w", self.iota_w, 0.0, 1.0)


@dataclass(frozen=True)
class ShockParameters:
    """ショックパラメータ"""

    # 持続性
    rho_a: float = 0.90  # 技術ショック
    rho_g: float = 0.90  # 政府支出ショック
    rho_tau_c: float = 0.95  # 消費税ショック
    rho_m: float = 0.50  # 金融政策ショック
    rho_risk: float = 0.75  # リスクプレミアムショック
    rho_fx: float = 0.90  # 円安ショック
    # 投資固有技術ショック: Smets-Wouters (2007) では ρ_i ≈ 0.71
    rho_i: float = 0.70
    # 賃金マークアップショック: Smets-Wouters (2007) では ρ_w ≈ 0.89
    rho_w: float = 0.90
    # 価格マークアップショック: Phase 3
    rho_p: float = 0.90

    # 標準偏差
    sigma_a: float = 0.01
    sigma_g: float = 0.01
    sigma_tau_c: float = 0.005
    sigma_m: float = 0.0025
    sigma_risk: float = 0.01
    sigma_fx: float = 0.01
    sigma_i: float = 0.01
    sigma_w: float = 0.01  # 賃金マークアップショック
    sigma_p: float = 0.01  # 価格マークアップショック

    def __post_init__(self) -> None:
        owner = type(self).__name__
        # AR(1)持続性は定常性のため [0, 1) に制限
        for name in (
            "rho_a",
            "rho_g",
            "rho_tau_c",
            "rho_m",
            "rho_risk",
            "rho_fx",
            "rho_i",
            "rho_w",
            "rho_p",
        ):
            _validate(owner, name, getattr(self, name), 0.0, 1.0, high_open=True)
        for name in (
            "sigma_a",
            "sigma_g",
            "sigma_tau_c",
            "sigma_m",
            "sigma_risk",
            "sigma_fx",
            "sigma_i",
            "sigma_w",
            "sigma_p",
        ):
            _validate(owner, name, getattr(self, name), 0.0)

    @property
    def persistent_non_state_shocks(self) -> dict[str, float]:
        """状態変数を持たないがIRF上でAR(1)持続性を持つショック"""
        return {
            "e_p": self.rho_p,
            "e_w": self.rho_w,
        }


@dataclass(frozen=True)
class DefaultParameters:
    """全パラメータを統合したデフォルト設定"""

    household: HouseholdParameters = field(default_factory=HouseholdParameters)
    firm: FirmParameters = field(default_factory=FirmParameters)
    government: GovernmentParameters = field(default_factory=GovernmentParameters)
    central_bank: CentralBankParameters = field(default_factory=CentralBankParameters)
    financial: FinancialParameters = field(default_factory=FinancialParameters)
    open_economy: OpenEconomyParameters = field(default_factory=OpenEconomyParameters)
    investment: InvestmentParameters = field(default_factory=InvestmentParameters)
    labor: LaborParameters = field(default_factory=LaborParameters)
    shocks: ShockParameters = field(default_factory=ShockParameters)

    def with_updates(
        self,
        household: HouseholdParameters | None = None,
        firm: FirmParameters | None = None,
        government: GovernmentParameters | None = None,
        central_bank: CentralBankParameters | None = None,
        financial: FinancialParameters | None = None,
        open_economy: OpenEconomyParameters | None = None,
        investment: InvestmentParameters | None = None,
        labor: LaborParameters | None = None,
        shocks: ShockParameters | None = None,
    ) -> "DefaultParameters":
        """パラメータの一部を更新した新しいインスタンスを返す"""
        updates: dict[str, Any] = {
            "household": household,
            "firm": firm,
            "government": government,
            "central_bank": central_bank,
            "financial": financial,
            "open_economy": open_economy,
            "investment": investment,
            "labor": labor,
            "shocks": shocks,
        }
        return replace(
            self, **{name: value for name, value in updates.items() if value is not None}
        )
