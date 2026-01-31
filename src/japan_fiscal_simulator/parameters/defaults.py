"""DSGEモデルのデフォルトパラメータ定義"""

from dataclasses import dataclass, field, replace
from typing import Self


@dataclass(frozen=True)
class HouseholdParameters:
    """家計部門パラメータ"""

    beta: float = 0.999  # 割引率（低金利環境向け）
    sigma: float = 1.5  # 異時点間代替弾力性の逆数
    phi: float = 2.0  # 労働供給弾力性の逆数（Frisch弾力性）
    habit: float = 0.7  # 習慣形成パラメータ
    chi: float = 1.0  # 労働の不効用パラメータ


@dataclass(frozen=True)
class FirmParameters:
    """企業部門パラメータ"""

    alpha: float = 0.33  # 資本分配率
    delta: float = 0.025  # 資本減耗率（四半期）
    theta: float = 0.75  # Calvo価格硬直性（75%が価格維持）
    epsilon: float = 6.0  # 財の代替弾力性
    psi: float = 0.5  # 価格インデクセーション


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


@dataclass(frozen=True)
class CentralBankParameters:
    """中央銀行パラメータ"""

    rho_r: float = 0.85  # 金利平滑化
    phi_pi: float = 1.5  # インフレ反応係数
    phi_y: float = 0.125  # 産出ギャップ反応係数
    pi_target: float = 0.005  # インフレ目標（四半期、年率2%）
    r_lower_bound: float = -0.001  # 名目金利下限（ZLB、若干のマイナス金利許容）


@dataclass(frozen=True)
class FinancialParameters:
    """金融部門パラメータ（BGG型簡略版）"""

    chi_b: float = 0.05  # 外部資金プレミアム弾力性
    leverage_ss: float = 2.0  # 定常状態レバレッジ
    survival_rate: float = 0.975  # 企業家生存率


@dataclass(frozen=True)
class ShockParameters:
    """ショックパラメータ"""

    # 持続性
    rho_a: float = 0.90  # 技術ショック
    rho_g: float = 0.90  # 政府支出ショック
    rho_tau_c: float = 0.95  # 消費税ショック
    rho_m: float = 0.50  # 金融政策ショック
    rho_risk: float = 0.75  # リスクプレミアムショック

    # 標準偏差
    sigma_a: float = 0.01
    sigma_g: float = 0.01
    sigma_tau_c: float = 0.005
    sigma_m: float = 0.0025
    sigma_risk: float = 0.01


@dataclass
class DefaultParameters:
    """全パラメータを統合したデフォルト設定"""

    household: HouseholdParameters = field(default_factory=HouseholdParameters)
    firm: FirmParameters = field(default_factory=FirmParameters)
    government: GovernmentParameters = field(default_factory=GovernmentParameters)
    central_bank: CentralBankParameters = field(default_factory=CentralBankParameters)
    financial: FinancialParameters = field(default_factory=FinancialParameters)
    shocks: ShockParameters = field(default_factory=ShockParameters)

    def with_updates(
        self,
        household: HouseholdParameters | None = None,
        firm: FirmParameters | None = None,
        government: GovernmentParameters | None = None,
        central_bank: CentralBankParameters | None = None,
        financial: FinancialParameters | None = None,
        shocks: ShockParameters | None = None,
    ) -> Self:
        """パラメータの一部を更新した新しいインスタンスを返す"""
        return replace(
            self,
            household=household if household is not None else self.household,
            firm=firm if firm is not None else self.firm,
            government=government if government is not None else self.government,
            central_bank=central_bank if central_bank is not None else self.central_bank,
            financial=financial if financial is not None else self.financial,
            shocks=shocks if shocks is not None else self.shocks,
        )


# 定数定義
QUARTERS_PER_YEAR = 4
DEFAULT_SIMULATION_PERIODS = 40  # 10年間
DEFAULT_IMPULSE_SIZE = 0.01  # 1%ショック
