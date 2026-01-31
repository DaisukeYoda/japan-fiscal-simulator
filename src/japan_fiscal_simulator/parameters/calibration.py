"""日本経済向けキャリブレーション"""

from dataclasses import dataclass, replace

from japan_fiscal_simulator.parameters.defaults import (
    CentralBankParameters,
    DefaultParameters,
    FinancialParameters,
    FirmParameters,
    GovernmentParameters,
    HouseholdParameters,
    ShockParameters,
)


@dataclass
class JapanCalibration:
    """日本経済向けにキャリブレーションされたパラメータセット"""

    parameters: DefaultParameters

    @classmethod
    def create(cls) -> JapanCalibration:
        """日本経済向けデフォルトキャリブレーションを作成"""
        params = DefaultParameters(
            household=HouseholdParameters(
                beta=0.999,  # 長期低金利環境
                sigma=1.5,
                phi=2.0,
                habit=0.7,  # 高い習慣形成
                chi=1.0,
            ),
            firm=FirmParameters(
                alpha=0.33,
                delta=0.025,
                theta=0.75,  # 価格硬直性
                epsilon=6.0,
                psi=0.5,
            ),
            government=GovernmentParameters(
                tau_c=0.10,  # 消費税10%
                tau_l=0.25,
                tau_k=0.30,
                g_y_ratio=0.20,
                b_y_ratio=2.00,  # 高債務
                transfer_y_ratio=0.15,
                rho_g=0.90,
                rho_tau=0.90,
                phi_b=0.02,
            ),
            central_bank=CentralBankParameters(
                rho_r=0.85,  # 高い金利平滑化
                phi_pi=1.5,
                phi_y=0.125,
                pi_target=0.005,
                r_lower_bound=-0.001,  # マイナス金利政策
            ),
            financial=FinancialParameters(
                chi_b=0.05,
                leverage_ss=2.0,
                survival_rate=0.975,
            ),
            shocks=ShockParameters(),
        )
        return cls(parameters=params)

    @classmethod
    def create_high_debt_scenario(cls) -> JapanCalibration:
        """高債務シナリオ"""
        base = cls.create()
        new_gov = replace(base.parameters.government, b_y_ratio=2.50, phi_b=0.03)
        return cls(parameters=base.parameters.with_updates(government=new_gov))

    @classmethod
    def create_zlb_scenario(cls) -> JapanCalibration:
        """ゼロ金利制約シナリオ"""
        base = cls.create()
        new_cb = replace(base.parameters.central_bank, r_lower_bound=0.0, rho_r=0.95)
        return cls(parameters=base.parameters.with_updates(central_bank=new_cb))

    def set_consumption_tax(self, rate: float) -> JapanCalibration:
        """消費税率を変更"""
        new_gov = replace(self.parameters.government, tau_c=rate)
        return JapanCalibration(parameters=self.parameters.with_updates(government=new_gov))

    def set_government_spending_ratio(self, ratio: float) -> JapanCalibration:
        """政府支出/GDP比率を変更"""
        new_gov = replace(self.parameters.government, g_y_ratio=ratio)
        return JapanCalibration(parameters=self.parameters.with_updates(government=new_gov))

    def set_transfer_ratio(self, ratio: float) -> JapanCalibration:
        """移転支払い/GDP比率を変更"""
        new_gov = replace(self.parameters.government, transfer_y_ratio=ratio)
        return JapanCalibration(parameters=self.parameters.with_updates(government=new_gov))


# プリセットシナリオ
JAPAN_BASELINE = JapanCalibration.create()
JAPAN_HIGH_DEBT = JapanCalibration.create_high_debt_scenario()
JAPAN_ZLB = JapanCalibration.create_zlb_scenario()
