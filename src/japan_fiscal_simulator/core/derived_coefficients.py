"""導出係数

構造パラメータから係数を計算する。

これらの係数は、経済学的な関係から導出される。
ハードコードされた定数の代わりに使用する。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@dataclass(frozen=True)
class ImpulseCoefficients:
    """インパルス応答の導出係数"""

    # 政府支出関連
    government_spending_investment_spillover: float
    crowding_out_consumption: float

    # 消費税関連
    consumption_tax_elasticity: float
    output_tax_multiplier_factor: float
    inflation_tax_passthrough: float
    debt_tax_effect: float

    # 技術ショック関連
    technology_investment_share: float

    # 金融政策関連
    monetary_investment_elasticity: float

    # リスクプレミアム関連
    risk_interest_rate_response: float
    risk_investment_response: float
    risk_output_response: float
    risk_consumption_response: float


@dataclass(frozen=True)
class TransitionCoefficients:
    """状態遷移の導出係数"""

    debt_persistence: float
    investment_accelerator: float


class DerivedCoefficients:
    """構造パラメータから係数を導出するクラス

    経済学的な関係に基づいて、インパルス応答係数や
    状態遷移係数を計算する。
    """

    def __init__(self, params: DefaultParameters) -> None:
        self.params = params

    def compute_impulse_coefficients(self) -> ImpulseCoefficients:
        """インパルス応答係数を計算"""
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        fin = self.params.financial

        # 投資の金利感応度は資本コストから導出
        # 投資の利子弾力性 ≈ 1 / (r_ss + δ) ≈ σ (近似)
        monetary_investment_elasticity = hh.sigma

        # 政府支出の投資へのスピルオーバー
        # 乗数効果を通じた波及（0.5程度）
        government_spending_investment_spillover = 0.5

        # 消費のクラウディングアウト
        # 政府支出1単位あたりの消費減少
        crowding_out_consumption = 1.0 - gov.g_y_ratio

        # 消費税弾力性は税率と代替弾力性から導出
        consumption_tax_elasticity = 1.0 / (1 + gov.tau_c)

        # 産出への税乗数係数
        output_tax_multiplier_factor = gov.g_y_ratio / (1 - gov.g_y_ratio)

        # インフレへの転嫁率（価格硬直性に依存）
        inflation_tax_passthrough = (1 - firm.theta) * gov.tau_c

        # 債務への税効果
        debt_tax_effect = gov.b_y_ratio * gov.tau_c

        # 技術ショックの投資シェア（加速度効果）
        # 技術向上で投資需要が増加
        technology_investment_share = 1.0 + firm.delta / (1 / hh.beta - 1 + firm.delta)

        # リスクプレミアムの影響係数
        risk_interest_rate_response = fin.chi_b
        risk_investment_response = fin.chi_b * fin.leverage_ss
        risk_output_response = risk_investment_response * firm.alpha
        risk_consumption_response = risk_output_response * (1 - gov.g_y_ratio)

        return ImpulseCoefficients(
            government_spending_investment_spillover=government_spending_investment_spillover,
            crowding_out_consumption=crowding_out_consumption,
            consumption_tax_elasticity=consumption_tax_elasticity,
            output_tax_multiplier_factor=output_tax_multiplier_factor,
            inflation_tax_passthrough=inflation_tax_passthrough,
            debt_tax_effect=debt_tax_effect,
            technology_investment_share=technology_investment_share,
            monetary_investment_elasticity=monetary_investment_elasticity,
            risk_interest_rate_response=risk_interest_rate_response,
            risk_investment_response=risk_investment_response,
            risk_output_response=risk_output_response,
            risk_consumption_response=risk_consumption_response,
        )

    def compute_transition_coefficients(self) -> TransitionCoefficients:
        """状態遷移係数を計算"""
        firm = self.params.firm
        hh = self.params.household

        # 債務持続性は金利と成長率から導出
        # 近似: (1 + r) / (1 + g) ≈ 1/β ≈ 1 - small
        # 高い持続性（0.99）は日本の低金利・低成長を反映
        debt_persistence = 1.0 - (1.0 / hh.beta - 1)

        # 投資加速度係数
        # 資本蓄積方程式から: I/K = δ + g_k
        # 加速度効果: dI/dY ≈ K/Y * (δ + adjustment)
        capital_output_ratio = firm.alpha / (1 / hh.beta - 1 + firm.delta)
        investment_accelerator = capital_output_ratio * firm.delta

        return TransitionCoefficients(
            debt_persistence=debt_persistence,
            investment_accelerator=investment_accelerator,
        )
