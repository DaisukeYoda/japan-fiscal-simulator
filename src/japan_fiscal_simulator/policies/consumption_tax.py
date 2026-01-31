"""消費税政策シナリオ"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.output.schemas import PolicyScenario, PolicyType, ShockType

if TYPE_CHECKING:
    from japan_fiscal_simulator.core.model import DSGEModel
    from japan_fiscal_simulator.core.simulation import ImpulseResponseResult


@dataclass
class ConsumptionTaxAnalysis:
    """消費税政策の分析結果"""

    scenario: PolicyScenario
    impulse_response: ImpulseResponseResult
    output_effect_peak: float  # 産出への最大効果（%）
    consumption_effect_peak: float  # 消費への最大効果（%）
    revenue_impact: float  # 税収への影響
    welfare_effect: float  # 厚生効果の近似


class ConsumptionTaxPolicy:
    """消費税政策のシナリオ生成と分析"""

    def __init__(self, model: DSGEModel) -> None:
        self.model = model

    @staticmethod
    def create_reduction_scenario(
        reduction_rate: float = 0.02,
        shock_type: ShockType = ShockType.TEMPORARY,
        periods: int = 40,
        name: str | None = None,
    ) -> PolicyScenario:
        """消費税減税シナリオを作成

        Args:
            reduction_rate: 減税幅（例: 0.02 = 2%pt）
            shock_type: ショックタイプ
            periods: シミュレーション期間
            name: シナリオ名
        """
        if name is None:
            name = f"消費税{int(reduction_rate * 100)}%pt減税"

        return PolicyScenario(
            name=name,
            description=f"消費税率を{reduction_rate * 100:.1f}%pt引き下げる政策",
            policy_type=PolicyType.CONSUMPTION_TAX,
            shock_type=shock_type,
            shock_size=-reduction_rate,  # 減税はマイナス
            periods=periods,
        )

    @staticmethod
    def create_increase_scenario(
        increase_rate: float = 0.02,
        shock_type: ShockType = ShockType.GRADUAL,
        periods: int = 40,
        name: str | None = None,
    ) -> PolicyScenario:
        """消費税増税シナリオを作成"""
        if name is None:
            name = f"消費税{int(increase_rate * 100)}%pt増税"

        return PolicyScenario(
            name=name,
            description=f"消費税率を{increase_rate * 100:.1f}%pt引き上げる政策",
            policy_type=PolicyType.CONSUMPTION_TAX,
            shock_type=shock_type,
            shock_size=increase_rate,
            periods=periods,
        )

    def analyze(self, scenario: PolicyScenario) -> ConsumptionTaxAnalysis:
        """シナリオを分析"""
        simulator = ImpulseResponseSimulator(self.model)
        irf = simulator.simulate(
            shock_name="e_tau",
            shock_size=scenario.shock_size,
            periods=scenario.periods,
        )

        # 産出への効果
        y_response = irf.get_response("y")
        output_effect_peak = float(y_response[irf.peak_response("y")[0]])

        # 消費への効果
        c_response = irf.get_response("c")
        consumption_effect_peak = float(c_response[irf.peak_response("c")[0]])

        # 税収への影響（簡易計算）
        # dTax/Tax ≈ dτ/τ + dC/C
        tau_response = irf.get_response("tau_c")
        revenue_impact = tau_response[0] + c_response[0]

        # 厚生効果の近似（消費等価変化）
        # 厳密な計算には2次近似が必要だが、ここでは簡易近似
        sigma = self.model.params.household.sigma
        welfare_effect = c_response.mean() / sigma

        return ConsumptionTaxAnalysis(
            scenario=scenario,
            impulse_response=irf,
            output_effect_peak=output_effect_peak,
            consumption_effect_peak=consumption_effect_peak,
            revenue_impact=revenue_impact,
            welfare_effect=welfare_effect,
        )


# プリセットシナリオ
SCENARIO_TAX_CUT_2PCT = ConsumptionTaxPolicy.create_reduction_scenario(0.02)
SCENARIO_TAX_CUT_5PCT = ConsumptionTaxPolicy.create_reduction_scenario(0.05)
SCENARIO_TAX_INCREASE_2PCT = ConsumptionTaxPolicy.create_increase_scenario(0.02)
