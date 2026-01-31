"""社会保障政策シナリオ"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.output.schemas import PolicyScenario, PolicyType, ShockType

if TYPE_CHECKING:
    from japan_fiscal_simulator.core.model import DSGEModel
    from japan_fiscal_simulator.core.simulation import ImpulseResponseResult


@dataclass
class SocialSecurityAnalysis:
    """社会保障政策の分析結果"""

    scenario: PolicyScenario
    impulse_response: ImpulseResponseResult
    output_effect_peak: float
    consumption_effect_peak: float
    debt_impact: float
    distributional_note: str


class SocialSecurityPolicy:
    """社会保障（移転支払い）政策のシナリオ生成と分析

    注: 本モデルは代表的家計モデルのため、
    異質性を持つモデル（HANKなど）と比較して
    移転支払いの効果が過小評価される可能性がある
    """

    def __init__(self, model: DSGEModel) -> None:
        self.model = model

    @staticmethod
    def create_transfer_increase_scenario(
        increase_rate: float = 0.01,
        shock_type: ShockType = ShockType.TEMPORARY,
        periods: int = 40,
        name: str | None = None,
    ) -> PolicyScenario:
        """移転支払い増加シナリオを作成

        Args:
            increase_rate: 増加率（GDP比、例: 0.01 = 1%）
            shock_type: ショックタイプ
            periods: シミュレーション期間
            name: シナリオ名
        """
        if name is None:
            name = f"社会保障給付{int(increase_rate * 100)}%増額"

        return PolicyScenario(
            name=name,
            description=f"移転支払いをGDP比{increase_rate * 100:.1f}%増加させる政策",
            policy_type=PolicyType.TRANSFER,
            shock_type=shock_type,
            shock_size=increase_rate,
            periods=periods,
        )

    @staticmethod
    def create_pension_reform_scenario(
        change_rate: float = -0.01,
        periods: int = 40,
    ) -> PolicyScenario:
        """年金改革シナリオ（給付調整）"""
        direction = "削減" if change_rate < 0 else "増額"
        return PolicyScenario(
            name=f"年金給付{direction}",
            description=f"年金給付をGDP比{abs(change_rate) * 100:.1f}%{direction}",
            policy_type=PolicyType.TRANSFER,
            shock_type=ShockType.GRADUAL,
            shock_size=change_rate,
            periods=periods,
        )

    def analyze(self, scenario: PolicyScenario) -> SocialSecurityAnalysis:
        """シナリオを分析"""
        simulator = ImpulseResponseSimulator(self.model)

        # 移転支払いは政府支出ショックとして近似
        # より精密なモデルでは専用のショックが必要
        irf = simulator.simulate(
            shock_name="e_g",
            shock_size=scenario.shock_size,
            periods=scenario.periods,
        )

        y_response = irf.get_response("y")
        c_response = irf.get_response("c")
        b_response = irf.get_response("b")

        output_effect_peak = float(y_response[irf.peak_response("y")[0]])
        consumption_effect_peak = float(c_response[irf.peak_response("c")[0]])
        debt_impact = float(b_response[-1])  # 長期的な債務への影響

        distributional_note = (
            "注: 本モデルは代表的家計を仮定しているため、"
            "移転支払いの再分配効果や流動性制約の影響は捕捉できません。"
            "HANKモデルなどの異質性モデルでは、より大きな効果が予想されます。"
        )

        return SocialSecurityAnalysis(
            scenario=scenario,
            impulse_response=irf,
            output_effect_peak=output_effect_peak,
            consumption_effect_peak=consumption_effect_peak,
            debt_impact=debt_impact,
            distributional_note=distributional_note,
        )


# プリセットシナリオ
SCENARIO_TRANSFER_INCREASE = SocialSecurityPolicy.create_transfer_increase_scenario(0.01)
SCENARIO_PENSION_CUT = SocialSecurityPolicy.create_pension_reform_scenario(-0.005)
