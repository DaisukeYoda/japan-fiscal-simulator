"""補助金政策シナリオ"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.output.schemas import PolicyScenario, PolicyType, ShockType

if TYPE_CHECKING:
    from japan_fiscal_simulator.core.model import DSGEModel
    from japan_fiscal_simulator.core.simulation import ImpulseResponseResult


@dataclass
class SubsidyAnalysis:
    """補助金政策の分析結果"""

    scenario: PolicyScenario
    impulse_response: ImpulseResponseResult
    output_effect_peak: float
    investment_effect_peak: float
    fiscal_cost: float
    crowding_out_ratio: float


class SubsidyPolicy:
    """補助金政策のシナリオ生成と分析

    補助金タイプ:
    1. 投資補助金（投資税額控除）
    2. 雇用補助金
    3. 産業補助金（特定セクター向け）

    本モデルでは政府支出を通じた効果として近似
    """

    def __init__(self, model: DSGEModel) -> None:
        self.model = model

    @staticmethod
    def create_investment_subsidy_scenario(
        subsidy_rate: float = 0.01,
        shock_type: ShockType = ShockType.TEMPORARY,
        periods: int = 40,
        name: str | None = None,
    ) -> PolicyScenario:
        """投資補助金シナリオを作成

        Args:
            subsidy_rate: 補助金規模（GDP比）
            shock_type: ショックタイプ
            periods: シミュレーション期間
            name: シナリオ名
        """
        if name is None:
            name = f"投資補助金{int(subsidy_rate * 100)}%"

        return PolicyScenario(
            name=name,
            description=f"GDP比{subsidy_rate * 100:.1f}%の投資補助金を実施",
            policy_type=PolicyType.SUBSIDY,
            shock_type=shock_type,
            shock_size=subsidy_rate,
            periods=periods,
        )

    @staticmethod
    def create_employment_subsidy_scenario(
        subsidy_rate: float = 0.005,
        periods: int = 40,
    ) -> PolicyScenario:
        """雇用補助金シナリオを作成"""
        return PolicyScenario(
            name="雇用補助金",
            description=f"GDP比{subsidy_rate * 100:.1f}%の雇用補助金を実施",
            policy_type=PolicyType.SUBSIDY,
            shock_type=ShockType.TEMPORARY,
            shock_size=subsidy_rate,
            periods=periods,
        )

    @staticmethod
    def create_green_subsidy_scenario(
        subsidy_rate: float = 0.01,
        periods: int = 40,
    ) -> PolicyScenario:
        """グリーン投資補助金シナリオ"""
        return PolicyScenario(
            name="グリーン投資補助金",
            description=f"GDP比{subsidy_rate * 100:.1f}%のグリーン投資補助金",
            policy_type=PolicyType.SUBSIDY,
            shock_type=ShockType.GRADUAL,
            shock_size=subsidy_rate,
            periods=periods,
        )

    def analyze(self, scenario: PolicyScenario) -> SubsidyAnalysis:
        """シナリオを分析"""
        simulator = ImpulseResponseSimulator(self.model)

        # 補助金は政府支出ショックとして実装
        irf = simulator.simulate(
            shock_name="e_g",
            shock_size=scenario.shock_size,
            periods=scenario.periods,
        )

        y_response = irf.get_response("y")
        i_response = irf.get_response("i")
        c_response = irf.get_response("c")
        g_response = irf.get_response("g")

        output_effect_peak = float(y_response[irf.peak_response("y")[0]])
        investment_effect_peak = float(i_response[irf.peak_response("i")[0]])

        # 財政コスト（累積政府支出）
        fiscal_cost = float(irf.cumulative_response("g", scenario.periods))

        # クラウディングアウト比率
        # 民間投資・消費の減少 / 政府支出増加
        private_decline = -(c_response.sum() + i_response.sum())
        if g_response.sum() > 0:
            crowding_out_ratio = private_decline / g_response.sum()
        else:
            crowding_out_ratio = 0.0

        return SubsidyAnalysis(
            scenario=scenario,
            impulse_response=irf,
            output_effect_peak=output_effect_peak,
            investment_effect_peak=investment_effect_peak,
            fiscal_cost=fiscal_cost,
            crowding_out_ratio=max(0, crowding_out_ratio),
        )


# プリセットシナリオ
SCENARIO_INVESTMENT_SUBSIDY = SubsidyPolicy.create_investment_subsidy_scenario(0.01)
SCENARIO_EMPLOYMENT_SUBSIDY = SubsidyPolicy.create_employment_subsidy_scenario(0.005)
SCENARIO_GREEN_SUBSIDY = SubsidyPolicy.create_green_subsidy_scenario(0.01)
