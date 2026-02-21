"""食料品・非食料品の複数税率政策シナリオ

理論的背景:
    消費をCobb-Douglas型で集計（C = C_f^s × C_nf^(1-s)）すると、
    対数線形化後の実効税率は加重平均に帰着する:

        τ_c_eff = s × τ_food + (1-s) × τ_nonfood

    よって既存の単一税率モデルに、実効税率を渡すことで
    2財モデルと同等の集計的効果を表現できる。
"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.output.schemas import PolicyScenario, PolicyType, ShockType

if TYPE_CHECKING:
    from japan_fiscal_simulator.core.model import DSGEModel
    from japan_fiscal_simulator.core.simulation import ImpulseResponseResult

# 日本の現行制度（2024年時点）
_CURRENT_FOOD_RATE: float = 0.08  # 食料品軽減税率 8%
_CURRENT_NONFOOD_RATE: float = 0.10  # 非食料品標準税率 10%
_JAPAN_FOOD_SHARE: float = 0.25  # 食料品消費シェア（総務省家計調査ベース）


@dataclass(frozen=True)
class DualRateTaxPolicy:
    """食料品と非食料品で税率を分けて指定し、モデルへの実効税率を計算するラッパー

    Attributes:
        tau_c_food: 食料品消費税率
        tau_c_nonfood: 非食料品消費税率
        food_share: 食料品の消費シェア（0-1）
    """

    tau_c_food: float = _CURRENT_FOOD_RATE
    tau_c_nonfood: float = _CURRENT_NONFOOD_RATE
    food_share: float = _JAPAN_FOOD_SHARE

    @property
    def tau_c_effective(self) -> float:
        """モデルに渡す実効消費税率（加重平均）"""
        return self.food_share * self.tau_c_food + (1 - self.food_share) * self.tau_c_nonfood

    def shock_size_vs(self, baseline: "DualRateTaxPolicy") -> float:
        """ベースライン制度からの実効税率の変化分"""
        return self.tau_c_effective - baseline.tau_c_effective

    def __str__(self) -> str:
        return (
            f"食料品{self.tau_c_food * 100:.0f}% / "
            f"非食料品{self.tau_c_nonfood * 100:.0f}% "
            f"→ 実効税率{self.tau_c_effective * 100:.2f}%"
        )


# 代表的な税制プリセット
CURRENT_SYSTEM = DualRateTaxPolicy(
    tau_c_food=0.08,
    tau_c_nonfood=0.10,
)
"""現行制度: 食料品8%、非食料品10%（実効9.5%）"""

FOOD_ZERO_RATE = DualRateTaxPolicy(
    tau_c_food=0.00,
    tau_c_nonfood=0.10,
)
"""食料品ゼロ税率案: 食料品0%、非食料品10%（実効7.5%）"""

UNIFORM_8PCT = DualRateTaxPolicy(
    tau_c_food=0.08,
    tau_c_nonfood=0.08,
)
"""一律8%案: 軽減税率を標準税率に揃えた場合（実効8.0%）"""

UNIFORM_10PCT = DualRateTaxPolicy(
    tau_c_food=0.10,
    tau_c_nonfood=0.10,
)
"""一律10%案: 軽減税率廃止（実効10.0%）"""


@dataclass
class DualRateTaxAnalysis:
    """複数税率政策の分析結果"""

    policy: DualRateTaxPolicy
    baseline: DualRateTaxPolicy
    scenario: PolicyScenario
    impulse_response: "ImpulseResponseResult"
    output_effect_peak: float
    consumption_effect_peak: float
    revenue_impact: float
    welfare_effect: float

    @property
    def effective_rate_change(self) -> float:
        """実効税率の変化幅（%pt）"""
        return self.policy.shock_size_vs(self.baseline) * 100


class DualRateTaxPolicyAnalyzer:
    """複数税率シナリオの生成と分析"""

    def __init__(self, model: "DSGEModel") -> None:
        self.model = model

    def create_scenario(
        self,
        policy: DualRateTaxPolicy,
        baseline: DualRateTaxPolicy = CURRENT_SYSTEM,
        shock_type: ShockType = ShockType.TEMPORARY,
        periods: int = 40,
        name: str | None = None,
    ) -> PolicyScenario:
        """複数税率シナリオを作成

        Args:
            policy: 分析する税制
            baseline: 比較基準となる税制（デフォルト: 現行制度）
            shock_type: ショックタイプ
            periods: シミュレーション期間
            name: シナリオ名（省略時は自動生成）
        """
        shock = policy.shock_size_vs(baseline)
        if name is None:
            direction = "減税" if shock < 0 else "増税"
            name = f"食料品{policy.tau_c_food * 100:.0f}%・非食料品{policy.tau_c_nonfood * 100:.0f}%（実効{direction}{abs(shock) * 100:.2f}%pt）"

        return PolicyScenario(
            name=name,
            description=str(policy),
            policy_type=PolicyType.CONSUMPTION_TAX,
            shock_type=shock_type,
            shock_size=shock,
            periods=periods,
        )

    def analyze(
        self,
        policy: DualRateTaxPolicy,
        baseline: DualRateTaxPolicy = CURRENT_SYSTEM,
        shock_type: ShockType = ShockType.TEMPORARY,
        periods: int = 40,
    ) -> DualRateTaxAnalysis:
        """複数税率シナリオを分析

        Args:
            policy: 分析する税制
            baseline: 比較基準となる税制
            shock_type: ショックタイプ
            periods: シミュレーション期間
        """
        scenario = self.create_scenario(policy, baseline, shock_type, periods)
        simulator = ImpulseResponseSimulator(self.model)
        irf = simulator.simulate(
            shock_name="e_tau",
            shock_size=scenario.shock_size,
            periods=scenario.periods,
        )

        y_response = irf.get_response("y")
        c_response = irf.get_response("c")
        tau_response = irf.get_response("tau_c")

        output_effect_peak = float(y_response[irf.peak_response("y")[0]])
        consumption_effect_peak = float(c_response[irf.peak_response("c")[0]])
        revenue_impact = tau_response[0] + c_response[0]

        sigma = self.model.params.household.sigma
        welfare_effect = c_response.mean() / sigma

        return DualRateTaxAnalysis(
            policy=policy,
            baseline=baseline,
            scenario=scenario,
            impulse_response=irf,
            output_effect_peak=output_effect_peak,
            consumption_effect_peak=consumption_effect_peak,
            revenue_impact=revenue_impact,
            welfare_effect=welfare_effect,
        )
