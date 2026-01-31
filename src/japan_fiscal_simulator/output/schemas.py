"""Pydantic schemas for JSON API output"""

from datetime import datetime
from enum import Enum
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, field_validator


class PolicyType(str, Enum):
    """政策タイプ"""

    CONSUMPTION_TAX = "consumption_tax"
    GOVERNMENT_SPENDING = "government_spending"
    TRANSFER = "transfer"
    SUBSIDY = "subsidy"
    MONETARY = "monetary"


class ShockType(str, Enum):
    """ショックタイプ"""

    PERMANENT = "permanent"
    TEMPORARY = "temporary"
    GRADUAL = "gradual"


class VariableTimeSeries(BaseModel):
    """変数の時系列データ"""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    name: str = Field(description="変数名")
    description: str = Field(description="変数の説明")
    values: list[float] = Field(description="時系列値（定常状態からの乖離率）")
    unit: str = Field(default="percent_deviation", description="単位")

    @field_validator("values", mode="before")
    @classmethod
    def convert_numpy(cls, v: Any) -> list[float]:
        if isinstance(v, np.ndarray):
            return list(v.tolist())
        return list(v) if not isinstance(v, list) else v


class SteadyStateValues(BaseModel):
    """定常状態値"""

    output: float = Field(description="産出（Y）")
    consumption: float = Field(description="消費（C）")
    investment: float = Field(description="投資（I）")
    capital: float = Field(description="資本（K）")
    labor: float = Field(description="労働（N）")
    real_wage: float = Field(description="実質賃金（W）")
    real_interest_rate: float = Field(description="実質金利（r）")
    inflation: float = Field(description="インフレ率（π）")
    government_spending: float = Field(description="政府支出（G）")
    government_debt: float = Field(description="政府債務（B）")
    tax_revenue: float = Field(description="税収（T）")
    primary_balance: float = Field(description="プライマリーバランス")


class ParameterSet(BaseModel):
    """パラメータセット"""

    beta: float = Field(description="割引率")
    sigma: float = Field(description="異時点間代替弾力性の逆数")
    phi: float = Field(description="労働供給弾力性の逆数")
    habit: float = Field(description="習慣形成パラメータ")
    alpha: float = Field(description="資本分配率")
    delta: float = Field(description="資本減耗率")
    theta: float = Field(description="Calvo価格硬直性")
    tau_c: float = Field(description="消費税率")
    tau_l: float = Field(description="労働所得税率")
    tau_k: float = Field(description="資本所得税率")
    g_y_ratio: float = Field(description="政府支出/GDP比率")
    b_y_ratio: float = Field(description="政府債務/GDP比率")
    rho_r: float = Field(description="金利平滑化")
    phi_pi: float = Field(description="インフレ反応係数")
    phi_y: float = Field(description="産出ギャップ反応係数")


class PolicyScenario(BaseModel):
    """政策シナリオ定義"""

    name: str = Field(description="シナリオ名")
    description: str = Field(description="シナリオの説明")
    policy_type: PolicyType = Field(description="政策タイプ")
    shock_type: ShockType = Field(default=ShockType.TEMPORARY, description="ショックタイプ")
    shock_size: float = Field(description="ショックサイズ（例: -0.02 = 2%減税）")
    shock_persistence: float = Field(default=0.9, ge=0.0, le=1.0, description="ショックの持続性")
    periods: int = Field(default=40, ge=1, description="シミュレーション期間")


class ImpulseResponse(BaseModel):
    """インパルス応答結果"""

    periods: int = Field(description="シミュレーション期間")
    time_axis: list[int] = Field(description="時間軸（四半期）")
    variables: dict[str, VariableTimeSeries] = Field(description="変数別の応答")

    def get_variable(self, name: str) -> VariableTimeSeries | None:
        """変数名で時系列を取得"""
        return self.variables.get(name)


class FiscalMultiplier(BaseModel):
    """財政乗数"""

    impact_multiplier: float = Field(description="インパクト乗数（即時効果）")
    peak_multiplier: float = Field(description="ピーク乗数")
    peak_period: int = Field(description="ピーク時期（四半期）")
    cumulative_multiplier_4q: float = Field(description="累積乗数（1年）")
    cumulative_multiplier_8q: float = Field(description="累積乗数（2年）")
    cumulative_multiplier_20q: float = Field(description="累積乗数（5年）")
    present_value_multiplier: float = Field(description="現在価値乗数")


class SimulationResult(BaseModel):
    """シミュレーション結果"""

    scenario: PolicyScenario = Field(description="実行されたシナリオ")
    parameters: ParameterSet = Field(description="使用されたパラメータ")
    steady_state: SteadyStateValues = Field(description="定常状態値")
    impulse_response: ImpulseResponse = Field(description="インパルス応答")
    fiscal_multiplier: FiscalMultiplier | None = Field(
        default=None, description="財政乗数（該当する場合）"
    )
    blanchard_kahn_satisfied: bool = Field(description="Blanchard-Kahn条件充足")
    num_stable_eigenvalues: int = Field(description="安定固有値の数")
    num_unstable_eigenvalues: int = Field(description="不安定固有値の数")
    timestamp: datetime = Field(default_factory=datetime.now, description="実行日時")
    computation_time_ms: float = Field(description="計算時間（ミリ秒）")


class ScenarioComparison(BaseModel):
    """シナリオ比較項目"""

    scenario_name: str
    impact_on_output: float = Field(description="産出への影響（%）")
    impact_on_consumption: float = Field(description="消費への影響（%）")
    impact_on_investment: float = Field(description="投資への影響（%）")
    impact_on_inflation: float = Field(description="インフレへの影響（%pt）")
    impact_on_debt: float = Field(description="政府債務への影響（%）")
    fiscal_multiplier: float = Field(description="財政乗数")


class ComparisonResult(BaseModel):
    """複数シナリオ比較結果"""

    baseline_scenario: str = Field(description="ベースラインシナリオ名")
    comparisons: list[ScenarioComparison] = Field(description="比較結果リスト")
    summary: str = Field(description="比較サマリー")
    timestamp: datetime = Field(default_factory=datetime.now)


class ReportContent(BaseModel):
    """レポート内容"""

    title: str
    executive_summary: str
    simulation_results: list[SimulationResult]
    comparison: ComparisonResult | None = None
    policy_recommendations: list[str]
    caveats: list[str]
    generated_at: datetime = Field(default_factory=datetime.now)
