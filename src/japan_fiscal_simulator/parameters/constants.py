"""モデル定数の定義

マジックナンバーを排除し、経済学的に意味のある名前を付ける
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class SteadyStateConstants:
    """定常状態の定数"""

    output_normalization: float = 1.0  # 産出を1に正規化
    tobin_q: float = 1.0  # 定常状態のTobin's Q
    external_finance_premium_quarterly: float = 0.005  # 外部資金プレミアム（四半期、年率2%）
    labor_hours_share: float = 0.33  # 労働時間の定常状態（1日の1/3）


@dataclass(frozen=True)
class SolverConstants:
    """ソルバーの定数"""

    default_tolerance: float = 1e-10
    default_max_iterations: int = 1000
    verification_tolerance: float = 1e-6

    # 反復法の初期値
    initial_output: float = 1.0
    initial_consumption: float = 0.6
    initial_labor: float = 0.33
    initial_capital: float = 10.0
    initial_wage: float = 1.0
    initial_interest_rate: float = 0.01


@dataclass(frozen=True)
class SteadyStateRatios:
    """定常状態の比率"""

    consumption_output: float = 0.60  # C/Y: 消費/GDP比率
    investment_output: float = 0.20  # I/Y: 投資/GDP比率
    labor_income_share: float = 0.67  # 労働所得シェア (1-α に近似)
    capital_income_share: float = 0.33  # 資本所得シェア (α)


@dataclass(frozen=True)
class ImpulseResponseCoefficients:
    """インパルス応答の係数

    経済学文献に基づくキャリブレーション値
    """

    # 政府支出乗数関連
    government_spending_multiplier: float = 1.2  # 短期財政乗数
    crowding_out_consumption: float = 0.4  # 消費のクラウディングアウト率
    government_spending_investment_spillover: float = 0.5  # 政府支出の投資への部分的波及
    labor_response_to_output: float = 0.7  # 産出増加時の労働応答

    # 消費税関連
    consumption_tax_elasticity: float = 0.8  # 消費税の消費弾力性
    output_tax_multiplier_factor: float = 0.6  # 消費税の産出乗数係数
    inflation_tax_passthrough: float = 0.3  # インフレへの転嫁率
    debt_tax_effect: float = 0.3  # 債務への効果

    # 技術ショック関連
    technology_output_elasticity: float = 1.0  # 技術の産出弾力性
    technology_consumption_share: float = 0.6  # 消費への波及
    technology_investment_share: float = 1.5  # 投資の加速度
    technology_labor_response: float = 0.3  # 労働応答
    technology_wage_response: float = 0.7  # 賃金応答
    technology_inflation_response: float = -0.1  # インフレ応答（負：デフレ圧力）

    # 金融政策関連
    monetary_output_elasticity: float = 0.5  # 金利の産出弾力性
    monetary_consumption_elasticity: float = 0.3  # 金利の消費弾力性
    monetary_investment_elasticity: float = 2.0  # 金利の投資弾力性（投資は金利に敏感）
    monetary_inflation_elasticity: float = 0.2  # 金利のインフレ弾力性

    # リスクプレミアム関連
    risk_interest_rate_response: float = 0.5  # リスクの金利応答
    risk_investment_response: float = 0.8  # リスクの投資応答
    risk_output_response: float = 0.3  # リスクの産出応答
    risk_consumption_response: float = 0.2  # リスクの消費応答


@dataclass(frozen=True)
class TransitionCoefficients:
    """状態遷移の係数"""

    debt_persistence: float = 0.99  # 政府債務の持続性
    consumption_gdp_elasticity: float = 0.6  # 消費のGDP弾力性
    consumption_tax_sensitivity: float = 0.5  # 消費の税率感応度
    investment_accelerator: float = 1.5  # 投資の加速度係数
    phillips_curve_slope_factor: float = 0.5  # Phillips曲線スロープの調整係数
    fallback_decay_rate: float = 0.9  # 次元不一致時の減衰率


# デフォルトインスタンス
STEADY_STATE_RATIOS = SteadyStateRatios()
IMPULSE_COEFFICIENTS = ImpulseResponseCoefficients()
TRANSITION_COEFFICIENTS = TransitionCoefficients()
