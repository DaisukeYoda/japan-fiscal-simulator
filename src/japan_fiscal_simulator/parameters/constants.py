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

    # シミュレーションのフォールバック
    fallback_decay_rate: float = 0.9  # 次元不一致時の減衰率


@dataclass(frozen=True)
class SteadyStateRatios:
    """定常状態の比率"""

    consumption_output: float = 0.60  # C/Y: 消費/GDP比率
    investment_output: float = 0.20  # I/Y: 投資/GDP比率
    labor_income_share: float = 0.67  # 労働所得シェア (1-α に近似)
    capital_income_share: float = 0.33  # 資本所得シェア (α)


# デフォルトインスタンス
STEADY_STATE_RATIOS = SteadyStateRatios()
SOLVER_CONSTANTS = SolverConstants()
