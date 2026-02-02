"""労働需要方程式

生産関数 Y = A × K^α × N^(1-α) から導出される労働需要。

対数線形化:
n̂_t = (1/(1-α))·(ŷ_t - â_t) - (α/(1-α))·k̂_{t-1}

標準化形式（=0）:
n̂_t - (1/(1-α))·ŷ_t + (1/(1-α))·â_t + (α/(1-α))·k̂_{t-1} = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class LaborDemandParameters:
    """労働需要のパラメータ"""

    alpha: float  # 資本分配率


class LaborDemand:
    """労働需要方程式

    コブ・ダグラス生産関数から導出される労働需要。
    生産量と技術水準、資本ストックから労働投入を決定。
    """

    def __init__(self, params: LaborDemandParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Labor Demand"

    @property
    def description(self) -> str:
        return "n̂_t = (1/(1-α))·(ŷ_t - â_t) - (α/(1-α))·k̂_{t-1}"

    def coefficients(self) -> EquationCoefficients:
        """労働需要方程式の係数を返す

        標準化形式:
        n̂_t - (1/(1-α))·ŷ_t + (1/(1-α))·â_t + (α/(1-α))·k̂_{t-1} = 0
        """
        alpha = self.params.alpha
        labor_share = 1 - alpha

        return EquationCoefficients(
            n_current=1.0,
            y_current=-1.0 / labor_share,
            a_current=1.0 / labor_share,
            k_lag=alpha / labor_share,
        )
