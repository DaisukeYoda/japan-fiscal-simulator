"""実質限界費用方程式

mc_t = α * rk_t + (1-α) * w_t - a_t

標準化形式（=0）:
mc_t - α * rk_t - (1-α) * w_t + a_t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class MarginalCostParameters:
    """限界費用方程式のパラメータ"""

    alpha: float  # 資本分配率


class MarginalCostEquation:
    """実質限界費用方程式"""

    def __init__(self, params: MarginalCostParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Marginal Cost"

    @property
    def description(self) -> str:
        return "mc_t = α * rk_t + (1-α) * w_t - a_t"

    def coefficients(self) -> EquationCoefficients:
        """限界費用方程式の係数を返す"""
        alpha = self.params.alpha
        return EquationCoefficients(
            mc_current=1.0,
            rk_current=-alpha,
            w_current=-(1.0 - alpha),
            a_current=1.0,
        )
