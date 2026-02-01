"""資本蓄積方程式

資本蓄積の法則:
k_t = (1-δ) × k_{t-1} + δ × i_t

標準化形式（=0）:
k_t - (1-δ) × k_{t-1} - δ × i_t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class CapitalAccumulationParameters:
    """資本蓄積のパラメータ"""

    delta: float  # 資本減耗率


class CapitalAccumulation:
    """資本蓄積方程式

    k_t = (1-δ) × k_{t-1} + δ × i_t

    標準化形式: k_t - (1-δ)×k_{t-1} - δ×i_t = 0
    """

    def __init__(self, params: CapitalAccumulationParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Capital Accumulation"

    @property
    def description(self) -> str:
        return "k_t = (1-δ)·k_{t-1} + δ·i_t"

    def coefficients(self) -> EquationCoefficients:
        """資本蓄積方程式の係数を返す"""
        delta = self.params.delta
        return EquationCoefficients(
            k_current=1.0,
            k_lag=-(1 - delta),
            i_current=-delta,
        )
