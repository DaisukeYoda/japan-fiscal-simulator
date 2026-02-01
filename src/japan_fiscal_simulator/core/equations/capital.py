"""資本蓄積方程式

資本蓄積の法則（対数線形化済み）:
k̂_t = (1-δ) × k̂_{t-1} + δ × î_t

水準での方程式 K_t = (1-δ)K_{t-1} + I_t を対数線形化。
定常状態で I̅ = δK̅ となるため、δ係数が現れる。

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
    """資本蓄積方程式（対数線形化済み）

    k̂_t = (1-δ)k̂_{t-1} + δî_t

    水準 K_t = (1-δ)K_{t-1} + I_t の対数線形化。
    定常状態で I̅ = δK̅ のため、投資にδ係数がかかる。

    標準化形式: k_t - (1-δ)k_{t-1} - δi_t = 0
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
