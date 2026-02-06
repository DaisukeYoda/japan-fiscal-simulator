"""資源制約

y_t = s_c*c_t + s_i*i_t + s_g*g_t

標準化形式（=0）:
y_t - s_c*c_t - s_i*i_t - s_g*g_t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class ResourceConstraintParameters:
    """資源制約のパラメータ"""

    s_c: float  # 消費シェア
    s_i: float  # 投資シェア
    s_g: float  # 政府支出シェア


class ResourceConstraint:
    """財市場均衡（資源制約）"""

    def __init__(self, params: ResourceConstraintParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Resource Constraint"

    @property
    def description(self) -> str:
        return "y_t = s_c*c_t + s_i*i_t + s_g*g_t"

    def coefficients(self) -> EquationCoefficients:
        return EquationCoefficients(
            y_current=1.0,
            c_current=-self.params.s_c,
            i_current=-self.params.s_i,
            g_current=-self.params.s_g,
        )
