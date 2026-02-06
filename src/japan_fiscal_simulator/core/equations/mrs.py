"""限界代替率（MRS）方程式

mrs_t = sigma * c_t + phi * n_t

標準化形式（=0）:
mrs_t - sigma*c_t - phi*n_t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class MRSEquationParameters:
    """MRS方程式のパラメータ"""

    sigma: float  # 異時点間代替弾力性の逆数
    phi: float  # 労働供給弾力性の逆数


class MRSEquation:
    """限界代替率方程式"""

    def __init__(self, params: MRSEquationParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "MRS"

    @property
    def description(self) -> str:
        return "mrs_t = sigma*c_t + phi*n_t"

    def coefficients(self) -> EquationCoefficients:
        return EquationCoefficients(
            mrs_current=1.0,
            c_current=-self.params.sigma,
            n_current=-self.params.phi,
        )
