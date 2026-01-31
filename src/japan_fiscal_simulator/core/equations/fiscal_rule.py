"""財政・外生プロセス

AR(1)プロセスとして表現される外生変数の方程式

政府支出: g_t = ρ_g·g_{t-1} + e_g,t
技術:     a_t = ρ_a·a_{t-1} + e_a,t

標準化形式（=0）:
g_t - ρ_g·g_{t-1} - e_g,t = 0
a_t - ρ_a·a_{t-1} - e_a,t = 0
"""

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


class GovernmentSpendingProcess:
    """政府支出のAR(1)プロセス

    g_t = ρ_g·g_{t-1} + e_g,t
    """

    def __init__(self, rho_g: float) -> None:
        self.rho_g = rho_g

    @property
    def name(self) -> str:
        return "Government Spending Process"

    @property
    def description(self) -> str:
        return "g_t = ρ_g·g_{t-1} + e_g,t"

    def coefficients(self) -> EquationCoefficients:
        """政府支出プロセスの係数を返す

        g_t - ρ_g·g_{t-1} - e_g,t = 0
        """
        return EquationCoefficients(
            g_current=1.0,
            g_lag=-self.rho_g,
            e_g=-1.0,
        )


class TechnologyProcess:
    """技術のAR(1)プロセス

    a_t = ρ_a·a_{t-1} + e_a,t
    """

    def __init__(self, rho_a: float) -> None:
        self.rho_a = rho_a

    @property
    def name(self) -> str:
        return "Technology Process"

    @property
    def description(self) -> str:
        return "a_t = ρ_a·a_{t-1} + e_a,t"

    def coefficients(self) -> EquationCoefficients:
        """技術プロセスの係数を返す

        a_t - ρ_a·a_{t-1} - e_a,t = 0
        """
        return EquationCoefficients(
            a_current=1.0,
            a_lag=-self.rho_a,
            e_a=-1.0,
        )
