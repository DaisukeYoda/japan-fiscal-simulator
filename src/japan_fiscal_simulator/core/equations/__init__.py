"""方程式モジュール

DSGEモデルの構造方程式を提供する。
"""

from japan_fiscal_simulator.core.equations.base import Equation, EquationCoefficients
from japan_fiscal_simulator.core.equations.fiscal_rule import (
    GovernmentSpendingProcess,
    TechnologyProcess,
)
from japan_fiscal_simulator.core.equations.is_curve import ISCurve, ISCurveParameters
from japan_fiscal_simulator.core.equations.phillips_curve import (
    PhillipsCurve,
    PhillipsCurveParameters,
    compute_phillips_slope,
)
from japan_fiscal_simulator.core.equations.taylor_rule import (
    TaylorRule,
    TaylorRuleParameters,
    check_taylor_principle,
)

__all__ = [
    "Equation",
    "EquationCoefficients",
    "GovernmentSpendingProcess",
    "ISCurve",
    "ISCurveParameters",
    "PhillipsCurve",
    "PhillipsCurveParameters",
    "TaylorRule",
    "TaylorRuleParameters",
    "TechnologyProcess",
    "check_taylor_principle",
    "compute_phillips_slope",
]
