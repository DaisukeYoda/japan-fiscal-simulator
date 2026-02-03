"""方程式モジュール

DSGEモデルの構造方程式を提供する。
"""

from japan_fiscal_simulator.core.equations.base import Equation, EquationCoefficients
from japan_fiscal_simulator.core.equations.capital import (
    CapitalAccumulation,
    CapitalAccumulationParameters,
)
from japan_fiscal_simulator.core.equations.fiscal_rule import (
    GovernmentSpendingProcess,
    TechnologyProcess,
)
from japan_fiscal_simulator.core.equations.investment import (
    CapitalRentalRateEquation,
    InvestmentAdjustmentEquation,
    InvestmentAdjustmentParameters,
    TobinsQEquation,
    TobinsQParameters,
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
    "CapitalAccumulation",
    "CapitalAccumulationParameters",
    "CapitalRentalRateEquation",
    "Equation",
    "EquationCoefficients",
    "GovernmentSpendingProcess",
    "ISCurve",
    "ISCurveParameters",
    "InvestmentAdjustmentEquation",
    "InvestmentAdjustmentParameters",
    "PhillipsCurve",
    "PhillipsCurveParameters",
    "TaylorRule",
    "TaylorRuleParameters",
    "TechnologyProcess",
    "TobinsQEquation",
    "TobinsQParameters",
    "check_taylor_principle",
    "compute_phillips_slope",
]
