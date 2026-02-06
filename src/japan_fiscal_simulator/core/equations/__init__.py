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
from japan_fiscal_simulator.core.equations.labor_demand import (
    LaborDemand,
    LaborDemandParameters,
)
from japan_fiscal_simulator.core.equations.marginal_cost import (
    MarginalCostEquation,
    MarginalCostParameters,
)
from japan_fiscal_simulator.core.equations.mrs import (
    MRSEquation,
    MRSEquationParameters,
)
from japan_fiscal_simulator.core.equations.phillips_curve import (
    PhillipsCurve,
    PhillipsCurveParameters,
    compute_phillips_slope,
)
from japan_fiscal_simulator.core.equations.resource_constraint import (
    ResourceConstraint,
    ResourceConstraintParameters,
)
from japan_fiscal_simulator.core.equations.taylor_rule import (
    TaylorRule,
    TaylorRuleParameters,
    check_taylor_principle,
)
from japan_fiscal_simulator.core.equations.wage_phillips import (
    WagePhillipsCurve,
    WagePhillipsCurveParameters,
    compute_wage_adjustment_speed,
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
    "LaborDemand",
    "LaborDemandParameters",
    "MarginalCostEquation",
    "MarginalCostParameters",
    "MRSEquation",
    "MRSEquationParameters",
    "PhillipsCurve",
    "PhillipsCurveParameters",
    "ResourceConstraint",
    "ResourceConstraintParameters",
    "TaylorRule",
    "TaylorRuleParameters",
    "TechnologyProcess",
    "TobinsQEquation",
    "TobinsQParameters",
    "WagePhillipsCurve",
    "WagePhillipsCurveParameters",
    "check_taylor_principle",
    "compute_phillips_slope",
    "compute_wage_adjustment_speed",
]
