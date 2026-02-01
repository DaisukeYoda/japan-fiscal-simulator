"""投資方程式

Tobin's Q方程式、投資調整コスト方程式、資本収益率方程式を定義する。

Tobin's Q:
q_t = β × E[(1-δ)q_{t+1} + rk_{t+1}] - r_t

投資調整コスト:
i_t = i_{t-1} + (1/S'') × q_t + e_i,t

資本収益率（限界生産物条件）:
rk_t = y_t - k_{t-1}  (log-linearized)
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class TobinsQParameters:
    """Tobin's Q方程式のパラメータ"""

    beta: float  # 割引率
    delta: float  # 資本減耗率


class TobinsQEquation:
    """Tobin's Q方程式

    q_t = β × E[(1-δ)q_{t+1} + rk_{t+1}] - r_t

    資本の影の価格（Tobin's Q）は、将来の資本収益率の
    割引現在価値で決定される。

    標準化形式: q_t - β(1-δ)×E[q_{t+1}] - β×E[rk_{t+1}] + r_t = 0
    """

    def __init__(self, params: TobinsQParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Tobin's Q"

    @property
    def description(self) -> str:
        return "q_t = β·E[(1-δ)q_{t+1} + rk_{t+1}] - r_t"

    def coefficients(self) -> EquationCoefficients:
        """Tobin's Q方程式の係数を返す"""
        beta = self.params.beta
        delta = self.params.delta
        return EquationCoefficients(
            q_current=1.0,
            q_forward=-beta * (1 - delta),
            rk_forward=-beta,
            r_current=1.0,
        )


@dataclass(frozen=True)
class InvestmentAdjustmentParameters:
    """投資調整コスト方程式のパラメータ"""

    S_double_prime: float  # 投資調整コスト曲率


class InvestmentAdjustmentEquation:
    """投資調整コスト方程式

    i_t = i_{t-1} + (1/S'') × q_t + e_i,t

    投資調整コストにより、投資はTobin's Qに対して
    緩やかに反応する。S''が大きいほど調整が遅い。

    標準化形式: i_t - i_{t-1} - (1/S'')×q_t - e_i,t = 0
    """

    def __init__(self, params: InvestmentAdjustmentParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Investment Adjustment"

    @property
    def description(self) -> str:
        return "i_t = i_{t-1} + (1/S'')·q_t + e_i,t"

    def coefficients(self) -> EquationCoefficients:
        """投資調整コスト方程式の係数を返す"""
        S_double_prime = self.params.S_double_prime
        return EquationCoefficients(
            i_current=1.0,
            i_lag=-1.0,
            q_current=-1.0 / S_double_prime,
            e_i=-1.0,
        )


class CapitalRentalRateEquation:
    """資本収益率方程式

    rk_t = y_t - k_{t-1}

    対数線形化された限界生産物条件。資本収益率は
    産出と資本ストックの差で決定される。

    標準化形式: rk_t - y_t + k_{t-1} = 0
    """

    @property
    def name(self) -> str:
        return "Capital Rental Rate"

    @property
    def description(self) -> str:
        return "rk_t = y_t - k_{t-1}"

    def coefficients(self) -> EquationCoefficients:
        """資本収益率方程式の係数を返す"""
        return EquationCoefficients(
            rk_current=1.0,
            y_current=-1.0,
            k_lag=1.0,
        )
