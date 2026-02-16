"""賃金のNew Keynesian Phillips Curve

Calvo型賃金硬直性に基づく賃金動学。MRS（限界代替率）は別方程式で与える。

ŵ_t = (β/(1+β))E[ŵ_{t+1}] + (1/(1+β))ŵ_{t-1}
    + (λ_w/(1+β))(mrs_t - ŵ_t) + e_w,t

ここで:
- λ_w = (1-θ_w)(1-β·θ_w)/θ_w（賃金調整速度）

標準化形式（=0）:
(1 + λ_w/(1+β))·ŵ_t - (β/(1+β))·E[ŵ_{t+1}] - (1/(1+β))·ŵ_{t-1}
    - (λ_w/(1+β))·mrs_t - e_w,t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class WagePhillipsCurveParameters:
    """賃金NKPCのパラメータ"""

    beta: float  # 割引率
    theta_w: float  # Calvo賃金硬直性（賃金を変更しない確率）
    sigma: float  # 互換性維持のため保持（係数計算には未使用）
    phi: float  # 互換性維持のため保持（係数計算には未使用）


def compute_wage_adjustment_speed(beta: float, theta_w: float) -> float:
    """賃金調整速度 λ_w を計算"""
    return (1 - theta_w) * (1 - beta * theta_w) / theta_w


class WagePhillipsCurve:
    """賃金のNew Keynesian Phillips Curve"""

    def __init__(self, params: WagePhillipsCurveParameters) -> None:
        self.params = params
        self._lambda_w = compute_wage_adjustment_speed(params.beta, params.theta_w)

    @property
    def name(self) -> str:
        return "Wage Phillips Curve"

    @property
    def description(self) -> str:
        return "ŵ_t = (β/(1+β))E[ŵ_{t+1}] + (1/(1+β))ŵ_{t-1} + (λ_w/(1+β))(mrs_t - ŵ_t) + e_w,t"

    @property
    def lambda_w(self) -> float:
        """賃金調整速度"""
        return self._lambda_w

    def coefficients(self) -> EquationCoefficients:
        """賃金NKPCの係数を返す"""
        beta = self.params.beta
        lambda_w = self._lambda_w
        denom = 1 + beta

        return EquationCoefficients(
            w_current=1.0 + lambda_w / denom,
            w_forward=-beta / denom,
            w_lag=-1.0 / denom,
            mrs_current=-lambda_w / denom,
            e_w=-1.0,
        )
