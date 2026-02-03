"""賃金のNew Keynesian Phillips Curve

Calvo型賃金硬直性に基づく賃金動学。MRS（限界代替率）を代入済み。

ŵ_t = (β/(1+β))E[ŵ_{t+1}] + (1/(1+β))ŵ_{t-1}
    + (λ_w/(1+β))(σ·ĉ_t + φ·n̂_t - ŵ_t) + e_w,t

ここで:
- λ_w = (1-θ_w)(1-β·θ_w)/θ_w（賃金調整速度）
- mrs_t = σ·ĉ_t + φ·n̂_t（限界代替率）

標準化形式（=0）:
(1 + λ_w/(1+β))·ŵ_t - (β/(1+β))·E[ŵ_{t+1}] - (1/(1+β))·ŵ_{t-1}
    - (λ_w·σ/(1+β))·ĉ_t - (λ_w·φ/(1+β))·n̂_t - e_w,t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class WagePhillipsCurveParameters:
    """賃金NKPCのパラメータ"""

    beta: float  # 割引率
    theta_w: float  # Calvo賃金硬直性（賃金を変更しない確率）
    sigma: float  # 異時点間代替弾力性の逆数
    phi: float  # 労働供給弾力性の逆数（Frisch弾力性）


def compute_wage_adjustment_speed(beta: float, theta_w: float) -> float:
    """賃金調整速度 λ_w を計算

    λ_w = (1-θ_w)(1-β·θ_w)/θ_w

    価格のPhillips曲線スロープ κ と同様の構造。
    θ_wが高いほど（賃金硬直性が高いほど）λ_wは小さくなる。

    Args:
        beta: 割引率
        theta_w: Calvo賃金硬直性

    Returns:
        賃金調整速度 λ_w
    """
    return (1 - theta_w) * (1 - beta * theta_w) / theta_w


class WagePhillipsCurve:
    """賃金のNew Keynesian Phillips Curve

    Calvo型賃金設定から導出される前向き・後向き賃金動学。
    MRSを代入済みの形式で、消費と労働に直接依存。
    """

    def __init__(self, params: WagePhillipsCurveParameters) -> None:
        self.params = params
        self._lambda_w = compute_wage_adjustment_speed(params.beta, params.theta_w)

    @property
    def name(self) -> str:
        return "Wage Phillips Curve"

    @property
    def description(self) -> str:
        return (
            "ŵ_t = (β/(1+β))E[ŵ_{t+1}] + (1/(1+β))ŵ_{t-1} "
            "+ (λ_w/(1+β))(mrs_t - ŵ_t) + e_w,t"
        )

    @property
    def lambda_w(self) -> float:
        """賃金調整速度"""
        return self._lambda_w

    def coefficients(self) -> EquationCoefficients:
        """賃金NKPCの係数を返す

        標準化形式:
        (1 + λ_w/(1+β))·ŵ_t - (β/(1+β))·E[ŵ_{t+1}] - (1/(1+β))·ŵ_{t-1}
            - (λ_w·σ/(1+β))·ĉ_t - (λ_w·φ/(1+β))·n̂_t - e_w,t = 0
        """
        beta = self.params.beta
        sigma = self.params.sigma
        phi = self.params.phi
        lambda_w = self._lambda_w

        denom = 1 + beta

        # 係数計算
        w_current_coef = 1.0 + lambda_w / denom
        w_forward_coef = -beta / denom
        w_lag_coef = -1.0 / denom
        c_current_coef = -lambda_w * sigma / denom
        n_current_coef = -lambda_w * phi / denom

        return EquationCoefficients(
            w_current=w_current_coef,
            w_forward=w_forward_coef,
            w_lag=w_lag_coef,
            c_current=c_current_coef,
            n_current=n_current_coef,
            e_w=-1.0,
        )
