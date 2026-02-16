"""Phillips曲線（NKPC: New Keynesian Phillips Curve）

価格インデクセーションつきの拡張形:
π_t = (ι_p/(1+βι_p)) * π_{t-1}
    + (β/(1+βι_p)) * E[π_{t+1}]
    + κ * mc_t
    + e_p,t

標準化形式（=0）:
π_t
- (ι_p/(1+βι_p)) * π_{t-1}
- (β/(1+βι_p)) * E[π_{t+1}]
- κ * mc_t
- e_p,t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class PhillipsCurveParameters:
    """Phillips曲線のパラメータ"""

    beta: float  # 割引率
    theta: float  # Calvo価格硬直性（価格を変更しない確率）
    iota_p: float = 0.0  # 価格インデクセーション
    rho_p: float = 0.0  # 価格マークアップショックの持続性


def compute_phillips_slope(beta: float, theta: float, iota_p: float = 0.0) -> float:
    """Phillips曲線のスロープκを計算

    κ = ((1-θ)(1-βθ)/θ) / (1+βι_p)

    Calvoモデルと価格インデクセーションから導出される、
    インフレと実質限界費用の関係を表す係数。
    """
    base_kappa = (1 - theta) * (1 - beta * theta) / theta
    denominator = 1 + beta * iota_p
    return base_kappa / denominator


class PhillipsCurve:
    """New Keynesian Phillips Curve

    Calvo型価格設定から導出される前向きPhillips曲線。
    インフレは期待インフレ、過去インフレ、実質限界費用に依存する。
    """

    def __init__(self, params: PhillipsCurveParameters) -> None:
        self.params = params
        self._kappa = compute_phillips_slope(params.beta, params.theta, params.iota_p)

    @property
    def name(self) -> str:
        return "Phillips Curve"

    @property
    def description(self) -> str:
        return "π_t = (ι_p/(1+βι_p))π_{t-1} + (β/(1+βι_p))E[π_{t+1}] + κ·mc_t + e_p,t"

    @property
    def kappa(self) -> float:
        """Phillips曲線のスロープ"""
        return self._kappa

    def coefficients(self) -> EquationCoefficients:
        """Phillips曲線の係数を返す

        π_t - c_lag·π_{t-1} - c_fwd·E[π_{t+1}] - κ·mc_t - e_p,t = 0
        """
        denom = 1 + self.params.beta * self.params.iota_p
        pi_lag_coef = self.params.iota_p / denom
        pi_forward_coef = self.params.beta / denom
        # e_p のAR(1)持続性が期待項を通じて当期へ与える影響を反映
        shock_scale_denom = 1.0 - pi_forward_coef * self.params.rho_p
        if abs(shock_scale_denom) < 1e-10:
            shock_scale = 1.0
        else:
            shock_scale = 1.0 / shock_scale_denom

        return EquationCoefficients(
            # 期待値（t+1期）
            pi_forward=-pi_forward_coef,
            # 当期（t期）
            pi_current=1.0,
            pi_lag=-pi_lag_coef,
            mc_current=-self._kappa,
            e_p=-shock_scale,
        )
