"""Phillips曲線（NKPC: New Keynesian Phillips Curve）

π_t = β·E[π_{t+1}] + κ·y_t

標準化形式（=0）:
π_t - β·E[π_{t+1}] - κ·y_t = 0

κ = (1-θ)(1-βθ)/θ （Calvoモデルから導出）
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class PhillipsCurveParameters:
    """Phillips曲線のパラメータ"""

    beta: float  # 割引率
    theta: float  # Calvo価格硬直性（価格を変更しない確率）


def compute_phillips_slope(beta: float, theta: float) -> float:
    """Phillips曲線のスロープκを計算

    κ = (1-θ)(1-βθ)/θ

    Calvoモデルにおいて、各期に確率(1-θ)で価格を調整できる場合の
    インフレと産出ギャップの関係を表す係数。
    """
    return (1 - theta) * (1 - beta * theta) / theta


class PhillipsCurve:
    """New Keynesian Phillips Curve

    Calvo型価格設定から導出される前向きPhillips曲線。
    インフレは期待インフレと産出ギャップに依存する。
    """

    def __init__(self, params: PhillipsCurveParameters) -> None:
        self.params = params
        self._kappa = compute_phillips_slope(params.beta, params.theta)

    @property
    def name(self) -> str:
        return "Phillips Curve"

    @property
    def description(self) -> str:
        return "π_t = β·E[π_{t+1}] + κ·y_t"

    @property
    def kappa(self) -> float:
        """Phillips曲線のスロープ"""
        return self._kappa

    def coefficients(self) -> EquationCoefficients:
        """Phillips曲線の係数を返す

        π_t - β·E[π_{t+1}] - κ·y_t = 0
        """
        return EquationCoefficients(
            # 期待値（t+1期）
            pi_forward=-self.params.beta,
            # 当期（t期）
            pi_current=1.0,
            y_current=-self._kappa,
        )
