"""IS曲線（動学的IS方程式）

y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y·g_t + a_t

標準化形式（=0）:
y_t - E[y_{t+1}] + σ^{-1}·r_t - σ^{-1}·E[π_{t+1}] - g_y·g_t - a_t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class ISCurveParameters:
    """IS曲線のパラメータ"""

    sigma: float  # 異時点間代替弾力性の逆数
    g_y: float  # 政府支出/GDP比率（政府支出の効果係数）


class ISCurve:
    """IS曲線

    消費のオイラー方程式から導出される動学的IS曲線。
    産出ギャップが実質金利と期待産出に依存する。
    """

    def __init__(self, params: ISCurveParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "IS Curve"

    @property
    def description(self) -> str:
        return "y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y·g_t + a_t"

    def coefficients(self) -> EquationCoefficients:
        """IS曲線の係数を返す

        y_t - E[y_{t+1}] + σ^{-1}·r_t - σ^{-1}·E[π_{t+1}] - g_y·g_t - a_t = 0
        """
        sigma_inv = 1.0 / self.params.sigma

        return EquationCoefficients(
            # 期待値（t+1期）
            y_forward=-1.0,
            pi_forward=-sigma_inv,
            # 当期（t期）
            y_current=1.0,
            r_current=sigma_inv,
            g_current=-self.params.g_y,
            a_current=-1.0,
        )
