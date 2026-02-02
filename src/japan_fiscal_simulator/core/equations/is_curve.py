"""IS曲線（動学的IS方程式）

習慣形成なし（h=0）:
y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y·g_t + a_t

習慣形成あり（h>0）:
y_t = h·y_{t-1} + (1-h)·E[y_{t+1}] - σ_h^{-1}(r_t - E[π_{t+1}]) + g_y·g_t + a_t

ここで σ_h = σ·(1-h)/(1+h) は習慣形成調整後の異時点間代替弾力性の逆数

標準化形式（=0、習慣形成あり）:
y_t - h·y_{t-1} - (1-h)·E[y_{t+1}] + σ_h^{-1}·r_t - σ_h^{-1}·E[π_{t+1}] - g_y·g_t - a_t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class ISCurveParameters:
    """IS曲線のパラメータ"""

    sigma: float  # 異時点間代替弾力性の逆数
    g_y: float  # 政府支出/GDP比率（政府支出の効果係数）
    habit: float = 0.0  # 習慣形成パラメータ（0-1）


class ISCurve:
    """IS曲線

    消費のオイラー方程式から導出される動学的IS曲線。
    産出ギャップが実質金利と期待産出に依存する。

    習慣形成（habit > 0）の場合、消費が前期の消費にも依存し、
    消費応答がハンプ型（hump-shaped）になる。
    """

    def __init__(self, params: ISCurveParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "IS Curve"

    @property
    def description(self) -> str:
        if self.params.habit > 0:
            return (
                "y_t = h·y_{t-1} + (1-h)·E[y_{t+1}] "
                "- σ_h^{-1}(r_t - E[π_{t+1}]) + g_y·g_t + a_t"
            )
        return "y_t = E[y_{t+1}] - σ^{-1}(r_t - E[π_{t+1}]) + g_y·g_t + a_t"

    def coefficients(self) -> EquationCoefficients:
        """IS曲線の係数を返す

        習慣形成なし:
        y_t - E[y_{t+1}] + σ^{-1}·r_t - σ^{-1}·E[π_{t+1}] - g_y·g_t - a_t = 0

        習慣形成あり:
        y_t - h·y_{t-1} - (1-h)·E[y_{t+1}] + σ_h^{-1}·r_t - σ_h^{-1}·E[π_{t+1}]
            - g_y·g_t - a_t = 0
        """
        h = self.params.habit
        sigma = self.params.sigma

        if h > 0:
            # 習慣形成あり: σ_h = σ·(1-h)/(1+h)
            sigma_h = sigma * (1 - h) / (1 + h)
            sigma_inv = 1.0 / sigma_h
            y_forward_coef = -(1 - h)
            y_lag_coef = -h
        else:
            # 習慣形成なし（従来）
            sigma_inv = 1.0 / sigma
            y_forward_coef = -1.0
            y_lag_coef = 0.0

        return EquationCoefficients(
            # 期待値（t+1期）
            y_forward=y_forward_coef,
            pi_forward=-sigma_inv,
            # 当期（t期）
            y_current=1.0,
            r_current=sigma_inv,
            g_current=-self.params.g_y,
            a_current=-1.0,
            # 前期（t-1期）
            y_lag=y_lag_coef,
        )
