"""方程式基盤クラス

DSGEモデルの方程式を構造化して表現する。
"""

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class EquationCoefficients:
    """方程式の係数を格納するデータクラス

    方程式形式: A @ E[y_{t+1}] + B @ y_t + C @ y_{t-1} + D @ ε_t = 0

    各属性はその変数の係数を表す。
    例: IS曲線 y_t - E[y_{t+1}] + σ^{-1}(r_t - E[π_{t+1}]) = 0
        -> y_current=1, y_forward=-1, r_current=1/σ, pi_forward=-1/σ
    """

    # 期待値（t+1期）の係数 (A行列への寄与)
    y_forward: float = 0.0
    pi_forward: float = 0.0
    r_forward: float = 0.0
    g_forward: float = 0.0
    a_forward: float = 0.0

    # 当期（t期）の係数 (B行列への寄与)
    y_current: float = 0.0
    pi_current: float = 0.0
    r_current: float = 0.0
    g_current: float = 0.0
    a_current: float = 0.0

    # 前期（t-1期）の係数 (C行列への寄与)
    y_lag: float = 0.0
    pi_lag: float = 0.0
    r_lag: float = 0.0
    g_lag: float = 0.0
    a_lag: float = 0.0

    # ショック係数 (D行列への寄与)
    e_g: float = 0.0
    e_a: float = 0.0
    e_m: float = 0.0


class Equation(Protocol):
    """方程式のProtocol

    すべての構造方程式はこのProtocolを実装する。
    """

    @property
    def name(self) -> str:
        """方程式の名前"""
        ...

    @property
    def description(self) -> str:
        """方程式の数学的表現"""
        ...

    def coefficients(self) -> EquationCoefficients:
        """パラメータから係数を計算して返す"""
        ...
