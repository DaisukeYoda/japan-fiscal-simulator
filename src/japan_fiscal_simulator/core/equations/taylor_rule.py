"""Taylor則

r_t = φ_π·π_t + φ_y·y_t + e_m,t

標準化形式（=0）:
r_t - φ_π·π_t - φ_y·y_t - e_m,t = 0
"""

from dataclasses import dataclass

from japan_fiscal_simulator.core.equations.base import EquationCoefficients


@dataclass(frozen=True)
class TaylorRuleParameters:
    """Taylor則のパラメータ"""

    phi_pi: float  # インフレ反応係数
    phi_y: float  # 産出ギャップ反応係数


def check_taylor_principle(
    phi_pi: float, phi_y: float, beta: float, kappa: float
) -> tuple[bool, float]:
    """Taylor原則（解の決定性条件）をチェック

    安定性条件: φ_π + (1-β)/κ·φ_y > 1

    Returns:
        (決定性が成立するか, Taylor criterion値)
    """
    criterion = phi_pi + (1 - beta) / kappa * phi_y
    return criterion > 1, criterion


class TaylorRule:
    """Taylor則

    中央銀行の金融政策ルール。
    名目金利をインフレと産出ギャップに反応させる。
    """

    def __init__(self, params: TaylorRuleParameters) -> None:
        self.params = params

    @property
    def name(self) -> str:
        return "Taylor Rule"

    @property
    def description(self) -> str:
        return "r_t = φ_π·π_t + φ_y·y_t + e_m,t"

    def coefficients(self) -> EquationCoefficients:
        """Taylor則の係数を返す

        r_t - φ_π·π_t - φ_y·y_t - e_m,t = 0
        """
        return EquationCoefficients(
            # 当期（t期）
            r_current=1.0,
            pi_current=-self.params.phi_pi,
            y_current=-self.params.phi_y,
            # ショック
            e_m=-1.0,
        )
