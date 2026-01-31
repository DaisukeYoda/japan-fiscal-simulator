"""定常状態ソルバー"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy.optimize import fsolve

from japan_fiscal_simulator.parameters.constants import (
    SolverConstants,
    SteadyStateConstants,
)

if TYPE_CHECKING:
    from japan_fiscal_simulator.parameters.defaults import DefaultParameters


# 定数インスタンス
SS_CONST = SteadyStateConstants()
SOLVER_CONST = SolverConstants()


@dataclass
class SteadyState:
    """モデル全体の定常状態"""

    # 実体経済変数
    output: float
    consumption: float
    investment: float
    capital: float
    labor: float

    # 価格変数
    real_wage: float
    real_interest_rate: float
    nominal_interest_rate: float
    inflation: float
    marginal_cost: float
    tobin_q: float

    # 政府変数
    government_spending: float
    government_debt: float
    tax_revenue: float
    transfers: float
    consumption_tax_rate: float
    primary_balance: float

    # 金融変数
    external_finance_premium: float
    net_worth: float
    capital_return: float

    def to_dict(self) -> dict[str, float]:
        """辞書形式で返す"""
        return {
            "y": self.output,
            "c": self.consumption,
            "i": self.investment,
            "k": self.capital,
            "n": self.labor,
            "w": self.real_wage,
            "r": self.real_interest_rate,
            "R": self.nominal_interest_rate,
            "pi": self.inflation,
            "mc": self.marginal_cost,
            "q": self.tobin_q,
            "g": self.government_spending,
            "b": self.government_debt,
            "tax": self.tax_revenue,
            "tr": self.transfers,
            "tau_c": self.consumption_tax_rate,
            "pb": self.primary_balance,
            "s": self.external_finance_premium,
            "nw": self.net_worth,
            "rk": self.capital_return,
        }


class SteadyStateSolver:
    """定常状態を数値的に求めるソルバー"""

    def __init__(self, params: DefaultParameters) -> None:
        self.params = params

    def solve(self) -> SteadyState:
        """定常状態を計算

        主要な定常状態関係:
        1. β * (1 + r) = 1  →  r = 1/β - 1
        2. π = π_target
        3. R = r + π
        4. mc = 1/markup = (ε-1)/ε
        5. r + δ = α * Y/K  (資本の限界生産性)
        6. w = (1-α) * Y/N  (労働の限界生産性)
        7. Y = C + I + G
        8. I = δ * K
        """
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        fin = self.params.financial

        # 実質金利（オイラー方程式から）
        r = 1 / hh.beta - 1

        # インフレ率（中央銀行の目標）
        pi = cb.pi_target

        # 名目金利
        R = r + pi

        # マークアップと限界費用
        markup = firm.epsilon / (firm.epsilon - 1)
        mc = 1 / markup

        # 資本の限界生産性 = r + δ
        mpk = r + firm.delta

        # 資本/産出比率
        k_y = firm.alpha / mpk

        # 投資/産出比率
        i_y = firm.delta * k_y

        # 政府支出/産出比率
        g_y = gov.g_y_ratio

        # 移転支払い/産出比率
        tr_y = gov.transfer_y_ratio

        # 消費/産出比率（財市場均衡から）
        c_y = 1 - i_y - g_y

        # 産出の正規化
        y = SS_CONST.output_normalization
        c = c_y * y
        i = i_y * y
        g = g_y * y
        k = k_y * y
        tr = tr_y * y

        # 労働（生産関数から）
        # Y = K^α * N^(1-α) → N = (Y / K^α)^(1/(1-α))
        n = (y / (k**firm.alpha)) ** (1 / (1 - firm.alpha))

        # 実質賃金（限界生産性条件から）
        w = mc * (1 - firm.alpha) * y / n

        # 税収
        consumption_tax = gov.tau_c * c
        labor_tax = gov.tau_l * w * n
        capital_tax = gov.tau_k * r * k
        total_tax = consumption_tax + labor_tax + capital_tax

        # 政府債務
        b = gov.b_y_ratio * y

        # プライマリーバランス
        pb = total_tax - g - tr

        # Tobin's Q（定常状態）
        q = SS_CONST.tobin_q

        # 外部資金プレミアム
        s = SS_CONST.external_finance_premium_quarterly

        # 純資産
        nw = k / fin.leverage_ss

        # 資本収益率
        rk = r + s

        return SteadyState(
            output=y,
            consumption=c,
            investment=i,
            capital=k,
            labor=n,
            real_wage=w,
            real_interest_rate=r,
            nominal_interest_rate=R,
            inflation=pi,
            marginal_cost=mc,
            tobin_q=q,
            government_spending=g,
            government_debt=b,
            tax_revenue=total_tax,
            transfers=tr,
            consumption_tax_rate=gov.tau_c,
            primary_balance=pb,
            external_finance_premium=s,
            net_worth=nw,
            capital_return=rk,
        )

    def solve_iterative(
        self,
        tol: float = SOLVER_CONST.default_tolerance,
        max_iter: int = SOLVER_CONST.default_max_iterations,
    ) -> SteadyState:
        """反復法で定常状態を求める（より複雑なモデル用）"""
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government

        def residuals(x: np.ndarray) -> np.ndarray:
            y, c, n, k, w, r = x

            # 生産関数
            res1 = y - (k**firm.alpha) * (n ** (1 - firm.alpha))

            # 資本の限界生産性
            mpk = firm.alpha * y / k
            res2 = mpk - (r + firm.delta)

            # 労働の限界生産性（限界費用考慮）
            mc = (firm.epsilon - 1) / firm.epsilon
            res3 = w - mc * (1 - firm.alpha) * y / n

            # オイラー方程式（定常状態）
            res4 = hh.beta * (1 + r) - 1

            # 財市場均衡
            i = firm.delta * k
            g = gov.g_y_ratio * y
            res5 = y - c - i - g

            # 労働供給（定常状態の労働時間）
            res6 = n - SS_CONST.labor_hours_share

            return np.array([res1, res2, res3, res4, res5, res6])

        # 初期値
        x0 = np.array(
            [
                SOLVER_CONST.initial_output,
                SOLVER_CONST.initial_consumption,
                SOLVER_CONST.initial_labor,
                SOLVER_CONST.initial_capital,
                SOLVER_CONST.initial_wage,
                SOLVER_CONST.initial_interest_rate,
            ]
        )

        solution, info, ier, mesg = fsolve(residuals, x0, full_output=True)

        if ier != 1:
            raise ValueError(f"定常状態の収束に失敗: {mesg}")

        y, c, n, k, w, r = solution

        return self._complete_steady_state(y, c, n, k, w, r)

    def _complete_steady_state(
        self, y: float, c: float, n: float, k: float, w: float, r: float
    ) -> SteadyState:
        """主要変数から残りの定常状態変数を計算"""
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        fin = self.params.financial

        pi = cb.pi_target
        R = r + pi
        mc = (firm.epsilon - 1) / firm.epsilon
        i = firm.delta * k
        g = gov.g_y_ratio * y
        tr = gov.transfer_y_ratio * y
        b = gov.b_y_ratio * y

        consumption_tax = gov.tau_c * c
        labor_tax = gov.tau_l * w * n
        capital_tax = gov.tau_k * r * k
        total_tax = consumption_tax + labor_tax + capital_tax
        pb = total_tax - g - tr

        q = SS_CONST.tobin_q
        s = SS_CONST.external_finance_premium_quarterly
        nw = k / fin.leverage_ss
        rk = r + s

        return SteadyState(
            output=y,
            consumption=c,
            investment=i,
            capital=k,
            labor=n,
            real_wage=w,
            real_interest_rate=r,
            nominal_interest_rate=R,
            inflation=pi,
            marginal_cost=mc,
            tobin_q=q,
            government_spending=g,
            government_debt=b,
            tax_revenue=total_tax,
            transfers=tr,
            consumption_tax_rate=gov.tau_c,
            primary_balance=pb,
            external_finance_premium=s,
            net_worth=nw,
            capital_return=rk,
        )

    def verify_steady_state(
        self, ss: SteadyState, tol: float = SOLVER_CONST.verification_tolerance
    ) -> bool:
        """定常状態が方程式を満たすか検証"""
        firm = self.params.firm

        errors = []

        # 生産関数
        y_check = (ss.capital**firm.alpha) * (ss.labor ** (1 - firm.alpha))
        errors.append(abs(ss.output - y_check))

        # 財市場均衡
        demand = ss.consumption + ss.investment + ss.government_spending
        errors.append(abs(ss.output - demand))

        # 資本蓄積（定常状態ではI = δK）
        errors.append(abs(ss.investment - firm.delta * ss.capital))

        max_error = max(errors)
        return bool(max_error < tol)
