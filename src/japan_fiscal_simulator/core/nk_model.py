"""New Keynesian DSGEモデル（14方程式・構造解）"""

from dataclasses import dataclass

import numpy as np

from japan_fiscal_simulator.core.equation_system import EquationSystem, SystemMatrices
from japan_fiscal_simulator.core.equations import (
    CapitalAccumulation,
    CapitalAccumulationParameters,
    CapitalRentalRateEquation,
    Equation,
    GovernmentSpendingProcess,
    InvestmentAdjustmentEquation,
    InvestmentAdjustmentParameters,
    ISCurve,
    ISCurveParameters,
    LaborDemand,
    LaborDemandParameters,
    MarginalCostEquation,
    MarginalCostParameters,
    MRSEquation,
    MRSEquationParameters,
    PhillipsCurve,
    PhillipsCurveParameters,
    ResourceConstraint,
    ResourceConstraintParameters,
    TaylorRule,
    TaylorRuleParameters,
    TechnologyProcess,
    TobinsQEquation,
    TobinsQParameters,
    WagePhillipsCurve,
    WagePhillipsCurveParameters,
    check_taylor_principle,
    compute_phillips_slope,
)
from japan_fiscal_simulator.core.exceptions import ValidationError
from japan_fiscal_simulator.core.solver import BlanchardKahnSolver
from japan_fiscal_simulator.core.steady_state import SteadyState, SteadyStateSolver
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@dataclass
class NKSolutionResult:
    """NKモデルの構造解"""

    # 状態遷移: s_t = P @ s_{t-1} + Q @ ε_t
    P: np.ndarray
    Q: np.ndarray

    # 制御変数: c_t = R @ s_t + S @ ε_t
    R: np.ndarray
    S: np.ndarray

    # 診断情報
    kappa: float
    determinacy: str
    message: str
    eigenvalues: np.ndarray
    n_stable: int
    n_unstable: int
    bk_satisfied: bool


@dataclass
class ModelVariables:
    """モデル変数定義（state + control + shocks）"""

    # 状態変数（先決変数）
    state_vars: tuple[str, ...] = ("g", "a", "k", "i", "w")

    # 制御変数（ジャンプ変数）
    control_vars: tuple[str, ...] = ("y", "pi", "r", "q", "rk", "n", "c", "mc", "mrs")

    # 構造ショック
    shocks: tuple[str, ...] = ("e_g", "e_a", "e_m", "e_i", "e_w", "e_p")

    @property
    def n_state(self) -> int:
        return len(self.state_vars)

    @property
    def n_control(self) -> int:
        return len(self.control_vars)

    @property
    def n_total(self) -> int:
        return self.n_state + self.n_control

    @property
    def n_shock(self) -> int:
        return len(self.shocks)

    def index(self, var: str) -> int:
        if var in self.state_vars:
            return self.state_vars.index(var)
        if var in self.control_vars:
            return self.n_state + self.control_vars.index(var)
        raise ValidationError(f"無効な変数名です: '{var}'")

    def shock_index(self, shock: str) -> int:
        if shock not in self.shocks:
            raise ValidationError(f"無効なショック名です: '{shock}'")
        return self.shocks.index(shock)


class NewKeynesianModel:
    """14方程式 NK DSGE モデル"""

    def __init__(self, params: DefaultParameters) -> None:
        self.params = params
        self.vars = ModelVariables()
        self._steady_state: SteadyState | None = None
        self._solution: NKSolutionResult | None = None

    @property
    def steady_state(self) -> SteadyState:
        if self._steady_state is None:
            solver = SteadyStateSolver(self.params)
            self._steady_state = solver.solve()
        return self._steady_state

    @property
    def solution(self) -> NKSolutionResult:
        if self._solution is None:
            self._solution = self._solve_structural_system()
        return self._solution

    def _resource_shares(self) -> tuple[float, float, float]:
        """資源制約のシェア (s_c, s_i, s_g) を返す"""
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government

        g_share = gov.g_y_ratio
        rental_rate_ss = 1.0 / hh.beta - 1.0 + firm.delta
        i_share = firm.delta * firm.alpha / rental_rate_ss
        c_share = 1.0 - g_share - i_share

        if c_share <= 0:
            raise ValidationError(
                f"資源制約シェアが不正です: c_share={c_share:.4f}, i_share={i_share:.4f}, g_share={g_share:.4f}"
            )

        return c_share, i_share, g_share

    def _create_equations(self) -> list[Equation]:
        """14方程式を構築"""
        hh = self.params.household
        firm = self.params.firm
        gov = self.params.government
        cb = self.params.central_bank
        inv = self.params.investment
        labor = self.params.labor
        shocks = self.params.shocks

        c_share, i_share, g_share = self._resource_shares()

        equations: list[Equation] = [
            # --- State block (5) ---
            GovernmentSpendingProcess(rho_g=shocks.rho_g),
            TechnologyProcess(rho_a=shocks.rho_a),
            CapitalAccumulation(CapitalAccumulationParameters(delta=firm.delta)),
            InvestmentAdjustmentEquation(
                InvestmentAdjustmentParameters(S_double_prime=inv.S_double_prime)
            ),
            WagePhillipsCurve(
                WagePhillipsCurveParameters(
                    beta=hh.beta,
                    theta_w=labor.theta_w,
                    sigma=hh.sigma,
                    phi=hh.phi,
                )
            ),
            # --- Control block (9) ---
            ISCurve(ISCurveParameters(sigma=hh.sigma, g_y=gov.g_y_ratio, habit=hh.habit)),
            PhillipsCurve(
                PhillipsCurveParameters(beta=hh.beta, theta=firm.theta, iota_p=firm.iota_p)
            ),
            TaylorRule(TaylorRuleParameters(phi_pi=cb.phi_pi, phi_y=cb.phi_y)),
            TobinsQEquation(TobinsQParameters(beta=hh.beta, delta=firm.delta)),
            CapitalRentalRateEquation(),
            LaborDemand(LaborDemandParameters(alpha=firm.alpha)),
            ResourceConstraint(
                ResourceConstraintParameters(
                    s_c=c_share,
                    s_i=i_share,
                    s_g=g_share,
                )
            ),
            MarginalCostEquation(MarginalCostParameters(alpha=firm.alpha)),
            MRSEquation(MRSEquationParameters(sigma=hh.sigma, phi=hh.phi)),
        ]

        if len(equations) != self.vars.n_total:
            raise ValidationError(
                f"方程式数が不一致です: {len(equations)} != {self.vars.n_total}"
            )

        return equations

    def _build_system_matrices(self) -> SystemMatrices:
        """14x14 / 14x6 のシステム行列を構築"""
        equation_system = EquationSystem(
            state_vars=self.vars.state_vars,
            control_vars=self.vars.control_vars,
            shocks=self.vars.shocks,
        )
        equations = [eq.coefficients() for eq in self._create_equations()]
        return equation_system.build_matrices(equations)

    def _solve_structural_system(self) -> NKSolutionResult:
        """構造行列をBK解法で解く"""
        matrices = self._build_system_matrices()
        solver = BlanchardKahnSolver(
            A=matrices.A,
            B=matrices.B,
            C=matrices.C,
            D=matrices.D,
            n_predetermined=self.vars.n_state,
            n_forward_looking=int(np.linalg.matrix_rank(matrices.A)),
        )
        result = solver.solve(tol=1e-8)

        hh = self.params.household
        firm = self.params.firm
        cb = self.params.central_bank
        kappa = compute_phillips_slope(hh.beta, firm.theta, firm.iota_p)
        is_determinate, _ = check_taylor_principle(cb.phi_pi, cb.phi_y, hh.beta, kappa)
        determinacy = "determinate" if is_determinate and result.bk_satisfied else "indeterminate"

        return NKSolutionResult(
            P=result.P,
            Q=result.Q,
            R=result.R,
            S=result.S,
            kappa=kappa,
            determinacy=determinacy,
            message=result.message,
            eigenvalues=result.eigenvalues,
            n_stable=result.n_stable,
            n_unstable=result.n_unstable,
            bk_satisfied=result.bk_satisfied,
        )

    def impulse_response(
        self,
        shock: str,
        size: float = 0.01,
        periods: int = 40,
    ) -> dict[str, np.ndarray]:
        """インパルス応答を計算"""
        sol = self.solution
        shock_idx = self.vars.shock_index(shock)

        n_s = self.vars.n_state
        n_c = self.vars.n_control

        state = np.zeros((periods + 1, n_s))
        control = np.zeros((periods + 1, n_c))

        epsilon = np.zeros(self.vars.n_shock)
        epsilon[shock_idx] = size

        # t=0
        state[0] = sol.Q @ epsilon

        rho_p = self.params.shocks.rho_p

        # t=1..T
        for t in range(1, periods + 1):
            state[t] = sol.P @ state[t - 1]
            if shock == "e_p":
                state[t] += sol.Q[:, shock_idx] * (size * (rho_p**t))

        for t in range(periods + 1):
            control[t] = sol.R @ state[t]
            if t == 0:
                control[t] += sol.S[:, shock_idx] * size
            elif shock == "e_p":
                control[t] += sol.S[:, shock_idx] * (size * (rho_p**t))

        result: dict[str, np.ndarray] = {}
        for i, var in enumerate(self.vars.state_vars):
            result[var] = state[:, i]
        for i, var in enumerate(self.vars.control_vars):
            result[var] = control[:, i]

        return result

    def fiscal_multiplier(self, horizon: int = 20) -> dict[str, float]:
        """財政乗数を計算"""
        irf = self.impulse_response("e_g", size=0.01, periods=horizon)
        y = irf["y"]
        g = irf["g"]
        g_y = self.params.government.g_y_ratio

        impact = y[0] / g[0] / g_y if abs(g[0]) > 1e-10 else 0.0

        def cumulative(h: int) -> float:
            y_cum = np.sum(y[:h])
            g_cum = np.sum(g[:h])
            return y_cum / g_cum / g_y if abs(g_cum) > 1e-10 else 0.0

        return {
            "impact": impact,
            "cumulative_4q": cumulative(4),
            "cumulative_8q": cumulative(8),
            "peak": np.max(np.abs(y)) / np.max(np.abs(g)) / g_y,
        }

    def invalidate_cache(self) -> None:
        self._steady_state = None
        self._solution = None
