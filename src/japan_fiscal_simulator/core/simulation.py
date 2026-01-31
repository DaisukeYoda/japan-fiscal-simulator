"""インパルス応答シミュレーション"""

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from japan_fiscal_simulator.core.model import N_SHOCKS, N_VARIABLES, SHOCK_VARS, VARIABLE_INDICES
from japan_fiscal_simulator.parameters.constants import SOLVER_CONSTANTS

if TYPE_CHECKING:
    from japan_fiscal_simulator.core.model import DSGEModel


@dataclass
class ImpulseResponseResult:
    """インパルス応答の結果"""

    periods: int
    shock_name: str
    shock_size: float
    responses: dict[str, np.ndarray]  # 変数名 → 応答時系列

    def get_response(self, variable: str) -> np.ndarray:
        """特定変数の応答を取得"""
        return self.responses.get(variable, np.zeros(self.periods))

    def peak_response(self, variable: str) -> tuple[int, float]:
        """ピーク応答（時期と大きさ）"""
        resp = self.get_response(variable)
        if len(resp) == 0:
            return 0, 0.0
        abs_resp = np.abs(resp)
        peak_period = int(np.argmax(abs_resp))
        peak_value = float(resp[peak_period])
        return peak_period, peak_value

    def cumulative_response(self, variable: str, horizon: int) -> float:
        """累積応答"""
        resp = self.get_response(variable)
        return float(np.sum(resp[:horizon]))


class ImpulseResponseSimulator:
    """インパルス応答関数のシミュレーター"""

    def __init__(self, model: DSGEModel) -> None:
        self.model = model

    def simulate(
        self,
        shock_name: str,
        shock_size: float = 0.01,
        periods: int = 40,
    ) -> ImpulseResponseResult:
        """インパルス応答をシミュレート

        Args:
            shock_name: ショック名（'e_a', 'e_g', 'e_m', 'e_tau', 'e_risk'）
            shock_size: ショックサイズ（デフォルト1%）
            periods: シミュレーション期間

        Returns:
            ImpulseResponseResult
        """
        policy = self.model.policy_function

        # ショックインデックス
        if shock_name not in SHOCK_VARS:
            raise ValueError(f"Unknown shock: {shock_name}. Available: {SHOCK_VARS}")
        shock_idx = SHOCK_VARS.index(shock_name)

        # 状態変数の時系列
        n_vars = N_VARIABLES
        x_history = np.zeros((periods + 1, n_vars))

        # 初期ショックベクトル
        epsilon = np.zeros(N_SHOCKS)
        epsilon[shock_idx] = shock_size

        # 初期インパクト: x_0 = Q * ε
        if policy.Q.shape[1] >= N_SHOCKS:
            x_history[0] = policy.Q[:, :N_SHOCKS] @ epsilon
        else:
            # Qの次元が足りない場合
            x_history[0, : policy.Q.shape[0]] = policy.Q @ epsilon[: policy.Q.shape[1]]

        # 時間発展: x_t = P * x_{t-1}
        P = policy.P
        for t in range(1, periods + 1):
            if P.shape[0] == n_vars and P.shape[1] == n_vars:
                x_history[t] = P @ x_history[t - 1]
            else:
                # Pの次元が合わない場合は減衰
                x_history[t] = x_history[t - 1] * SOLVER_CONSTANTS.fallback_decay_rate

        # 結果を変数名でマッピング（t=0のインパクトを含む）
        responses = {}
        for name, idx in VARIABLE_INDICES.items():
            if idx < n_vars:
                responses[name] = x_history[:, idx].copy()
            else:
                responses[name] = np.zeros(periods + 1)

        return ImpulseResponseResult(
            periods=periods + 1,  # t=0を含む
            shock_name=shock_name,
            shock_size=shock_size,
            responses=responses,
        )

    def simulate_consumption_tax_cut(
        self, tax_cut: float = 0.02, periods: int = 40
    ) -> ImpulseResponseResult:
        """消費税減税のシミュレーション

        Args:
            tax_cut: 減税幅（例: 0.02 = 2%pt減税）
            periods: シミュレーション期間
        """
        return self.simulate("e_tau", shock_size=-tax_cut, periods=periods)

    def simulate_government_spending(
        self, spending_increase: float = 0.01, periods: int = 40
    ) -> ImpulseResponseResult:
        """政府支出増加のシミュレーション

        Args:
            spending_increase: 支出増加率（GDP比、例: 0.01 = 1%）
            periods: シミュレーション期間
        """
        return self.simulate("e_g", shock_size=spending_increase, periods=periods)

    def simulate_monetary_shock(
        self, rate_change: float = 0.0025, periods: int = 40
    ) -> ImpulseResponseResult:
        """金融政策ショックのシミュレーション

        Args:
            rate_change: 金利変更（例: 0.0025 = 25bp）
            periods: シミュレーション期間
        """
        return self.simulate("e_m", shock_size=rate_change, periods=periods)

    def simulate_technology_shock(
        self, productivity_increase: float = 0.01, periods: int = 40
    ) -> ImpulseResponseResult:
        """技術ショックのシミュレーション

        Args:
            productivity_increase: 生産性上昇率
            periods: シミュレーション期間
        """
        return self.simulate("e_a", shock_size=productivity_increase, periods=periods)


@dataclass
class FiscalMultiplierResult:
    """財政乗数の計算結果"""

    impact: float  # インパクト乗数
    peak: float  # ピーク乗数
    peak_period: int  # ピーク時期
    cumulative_4q: float  # 1年累積
    cumulative_8q: float  # 2年累積
    cumulative_20q: float  # 5年累積
    present_value: float  # 現在価値乗数


class FiscalMultiplierCalculator:
    """財政乗数の計算"""

    def __init__(self, model: DSGEModel) -> None:
        self.model = model
        self.simulator = ImpulseResponseSimulator(model)

    def compute_spending_multiplier(self, horizon: int = 40) -> FiscalMultiplierResult:
        """政府支出乗数を計算"""
        result = self.simulator.simulate_government_spending(
            spending_increase=0.01, periods=horizon
        )

        y_response = result.get_response("y")
        g_response = result.get_response("g")

        # 政府支出の定常状態比率
        g_y_ratio = self.model.params.government.g_y_ratio

        return self._compute_multiplier(y_response, g_response, g_y_ratio, horizon)

    def compute_tax_multiplier(self, horizon: int = 40) -> FiscalMultiplierResult:
        """消費税乗数を計算（減税の効果）"""
        result = self.simulator.simulate_consumption_tax_cut(tax_cut=0.01, periods=horizon)

        y_response = result.get_response("y")
        tau_response = result.get_response("tau_c")

        # 消費税の定常状態
        tau_c = self.model.params.government.tau_c

        # 乗数計算（減税の効果なので符号を調整）
        return self._compute_multiplier(-y_response, tau_response, tau_c, horizon)

    def _compute_multiplier(
        self,
        y_response: np.ndarray,
        policy_response: np.ndarray,
        policy_ratio: float,
        horizon: int,
    ) -> FiscalMultiplierResult:
        """乗数を計算する共通ロジック"""
        # インパクト乗数（t=0）
        if len(policy_response) > 0 and abs(policy_response[0]) > 1e-10:
            impact = y_response[0] / policy_response[0] / policy_ratio
        else:
            impact = y_response[0] / 0.01 / policy_ratio if len(y_response) > 0 else 0.0

        # ピーク乗数
        if len(y_response) > 0:
            abs_y = np.abs(y_response)
            peak_period = int(np.argmax(abs_y))
            if abs(policy_response[peak_period]) > 1e-10:
                peak = y_response[peak_period] / policy_response[peak_period] / policy_ratio
            else:
                peak = y_response[peak_period] / 0.01 / policy_ratio
        else:
            peak_period = 0
            peak = 0.0

        # 累積乗数
        def cumulative_mult(h: int) -> float:
            if h > len(y_response):
                h = len(y_response)
            y_cum = np.sum(y_response[:h])
            policy_cum = np.sum(policy_response[:h])
            if abs(policy_cum) > 1e-10:
                return float(y_cum / policy_cum / policy_ratio)
            return float(y_cum / (0.01 * h) / policy_ratio)

        cumulative_4q = cumulative_mult(4)
        cumulative_8q = cumulative_mult(8)
        cumulative_20q = cumulative_mult(20)

        # 現在価値乗数
        beta = self.model.params.household.beta
        discount_factors = np.array([beta**t for t in range(len(y_response))])
        y_pv = np.sum(y_response * discount_factors)
        policy_pv = np.sum(policy_response * discount_factors)
        if abs(policy_pv) > 1e-10:
            pv_mult = y_pv / policy_pv / policy_ratio
        else:
            pv_mult = y_pv / (0.01 * np.sum(discount_factors)) / policy_ratio

        return FiscalMultiplierResult(
            impact=float(impact),
            peak=float(peak),
            peak_period=peak_period,
            cumulative_4q=float(cumulative_4q),
            cumulative_8q=float(cumulative_8q),
            cumulative_20q=float(cumulative_20q),
            present_value=float(pv_mult),
        )
