"""推定パラメータとDefaultParametersの双方向マッピング

MCMCで使用するフラットなθベクトルと、モデルのDefaultParametersデータクラス階層との
変換を担当する。
"""

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@dataclass(frozen=True)
class ParameterSpec:
    """推定パラメータの仕様

    Attributes:
        name: パラメータ名（一意識別子）
        section: DefaultParameters内のセクション名（"household"等）。
                 "measurement"の場合はDefaultParametersに含まれない。
        field: セクション内のフィールド名
        default: デフォルト値
        lower_bound: 下限
        upper_bound: 上限
    """

    name: str
    section: str
    field: str
    default: float
    lower_bound: float
    upper_bound: float


class ParameterMapping:
    """MCMCθベクトルとDefaultParametersの双方向マッピング

    推定対象パラメータ（約30個）をフラットなnumpy配列とモデルパラメータ間で変換する。
    観測誤差パラメータはDefaultParametersに含まれないため、別途取得する。
    """

    ESTIMATED_PARAMS: list[ParameterSpec] = [
        # --- 家計パラメータ ---
        ParameterSpec("sigma", "household", "sigma", 1.5, 0.5, 5.0),
        ParameterSpec("phi", "household", "phi", 2.0, 0.5, 5.0),
        ParameterSpec("habit", "household", "habit", 0.7, 0.01, 0.99),
        # --- 企業パラメータ ---
        ParameterSpec("theta", "firm", "theta", 0.75, 0.1, 0.99),
        ParameterSpec("psi", "firm", "psi", 0.5, 0.01, 0.99),
        # --- 投資パラメータ ---
        ParameterSpec("S_double_prime", "investment", "S_double_prime", 5.0, 0.1, 15.0),
        # --- 労働市場パラメータ ---
        ParameterSpec("theta_w", "labor", "theta_w", 0.75, 0.1, 0.99),
        ParameterSpec("iota_w", "labor", "iota_w", 0.5, 0.01, 0.99),
        # --- 中央銀行パラメータ ---
        ParameterSpec("phi_pi", "central_bank", "phi_pi", 1.5, 1.01, 3.0),
        ParameterSpec("phi_y", "central_bank", "phi_y", 0.125, 0.01, 0.5),
        ParameterSpec("rho_r", "central_bank", "rho_r", 0.85, 0.5, 0.99),
        # --- ショック持続性 ---
        ParameterSpec("rho_a", "shocks", "rho_a", 0.90, 0.01, 0.99),
        ParameterSpec("rho_g", "shocks", "rho_g", 0.90, 0.01, 0.99),
        ParameterSpec("rho_p", "shocks", "rho_p", 0.90, 0.01, 0.99),
        # --- ショック標準偏差 ---
        ParameterSpec("sigma_a", "shocks", "sigma_a", 0.01, 0.001, 0.1),
        ParameterSpec("sigma_g", "shocks", "sigma_g", 0.01, 0.001, 0.1),
        ParameterSpec("sigma_i", "shocks", "sigma_i", 0.01, 0.001, 0.1),
        ParameterSpec("sigma_w", "shocks", "sigma_w", 0.01, 0.001, 0.1),
        ParameterSpec("sigma_p", "shocks", "sigma_p", 0.01, 0.001, 0.1),
        ParameterSpec("sigma_m", "shocks", "sigma_m", 0.0025, 0.001, 0.1),
        # --- 観測誤差 ---
        ParameterSpec("me_y", "measurement", "me_y", 0.01, 0.001, 0.1),
        ParameterSpec("me_c", "measurement", "me_c", 0.01, 0.001, 0.1),
        ParameterSpec("me_i", "measurement", "me_i", 0.01, 0.001, 0.1),
        ParameterSpec("me_pi", "measurement", "me_pi", 0.01, 0.001, 0.1),
        ParameterSpec("me_w", "measurement", "me_w", 0.01, 0.001, 0.1),
        ParameterSpec("me_n", "measurement", "me_n", 0.01, 0.001, 0.1),
        ParameterSpec("me_r", "measurement", "me_r", 0.01, 0.001, 0.1),
        # --- 定常状態パラメータ（観測方程式の定数項） ---
        ParameterSpec("gamma_bar", "steady_state", "gamma_bar", 0.4, -1.0, 3.0),
        ParameterSpec("pi_bar", "steady_state", "pi_bar", 0.0, -2.0, 5.0),
        ParameterSpec("n_bar", "steady_state", "n_bar", 0.0, -10.0, 10.0),
        ParameterSpec("r_bar", "steady_state", "r_bar", 0.1, -2.0, 5.0),
    ]

    # DefaultParametersに含まれないセクション
    _NON_MODEL_SECTIONS: set[str] = {"measurement", "steady_state"}

    def __init__(self) -> None:
        """マッピングの初期化と内部インデックスの構築"""
        self._name_to_index: dict[str, int] = {
            spec.name: i for i, spec in enumerate(self.ESTIMATED_PARAMS)
        }
        self._measurement_indices: list[int] = [
            i for i, spec in enumerate(self.ESTIMATED_PARAMS) if spec.section == "measurement"
        ]
        self._steady_state_indices: list[int] = [
            i for i, spec in enumerate(self.ESTIMATED_PARAMS) if spec.section == "steady_state"
        ]
        self._model_indices: list[int] = [
            i
            for i, spec in enumerate(self.ESTIMATED_PARAMS)
            if spec.section not in self._NON_MODEL_SECTIONS
        ]

    @property
    def names(self) -> list[str]:
        """推定パラメータ名のリスト"""
        return [spec.name for spec in self.ESTIMATED_PARAMS]

    @property
    def n_params(self) -> int:
        """推定パラメータ数"""
        return len(self.ESTIMATED_PARAMS)

    @property
    def n_measurement(self) -> int:
        """観測誤差パラメータ数"""
        return len(self._measurement_indices)

    def defaults(self) -> np.ndarray:
        """デフォルト値のベクトルを返す"""
        return np.array([spec.default for spec in self.ESTIMATED_PARAMS])

    def bounds(self) -> list[tuple[float, float]]:
        """各パラメータの(下限, 上限)タプルのリストを返す"""
        return [(spec.lower_bound, spec.upper_bound) for spec in self.ESTIMATED_PARAMS]

    def theta_to_params(
        self, theta: np.ndarray, base_params: DefaultParameters | None = None
    ) -> DefaultParameters:
        """θベクトルからDefaultParametersを構築する

        観測誤差パラメータはDefaultParametersに含まれないため無視される。
        観測誤差の取得にはtheta_to_measurement_errorsを使用すること。

        Args:
            theta: 推定パラメータベクトル（長さ n_params）
            base_params: ベースとなるパラメータ。Noneの場合はデフォルト値を使用。

        Returns:
            更新されたDefaultParameters
        """
        if len(theta) != self.n_params:
            msg = f"θの長さが不正: {len(theta)} != {self.n_params}"
            raise ValueError(msg)

        params = base_params or DefaultParameters()

        # セクションごとにフィールド更新をまとめる
        section_updates: dict[str, dict[str, float]] = {}
        for i in self._model_indices:
            spec = self.ESTIMATED_PARAMS[i]
            if spec.section not in section_updates:
                section_updates[spec.section] = {}
            section_updates[spec.section][spec.field] = float(theta[i])

        # 各セクションのfrozenデータクラスをreplaceで更新
        section_to_attr: dict[str, Any] = {
            "household": params.household,
            "firm": params.firm,
            "investment": params.investment,
            "labor": params.labor,
            "central_bank": params.central_bank,
            "shocks": params.shocks,
        }

        kwargs: dict[str, Any] = {}
        for section_name, updates in section_updates.items():
            current = section_to_attr[section_name]
            kwargs[section_name] = replace(current, **updates)

        return params.with_updates(**kwargs)

    def theta_to_measurement_errors(self, theta: np.ndarray) -> np.ndarray:
        """θベクトルから観測誤差標準偏差を抽出する

        Args:
            theta: 推定パラメータベクトル（長さ n_params）

        Returns:
            観測誤差標準偏差のベクトル（長さ n_measurement）
        """
        if len(theta) != self.n_params:
            msg = f"θの長さが不正: {len(theta)} != {self.n_params}"
            raise ValueError(msg)

        return np.array([theta[i] for i in self._measurement_indices])

    def theta_to_steady_state_means(self, theta: np.ndarray) -> np.ndarray:
        """θベクトルから観測方程式の定常状態定数ベクトル d を構築する

        SW2007方式の観測方程式:
            Δy_t = γ + ŷ_t - ŷ_{t-1}
            Δc_t = γ + ĉ_t - ĉ_{t-1}
            Δi_t = γ + î_t - î_{t-1}
            π_t  = π* + π̂_t
            Δw_t = γ + ŵ_t - ŵ_{t-1}
            n_t  = n* + n̂_t
            r_t  = r* + r̂_t

        Args:
            theta: 推定パラメータベクトル（長さ n_params）

        Returns:
            定常状態定数ベクトル d (7,)
        """
        if len(theta) != self.n_params:
            msg = f"θの長さが不正: {len(theta)} != {self.n_params}"
            raise ValueError(msg)

        gamma_bar = theta[self._name_to_index["gamma_bar"]]
        pi_bar = theta[self._name_to_index["pi_bar"]]
        n_bar = theta[self._name_to_index["n_bar"]]
        r_bar = theta[self._name_to_index["r_bar"]]

        # d = [γ, γ, γ, π*, γ, n*, r*]
        return np.array([gamma_bar, gamma_bar, gamma_bar, pi_bar, gamma_bar, n_bar, r_bar])

    def params_to_theta(self, params: DefaultParameters) -> np.ndarray:
        """DefaultParametersからθベクトルを構築する

        観測誤差パラメータはデフォルト値が使用される。

        Args:
            params: モデルパラメータ

        Returns:
            θベクトル（長さ n_params）
        """
        theta = np.zeros(self.n_params)

        section_to_attr: dict[str, Any] = {
            "household": params.household,
            "firm": params.firm,
            "investment": params.investment,
            "labor": params.labor,
            "central_bank": params.central_bank,
            "shocks": params.shocks,
        }

        for i, spec in enumerate(self.ESTIMATED_PARAMS):
            if spec.section in self._NON_MODEL_SECTIONS:
                theta[i] = spec.default
            else:
                section_obj = section_to_attr[spec.section]
                theta[i] = getattr(section_obj, spec.field)

        return theta

    @property
    def measurement_names(self) -> list[str]:
        """観測誤差パラメータ名のリスト"""
        return [self.ESTIMATED_PARAMS[i].name for i in self._measurement_indices]

    @property
    def n_steady_state(self) -> int:
        """定常状態パラメータ数"""
        return len(self._steady_state_indices)

    @property
    def steady_state_names(self) -> list[str]:
        """定常状態パラメータ名のリスト"""
        return [self.ESTIMATED_PARAMS[i].name for i in self._steady_state_indices]
