"""合成データ生成モジュール

DSGEモデルからシミュレーションして合成観測データを生成する。
テスト・検証に使用する。
"""

from pathlib import Path

import numpy as np

from japan_fiscal_simulator.core.nk_model import ModelVariables, NewKeynesianModel
from japan_fiscal_simulator.estimation.data_loader import (
    DEFAULT_OBSERVABLES,
    EstimationData,
    ObservableDefinition,
)
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class SyntheticDataGenerator:
    """合成データ生成（テスト・検証用）

    DSGEモデルをシミュレートし、観測変数に対応する合成データを生成する。
    """

    def __init__(
        self,
        observables: list[ObservableDefinition] | None = None,
    ) -> None:
        """初期化

        Args:
            observables: 観測変数定義リスト。Noneの場合はDEFAULT_OBSERVABLESを使用。
        """
        self.observables = observables or DEFAULT_OBSERVABLES

    def generate(
        self,
        params: DefaultParameters,
        n_periods: int = 200,
        measurement_error_std: float = 0.001,
        rng: np.random.Generator | None = None,
    ) -> EstimationData:
        """モデルからシミュレーションして合成観測データを生成する

        1. NewKeynesianModel(params) を構築し solution を取得
        2. ショックをランダム生成
        3. 状態方程式で state path をシミュレート
        4. 観測方程式で observation を生成
        5. 測定誤差を付加
        6. EstimationData として返す

        Args:
            params: モデルパラメータ
            n_periods: シミュレーション期間数
            measurement_error_std: 測定誤差の標準偏差
            rng: 乱数生成器。Noneの場合はデフォルトを使用。

        Returns:
            合成観測データ
        """
        if rng is None:
            rng = np.random.default_rng()

        model = NewKeynesianModel(params)
        sol = model.solution
        mv = model.vars

        P, Q, R, S = sol.P, sol.Q, sol.R, sol.S
        n_state = mv.n_state
        n_control = mv.n_control
        n_shock = mv.n_shock

        # ショック標準偏差を取得
        shock_stds = self._get_shock_stds(params, mv)

        # +1 期分をバーンイン期間として追加（成長率計算のため）
        total_periods = n_periods + 1

        # ショック系列を生成
        shocks = np.zeros((total_periods, n_shock))
        for t in range(total_periods):
            shocks[t] = rng.normal(0.0, shock_stds)

        # 状態・制御変数のパスをシミュレート
        states = np.zeros((total_periods, n_state))
        controls = np.zeros((total_periods, n_control))

        # t=0: 初期ショック
        states[0] = Q @ shocks[0]
        controls[0] = R @ states[0] + S @ shocks[0]

        # t=1..T: 状態遷移
        for t in range(1, total_periods):
            states[t] = P @ states[t - 1] + Q @ shocks[t]
            controls[t] = R @ states[t] + S @ shocks[t]

        # 観測変数を抽出・変換
        obs_data = self._extract_observables(states, controls, mv, rng, measurement_error_std)

        # 日付ラベルを生成（1994Q1から開始）
        dates = self._generate_dates(n_periods, start_year=1994, start_quarter=1)

        variable_names = [obs.name for obs in self.observables]
        n_obs = len(variable_names)

        return EstimationData(
            data=obs_data,
            variable_names=variable_names,
            dates=dates,
            n_obs=n_obs,
            n_periods=n_periods,
        )

    def to_csv(self, data: EstimationData, path: str | Path) -> None:
        """EstimationDataをCSVファイルに出力する

        Args:
            data: 推定用データ
            path: 出力先ファイルパス
        """
        filepath = Path(path)
        header = "date," + ",".join(data.variable_names)
        lines = [header]
        for t in range(data.n_periods):
            values = ",".join(f"{data.data[t, j]:.8f}" for j in range(data.n_obs))
            lines.append(f"{data.dates[t]},{values}")

        filepath.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _get_shock_stds(self, params: DefaultParameters, mv: ModelVariables) -> np.ndarray:
        """ショック標準偏差ベクトルを構築する"""
        shock_std_map: dict[str, float] = {
            "e_g": params.shocks.sigma_g,
            "e_a": params.shocks.sigma_a,
            "e_m": params.shocks.sigma_m,
            "e_i": params.shocks.sigma_i,
            "e_w": params.shocks.sigma_w,
            "e_p": params.shocks.sigma_p,
        }
        stds = np.zeros(mv.n_shock)
        for i, shock_name in enumerate(mv.shocks):
            stds[i] = shock_std_map.get(shock_name, 0.01)
        return stds

    def _extract_observables(
        self,
        states: np.ndarray,
        controls: np.ndarray,
        mv: ModelVariables,
        rng: np.random.Generator,
        measurement_error_std: float,
    ) -> np.ndarray:
        """状態・制御変数パスから観測変数を抽出・変換する

        states, controls は total_periods = n_periods + 1 の長さ。
        成長率変数は差分を取るので最終的に n_periods の長さになる。

        Returns:
            (n_periods, n_obs) の観測データ行列
        """
        total_periods = states.shape[0]
        n_periods = total_periods - 1

        # 全変数を (total_periods, n_total) にまとめる
        all_vars = np.column_stack([states, controls])

        obs_list: list[np.ndarray] = []
        for obs in self.observables:
            var_idx = mv.index(obs.model_variable)
            series = all_vars[:, var_idx]

            if obs.transform == "dlog":
                # 成長率: x_t - x_{t-1}（モデル変数は既にlog偏差）
                transformed = np.diff(series)
            elif obs.transform in ("level", "rate"):
                # 水準・金利: 先頭1期をバーンインとして除外
                transformed = series[1:]
            else:
                msg = f"未対応の変換方法: '{obs.transform}'"
                raise ValueError(msg)

            # 測定誤差を付加
            noise = rng.normal(0.0, measurement_error_std, size=n_periods)
            obs_list.append(transformed + noise)

        return np.column_stack(obs_list)

    @staticmethod
    def _generate_dates(
        n_periods: int, start_year: int = 1994, start_quarter: int = 1
    ) -> list[str]:
        """四半期日付ラベルを生成する"""
        dates: list[str] = []
        year = start_year
        quarter = start_quarter
        for _ in range(n_periods):
            dates.append(f"{year}Q{quarter}")
            quarter += 1
            if quarter > 4:
                quarter = 1
                year += 1
        return dates
