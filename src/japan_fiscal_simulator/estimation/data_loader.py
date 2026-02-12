"""推定用データの読込・変換モジュール

観測変数定義、CSV読込、データ変換（dlog、HPフィルタ、demean）を提供する。
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


@dataclass(frozen=True)
class ObservableDefinition:
    """観測変数定義

    Attributes:
        name: 観測変数名（例: "output_growth"）
        description: 日本語の説明
        transform: 変換方法（"dlog", "level", "rate"）
        model_variable: 対応するモデル変数名
    """

    name: str
    description: str
    transform: str
    model_variable: str


# Smets-Wouters (2007) 標準の7観測変数
DEFAULT_OBSERVABLES: list[ObservableDefinition] = [
    ObservableDefinition("output_growth", "実質GDP成長率", "dlog", "y"),
    ObservableDefinition("consumption_growth", "消費成長率", "dlog", "c"),
    ObservableDefinition("investment_growth", "投資成長率", "dlog", "i"),
    ObservableDefinition("inflation", "インフレ率", "dlog", "pi"),
    ObservableDefinition("wage_growth", "実質賃金成長率", "dlog", "w"),
    ObservableDefinition("hours", "労働時間", "level", "n"),
    ObservableDefinition("interest_rate", "名目金利", "rate", "r"),
]


@dataclass
class EstimationData:
    """推定用データ

    Attributes:
        data: 変換済みデータ (T, n_obs)
        variable_names: 観測変数名のリスト
        dates: 日付ラベル（例: ["1994Q1", "1994Q2", ...]）
        n_obs: 観測変数数
        n_periods: 時系列の長さ
    """

    data: np.ndarray
    variable_names: list[str]
    dates: list[str]
    n_obs: int
    n_periods: int

    @classmethod
    def from_csv(
        cls,
        path: str | Path,
        observables: list[ObservableDefinition] | None = None,
    ) -> EstimationData:
        """CSVファイルから推定データを構築する

        Args:
            path: CSVファイルパス
            observables: 観測変数定義リスト。Noneの場合はDEFAULT_OBSERVABLESを使用。

        Returns:
            EstimationData インスタンス
        """
        loader = DataLoader(observables=observables)
        return loader.load_csv(path)


class DataLoader:
    """データ読込・変換クラス

    CSVファイルからデータを読み込み、変換を適用してEstimationDataを返す。
    CSVフォーマット: date, gdp, consumption, investment, deflator, wage, employment, rate
    """

    # CSV列名から観測変数名へのマッピング
    _COLUMN_MAP: dict[str, str] = {
        "output_growth": "gdp",
        "consumption_growth": "consumption",
        "investment_growth": "investment",
        "inflation": "deflator",
        "wage_growth": "wage",
        "hours": "employment",
        "interest_rate": "rate",
    }

    def __init__(self, observables: list[ObservableDefinition] | None = None) -> None:
        """初期化

        Args:
            observables: 観測変数定義リスト。Noneの場合はDEFAULT_OBSERVABLESを使用。
        """
        self.observables = observables or DEFAULT_OBSERVABLES

    def load_csv(self, path: str | Path) -> EstimationData:
        """CSV読込→変換→EstimationData

        CSVフォーマット: date, gdp, consumption, investment, deflator, wage, employment, rate
        dlog変換後に1期分短くなるため、日付列も対応して短縮される。

        Args:
            path: CSVファイルパス

        Returns:
            変換・demean済みのEstimationData
        """
        filepath = Path(path)
        raw_text = filepath.read_text(encoding="utf-8")
        lines = [line.strip() for line in raw_text.strip().split("\n") if line.strip()]

        header = [col.strip() for col in lines[0].split(",")]
        n_rows = len(lines) - 1

        # データを解析
        dates: list[str] = []
        columns: dict[str, np.ndarray] = {col: np.zeros(n_rows) for col in header if col != "date"}

        for row_idx, line in enumerate(lines[1:]):
            values = [v.strip() for v in line.split(",")]
            dates.append(values[0])
            for col_idx, col_name in enumerate(header):
                if col_name != "date":
                    columns[col_name][row_idx] = float(values[col_idx])

        # 各観測変数を変換
        transformed_series: list[np.ndarray] = []
        has_dlog = any(obs.transform == "dlog" for obs in self.observables)

        for obs in self.observables:
            csv_col = self._COLUMN_MAP.get(obs.name, obs.name)
            if csv_col not in columns:
                msg = f"CSV列 '{csv_col}' が見つかりません（観測変数 '{obs.name}'）"
                raise ValueError(msg)

            series = columns[csv_col]
            if obs.transform == "dlog":
                transformed_series.append(self.dlog_transform(series))
            elif obs.transform == "level":
                # dlog変換で1期短くなるため、levelとrateも1期短くする
                if has_dlog:
                    transformed_series.append(series[1:])
                else:
                    transformed_series.append(series)
            elif obs.transform == "rate":
                if has_dlog:
                    transformed_series.append(series[1:])
                else:
                    transformed_series.append(series)
            else:
                msg = f"未対応の変換方法: '{obs.transform}'"
                raise ValueError(msg)

        # dlog変換で先頭1期分が失われるため日付を調整
        if has_dlog:
            dates = dates[1:]

        # (T, n_obs) 行列に結合
        data = np.column_stack(transformed_series)

        # demean
        data = self.demean(data)

        variable_names = [obs.name for obs in self.observables]
        n_periods, n_obs = data.shape

        return EstimationData(
            data=data,
            variable_names=variable_names,
            dates=dates,
            n_obs=n_obs,
            n_periods=n_periods,
        )

    @staticmethod
    def dlog_transform(series: np.ndarray) -> np.ndarray:
        """100 * delta-log 変換

        Args:
            series: 水準値の時系列

        Returns:
            100 * (log(x_t) - log(x_{t-1})) の時系列（元の長さ-1）
        """
        log_series = np.log(series)
        result: np.ndarray = 100.0 * np.diff(log_series)
        return result

    @staticmethod
    def hp_filter(series: np.ndarray, lamb: float = 1600.0) -> tuple[np.ndarray, np.ndarray]:
        """Hodrick-Prescottフィルタ

        スパース行列を用いた効率的な実装。

        Args:
            series: 入力時系列
            lamb: 平滑化パラメータ（四半期データの場合 1600）

        Returns:
            (trend, cycle) のタプル
        """
        n = len(series)
        if n < 3:
            return series.copy(), np.zeros(n)

        # 2階差分行列 D を構築
        e = np.ones(n)
        diags = np.array([e, -2.0 * e, e])
        D = sp.spdiags(diags, [0, 1, 2], n - 2, n).tocsc()

        # (eye + lambda * D'D) * trend = y
        eye = sp.eye(n, format="csc")
        A = eye + lamb * (D.T @ D)

        trend = spla.spsolve(A, series)
        cycle = series - trend
        return trend, cycle

    @staticmethod
    def demean(data: np.ndarray) -> np.ndarray:
        """各列からNaNを除いた平均を引く

        Args:
            data: (T, n_obs) の行列

        Returns:
            各列の平均が0に近い行列
        """
        result = data.copy()
        if result.ndim == 1:
            mask = ~np.isnan(result)
            result[mask] -= np.mean(result[mask])
        else:
            for j in range(result.shape[1]):
                col = result[:, j]
                mask = ~np.isnan(col)
                if np.any(mask):
                    col[mask] -= np.mean(col[mask])
        return result
