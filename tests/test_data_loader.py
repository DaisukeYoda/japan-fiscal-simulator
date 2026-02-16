"""データ読込・変換・合成データ生成のテスト"""

import tempfile
from pathlib import Path

import numpy as np
import pytest

from japan_fiscal_simulator.estimation.data_fetcher import SyntheticDataGenerator
from japan_fiscal_simulator.estimation.data_loader import (
    DEFAULT_OBSERVABLES,
    DataLoader,
    EstimationData,
)
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestDlogTransform:
    """dlog_transform のテスト"""

    def test_known_values(self) -> None:
        """既知の入力・出力で正しく変換されることを確認"""
        series = np.array([100.0, 101.0, 102.01])
        result = DataLoader.dlog_transform(series)
        # 100 * (log(101) - log(100)) ≈ 0.995
        # 100 * (log(102.01) - log(101)) ≈ 0.995
        assert len(result) == 2
        np.testing.assert_allclose(result[0], 100.0 * np.log(101.0 / 100.0), rtol=1e-10)
        np.testing.assert_allclose(result[1], 100.0 * np.log(102.01 / 101.0), rtol=1e-10)

    def test_output_length(self) -> None:
        """出力の長さが入力の長さ-1であることを確認"""
        n = 50
        series = np.exp(np.cumsum(np.random.default_rng(42).normal(0, 0.01, n)))
        result = DataLoader.dlog_transform(series)
        assert len(result) == n - 1

    def test_constant_series(self) -> None:
        """定常系列の場合、変換結果がゼロになることを確認"""
        series = np.ones(10) * 5.0
        result = DataLoader.dlog_transform(series)
        np.testing.assert_allclose(result, 0.0, atol=1e-15)

    def test_doubling_series(self) -> None:
        """毎期2倍になる系列の場合、100*log(2)になることを確認"""
        series = np.array([1.0, 2.0, 4.0, 8.0])
        result = DataLoader.dlog_transform(series)
        expected = 100.0 * np.log(2.0)
        np.testing.assert_allclose(result, expected, rtol=1e-10)


class TestHPFilter:
    """hp_filter のテスト"""

    def test_trend_is_smooth(self) -> None:
        """トレンドが元の系列より滑らかであることを確認"""
        rng = np.random.default_rng(42)
        n = 100
        series = np.cumsum(rng.normal(0, 1, n)) + rng.normal(0, 0.5, n)
        trend, cycle = DataLoader.hp_filter(series)

        # トレンドの2階差分の分散が元系列の2階差分の分散より小さい
        trend_dd = np.diff(trend, n=2)
        series_dd = np.diff(series, n=2)
        assert np.var(trend_dd) < np.var(series_dd)

    def test_cycle_mean_near_zero(self) -> None:
        """循環成分の平均がゼロに近いことを確認"""
        rng = np.random.default_rng(42)
        n = 200
        series = np.arange(n, dtype=float) + rng.normal(0, 1, n)
        _, cycle = DataLoader.hp_filter(series)
        assert abs(np.mean(cycle)) < 1.0

    def test_trend_plus_cycle_equals_original(self) -> None:
        """トレンド+循環成分が元の系列と一致することを確認"""
        rng = np.random.default_rng(42)
        series = np.cumsum(rng.normal(0, 1, 50))
        trend, cycle = DataLoader.hp_filter(series)
        np.testing.assert_allclose(trend + cycle, series, atol=1e-10)

    def test_short_series(self) -> None:
        """短い系列（3未満）でも動作することを確認"""
        series = np.array([1.0, 2.0])
        trend, cycle = DataLoader.hp_filter(series)
        np.testing.assert_allclose(trend, series)
        np.testing.assert_allclose(cycle, 0.0, atol=1e-15)

    def test_output_lengths_match(self) -> None:
        """出力の長さが入力と一致することを確認"""
        series = np.random.default_rng(42).normal(0, 1, 100)
        trend, cycle = DataLoader.hp_filter(series)
        assert len(trend) == len(series)
        assert len(cycle) == len(series)


class TestCSVRoundTrip:
    """CSV書き出し→読み込みのラウンドトリップテスト"""

    def test_write_and_read(self) -> None:
        """SyntheticDataGeneratorで生成→CSV書き出し→DataLoaderで読み込み"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        synth = gen.generate(params, n_periods=50, rng=rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_data.csv"
            gen.to_csv(synth, csv_path)

            # CSVが作成されたことを確認
            assert csv_path.exists()

            # 内容を読み込んで確認
            text = csv_path.read_text(encoding="utf-8")
            lines = text.strip().split("\n")
            # ヘッダー + (n_periods+1)行（dlog復元で1期多い）
            assert len(lines) == synth.n_periods + 2

    def test_csv_round_trip_through_loader(self) -> None:
        """to_csv → DataLoader.load_csv のラウンドトリップでスケール一致"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        synth = gen.generate(params, n_periods=50, rng=rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "round_trip.csv"
            gen.to_csv(synth, csv_path)

            # DataLoaderで再読み込み
            loaded = EstimationData.from_csv(csv_path)

            assert loaded.n_obs == 7
            assert loaded.n_periods == synth.n_periods
            assert np.all(np.isfinite(loaded.data))

            # dlog変数のスケールが一致することを確認
            for j in range(loaded.n_obs):
                std_loaded = np.std(loaded.data[:, j])
                std_orig = np.std(synth.data[:, j])
                if std_orig > 1e-10:
                    ratio = std_loaded / std_orig
                    assert 0.95 < ratio < 1.05, f"変数{j}: std比={ratio:.2f} (期待: ~1.0)"

    def test_csv_header(self) -> None:
        """CSVヘッダーが正しいことを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        synth = gen.generate(params, n_periods=20, rng=rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test_header.csv"
            gen.to_csv(synth, csv_path)

            text = csv_path.read_text(encoding="utf-8")
            header = text.strip().split("\n")[0]
            # DataLoader互換のカラム名で出力される
            expected_cols = [
                "date",
                "gdp",
                "consumption",
                "investment",
                "deflator",
                "wage",
                "employment",
                "rate",
            ]
            assert header == ",".join(expected_cols)


class TestSyntheticDataGenerator:
    """SyntheticDataGenerator のテスト"""

    def test_correct_shape(self) -> None:
        """生成データが正しい次元を持つことを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        data = gen.generate(params, n_periods=100, rng=rng)

        assert data.n_periods == 100
        assert data.n_obs == 7  # 7 observables
        assert data.data.shape == (100, 7)

    def test_finite_values(self) -> None:
        """生成データが有限値であることを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        data = gen.generate(params, n_periods=200, rng=rng)

        assert np.all(np.isfinite(data.data))

    def test_dates_format(self) -> None:
        """日付ラベルが正しいフォーマットであることを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        data = gen.generate(params, n_periods=8, rng=rng)

        assert len(data.dates) == 8
        assert data.dates[0] == "1994Q1"
        assert data.dates[3] == "1994Q4"
        assert data.dates[4] == "1995Q1"

    def test_variable_names(self) -> None:
        """変数名がデフォルト観測変数と一致することを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        data = gen.generate(params, n_periods=10, rng=rng)

        expected_names = [obs.name for obs in DEFAULT_OBSERVABLES]
        assert data.variable_names == expected_names

    def test_reproducibility(self) -> None:
        """同じシードで再現性があることを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()

        data1 = gen.generate(params, n_periods=50, rng=np.random.default_rng(123))
        data2 = gen.generate(params, n_periods=50, rng=np.random.default_rng(123))

        np.testing.assert_array_equal(data1.data, data2.data)

    def test_measurement_error_affects_data(self) -> None:
        """測定誤差の大きさがデータに反映されることを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()

        data_small = gen.generate(
            params, n_periods=200, measurement_error_std=0.0001, rng=np.random.default_rng(42)
        )
        data_large = gen.generate(
            params, n_periods=200, measurement_error_std=0.1, rng=np.random.default_rng(42)
        )

        # 大きい測定誤差のほうがデータの分散が大きい
        var_small = np.var(data_small.data, axis=0)
        var_large = np.var(data_large.data, axis=0)
        # 少なくとも1つの変数で大きい測定誤差のほうが分散が大きいはず
        assert np.any(var_large > var_small)

    def test_zero_measurement_error(self) -> None:
        """測定誤差ゼロでも正常に動作することを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        data = gen.generate(params, n_periods=50, measurement_error_std=0.0, rng=rng)

        assert np.all(np.isfinite(data.data))
        assert data.data.shape == (50, 7)


class TestEstimationData:
    """EstimationData のテスト"""

    def test_dimensions_consistency(self) -> None:
        """n_obs, n_periods がデータ配列と一致することを確認"""
        data = np.random.default_rng(42).normal(0, 1, (30, 5))
        ed = EstimationData(
            data=data,
            variable_names=["a", "b", "c", "d", "e"],
            dates=[f"Q{i}" for i in range(30)],
            n_obs=5,
            n_periods=30,
        )
        assert ed.data.shape[0] == ed.n_periods
        assert ed.data.shape[1] == ed.n_obs
        assert len(ed.variable_names) == ed.n_obs
        assert len(ed.dates) == ed.n_periods

    def test_from_csv_classmethod(self) -> None:
        """from_csv → DataLoader.load_csv でエラーなく読み込めることを確認"""
        params = DefaultParameters()
        gen = SyntheticDataGenerator()
        rng = np.random.default_rng(42)
        synth = gen.generate(params, n_periods=20, rng=rng)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "test.csv"
            gen.to_csv(synth, csv_path)

            # to_csv は DataLoader 互換フォーマットで書き出すため
            # from_csv で正常に読み込める
            loaded = EstimationData.from_csv(csv_path)
            assert loaded.n_obs == 7
            assert loaded.n_periods == synth.n_periods
            assert np.all(np.isfinite(loaded.data))


class TestObservableDefinition:
    """ObservableDefinition のテスト"""

    def test_default_observables_count(self) -> None:
        """デフォルト観測変数が7つであることを確認"""
        assert len(DEFAULT_OBSERVABLES) == 7

    def test_default_observables_transforms(self) -> None:
        """デフォルト観測変数の変換方法が正しいことを確認"""
        transforms = {obs.name: obs.transform for obs in DEFAULT_OBSERVABLES}
        assert transforms["output_growth"] == "dlog"
        assert transforms["consumption_growth"] == "dlog"
        assert transforms["investment_growth"] == "dlog"
        assert transforms["inflation"] == "level"
        assert transforms["wage_growth"] == "dlog"
        assert transforms["hours"] == "level"
        assert transforms["interest_rate"] == "rate"

    def test_default_observables_model_variables(self) -> None:
        """デフォルト観測変数のモデル変数名が正しいことを確認"""
        model_vars = {obs.name: obs.model_variable for obs in DEFAULT_OBSERVABLES}
        assert model_vars["output_growth"] == "y"
        assert model_vars["consumption_growth"] == "c"
        assert model_vars["investment_growth"] == "i"
        assert model_vars["inflation"] == "pi"
        assert model_vars["wage_growth"] == "w"
        assert model_vars["hours"] == "n"
        assert model_vars["interest_rate"] == "r"

    def test_frozen_dataclass(self) -> None:
        """ObservableDefinitionが不変であることを確認"""
        obs = DEFAULT_OBSERVABLES[0]
        with pytest.raises(AttributeError):
            obs.name = "modified"  # type: ignore[misc]
