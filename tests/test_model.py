"""DSGEモデルのテスト"""

from japan_fiscal_simulator.core.model import N_VARIABLES, VARIABLE_INDICES, DSGEModel
from japan_fiscal_simulator.parameters.calibration import JapanCalibration
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


class TestDSGEModel:
    """DSGEModelのテスト"""

    def test_model_initialization(self) -> None:
        """モデルが初期化できることを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)

        assert model.params is params

    def test_steady_state_computation(self) -> None:
        """定常状態が計算できることを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)

        ss = model.steady_state
        assert ss.output > 0
        assert ss.consumption > 0

    def test_policy_function(self) -> None:
        """政策関数が計算できることを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)

        pf = model.policy_function

        assert pf.P.shape == (N_VARIABLES, N_VARIABLES)
        assert pf.Q.shape[0] == N_VARIABLES

    def test_variable_indices(self) -> None:
        """変数インデックスが正しく設定されていることを確認"""
        assert "y" in VARIABLE_INDICES
        assert "c" in VARIABLE_INDICES
        assert "pi" in VARIABLE_INDICES
        assert "r" in VARIABLE_INDICES

    def test_get_variable_index(self) -> None:
        """変数インデックス取得が機能することを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)

        y_idx = model.get_variable_index("y")
        assert y_idx == VARIABLE_INDICES["y"]

    def test_get_variable_name(self) -> None:
        """変数名取得が機能することを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)

        name = model.get_variable_name(0)
        assert name == "y"

    def test_cache_invalidation(self) -> None:
        """キャッシュ無効化が機能することを確認"""
        params = DefaultParameters()
        model = DSGEModel(params)

        # キャッシュを生成
        _ = model.steady_state
        assert model._steady_state is not None

        # キャッシュを無効化
        model.invalidate_cache()
        assert model._steady_state is None


class TestModelWithJapanCalibration:
    """日本キャリブレーションでのモデルテスト"""

    def test_model_with_japan_calibration(self) -> None:
        """日本キャリブレーションでモデルが動作することを確認"""
        calibration = JapanCalibration.create()
        model = DSGEModel(calibration.parameters)

        ss = model.steady_state

        # 定常状態の消費税率が10%であることを確認
        assert ss.consumption_tax_rate == 0.10

        # 政府債務比率が高いことを確認
        debt_ratio = ss.government_debt / ss.output
        assert debt_ratio > 1.5
