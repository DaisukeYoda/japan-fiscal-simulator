"""パラメータ境界値とエラーケースのテスト"""

from dataclasses import FrozenInstanceError

import pytest

from japan_fiscal_simulator.core.exceptions import (
    ParameterValidationError,
    ShockValidationError,
    ValidationError,
)
from japan_fiscal_simulator.core.model import VARIABLE_INDICES, DSGEModel
from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.parameters.calibration import JapanCalibration
from japan_fiscal_simulator.parameters.defaults import (
    DefaultParameters,
    FirmParameters,
    HouseholdParameters,
)


class TestShockValidation:
    """ショックバリデーションのテスト"""

    @pytest.fixture
    def simulator(self) -> ImpulseResponseSimulator:
        model = DSGEModel(DefaultParameters())
        return ImpulseResponseSimulator(model)

    def test_invalid_shock_name_raises_error(self, simulator: ImpulseResponseSimulator) -> None:
        """無効なショック名はShockValidationErrorを発生させる"""
        with pytest.raises(ShockValidationError):
            simulator.simulate("invalid_shock", 0.01)

    def test_shock_size_exceeds_max_raises_error(self, simulator: ImpulseResponseSimulator) -> None:
        """ショックサイズが最大値を超えるとShockValidationErrorを発生させる"""
        with pytest.raises(ShockValidationError):
            simulator.simulate("e_g", 0.51)

    def test_negative_shock_at_boundary_works(self, simulator: ImpulseResponseSimulator) -> None:
        """負のショックサイズが境界値（abs=0.50）で正常動作する"""
        result = simulator.simulate("e_g", -0.50)
        assert result.shock_size == -0.50

    def test_positive_shock_at_boundary_works(self, simulator: ImpulseResponseSimulator) -> None:
        """正のショックサイズが境界値（0.50）で正常動作する"""
        result = simulator.simulate("e_g", 0.50)
        assert result.shock_size == 0.50

    def test_negative_shock_exceeds_max_raises_error(
        self, simulator: ImpulseResponseSimulator
    ) -> None:
        """負のショックサイズの絶対値が最大値を超えるとエラー"""
        with pytest.raises(ShockValidationError):
            simulator.simulate("e_g", -0.51)


class TestPeriodValidation:
    """シミュレーション期間バリデーションのテスト"""

    @pytest.fixture
    def simulator(self) -> ImpulseResponseSimulator:
        model = DSGEModel(DefaultParameters())
        return ImpulseResponseSimulator(model)

    def test_zero_periods_raises_error(self, simulator: ImpulseResponseSimulator) -> None:
        """期間数0はValidationErrorを発生させる"""
        with pytest.raises(ValidationError):
            simulator.simulate("e_g", 0.01, periods=0)

    def test_periods_exceeds_max_raises_error(self, simulator: ImpulseResponseSimulator) -> None:
        """期間数が最大値を超えるとValidationErrorを発生させる"""
        with pytest.raises(ValidationError):
            simulator.simulate("e_g", 0.01, periods=201)

    def test_min_periods_works(self, simulator: ImpulseResponseSimulator) -> None:
        """最小期間数（1）で正常動作する"""
        result = simulator.simulate("e_g", 0.01, periods=1)
        assert result.periods == 2  # t=0を含む

    def test_max_periods_works(self, simulator: ImpulseResponseSimulator) -> None:
        """最大期間数（200）で正常動作する"""
        result = simulator.simulate("e_g", 0.01, periods=200)
        assert result.periods == 201  # t=0を含む

    def test_negative_periods_raises_error(self, simulator: ImpulseResponseSimulator) -> None:
        """負の期間数はValidationErrorを発生させる"""
        with pytest.raises(ValidationError):
            simulator.simulate("e_g", 0.01, periods=-1)


class TestConsumptionTaxBoundary:
    """消費税パラメータ境界値のテスト"""

    def test_min_consumption_tax_works(self) -> None:
        """消費税率0%（下限）で正常動作する"""
        cal = JapanCalibration.create().set_consumption_tax(0.0)
        assert cal.parameters.government.tau_c == 0.0

    def test_max_consumption_tax_works(self) -> None:
        """消費税率50%（上限）で正常動作する"""
        cal = JapanCalibration.create().set_consumption_tax(0.50)
        assert cal.parameters.government.tau_c == 0.50

    def test_consumption_tax_above_max_raises_error(self) -> None:
        """消費税率が上限を超えるとParameterValidationErrorを発生させる"""
        with pytest.raises(ParameterValidationError):
            JapanCalibration.create().set_consumption_tax(0.51)

    def test_consumption_tax_below_min_raises_error(self) -> None:
        """消費税率が下限を下回るとParameterValidationErrorを発生させる"""
        with pytest.raises(ParameterValidationError):
            JapanCalibration.create().set_consumption_tax(-0.01)


class TestGovernmentSpendingRatioBoundary:
    """政府支出比率パラメータ境界値のテスト"""

    def test_min_spending_ratio_works(self) -> None:
        """政府支出比率0%（下限）で正常動作する"""
        cal = JapanCalibration.create().set_government_spending_ratio(0.0)
        assert cal.parameters.government.g_y_ratio == 0.0

    def test_max_spending_ratio_works(self) -> None:
        """政府支出比率60%（上限）で正常動作する"""
        cal = JapanCalibration.create().set_government_spending_ratio(0.60)
        assert cal.parameters.government.g_y_ratio == 0.60

    def test_spending_ratio_above_max_raises_error(self) -> None:
        """政府支出比率が上限を超えるとParameterValidationErrorを発生させる"""
        with pytest.raises(ParameterValidationError):
            JapanCalibration.create().set_government_spending_ratio(0.61)

    def test_spending_ratio_below_min_raises_error(self) -> None:
        """政府支出比率が下限を下回るとParameterValidationErrorを発生させる"""
        with pytest.raises(ParameterValidationError):
            JapanCalibration.create().set_government_spending_ratio(-0.01)


class TestFrozenDataclassImmutability:
    """frozenデータクラスの不変性テスト"""

    def test_household_params_frozen(self) -> None:
        """HouseholdParametersは変更不可"""
        params = HouseholdParameters()
        with pytest.raises(FrozenInstanceError):
            params.beta = 0.5  # type: ignore[misc]

    def test_firm_params_frozen(self) -> None:
        """FirmParametersは変更不可"""
        params = FirmParameters()
        with pytest.raises(FrozenInstanceError):
            params.alpha = 0.5  # type: ignore[misc]


class TestCacheInvalidation:
    """キャッシュ無効化のテスト"""

    def test_invalidate_cache_clears_all(self) -> None:
        """invalidate_cacheが全キャッシュをクリアする"""
        model = DSGEModel(DefaultParameters())

        # キャッシュを生成
        _ = model.steady_state
        _ = model.policy_function

        # キャッシュが存在することを確認
        assert model._steady_state is not None
        assert model._policy_result is not None
        assert model._nk_model is not None

        # キャッシュ無効化
        model.invalidate_cache()

        assert model._steady_state is None
        assert model._policy_result is None
        assert model._nk_model is None

    def test_recomputation_after_invalidation(self) -> None:
        """キャッシュ無効化後にプロパティが再計算される"""
        model = DSGEModel(DefaultParameters())

        _ = model.steady_state
        model.invalidate_cache()

        # 再計算が正常に動作する
        ss = model.steady_state
        assert ss is not None


class TestVariableIndexBoundaries:
    """変数インデックス境界のテスト"""

    def test_invalid_variable_name_raises_key_error(self) -> None:
        """無効な変数名はKeyErrorを発生させる"""
        with pytest.raises(KeyError):
            _ = VARIABLE_INDICES["invalid"]

    def test_get_variable_name_invalid_index_raises_error(self) -> None:
        """無効なインデックスはValidationErrorを発生させる"""
        model = DSGEModel(DefaultParameters())
        with pytest.raises(ValidationError):
            model.get_variable_name(999)

    def test_get_variable_index_roundtrip(self) -> None:
        """変数名→インデックス→変数名の往復が一致する"""
        model = DSGEModel(DefaultParameters())
        for name in VARIABLE_INDICES:
            idx = model.get_variable_index(name)
            assert model.get_variable_name(idx) == name
