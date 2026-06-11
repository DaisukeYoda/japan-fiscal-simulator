"""パラメータ生成時のfail-fastバリデーションのテスト（issue #30）"""

import math
from dataclasses import replace

import pytest

from japan_fiscal_simulator.core.exceptions import ParameterValidationError
from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping
from japan_fiscal_simulator.parameters.calibration import JapanCalibration
from japan_fiscal_simulator.parameters.defaults import (
    CentralBankParameters,
    DefaultParameters,
    FinancialParameters,
    FirmParameters,
    GovernmentParameters,
    HouseholdParameters,
    InvestmentParameters,
    LaborParameters,
    OpenEconomyParameters,
    ShockParameters,
)


class TestInvalidValuesRejected:
    """issue #30 で確認された不正値が生成時に拒否される"""

    def test_negative_consumption_tax_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            GovernmentParameters(tau_c=-0.1)

    def test_beta_zero_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            HouseholdParameters(beta=0.0)

    def test_beta_one_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            HouseholdParameters(beta=1.0)

    @pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
    def test_non_finite_rejected(self, value: float) -> None:
        with pytest.raises(ParameterValidationError):
            HouseholdParameters(beta=value)

    def test_replace_path_also_validated(self) -> None:
        """dataclasses.replace経由でも__post_init__が再実行され検証される"""
        with pytest.raises(ParameterValidationError):
            replace(HouseholdParameters(), habit=1.5)

    def test_theta_zero_rejected(self) -> None:
        """theta=0はPhillips曲線スロープのゼロ除算になるため拒否"""
        with pytest.raises(ParameterValidationError):
            FirmParameters(theta=0.0)

    def test_epsilon_at_most_one_rejected(self) -> None:
        """ε<=1ではマークアップ ε/(ε-1) が定義できない"""
        with pytest.raises(ParameterValidationError):
            FirmParameters(epsilon=1.0)

    def test_unit_root_shock_persistence_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            ShockParameters(rho_a=1.0)

    def test_negative_shock_std_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            ShockParameters(sigma_g=-0.01)

    def test_zero_investment_adjustment_cost_rejected(self) -> None:
        """投資方程式の係数 1/S'' が計算できないため拒否"""
        with pytest.raises(ParameterValidationError):
            InvestmentParameters(S_double_prime=0.0)

    def test_zero_leverage_rejected(self) -> None:
        """定常状態の nw = k / leverage_ss が計算できないため拒否"""
        with pytest.raises(ParameterValidationError):
            FinancialParameters(leverage_ss=0.0)

    def test_labor_elasticity_at_most_one_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            LaborParameters(epsilon_w=1.0)

    def test_import_share_one_rejected(self) -> None:
        with pytest.raises(ParameterValidationError):
            OpenEconomyParameters(import_share=1.0)


class TestErrorMessages:
    """エラーメッセージから対象パラメータと値を特定できる"""

    def test_message_contains_class_field_and_value(self) -> None:
        with pytest.raises(ParameterValidationError, match=r"GovernmentParameters\.tau_c=-0\.1"):
            GovernmentParameters(tau_c=-0.1)

    def test_message_contains_range(self) -> None:
        with pytest.raises(ParameterValidationError, match=r"0\.0 < beta < 1\.0"):
            HouseholdParameters(beta=0.0)

    def test_non_finite_message_identifies_parameter(self) -> None:
        with pytest.raises(ParameterValidationError, match=r"HouseholdParameters\.beta=nan"):
            HouseholdParameters(beta=float("nan"))


class TestBoundaryValuesAccepted:
    """境界上の正常値は受理される"""

    def test_zero_habit_accepted(self) -> None:
        assert HouseholdParameters(habit=0.0).habit == 0.0

    def test_zero_taxes_accepted(self) -> None:
        params = GovernmentParameters(tau_c=0.0, tau_l=0.0, tau_k=0.0)
        assert params.tau_c == 0.0

    def test_full_depreciation_accepted(self) -> None:
        assert FirmParameters(delta=1.0).delta == 1.0

    def test_psi_bounds_accepted(self) -> None:
        assert FirmParameters(psi=0.0).psi == 0.0
        assert FirmParameters(psi=1.0).psi == 1.0

    def test_zero_shock_std_accepted(self) -> None:
        assert ShockParameters(sigma_a=0.0).sigma_a == 0.0

    def test_taylor_principle_violation_allowed(self) -> None:
        """phi_pi < 1は不決定性の検証に使うため拒否しない（test_solver.py参照）"""
        assert CentralBankParameters(phi_pi=0.8).phi_pi == 0.8

    def test_negative_policy_coefficients_allowed(self) -> None:
        assert CentralBankParameters(phi_y=-0.1, r_lower_bound=-0.01).phi_y == -0.1


class TestExistingCalibrationsStillValid:
    """既存のキャリブレーションがすべて検証を通過する"""

    def test_default_parameters(self) -> None:
        assert DefaultParameters() is not None

    def test_japan_baseline(self) -> None:
        assert JapanCalibration.create() is not None

    def test_japan_high_debt(self) -> None:
        assert JapanCalibration.create_high_debt_scenario() is not None

    def test_japan_zlb(self) -> None:
        assert JapanCalibration.create_zlb_scenario() is not None


class TestMCMCBoundsCompatibility:
    """検証境界がMCMCの探索範囲を内包する"""

    def test_theta_at_mcmc_lower_bounds_accepted(self) -> None:
        mapping = ParameterMapping()
        theta = mapping.defaults()
        for i, (low, _) in enumerate(mapping.bounds()):
            if math.isfinite(low):
                theta[i] = low
        assert mapping.theta_to_params(theta) is not None

    def test_theta_at_mcmc_upper_bounds_accepted(self) -> None:
        mapping = ParameterMapping()
        theta = mapping.defaults()
        for i, (_, high) in enumerate(mapping.bounds()):
            if math.isfinite(high):
                theta[i] = high
        assert mapping.theta_to_params(theta) is not None
