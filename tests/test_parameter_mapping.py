"""パラメータマッピングのテスト"""

import numpy as np
import pytest

from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping, ParameterSpec
from japan_fiscal_simulator.parameters.defaults import (
    DefaultParameters,
    HouseholdParameters,
)


class TestParameterSpec:
    """ParameterSpecのテスト"""

    def test_frozen(self) -> None:
        """immutableであることを確認"""
        spec = ParameterSpec("test", "household", "sigma", 1.5, 0.5, 5.0)
        with pytest.raises(AttributeError):
            spec.name = "changed"  # type: ignore[misc]


class TestParameterMapping:
    """ParameterMappingのテスト"""

    def setup_method(self) -> None:
        self.mapping = ParameterMapping()

    def test_n_params_consistency(self) -> None:
        """n_paramsがESTIMATED_PARAMSのリスト長と一致"""
        assert self.mapping.n_params == len(ParameterMapping.ESTIMATED_PARAMS)
        assert self.mapping.n_params == len(self.mapping.names)
        assert self.mapping.n_params == len(self.mapping.defaults())
        assert self.mapping.n_params == len(self.mapping.bounds())

    def test_names_unique(self) -> None:
        """パラメータ名が一意であること"""
        names = self.mapping.names
        assert len(names) == len(set(names))

    def test_defaults_shape(self) -> None:
        """デフォルト値ベクトルの形状"""
        defaults = self.mapping.defaults()
        assert defaults.shape == (self.mapping.n_params,)
        assert defaults.dtype == np.float64

    def test_bounds_valid(self) -> None:
        """全バウンドで下限<上限、デフォルト値がバウンド内"""
        defaults = self.mapping.defaults()
        for i, (lb, ub) in enumerate(self.mapping.bounds()):
            assert lb < ub, f"パラメータ {self.mapping.names[i]}: 下限({lb}) >= 上限({ub})"
            assert lb <= defaults[i] <= ub, (
                f"パラメータ {self.mapping.names[i]}: "
                f"デフォルト値({defaults[i]})がバウンド[{lb}, {ub}]の範囲外"
            )

    def test_round_trip(self) -> None:
        """params → theta → params のラウンドトリップ"""
        original = DefaultParameters()
        theta = self.mapping.params_to_theta(original)
        reconstructed = self.mapping.theta_to_params(theta)

        # 推定対象パラメータが一致することを確認
        assert reconstructed.household.sigma == original.household.sigma
        assert reconstructed.household.phi == original.household.phi
        assert reconstructed.household.habit == original.household.habit
        assert reconstructed.firm.theta == original.firm.theta
        assert reconstructed.firm.psi == original.firm.psi
        assert reconstructed.investment.S_double_prime == original.investment.S_double_prime
        assert reconstructed.labor.theta_w == original.labor.theta_w
        assert reconstructed.labor.iota_w == original.labor.iota_w
        assert reconstructed.central_bank.phi_pi == original.central_bank.phi_pi
        assert reconstructed.central_bank.phi_y == original.central_bank.phi_y
        assert reconstructed.central_bank.rho_r == original.central_bank.rho_r
        assert reconstructed.shocks.rho_a == original.shocks.rho_a
        assert reconstructed.shocks.sigma_a == original.shocks.sigma_a

    def test_round_trip_theta(self) -> None:
        """theta → params → theta のラウンドトリップ"""
        original_theta = self.mapping.defaults()
        params = self.mapping.theta_to_params(original_theta)
        recovered_theta = self.mapping.params_to_theta(params)

        # モデルパラメータ部分が一致（観測誤差はデフォルト値で埋められる）
        np.testing.assert_allclose(recovered_theta, original_theta)

    def test_theta_to_params_with_default_theta(self) -> None:
        """デフォルトθでDefaultParametersのデフォルト値と一致"""
        theta = self.mapping.defaults()
        params = self.mapping.theta_to_params(theta)
        default_params = DefaultParameters()

        assert params.household.sigma == default_params.household.sigma
        assert params.firm.theta == default_params.firm.theta
        assert params.central_bank.phi_pi == default_params.central_bank.phi_pi
        assert params.shocks.rho_a == default_params.shocks.rho_a

    def test_theta_to_params_modifies_values(self) -> None:
        """θを変更するとパラメータが変わることを確認"""
        theta = self.mapping.defaults().copy()
        # sigma を変更
        sigma_idx = self.mapping.names.index("sigma")
        theta[sigma_idx] = 3.0

        params = self.mapping.theta_to_params(theta)
        assert params.household.sigma == 3.0
        # 他のパラメータはデフォルト値のまま
        assert params.household.phi == DefaultParameters().household.phi

    def test_theta_to_params_with_base_params(self) -> None:
        """base_paramsを指定して非推定パラメータを維持"""
        base = DefaultParameters().with_updates(
            household=HouseholdParameters(beta=0.995, sigma=1.5, phi=2.0, habit=0.7, chi=1.0)
        )
        theta = self.mapping.defaults()
        params = self.mapping.theta_to_params(theta, base_params=base)

        # 非推定パラメータ（beta）が保持される
        assert params.household.beta == 0.995

    def test_theta_to_params_preserves_fixed_params(self) -> None:
        """固定パラメータ（beta, delta, alpha等）が変更されないことを確認"""
        theta = self.mapping.defaults()
        params = self.mapping.theta_to_params(theta)
        default = DefaultParameters()

        assert params.household.beta == default.household.beta
        assert params.firm.alpha == default.firm.alpha
        assert params.firm.delta == default.firm.delta
        assert params.firm.epsilon == default.firm.epsilon
        assert params.labor.epsilon_w == default.labor.epsilon_w
        assert params.government.g_y_ratio == default.government.g_y_ratio
        assert params.central_bank.pi_target == default.central_bank.pi_target

    def test_measurement_errors_extraction(self) -> None:
        """観測誤差の抽出"""
        theta = self.mapping.defaults()
        me = self.mapping.theta_to_measurement_errors(theta)

        assert me.shape == (self.mapping.n_measurement,)
        assert self.mapping.n_measurement == 7  # y, c, i, pi, w, n, r
        assert len(self.mapping.measurement_names) == 7

        # デフォルト値の確認
        for val in me:
            assert val == pytest.approx(0.01)

    def test_measurement_errors_modified(self) -> None:
        """観測誤差パラメータの変更が反映される"""
        theta = self.mapping.defaults().copy()
        me_y_idx = self.mapping.names.index("me_y")
        theta[me_y_idx] = 0.05

        me = self.mapping.theta_to_measurement_errors(theta)
        assert me[0] == pytest.approx(0.05)

    def test_invalid_theta_length(self) -> None:
        """不正な長さのθでValueError"""
        with pytest.raises(ValueError, match="θの長さが不正"):
            self.mapping.theta_to_params(np.zeros(5))

        with pytest.raises(ValueError, match="θの長さが不正"):
            self.mapping.theta_to_measurement_errors(np.zeros(5))

    def test_params_to_theta_measurement_defaults(self) -> None:
        """params_to_thetaで観測誤差はデフォルト値が返る"""
        params = DefaultParameters()
        theta = self.mapping.params_to_theta(params)
        me = self.mapping.theta_to_measurement_errors(theta)

        for val in me:
            assert val == pytest.approx(0.01)

    def test_all_sections_covered(self) -> None:
        """推定対象の全セクションがカバーされている"""
        sections = {spec.section for spec in ParameterMapping.ESTIMATED_PARAMS}
        expected = {
            "household",
            "firm",
            "investment",
            "labor",
            "central_bank",
            "shocks",
            "measurement",
            "steady_state",
        }
        assert sections == expected

    def test_parameter_count(self) -> None:
        """推定パラメータが約30個"""
        assert 25 <= self.mapping.n_params <= 35
