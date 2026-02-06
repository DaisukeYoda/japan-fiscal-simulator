"""NKモデルのシステム行列構築テスト (Phase 2)"""

import numpy as np

from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


def test_system_matrices_shape_phase2() -> None:
    model = NewKeynesianModel(DefaultParameters())
    matrices = model._build_system_matrices()

    assert matrices.A.shape == (11, 11)
    assert matrices.B.shape == (11, 11)
    assert matrices.C.shape == (11, 11)
    assert matrices.D.shape == (11, 5)


def test_system_matrices_wage_shock_column() -> None:
    model = NewKeynesianModel(DefaultParameters())
    matrices = model._build_system_matrices()

    # e_w 列に少なくとも1つは非ゼロがある
    assert np.any(np.abs(matrices.D[:, 4]) > 0.0)
