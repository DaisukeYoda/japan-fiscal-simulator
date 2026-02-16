"""Japan Fiscal Simulator - 日本財政政策DSGEシミュレーター"""

from importlib.metadata import PackageNotFoundError, version


def _resolve_version() -> str:
    """配布メタデータからバージョンを解決する。"""
    try:
        return version("jpfs")
    except PackageNotFoundError:
        # インストール前のローカル実行時フォールバック
        return "0+unknown"


__version__ = _resolve_version()

from japan_fiscal_simulator.core.model import DSGEModel
from japan_fiscal_simulator.core.simulation import ImpulseResponseSimulator
from japan_fiscal_simulator.estimation.mcmc import MCMCConfig, MetropolisHastings
from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping
from japan_fiscal_simulator.estimation.priors import PriorConfig
from japan_fiscal_simulator.estimation.results import EstimationResult
from japan_fiscal_simulator.parameters.calibration import JapanCalibration

__all__ = [
    "DSGEModel",
    "EstimationResult",
    "ImpulseResponseSimulator",
    "JapanCalibration",
    "MCMCConfig",
    "MetropolisHastings",
    "ParameterMapping",
    "PriorConfig",
]
