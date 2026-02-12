"""ベイズ推定の事前分布

Smets-Wouters (2007) および Sugo-Ueda (2008) に基づく
パラメータの事前分布を定義する。
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
import scipy.stats


class DistributionType(Enum):
    """事前分布の種類"""

    BETA = "beta"
    GAMMA = "gamma"
    NORMAL = "normal"
    INV_GAMMA = "inv_gamma"


@dataclass(frozen=True)
class ParameterPrior:
    """単一パラメータの事前分布

    Attributes:
        name: パラメータ名
        dist_type: 分布の種類
        mean: 事前分布の平均
        std: 事前分布の標準偏差
        lower_bound: 下限値
        upper_bound: 上限値
    """

    name: str
    dist_type: DistributionType
    mean: float
    std: float
    lower_bound: float = 0.0
    upper_bound: float = float("inf")

    def _get_scipy_dist(self) -> Any:
        """scipy frozen分布オブジェクトを返す"""
        match self.dist_type:
            case DistributionType.BETA:
                variance = self.std**2
                common = self.mean * (1 - self.mean) / variance - 1
                a = self.mean * common
                b = (1 - self.mean) * common
                return scipy.stats.beta(a, b)
            case DistributionType.GAMMA:
                a = (self.mean / self.std) ** 2
                scale = self.std**2 / self.mean
                return scipy.stats.gamma(a, scale=scale)
            case DistributionType.NORMAL:
                return scipy.stats.norm(loc=self.mean, scale=self.std)
            case DistributionType.INV_GAMMA:
                shape = (self.mean / self.std) ** 2 + 2
                scale = self.mean * (shape - 1)
                return scipy.stats.invgamma(shape, scale=scale)

    def log_pdf(self, value: float) -> float:
        """対数確率密度を計算する

        Args:
            value: パラメータの値

        Returns:
            対数確率密度。範囲外の場合は -inf
        """
        if value <= self.lower_bound or value >= self.upper_bound:
            return -np.inf
        dist = self._get_scipy_dist()
        lp = float(dist.logpdf(value))
        if not np.isfinite(lp):
            return -np.inf
        return lp

    def sample(self, rng: np.random.Generator, size: int = 1) -> np.ndarray:
        """事前分布からサンプルを生成する

        Args:
            rng: NumPy乱数生成器
            size: サンプル数

        Returns:
            サンプル配列
        """
        dist = self._get_scipy_dist()
        samples = np.asarray(dist.rvs(size=size, random_state=rng), dtype=np.float64)
        clipped: np.ndarray = np.clip(
            samples,
            self.lower_bound + 1e-10,
            self.upper_bound - 1e-10 if np.isfinite(self.upper_bound) else None,
        )
        return clipped


class PriorConfig:
    """事前分布の集合

    Attributes:
        priors: パラメータ事前分布のリスト
    """

    def __init__(self, priors: list[ParameterPrior]) -> None:
        self.priors = priors
        self._name_to_index: dict[str, int] = {p.name: i for i, p in enumerate(priors)}

    @property
    def n_params(self) -> int:
        """パラメータ数"""
        return len(self.priors)

    @property
    def names(self) -> list[str]:
        """パラメータ名のリスト"""
        return [p.name for p in self.priors]

    def get_prior(self, name: str) -> ParameterPrior:
        """名前でパラメータ事前分布を取得する"""
        idx = self._name_to_index[name]
        return self.priors[idx]

    def log_prior(self, theta: np.ndarray) -> float:
        """パラメータベクトルに対する対数事前確率を計算する

        Args:
            theta: パラメータベクトル（priors と同じ順序）

        Returns:
            対数事前確率の合計
        """
        total = 0.0
        for i, prior in enumerate(self.priors):
            lp = prior.log_pdf(theta[i])
            if not np.isfinite(lp):
                return -np.inf
            total += lp
        return total

    def sample(self, rng: np.random.Generator) -> np.ndarray:
        """全パラメータの事前分布からサンプルを生成する

        Args:
            rng: NumPy乱数生成器

        Returns:
            パラメータベクトル
        """
        return np.array([p.sample(rng, size=1)[0] for p in self.priors])

    def means(self) -> np.ndarray:
        """全パラメータの事前分布の平均値を返す"""
        return np.array([p.mean for p in self.priors])

    @classmethod
    def smets_wouters_japan(cls) -> PriorConfig:
        """Smets-Wouters (2007) + 日本経済向け事前分布を返す

        Sugo & Ueda (2008) の日本向けキャリブレーションを参考に調整。
        """
        priors: list[ParameterPrior] = [
            # 構造パラメータ
            ParameterPrior(
                name="sigma",
                dist_type=DistributionType.GAMMA,
                mean=1.5,
                std=0.37,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="phi",
                dist_type=DistributionType.GAMMA,
                mean=2.0,
                std=0.75,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="habit",
                dist_type=DistributionType.BETA,
                mean=0.7,
                std=0.1,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="theta",
                dist_type=DistributionType.BETA,
                mean=0.75,
                std=0.05,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="psi",
                dist_type=DistributionType.BETA,
                mean=0.5,
                std=0.15,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            # 投資パラメータ
            ParameterPrior(
                name="S_double_prime",
                dist_type=DistributionType.GAMMA,
                mean=5.0,
                std=1.5,
                lower_bound=0.0,
            ),
            # 労働市場パラメータ
            ParameterPrior(
                name="theta_w",
                dist_type=DistributionType.BETA,
                mean=0.75,
                std=0.05,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="iota_w",
                dist_type=DistributionType.BETA,
                mean=0.5,
                std=0.15,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            # 金融政策パラメータ
            ParameterPrior(
                name="phi_pi",
                dist_type=DistributionType.NORMAL,
                mean=1.5,
                std=0.25,
                lower_bound=-np.inf,
                upper_bound=np.inf,
            ),
            ParameterPrior(
                name="phi_y",
                dist_type=DistributionType.NORMAL,
                mean=0.125,
                std=0.05,
                lower_bound=-np.inf,
                upper_bound=np.inf,
            ),
            ParameterPrior(
                name="rho_r",
                dist_type=DistributionType.BETA,
                mean=0.85,
                std=0.1,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            # ショック持続性パラメータ
            ParameterPrior(
                name="rho_a",
                dist_type=DistributionType.BETA,
                mean=0.9,
                std=0.05,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="rho_g",
                dist_type=DistributionType.BETA,
                mean=0.9,
                std=0.05,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="rho_i",
                dist_type=DistributionType.BETA,
                mean=0.7,
                std=0.1,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="rho_w",
                dist_type=DistributionType.BETA,
                mean=0.9,
                std=0.05,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="rho_p",
                dist_type=DistributionType.BETA,
                mean=0.9,
                std=0.05,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            ParameterPrior(
                name="rho_m",
                dist_type=DistributionType.BETA,
                mean=0.5,
                std=0.1,
                lower_bound=0.0,
                upper_bound=1.0,
            ),
            # ショック標準偏差
            ParameterPrior(
                name="sigma_a",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="sigma_g",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="sigma_i",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="sigma_w",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="sigma_p",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="sigma_m",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            # 観測誤差標準偏差
            ParameterPrior(
                name="me_y",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="me_c",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="me_i",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="me_pi",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="me_w",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="me_n",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
            ParameterPrior(
                name="me_r",
                dist_type=DistributionType.INV_GAMMA,
                mean=0.01,
                std=0.01,
                lower_bound=0.0,
            ),
        ]
        return cls(priors)
