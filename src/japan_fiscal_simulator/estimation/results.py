"""ベイズ推定結果

MCMCサンプリング結果の集約、事後分布のサマリー、周辺尤度の近似計算を提供する。
"""

from dataclasses import dataclass

import numpy as np

from japan_fiscal_simulator.estimation.diagnostics import ConvergenceDiagnostics, run_diagnostics
from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping
from japan_fiscal_simulator.estimation.priors import PriorConfig
from japan_fiscal_simulator.parameters.defaults import DefaultParameters


@dataclass
class PosteriorSummary:
    """単一パラメータの事後分布サマリー

    Attributes:
        name: パラメータ名
        mean: 事後平均
        median: 事後中央値
        std: 事後標準偏差
        hpd_lower: 90% HPD下限
        hpd_upper: 90% HPD上限
        prior_mean: 事前平均
        prior_std: 事前標準偏差
    """

    name: str
    mean: float
    median: float
    std: float
    hpd_lower: float
    hpd_upper: float
    prior_mean: float
    prior_std: float


@dataclass
class EstimationResult:
    """推定結果

    Attributes:
        posterior_samples: 全チェーン結合後の事後サンプル (n_total_samples, n_params)
        parameter_names: パラメータ名のリスト
        log_marginal_likelihood: Laplace近似による対数周辺尤度
        diagnostics: 収束診断結果
        summaries: パラメータごとの事後分布サマリー
        mode: 事後分布のモード（最大事後確率点）
        mode_log_posterior: モードでの対数事後確率
        n_chains: チェーン数
        n_draws: チェーンあたりのドロー数
        n_burnin: バーンイン数
    """

    posterior_samples: np.ndarray
    parameter_names: list[str]
    log_marginal_likelihood: float
    diagnostics: ConvergenceDiagnostics
    summaries: list[PosteriorSummary]
    mode: np.ndarray
    mode_log_posterior: float
    n_chains: int
    n_draws: int
    n_burnin: int

    def get_summary(self, name: str) -> PosteriorSummary:
        """名前でパラメータサマリーを取得する

        Args:
            name: パラメータ名

        Returns:
            指定パラメータの PosteriorSummary

        Raises:
            KeyError: パラメータが見つからない場合
        """
        for s in self.summaries:
            if s.name == name:
                return s
        msg = f"パラメータ '{name}' が見つかりません"
        raise KeyError(msg)

    def to_default_params(
        self,
        use: str = "posterior_mean",
        mapping: ParameterMapping | None = None,
    ) -> DefaultParameters:
        """推定値でDefaultParametersを生成する

        Args:
            use: "posterior_mean" または "mode"
            mapping: ParameterMapping インスタンス。Noneの場合は新規作成。

        Returns:
            推定値を反映した DefaultParameters
        """
        if mapping is None:
            mapping = ParameterMapping()

        if use == "posterior_mean":
            theta = self.posterior_samples.mean(axis=0)
        elif use == "mode":
            theta = self.mode
        else:
            msg = f"use は 'posterior_mean' または 'mode' のみ: {use}"
            raise ValueError(msg)

        return mapping.theta_to_params(theta)

    def summary_table(self) -> str:
        """マークダウン形式のサマリーテーブルを生成する

        Returns:
            パラメータ推定結果のマークダウンテーブル文字列
        """
        header = (
            "| Parameter | Prior Mean | Prior Std | Post. Mean | Post. Std "
            "| 90% HPD Lower | 90% HPD Upper |"
        )
        separator = "|-----------|-----------|----------|-----------|----------|--------------|--------------|"

        rows = [header, separator]
        for s in self.summaries:
            row = (
                f"| {s.name:9s} "
                f"| {s.prior_mean:9.4f} "
                f"| {s.prior_std:8.4f} "
                f"| {s.mean:9.4f} "
                f"| {s.std:8.4f} "
                f"| {s.hpd_lower:12.4f} "
                f"| {s.hpd_upper:12.4f} |"
            )
            rows.append(row)

        return "\n".join(rows)


def compute_hpd(samples: np.ndarray, alpha: float = 0.1) -> tuple[float, float]:
    """Highest Posterior Density (HPD) 区間を計算する

    (1-alpha)*100% の確率質量を含む最短の区間を求める。

    Args:
        samples: 1次元の事後サンプル配列
        alpha: 有意水準 (0.1 で 90% HPD)

    Returns:
        (lower, upper) HPD区間の下限と上限
    """
    sorted_samples = np.sort(samples)
    n = len(sorted_samples)
    interval_size = max(int(np.ceil((1.0 - alpha) * n)), 2)

    # 全候補区間の幅を計算
    widths = sorted_samples[interval_size - 1 :] - sorted_samples[: n - interval_size + 1]
    best_idx = int(np.argmin(widths))

    return float(sorted_samples[best_idx]), float(sorted_samples[best_idx + interval_size - 1])


def compute_marginal_likelihood_laplace(
    mode: np.ndarray,
    mode_log_posterior: float,
    hessian: np.ndarray,
) -> float:
    """Laplace近似による周辺尤度を計算する

    log p(y) ≈ log p(θ*|y) + (d/2) * log(2π) - 0.5 * log|H|

    ここで θ* はモード、H はモードでの負のヘッセ行列。

    Args:
        mode: 事後分布のモード (n_params,)
        mode_log_posterior: モードでの対数事後確率
        hessian: モードでの負のヘッセ行列 (n_params, n_params)

    Returns:
        対数周辺尤度の近似値
    """
    d = len(mode)
    sign, log_det = np.linalg.slogdet(hessian)

    if sign <= 0:
        # ヘッセ行列が正定値でない場合は対角要素で近似
        diag = np.diag(hessian)
        diag = np.maximum(diag, 1e-10)
        log_det = np.sum(np.log(diag))

    return float(mode_log_posterior + 0.5 * d * np.log(2.0 * np.pi) - 0.5 * log_det)


def build_estimation_result(
    chains: np.ndarray,
    acceptance_rates: np.ndarray,
    mode: np.ndarray,
    mode_log_posterior: float,
    hessian: np.ndarray,
    prior_config: PriorConfig,
    mapping: ParameterMapping,
    n_burnin: int,
) -> EstimationResult:
    """MCMCチェーンから EstimationResult を構築する

    Args:
        chains: MCMCチェーン配列 (n_chains, n_draws, n_params)
        acceptance_rates: チェーンごとの採択率 (n_chains,)
        mode: 事後分布のモード
        mode_log_posterior: モードでの対数事後確率
        hessian: モードでの負のヘッセ行列
        prior_config: 事前分布設定
        mapping: パラメータマッピング
        n_burnin: バーンイン数

    Returns:
        完成した EstimationResult
    """
    n_chains, n_draws, n_params = chains.shape
    parameter_names = mapping.names

    # 収束診断
    diagnostics = run_diagnostics(chains, acceptance_rates, parameter_names)

    # 全チェーン結合
    posterior_samples = chains.reshape(n_chains * n_draws, n_params)

    # 周辺尤度
    log_ml = compute_marginal_likelihood_laplace(mode, mode_log_posterior, hessian)

    # パラメータごとのサマリー
    summaries: list[PosteriorSummary] = []
    for i, name in enumerate(parameter_names):
        samples_i = posterior_samples[:, i]
        prior = prior_config.get_prior(name)
        hpd_lower, hpd_upper = compute_hpd(samples_i, alpha=0.1)

        summaries.append(
            PosteriorSummary(
                name=name,
                mean=float(np.mean(samples_i)),
                median=float(np.median(samples_i)),
                std=float(np.std(samples_i)),
                hpd_lower=hpd_lower,
                hpd_upper=hpd_upper,
                prior_mean=prior.mean,
                prior_std=prior.std,
            )
        )

    return EstimationResult(
        posterior_samples=posterior_samples,
        parameter_names=parameter_names,
        log_marginal_likelihood=log_ml,
        diagnostics=diagnostics,
        summaries=summaries,
        mode=mode,
        mode_log_posterior=mode_log_posterior,
        n_chains=n_chains,
        n_draws=n_draws,
        n_burnin=n_burnin,
    )
