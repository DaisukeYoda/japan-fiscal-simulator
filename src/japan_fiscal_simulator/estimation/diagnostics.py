"""MCMC収束診断

Gelman-Rubin R-hat, 有効サンプルサイズ (ESS), Geweke検定を実装する。
"""

from dataclasses import dataclass

import numpy as np
import scipy.stats


@dataclass
class ConvergenceDiagnostics:
    """MCMC収束診断結果

    Attributes:
        r_hat: Gelman-Rubin R-hat統計量 (n_params,)
        ess: 有効サンプルサイズ (n_params,)
        acceptance_rates: チェーンごとの採択率 (n_chains,)
        converged: 全パラメータが R-hat < 1.1 かどうか
        geweke_z: Geweke z-scores (n_params,)
        geweke_p: Geweke p-values (n_params,)
        parameter_names: パラメータ名のリスト
    """

    r_hat: np.ndarray
    ess: np.ndarray
    acceptance_rates: np.ndarray
    converged: bool
    geweke_z: np.ndarray
    geweke_p: np.ndarray
    parameter_names: list[str]


def compute_rhat(chains: np.ndarray) -> np.ndarray:
    """Gelman-Rubin R-hat統計量を計算する

    Args:
        chains: MCMCチェーン配列 (n_chains, n_draws, n_params)

    Returns:
        R-hat値 (n_params,)

    チェーン間の分散とチェーン内の分散の比に基づく収束診断指標。
    R-hat < 1.1 が収束の目安。

    Formula:
        W = チェーン内分散の平均
        B = チェーン平均の分散 * n_draws
        V_hat = (1 - 1/n) * W + (1/n) * B
        R_hat = sqrt(V_hat / W)
    """
    n_chains, n_draws, n_params = chains.shape

    # チェーンごとの平均: (n_chains, n_params)
    chain_means = chains.mean(axis=1)

    # チェーンごとの分散: (n_chains, n_params)
    chain_vars = chains.var(axis=1, ddof=1)

    # W: チェーン内分散の平均
    w = chain_vars.mean(axis=0)

    # B: チェーン平均の分散 * n_draws
    b = chain_means.var(axis=0, ddof=1) * n_draws

    # V_hat: 推定分散
    v_hat = (1.0 - 1.0 / n_draws) * w + (1.0 / n_draws) * b

    # W が 0 の場合（全チェーンが定数）は R-hat = 1.0
    r_hat = np.where(w > 0, np.sqrt(v_hat / w), 1.0)
    return r_hat


def compute_ess(chains: np.ndarray, max_lag: int | None = None) -> np.ndarray:
    """自己相関に基づく有効サンプルサイズを計算する

    Args:
        chains: MCMCチェーン配列 (n_chains, n_draws, n_params)
        max_lag: 自己相関計算の最大ラグ。Noneの場合は n_draws を使用。

    Returns:
        ESS値 (n_params,)

    各パラメータについて:
    1. 全チェーンを結合
    2. 自己相関を計算
    3. 正のペアの合計で ESS = n_total / (1 + 2 * sum) を算出
    """
    n_chains, n_draws, n_params = chains.shape
    n_total = n_chains * n_draws

    if max_lag is None:
        max_lag = n_draws

    ess = np.zeros(n_params)

    for p in range(n_params):
        # 全チェーンを結合
        combined = chains[:, :, p].ravel()
        mean = combined.mean()
        var = combined.var()

        if var < 1e-30:
            ess[p] = float(n_total)
            continue

        # 自己相関を計算
        autocorr_sum = 0.0
        for lag in range(1, max_lag):
            c = np.mean((combined[:-lag] - mean) * (combined[lag:] - mean)) / var
            if c < 0:
                break
            autocorr_sum += c

        denominator = 1.0 + 2.0 * autocorr_sum
        ess[p] = n_total / max(denominator, 1.0)

    return ess


def geweke_test(
    chain: np.ndarray,
    first_frac: float = 0.1,
    last_frac: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Geweke収束診断

    チェーンの最初 first_frac 部分と最後 last_frac 部分の平均を比較する。
    収束していれば z-score は標準正規分布に従う。

    Args:
        chain: MCMCチェーン (n_draws, n_params)
        first_frac: 前半ウィンドウの割合
        last_frac: 後半ウィンドウの割合

    Returns:
        (z_scores, p_values) 各 (n_params,)
    """
    n_draws = chain.shape[0]
    if chain.ndim == 1:
        chain = chain[:, np.newaxis]

    n_first = max(int(n_draws * first_frac), 1)
    n_last = max(int(n_draws * last_frac), 1)

    first_part = chain[:n_first]
    last_part = chain[-n_last:]

    mean_first = first_part.mean(axis=0)
    mean_last = last_part.mean(axis=0)

    # スペクトル密度の推定にはサンプル分散を使用（簡易版）
    var_first = first_part.var(axis=0, ddof=1) / n_first
    var_last = last_part.var(axis=0, ddof=1) / n_last

    se = np.sqrt(var_first + var_last)
    z_scores = np.where(se > 0, (mean_first - mean_last) / se, 0.0)
    p_values = 2.0 * (1.0 - scipy.stats.norm.cdf(np.abs(z_scores)))

    return z_scores, p_values


def run_diagnostics(
    chains: np.ndarray,
    acceptance_rates: np.ndarray,
    parameter_names: list[str],
) -> ConvergenceDiagnostics:
    """全ての収束診断を実行する

    Args:
        chains: MCMCチェーン配列 (n_chains, n_draws, n_params)
        acceptance_rates: チェーンごとの採択率 (n_chains,)
        parameter_names: パラメータ名のリスト

    Returns:
        全診断結果を含む ConvergenceDiagnostics
    """
    r_hat = compute_rhat(chains)
    ess = compute_ess(chains)

    # Gewekeテストは全チェーン結合で実行
    n_chains, n_draws, n_params = chains.shape
    combined = chains.reshape(n_chains * n_draws, n_params)
    geweke_z, geweke_p = geweke_test(combined)

    converged = bool(np.all(r_hat < 1.1))

    return ConvergenceDiagnostics(
        r_hat=r_hat,
        ess=ess,
        acceptance_rates=acceptance_rates,
        converged=converged,
        geweke_z=geweke_z,
        geweke_p=geweke_p,
        parameter_names=parameter_names,
    )
