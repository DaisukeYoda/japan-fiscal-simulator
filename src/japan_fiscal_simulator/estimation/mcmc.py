"""Random Walk Metropolis-Hastingsサンプラー

ベイズ推定の事後分布からのサンプリングを行う。
事後モード探索 → 提案共分散行列の構築 → 複数チェーンによるMCMCサンプリング
の手順で実行する。
"""

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import scipy.optimize

from japan_fiscal_simulator.core.exceptions import EstimationError
from japan_fiscal_simulator.core.nk_model import NewKeynesianModel
from japan_fiscal_simulator.estimation.kalman_filter import kalman_filter
from japan_fiscal_simulator.estimation.parameter_mapping import ParameterMapping
from japan_fiscal_simulator.estimation.priors import PriorConfig
from japan_fiscal_simulator.estimation.state_space import StateSpaceBuilder

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MCMCConfig:
    """MCMC設定"""

    n_chains: int = 4
    n_draws: int = 100_000
    n_burnin: int = 50_000
    thinning: int = 10
    target_acceptance: float = 0.234
    adaptive_interval: int = 100
    mode_search_max_iter: int = 500


@dataclass
class MCMCResult:
    """MCMC結果"""

    chains: np.ndarray  # (n_chains, n_kept, n_params) thinned chains
    acceptance_rates: np.ndarray  # (n_chains,)
    log_posteriors: np.ndarray  # (n_chains, n_kept)
    mode: np.ndarray  # (n_params,)
    mode_hessian: np.ndarray  # (n_params, n_params)
    mode_log_posterior: float
    parameter_names: list[str] = field(default_factory=list)


class MetropolisHastings:
    """Random Walk Metropolis-Hastingsサンプラー"""

    def __init__(
        self,
        log_posterior_fn: Callable[[np.ndarray], float],
        n_params: int,
        config: MCMCConfig | None = None,
        parameter_names: list[str] | None = None,
        bounds: list[tuple[float, float]] | None = None,
    ) -> None:
        self._log_posterior_fn = log_posterior_fn
        self._n_params = n_params
        self._config = config or MCMCConfig()
        self._parameter_names = parameter_names or [f"param_{i}" for i in range(n_params)]
        self._bounds = bounds

    def find_mode(self, theta0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """事後モードとヘシアン逆行列を求める

        scipy.optimize.minimize (L-BFGS-B) で負の対数事後確率を最小化する。
        ヘシアンは有限差分で数値近似する。正定値でなければ単位行列にフォールバック。

        Args:
            theta0: 初期パラメータベクトル

        Returns:
            (mode, hessian_inverse) のタプル
        """

        def neg_log_posterior(theta: np.ndarray) -> float:
            lp = self._log_posterior_fn(theta)
            if not np.isfinite(lp):
                return 1e10
            return -lp

        bounds = self._bounds
        result = scipy.optimize.minimize(
            neg_log_posterior,
            theta0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": self._config.mode_search_max_iter},
        )

        mode = result.x

        # ヘシアンを数値的に近似
        hessian_inv = self._compute_hessian_inverse(neg_log_posterior, mode)

        return mode, hessian_inv

    def _compute_hessian_inverse(
        self,
        neg_log_posterior: Callable[[np.ndarray], float],
        mode: np.ndarray,
    ) -> np.ndarray:
        """ヘシアン逆行列を数値的に計算する

        有限差分で二階微分を近似し、逆行列を計算する。
        正定値でない場合は単位行列にフォールバックする。
        """
        n = self._n_params
        eps = 1e-5
        hessian = np.zeros((n, n))

        for i in range(n):
            ei = np.zeros(n)
            ei[i] = eps * max(1.0, abs(mode[i]))
            for j in range(i, n):
                ej = np.zeros(n)
                ej[j] = eps * max(1.0, abs(mode[j]))

                fpp = neg_log_posterior(mode + ei + ej)
                fpm = neg_log_posterior(mode + ei - ej)
                fmp = neg_log_posterior(mode - ei + ej)
                fmm = neg_log_posterior(mode - ei - ej)

                hessian[i, j] = (fpp - fpm - fmp + fmm) / (4.0 * ei[i] * ej[j])
                hessian[j, i] = hessian[i, j]

        # 正定値チェックと逆行列計算
        try:
            eigvals = np.linalg.eigvalsh(hessian)
            if np.all(eigvals > 0):
                hessian_inv: np.ndarray = np.linalg.inv(hessian)
                # 対称性を保証
                hessian_inv = 0.5 * (hessian_inv + hessian_inv.T)
                return hessian_inv
        except np.linalg.LinAlgError:
            pass

        logger.warning("ヘシアンが正定値でないため単位行列にフォールバック")
        return np.eye(n)

    def run(self, theta0: np.ndarray | None = None) -> MCMCResult:
        """MCMCサンプリングを実行する

        1. find_mode()でモードとヘシアンを求める
        2. 各チェーンの初期値: mode + small perturbation
        3. 提案共分散: hessian_inverse * (2.38^2 / n_params)
        4. Random Walk MH: theta_new = theta_old + N(0, proposal_cov)
        5. Accept/reject: log(u) < log_post(new) - log_post(old)
        6. Adaptive: 毎adaptive_intervalドローで経験共分散に更新（burnin中のみ）
        7. Burnin除去 + thinning

        Args:
            theta0: 初期パラメータベクトル。Noneの場合はモード探索のみ。

        Returns:
            MCMCResult
        """
        if theta0 is None:
            msg = "theta0を指定してください"
            raise EstimationError(msg)

        # モード探索
        mode, hessian_inv = self.find_mode(theta0)
        mode_log_post = self._log_posterior_fn(mode)

        # 提案共分散行列
        scale = (2.38**2) / self._n_params
        proposal_cov = scale * hessian_inv

        # 対称正定値を保証
        proposal_cov = 0.5 * (proposal_cov + proposal_cov.T)
        eigvals = np.linalg.eigvalsh(proposal_cov)
        if np.any(eigvals <= 0):
            proposal_cov += (abs(eigvals.min()) + 1e-8) * np.eye(self._n_params)

        cfg = self._config
        n_kept = (cfg.n_draws - cfg.n_burnin) // cfg.thinning

        all_chains = np.empty((cfg.n_chains, n_kept, self._n_params))
        all_log_posts = np.empty((cfg.n_chains, n_kept))
        all_acceptance = np.empty(cfg.n_chains)

        for chain_id in range(cfg.n_chains):
            # 各チェーンの初期値にランダム摂動を加える
            rng = np.random.default_rng(seed=chain_id * 1000 + 42)
            perturbation = rng.multivariate_normal(np.zeros(self._n_params), 0.01 * proposal_cov)
            chain_init = mode + perturbation

            # 境界内にクリップ
            if self._bounds is not None:
                for k, (lb, ub) in enumerate(self._bounds):
                    chain_init[k] = np.clip(chain_init[k], lb + 1e-10, ub - 1e-10)

            draws, log_posts, acc_rate = self._run_chain(chain_id, chain_init, proposal_cov, rng)

            # Burnin除去 + thinning
            post_burnin = draws[cfg.n_burnin :]
            post_burnin_lp = log_posts[cfg.n_burnin :]

            thinned = post_burnin[:: cfg.thinning][:n_kept]
            thinned_lp = post_burnin_lp[:: cfg.thinning][:n_kept]

            all_chains[chain_id] = thinned
            all_log_posts[chain_id] = thinned_lp
            all_acceptance[chain_id] = acc_rate

        return MCMCResult(
            chains=all_chains,
            acceptance_rates=all_acceptance,
            log_posteriors=all_log_posts,
            mode=mode,
            mode_hessian=hessian_inv,
            mode_log_posterior=mode_log_post,
            parameter_names=list(self._parameter_names),
        )

    def _run_chain(
        self,
        chain_id: int,
        theta0: np.ndarray,
        proposal_cov: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, float]:
        """1チェーンを実行する

        Args:
            chain_id: チェーン番号
            theta0: 初期パラメータベクトル
            proposal_cov: 提案共分散行列
            rng: 乱数生成器

        Returns:
            (draws, log_posteriors, acceptance_rate) のタプル
            draws: (n_draws, n_params)
            log_posteriors: (n_draws,)
        """
        cfg = self._config
        n_draws = cfg.n_draws
        n_params = self._n_params

        draws = np.empty((n_draws, n_params))
        log_posts = np.empty(n_draws)

        theta_cur = theta0.copy()
        lp_cur = self._log_posterior_fn(theta_cur)
        if not np.isfinite(lp_cur):
            lp_cur = -1e100

        n_accepted = 0
        current_cov = proposal_cov.copy()

        # Cholesky分解（提案分布の効率的サンプリング用）
        try:
            chol = np.linalg.cholesky(current_cov)
        except np.linalg.LinAlgError:
            current_cov += 1e-6 * np.eye(n_params)
            chol = np.linalg.cholesky(current_cov)

        for t in range(n_draws):
            # 提案値の生成
            z = rng.standard_normal(n_params)
            theta_prop = theta_cur + chol @ z

            # 境界チェック
            in_bounds = True
            if self._bounds is not None:
                for k, (lb, ub) in enumerate(self._bounds):
                    if theta_prop[k] <= lb or theta_prop[k] >= ub:
                        in_bounds = False
                        break

            if in_bounds:
                lp_prop = self._log_posterior_fn(theta_prop)
            else:
                lp_prop = -np.inf

            # Accept/reject
            log_alpha = lp_prop - lp_cur
            if np.isfinite(log_alpha) and np.log(rng.uniform()) < log_alpha:
                theta_cur = theta_prop
                lp_cur = lp_prop
                n_accepted += 1

            draws[t] = theta_cur
            log_posts[t] = lp_cur

            # Adaptive更新（burnin中のみ）
            if t > 0 and t < cfg.n_burnin and t % cfg.adaptive_interval == 0 and t >= 2 * n_params:
                recent = draws[max(0, t - 5 * cfg.adaptive_interval) : t + 1]
                if len(recent) > n_params + 1:
                    emp_cov = np.cov(recent.T)
                    if emp_cov.ndim == 2 and np.all(np.isfinite(emp_cov)):
                        scale = (2.38**2) / n_params
                        new_cov = scale * emp_cov
                        # 正定値を保証
                        new_cov = 0.5 * (new_cov + new_cov.T)
                        new_cov += 1e-8 * np.eye(n_params)
                        try:
                            chol = np.linalg.cholesky(new_cov)
                            current_cov = new_cov
                        except np.linalg.LinAlgError:
                            pass  # Cholesky失敗時は以前の共分散を維持

        acceptance_rate = n_accepted / n_draws
        logger.info("チェーン %d: 受容率 %.3f", chain_id, acceptance_rate)

        return draws, log_posts, acceptance_rate


def make_log_posterior(
    mapping: ParameterMapping,
    prior_config: PriorConfig,
    data_y: np.ndarray,
) -> Callable[[np.ndarray], float]:
    """事後確率関数を構築する

    θベクトルから対数事後確率 log p(θ|y) = log p(y|θ) + log p(θ) を計算する
    関数を返す。

    Args:
        mapping: パラメータマッピング
        prior_config: 事前分布設定
        data_y: 観測データ (T_obs, n_obs)

    Returns:
        log_posterior(theta) → float を返す関数
    """
    # ショック標準偏差のパラメータ名→インデックスマッピング
    shock_std_names = [
        "sigma_g",
        "sigma_a",
        "sigma_m",
        "sigma_i",
        "sigma_w",
        "sigma_p",
    ]
    shock_std_indices = [mapping._name_to_index[name] for name in shock_std_names]

    def log_posterior(theta: np.ndarray) -> float:
        """対数事後確率を計算する"""
        # 事前確率
        lp = prior_config.log_prior(theta)
        if not np.isfinite(lp):
            return -np.inf

        try:
            # θからモデルパラメータに変換
            params = mapping.theta_to_params(theta)

            # モデルを解く
            model = NewKeynesianModel(params)
            solution = model.solution

            # ショック標準偏差を抽出
            shock_stds = np.array([theta[i] for i in shock_std_indices])

            # 測定誤差を抽出
            measurement_errors = mapping.theta_to_measurement_errors(theta)

            # 状態空間行列を構築
            ss = StateSpaceBuilder.build(solution, shock_stds, measurement_errors)

            # Kalmanフィルタで対数尤度を計算
            result = kalman_filter(data_y, ss.T, ss.Z, ss.R_aug, ss.Q_cov, ss.H)

            ll = result.log_likelihood
            if not np.isfinite(ll):
                return -np.inf

            return lp + ll
        except Exception:
            return -np.inf

    return log_posterior
