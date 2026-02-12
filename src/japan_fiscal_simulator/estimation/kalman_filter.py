"""標準Kalmanフィルタ

線形ガウス状態空間モデルに対するKalmanフィルタを実装する。

状態空間モデル:
    α_t = T @ α_{t-1} + R @ ε_t,    ε_t ~ N(0, Q)
    z_t = Z @ α_t + η_t,             η_t ~ N(0, H)
"""

from dataclasses import dataclass

import numpy as np
import scipy.linalg

from japan_fiscal_simulator.core.exceptions import KalmanFilterError


@dataclass
class KalmanFilterResult:
    """Kalmanフィルタの結果

    Attributes:
        log_likelihood: 対数尤度
        filtered_states: フィルタ済み状態推定値 (T_obs, n_state)
        prediction_errors: 予測誤差 (T_obs, n_obs)
        filtered_covariances: フィルタ済み共分散の対角要素 (T_obs, n_state)
    """

    log_likelihood: float
    filtered_states: np.ndarray
    prediction_errors: np.ndarray
    filtered_covariances: np.ndarray


def kalman_filter(
    y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    a0: np.ndarray | None = None,
    P0: np.ndarray | None = None,
) -> KalmanFilterResult:
    """標準Kalmanフィルタ

    Args:
        y: (T_obs, n_obs) 観測データ。欠損値はNaN。
        T: (n_state, n_state) 状態遷移行列
        Z: (n_obs, n_state) 観測行列
        R: (n_state, n_shock) ショック負荷行列
        Q: (n_shock, n_shock) ショック共分散行列
        H: (n_obs, n_obs) 測定誤差共分散行列
        a0: (n_state,) 初期状態。Noneの場合はゼロ。
        P0: (n_state, n_state) 初期共分散。Noneの場合はLyapunov方程式から計算。

    Returns:
        KalmanFilterResult

    Raises:
        KalmanFilterError: フィルタ計算中にNaN/infが発生した場合
    """
    _validate_dimensions(y, T, Z, R, Q, H, a0, P0)

    T_obs, n_obs = y.shape
    n_state = T.shape[0]

    # 初期状態
    a_filt = np.zeros(n_state) if a0 is None else a0.copy()
    P_filt = _initialize_covariance(T, R, Q, n_state) if P0 is None else P0.copy()

    # 結果格納用配列
    filtered_states = np.empty((T_obs, n_state))
    prediction_errors = np.full((T_obs, n_obs), np.nan)
    filtered_covariances = np.empty((T_obs, n_state))
    total_ll = 0.0

    RQR = R @ Q @ R.T

    for t in range(T_obs):
        # --- Predict ---
        a_pred = T @ a_filt
        P_pred = T @ P_filt @ T.T + RQR

        # 対称性の強制
        P_pred = 0.5 * (P_pred + P_pred.T)

        # --- 欠損値処理 ---
        obs_t = y[t]
        valid = ~np.isnan(obs_t)
        n_valid = int(np.sum(valid))

        if n_valid == 0:
            # 全欠損: 予測のみ（尤度貢献なし）
            a_filt = a_pred
            P_filt = P_pred
        else:
            # 有効な観測のみ抽出
            if n_valid < n_obs:
                Z_t = Z[valid, :]
                H_t = H[np.ix_(valid, valid)]
                obs_valid = obs_t[valid]
            else:
                Z_t = Z
                H_t = H
                obs_valid = obs_t

            # --- Innovation ---
            v = obs_valid - Z_t @ a_pred
            F = Z_t @ P_pred @ Z_t.T + H_t

            # 対称性の強制
            F = 0.5 * (F + F.T)

            # --- Update (Cholesky) ---
            ll_t = _update_step(a_pred, P_pred, v, F, Z_t, n_valid)

            if ll_t is None:
                # Cholesky失敗 → -inf尤度を返す
                return KalmanFilterResult(
                    log_likelihood=-np.inf,
                    filtered_states=filtered_states,
                    prediction_errors=prediction_errors,
                    filtered_covariances=filtered_covariances,
                )

            a_filt, P_filt, ll_contrib = ll_t
            total_ll += ll_contrib

            # 予測誤差を格納（有効な観測のみ）
            if n_valid < n_obs:
                prediction_errors[t, valid] = v
            else:
                prediction_errors[t] = v

        # 対称性の強制
        P_filt = 0.5 * (P_filt + P_filt.T)

        # NaN/inf検査
        if not np.all(np.isfinite(a_filt)) or not np.all(np.isfinite(P_filt)):
            raise KalmanFilterError(f"フィルタ済み状態にNaN/infが発生 (t={t})")

        filtered_states[t] = a_filt
        filtered_covariances[t] = np.diag(P_filt)

    return KalmanFilterResult(
        log_likelihood=total_ll,
        filtered_states=filtered_states,
        prediction_errors=prediction_errors,
        filtered_covariances=filtered_covariances,
    )


def _initialize_covariance(T: np.ndarray, R: np.ndarray, Q: np.ndarray, n_state: int) -> np.ndarray:
    """初期共分散行列を計算する

    Lyapunov方程式 P0 = T @ P0 @ T' + R @ Q @ R' を解く。
    失敗した場合は大きな対角行列をフォールバックとして返す。
    """
    RQR = R @ Q @ R.T
    try:
        P0 = scipy.linalg.solve_discrete_lyapunov(T, RQR)
        if np.all(np.isfinite(P0)):
            return 0.5 * (P0 + P0.T)
    except (np.linalg.LinAlgError, ValueError):
        pass
    return 10.0 * np.eye(n_state)


def _update_step(
    a_pred: np.ndarray,
    P_pred: np.ndarray,
    v: np.ndarray,
    F: np.ndarray,
    Z_t: np.ndarray,
    n_valid: int,
) -> tuple[np.ndarray, np.ndarray, float] | None:
    """Kalmanフィルタの更新ステップ

    Cholesky分解を用いて数値的に安定なKalmanゲインと尤度貢献を計算する。
    Cholesky分解が失敗した場合は小さな正則化を加えて再試行する。

    Returns:
        (a_filt, P_filt, ll_contrib) のタプル。失敗した場合はNone。
    """
    try:
        CF = scipy.linalg.cho_factor(F)
    except np.linalg.LinAlgError:
        # 正則化して再試行
        F_reg = F + 1e-8 * np.eye(n_valid)
        try:
            CF = scipy.linalg.cho_factor(F_reg)
        except np.linalg.LinAlgError:
            return None

    # Kalmanゲイン: K = P_pred @ Z' @ F^{-1}
    K = P_pred @ Z_t.T @ scipy.linalg.cho_solve(CF, np.eye(n_valid))

    # 状態更新
    a_filt = a_pred + K @ v

    # 共分散更新
    P_filt = P_pred - K @ Z_t @ P_pred

    # 対数尤度貢献
    log_det_F = 2.0 * np.sum(np.log(np.diag(CF[0])))
    Finv_v = scipy.linalg.cho_solve(CF, v)
    ll_contrib = -0.5 * (n_valid * np.log(2.0 * np.pi) + log_det_F + float(v @ Finv_v))

    return a_filt, P_filt, ll_contrib


def _validate_dimensions(
    y: np.ndarray,
    T: np.ndarray,
    Z: np.ndarray,
    R: np.ndarray,
    Q: np.ndarray,
    H: np.ndarray,
    a0: np.ndarray | None,
    P0: np.ndarray | None,
) -> None:
    """入力行列の次元整合性を検証する"""
    if y.ndim != 2:
        raise KalmanFilterError(f"yは2次元配列が必要 (got {y.ndim}D)")

    n_obs = y.shape[1]
    n_state = T.shape[0]
    n_shock = R.shape[1]

    if T.shape != (n_state, n_state):
        raise KalmanFilterError(f"Tは({n_state}, {n_state})が必要 (got {T.shape})")
    if Z.shape != (n_obs, n_state):
        raise KalmanFilterError(f"Zは({n_obs}, {n_state})が必要 (got {Z.shape})")
    if R.shape[0] != n_state:
        raise KalmanFilterError(f"Rの行数は{n_state}が必要 (got {R.shape[0]})")
    if Q.shape != (n_shock, n_shock):
        raise KalmanFilterError(f"Qは({n_shock}, {n_shock})が必要 (got {Q.shape})")
    if H.shape != (n_obs, n_obs):
        raise KalmanFilterError(f"Hは({n_obs}, {n_obs})が必要 (got {H.shape})")
    if a0 is not None and a0.shape != (n_state,):
        raise KalmanFilterError(f"a0は({n_state},)が必要 (got {a0.shape})")
    if P0 is not None and P0.shape != (n_state, n_state):
        raise KalmanFilterError(f"P0は({n_state}, {n_state})が必要 (got {P0.shape})")
