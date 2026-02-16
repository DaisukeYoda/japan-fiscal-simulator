"""NKモデル解からKalmanフィルタ用状態空間形式への変換

NKSolutionResult (P, Q, R, S) を拡張状態空間形式に変換する。
6ショック vs 7観測変数の確率的特異性を測定誤差で解消する（SW2007方式）。

拡張状態ベクトル α_t (dim=14):
    [s_t(5), y_t, c_t, π_t, n_t, r_t, y_{t-1}, c_{t-1}, i_{t-1}, w_{t-1}]

制御変数(y, c, π, n, r)を補助状態に含めることで、
観測方程式が同時ショック効果(S行列)を正確に反映する。

状態方程式: α_t = T @ α_{t-1} + R_aug @ ε_t
観測方程式: z_t = Z @ α_t + η_t,  η_t ~ N(0, H)
"""

from dataclasses import dataclass

import numpy as np

from japan_fiscal_simulator.core.nk_model import NKSolutionResult


@dataclass
class StateSpaceMatrices:
    """Kalmanフィルタ用状態空間行列

    Attributes:
        T: 状態遷移行列 (n_aug, n_aug)
        R_aug: ショック負荷行列 (n_aug, n_shock)
        Q_cov: ショック共分散行列 (n_shock, n_shock)
        Z: 観測行列 (n_obs, n_aug)
        H: 測定誤差共分散行列 (n_obs, n_obs)
        d: 観測方程式の定数項 (n_obs,)。定常状態の平均値。
        n_aug: 拡張状態次元
        n_obs: 観測変数数
        n_shock: ショック数
    """

    T: np.ndarray
    R_aug: np.ndarray
    Q_cov: np.ndarray
    Z: np.ndarray
    H: np.ndarray
    d: np.ndarray
    n_aug: int
    n_obs: int
    n_shock: int


class StateSpaceBuilder:
    """NKモデル解から状態空間形式への変換

    NKSolutionResult の (P, Q, R, S) を Kalman フィルタで使用可能な
    拡張状態空間形式に変換する。

    モデル変数の配置:
        状態変数 s_t: g(0), a(1), k(2), i(3), w(4)
        制御変数 c_t: y(0), pi(1), r(2), q(3), rk(4), n(5), c(6), mc(7), mrs(8)
        ショック: e_g(0), e_a(1), e_m(2), e_i(3), e_w(4), e_p(5)

    観測変数 (7):
        output_growth(Δy), consumption_growth(Δc), investment_growth(Δi),
        inflation(π), wage_growth(Δw), hours(n), interest_rate(r)

    拡張状態ベクトル α_t (dim=14):
        [s_t(5), y_t, c_t, π_t, n_t, r_t, y_{t-1}, c_{t-1}, i_{t-1}, w_{t-1}]

    制御変数を補助状態に含めることで、同時ショック効果 S@ε_t を
    R_aug行列に反映し、観測方程式を正確にする。
    """

    # 制御変数のR/S行列におけるインデックス
    Y_IDX: int = 0  # output
    PI_IDX: int = 1  # inflation
    R_IDX: int = 2  # interest rate
    N_IDX: int = 5  # labor hours
    C_IDX: int = 6  # consumption

    # 状態変数のインデックス
    I_STATE_IDX: int = 3  # investment
    W_STATE_IDX: int = 4  # wage

    N_STATE: int = 5
    N_AUX: int = 5  # 補助変数: y_t, c_t, π_t, n_t, r_t
    N_LAGS: int = 4  # ラグ変数: y_{t-1}, c_{t-1}, i_{t-1}, w_{t-1}
    N_AUG: int = 14  # N_STATE + N_AUX + N_LAGS
    N_OBS: int = 7
    N_SHOCK: int = 6

    # 拡張状態ベクトルにおけるインデックス
    _Y_CUR: int = 5  # y_t
    _C_CUR: int = 6  # c_t (consumption)
    _PI_CUR: int = 7  # π_t
    _N_CUR: int = 8  # n_t
    _R_CUR: int = 9  # r_t
    _Y_LAG: int = 10  # y_{t-1}
    _C_LAG: int = 11  # c_{t-1}
    _I_LAG: int = 12  # i_{t-1}
    _W_LAG: int = 13  # w_{t-1}

    @classmethod
    def build(
        cls,
        solution: NKSolutionResult,
        shock_stds: np.ndarray,
        measurement_errors: np.ndarray,
        steady_state_means: np.ndarray | None = None,
    ) -> StateSpaceMatrices:
        """NKモデル解から状態空間行列を構築する

        Args:
            solution: NKSolutionResult (P, Q, R, S)
            shock_stds: ショック標準偏差 (6,) [σ_g, σ_a, σ_m, σ_i, σ_w, σ_p]
            measurement_errors: 測定誤差標準偏差 (7,)
                [me_y, me_c, me_i, me_pi, me_w, me_n, me_r]
            steady_state_means: 観測方程式の定常状態定数ベクトル (7,)
                [γ, γ, γ, π*, γ, n*, r*]。Noneの場合はゼロベクトル（従来互換）。

        Returns:
            StateSpaceMatrices

        Raises:
            ValueError: 入力次元が不正な場合
        """
        cls._validate_inputs(solution, shock_stds, measurement_errors)

        T = cls._build_T(solution)
        R_aug = cls._build_R_aug(solution)
        Q_cov = cls._build_Q_cov(shock_stds)
        Z = cls._build_Z()
        H = cls._build_H(measurement_errors)
        d = cls._build_d(steady_state_means)

        return StateSpaceMatrices(
            T=T,
            R_aug=R_aug,
            Q_cov=Q_cov,
            Z=Z,
            H=H,
            d=d,
            n_aug=cls.N_AUG,
            n_obs=cls.N_OBS,
            n_shock=cls.N_SHOCK,
        )

    @classmethod
    def _validate_inputs(
        cls,
        solution: NKSolutionResult,
        shock_stds: np.ndarray,
        measurement_errors: np.ndarray,
    ) -> None:
        """入力の次元チェック"""
        if solution.P.shape != (cls.N_STATE, cls.N_STATE):
            raise ValueError(
                f"P行列のサイズが不正: {solution.P.shape} != ({cls.N_STATE}, {cls.N_STATE})"
            )
        if solution.Q.shape != (cls.N_STATE, cls.N_SHOCK):
            raise ValueError(
                f"Q行列のサイズが不正: {solution.Q.shape} != ({cls.N_STATE}, {cls.N_SHOCK})"
            )
        if solution.R.shape[1] != cls.N_STATE:
            raise ValueError(f"R行列の列数が不正: {solution.R.shape[1]} != {cls.N_STATE}")
        if solution.S.shape[1] != cls.N_SHOCK:
            raise ValueError(f"S行列の列数が不正: {solution.S.shape[1]} != {cls.N_SHOCK}")
        if shock_stds.shape != (cls.N_SHOCK,):
            raise ValueError(f"shock_stdsのサイズが不正: {shock_stds.shape} != ({cls.N_SHOCK},)")
        if measurement_errors.shape != (cls.N_OBS,):
            raise ValueError(
                f"measurement_errorsのサイズが不正: {measurement_errors.shape} != ({cls.N_OBS},)"
            )

    @classmethod
    def _build_T(cls, solution: NKSolutionResult) -> np.ndarray:
        """状態遷移行列 T (14x14) を構築する

        α_t = T @ α_{t-1} + R_aug @ ε_t

        構造:
            Row 0-4  (s_t):      [P, ...]
            Row 5    (y_t):      [R_y @ P, ...]
            Row 6    (c_t):      [R_c @ P, ...]
            Row 7    (π_t):      [R_π @ P, ...]
            Row 8    (n_t):      [R_n @ P, ...]
            Row 9    (r_t):      [R_r @ P, ...]
            Row 10   (y_{t-1}):  [0..., 1(col5), ...]
            Row 11   (c_{t-1}):  [0..., 0, 1(col6), ...]
            Row 12   (i_{t-1}):  [0, 0, 0, 1(col3), ...]
            Row 13   (w_{t-1}):  [0, 0, 0, 0, 1(col4), ...]
        """
        P = solution.P
        R = solution.R

        T = np.zeros((cls.N_AUG, cls.N_AUG))

        # s_t = P @ s_{t-1}
        T[: cls.N_STATE, : cls.N_STATE] = P

        # 補助制御変数: x_t = R[idx,:] @ P @ s_{t-1}
        # (ショック同時効果は R_aug で処理)
        aux_control_indices = [cls.Y_IDX, cls.C_IDX, cls.PI_IDX, cls.N_IDX, cls.R_IDX]
        aux_state_rows = [cls._Y_CUR, cls._C_CUR, cls._PI_CUR, cls._N_CUR, cls._R_CUR]
        for state_row, ctrl_idx in zip(aux_state_rows, aux_control_indices, strict=True):
            T[state_row, : cls.N_STATE] = R[ctrl_idx, :] @ P

        # ラグ変数は前期の対応する値をコピー
        T[cls._Y_LAG, cls._Y_CUR] = 1.0  # y_{t-1} ← y_t
        T[cls._C_LAG, cls._C_CUR] = 1.0  # c_{t-1} ← c_t
        T[cls._I_LAG, cls.I_STATE_IDX] = 1.0  # i_{t-1} ← s_t[3]
        T[cls._W_LAG, cls.W_STATE_IDX] = 1.0  # w_{t-1} ← s_t[4]

        return T

    @classmethod
    def _build_R_aug(cls, solution: NKSolutionResult) -> np.ndarray:
        """ショック負荷行列 R_aug (14x6) を構築する

        構造:
            Row 0-4:  Q
            Row 5:    R[0,:] @ Q + S[0,:]   (y_t のショック応答)
            Row 6:    R[6,:] @ Q + S[6,:]   (c_t のショック応答)
            Row 7:    R[1,:] @ Q + S[1,:]   (π_t のショック応答)
            Row 8:    R[5,:] @ Q + S[5,:]   (n_t のショック応答)
            Row 9:    R[2,:] @ Q + S[2,:]   (r_t のショック応答)
            Row 10-13: 0                     (ラグ変数)
        """
        Q = solution.Q
        R = solution.R
        S = solution.S

        R_aug = np.zeros((cls.N_AUG, cls.N_SHOCK))

        # s_t のショック応答
        R_aug[: cls.N_STATE, :] = Q

        # 補助制御変数のショック応答: R[idx,:] @ Q + S[idx,:]
        aux_control_indices = [cls.Y_IDX, cls.C_IDX, cls.PI_IDX, cls.N_IDX, cls.R_IDX]
        aux_state_rows = [cls._Y_CUR, cls._C_CUR, cls._PI_CUR, cls._N_CUR, cls._R_CUR]
        for state_row, ctrl_idx in zip(aux_state_rows, aux_control_indices, strict=True):
            R_aug[state_row, :] = R[ctrl_idx, :] @ Q + S[ctrl_idx, :]

        # Row 10-13 はゼロ（ラグ変数）

        return R_aug

    @classmethod
    def _build_Q_cov(cls, shock_stds: np.ndarray) -> np.ndarray:
        """ショック共分散行列 Q_cov (6x6) を構築する

        Q_cov = diag(σ_g², σ_a², σ_m², σ_i², σ_w², σ_p²)
        """
        return np.diag(shock_stds**2)

    @classmethod
    def _build_Z(cls) -> np.ndarray:
        """観測行列 Z (7x14) を構築する

        全ての観測変数を補助状態変数から直接読み出す。
        同時ショック効果は R_aug 経由で補助状態に既に反映済み。

        観測変数と拡張状態の対応:
            0. Δy_t = y_t - y_{t-1}     = α_t[5] - α_t[10]
            1. Δc_t = c_t - c_{t-1}     = α_t[6] - α_t[11]
            2. Δi_t = i_t - i_{t-1}     = α_t[3] - α_t[12]
            3. π_t                       = α_t[7]
            4. Δw_t = w_t - w_{t-1}     = α_t[4] - α_t[13]
            5. n_t                       = α_t[8]
            6. r_t                       = α_t[9]
        """
        Z = np.zeros((cls.N_OBS, cls.N_AUG))

        # Δy_t = y_t - y_{t-1}
        Z[0, cls._Y_CUR] = 1.0
        Z[0, cls._Y_LAG] = -1.0

        # Δc_t = c_t - c_{t-1}
        Z[1, cls._C_CUR] = 1.0
        Z[1, cls._C_LAG] = -1.0

        # Δi_t = i_t - i_{t-1} (both from state/lag)
        Z[2, cls.I_STATE_IDX] = 1.0
        Z[2, cls._I_LAG] = -1.0

        # π_t (from auxiliary state)
        Z[3, cls._PI_CUR] = 1.0

        # Δw_t = w_t - w_{t-1} (both from state/lag)
        Z[4, cls.W_STATE_IDX] = 1.0
        Z[4, cls._W_LAG] = -1.0

        # n_t (from auxiliary state)
        Z[5, cls._N_CUR] = 1.0

        # r_t (from auxiliary state)
        Z[6, cls._R_CUR] = 1.0

        return Z

    @classmethod
    def _build_H(cls, measurement_errors: np.ndarray) -> np.ndarray:
        """測定誤差共分散行列 H (7x7) を構築する

        H = diag(me_y², me_c², me_i², me_pi², me_w², me_n², me_r²)
        """
        return np.diag(measurement_errors**2)

    @classmethod
    def _build_d(cls, steady_state_means: np.ndarray | None) -> np.ndarray:
        """観測方程式の定常状態定数ベクトル d (7,) を構築する

        観測方程式: z_t = d + Z @ α_t + η_t

        d = [γ, γ, γ, π*, γ, n*, r*] の形式。
        Noneの場合はゼロベクトル。
        """
        if steady_state_means is None:
            return np.zeros(cls.N_OBS)
        d = np.asarray(steady_state_means, dtype=np.float64)
        if d.shape != (cls.N_OBS,):
            raise ValueError(
                f"steady_state_meansのサイズが不正: {d.shape} != ({cls.N_OBS},)"
            )
        return d.copy()
