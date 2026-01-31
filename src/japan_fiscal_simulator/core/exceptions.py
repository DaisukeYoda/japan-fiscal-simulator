"""JPFSカスタム例外階層

FailFast原則に従い、エラーは即座に報告される。
"""


class JPFSError(Exception):
    """JPFSの基底例外クラス"""

    pass


class SolverError(JPFSError):
    """ソルバー関連のエラー"""

    pass


class BlanchardKahnError(SolverError):
    """Blanchard-Kahn条件違反エラー

    不安定固有値の数がジャンプ変数の数と一致しない場合に発生。
    """

    pass


class SingularMatrixError(SolverError):
    """特異行列エラー

    行列が特異で逆行列を計算できない場合に発生。
    """

    pass


class ConvergenceError(SolverError):
    """収束エラー

    反復ソルバーが収束しなかった場合に発生。
    """

    pass


class ValidationError(JPFSError):
    """入力バリデーションエラー"""

    pass


class ParameterValidationError(ValidationError):
    """パラメータ値が有効範囲外のエラー"""

    pass


class ShockValidationError(ValidationError):
    """ショック指定が無効なエラー"""

    pass
