"""グラフ生成"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from japan_fiscal_simulator.core.simulation import ImpulseResponseResult


# デフォルトのグラフスタイル設定
DEFAULT_STYLE_CONFIG: dict[str, object] = {
    "figure.figsize": (12, 8),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 2,
    "axes.labelsize": 12,
    "axes.titlesize": 14,
    "legend.fontsize": 10,
}

# デフォルトのカラーパレット
DEFAULT_COLORS: dict[str, str] = {
    "output": "#1f77b4",  # 青
    "consumption": "#ff7f0e",  # オレンジ
    "investment": "#2ca02c",  # 緑
    "inflation": "#d62728",  # 赤
    "interest_rate": "#9467bd",  # 紫
    "debt": "#8c564b",  # 茶
    "labor": "#e377c2",  # ピンク
    "wage": "#7f7f7f",  # グレー
}

# デフォルトの変数ラベル
DEFAULT_VARIABLE_LABELS: dict[str, str] = {
    "y": "産出（GDP）",
    "c": "消費",
    "i": "投資",
    "n": "労働",
    "k": "資本",
    "w": "実質賃金",
    "pi": "インフレ率",
    "r": "実質金利",
    "R": "名目金利",
    "g": "政府支出",
    "b": "政府債務",
    "tau_c": "消費税率",
    "mc": "限界費用",
    "q": "Tobin's Q",
    "s": "外部資金プレミアム",
    "nw": "純資産",
}

# 日本語フォント候補
JAPANESE_FONTS = ["Hiragino Sans", "Yu Gothic", "Meiryo", "Takao Gothic"]


@dataclass
class GraphStyle:
    """グラフスタイル設定"""

    style_config: dict[str, object] = field(default_factory=lambda: dict(DEFAULT_STYLE_CONFIG))
    colors: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_COLORS))
    variable_labels: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VARIABLE_LABELS))
    font_family: str | None = None

    def apply(self) -> None:
        """スタイルをmatplotlibに適用"""
        # 日本語フォント設定
        if self.font_family:
            matplotlib.rcParams["font.family"] = self.font_family
        else:
            for font in JAPANESE_FONTS:
                try:
                    matplotlib.rcParams["font.family"] = font
                    break
                except Exception:
                    continue
            else:
                matplotlib.rcParams["font.family"] = "sans-serif"

        # スタイル設定を適用
        plt.rcParams.update(self.style_config)


class GraphGenerator:
    """インパルス応答グラフの生成"""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        style: GraphStyle | None = None,
    ) -> None:
        self.output_dir = Path(output_dir) if output_dir else Path("./output/graphs")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.style = style or GraphStyle()
        # スタイルを適用
        self.style.apply()

    def plot_impulse_response(
        self,
        result: ImpulseResponseResult,
        variables: list[str] | None = None,
        title: str | None = None,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> Figure:
        """インパルス応答をプロット

        Args:
            result: シミュレーション結果
            variables: プロットする変数リスト（Noneの場合はデフォルト）
            title: グラフタイトル
            save_path: 保存先パス
            show: プロットを表示するか
        """
        if variables is None:
            variables = ["y", "c", "i", "pi", "r", "b"]

        n_vars = len(variables)
        n_cols = min(3, n_vars)
        n_rows = (n_vars + n_cols - 1) // n_cols

        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        if n_vars == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = axes.reshape(1, -1)

        time_axis = np.arange(result.periods)

        for idx, var in enumerate(variables):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]

            response = result.get_response(var)
            color = self.style.colors.get(var.replace("_", ""), "#1f77b4")
            label = self.style.variable_labels.get(var, var)

            ax.plot(time_axis, response * 100, color=color, linewidth=2)
            ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
            ax.fill_between(
                time_axis,
                response * 100,
                0,
                alpha=0.2,
                color=color,
            )

            ax.set_xlabel("四半期")
            ax.set_ylabel("定常状態からの乖離（%）")
            ax.set_title(label)
            ax.grid(True, alpha=0.3)

        # 使用していないサブプロットを非表示
        for idx in range(n_vars, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            axes[row, col].set_visible(False)

        if title is None:
            title = f"インパルス応答: {result.shock_name} ショック ({result.shock_size * 100:.1f}%)"
        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")
        elif self.output_dir:
            save_file = self.output_dir / f"irf_{result.shock_name}.png"
            fig.savefig(save_file, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_comparison(
        self,
        results: list[ImpulseResponseResult],
        variable: str,
        labels: list[str] | None = None,
        title: str | None = None,
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> Figure:
        """複数シナリオの比較プロット

        Args:
            results: 複数のシミュレーション結果
            variable: 比較する変数
            labels: 各シナリオのラベル
            title: グラフタイトル
            save_path: 保存先パス
            show: プロットを表示するか
        """
        fig, ax = plt.subplots(figsize=(10, 6))

        if labels is None:
            labels = [f"シナリオ {i + 1}" for i in range(len(results))]

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in np.linspace(0, 1, len(results))]

        for idx, (result, label) in enumerate(zip(results, labels, strict=True)):
            response = result.get_response(variable)
            time_axis = np.arange(result.periods)
            ax.plot(time_axis, response * 100, label=label, color=colors[idx], linewidth=2)

        ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax.set_xlabel("四半期")
        ax.set_ylabel("定常状態からの乖離（%）")

        var_label = self.style.variable_labels.get(variable, variable)
        if title is None:
            title = f"{var_label}の応答比較"
        ax.set_title(title)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def plot_fiscal_multiplier(
        self,
        y_response: np.ndarray,
        g_response: np.ndarray,
        g_y_ratio: float,
        title: str = "財政乗数の時間推移",
        save_path: Path | str | None = None,
        show: bool = False,
    ) -> Figure:
        """財政乗数の時間推移をプロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        periods = len(y_response)
        time_axis = np.arange(periods)

        # 左: 産出と政府支出の応答
        ax1.plot(time_axis, y_response * 100, label="産出（Y）", color=self.style.colors["output"])
        ax1.plot(
            time_axis,
            g_response * 100,
            label="政府支出（G）",
            color=self.style.colors["debt"],
            linestyle="--",
        )
        ax1.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        ax1.set_xlabel("四半期")
        ax1.set_ylabel("定常状態からの乖離（%）")
        ax1.set_title("産出と政府支出の応答")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 右: 累積乗数
        y_cumulative = np.cumsum(y_response)
        g_cumulative = np.cumsum(g_response)

        with np.errstate(divide="ignore", invalid="ignore"):
            cumulative_multiplier = np.where(
                np.abs(g_cumulative) > 1e-10,
                y_cumulative / g_cumulative / g_y_ratio,
                0,
            )

        ax2.plot(time_axis, cumulative_multiplier, color=self.style.colors["output"], linewidth=2)
        ax2.axhline(y=1, color="gray", linestyle="--", linewidth=1, label="乗数 = 1")
        ax2.set_xlabel("四半期")
        ax2.set_ylabel("累積財政乗数")
        ax2.set_title("累積財政乗数の推移")
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        fig.suptitle(title, fontsize=14, fontweight="bold")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=150, bbox_inches="tight")

        if show:
            plt.show()

        return fig

    def close_all(self) -> None:
        """全てのフィギュアを閉じる"""
        plt.close("all")
