"""レポート生成"""

from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import BaseLoader, Environment

if TYPE_CHECKING:
    from japan_fiscal_simulator.output.schemas import ComparisonResult, SimulationResult


# デフォルトのレポートテンプレート（Markdown）
DEFAULT_REPORT_TEMPLATE = """# {{ title }}

生成日時: {{ generated_at }}

## エグゼクティブサマリー

{{ executive_summary }}

## シミュレーション設定

### パラメータ

| パラメータ | 値 | 説明 |
|-----------|-----|------|
| 割引率 (β) | {{ params.beta }} | 家計の時間選好 |
| 消費税率 (τ_c) | {{ "%.1f"|format(params.tau_c * 100) }}% | 現行消費税率 |
| 政府支出/GDP | {{ "%.1f"|format(params.g_y_ratio * 100) }}% | 政府支出の対GDP比 |
| 政府債務/GDP | {{ "%.0f"|format(params.b_y_ratio * 100) }}% | 政府債務の対GDP比 |
| 金利平滑化 (ρ_R) | {{ params.rho_r }} | 中央銀行の金利平滑化 |

### シナリオ

**{{ scenario.name }}**

- 政策タイプ: {{ scenario.policy_type }}
- ショックサイズ: {{ "%.1f"|format(scenario.shock_size * 100) }}%
- シミュレーション期間: {{ scenario.periods }}四半期

## 主要結果

### 定常状態

| 変数 | 値 |
|------|-----|
| 産出 (Y) | {{ "%.4f"|format(steady_state.output) }} |
| 消費 (C) | {{ "%.4f"|format(steady_state.consumption) }} |
| 投資 (I) | {{ "%.4f"|format(steady_state.investment) }} |
| 実質金利 (r) | {{ "%.2f"|format(steady_state.real_interest_rate * 100) }}% |
| インフレ率 (π) | {{ "%.2f"|format(steady_state.inflation * 100) }}% |

### インパルス応答

{% for var_name, var_data in impulse_response.items() %}
#### {{ var_data.description }}

- ピーク効果: {{ "%.3f"|format(var_data.peak * 100) }}%（{{ var_data.peak_period }}四半期目）
- 累積効果（1年）: {{ "%.3f"|format(var_data.cumulative_4q * 100) }}%
{% endfor %}

{% if fiscal_multiplier %}
### 財政乗数

| 乗数タイプ | 値 |
|-----------|-----|
| インパクト乗数 | {{ "%.2f"|format(fiscal_multiplier.impact_multiplier) }} |
| ピーク乗数 | {{ "%.2f"|format(fiscal_multiplier.peak_multiplier) }}（{{ fiscal_multiplier.peak_period }}四半期目） |
| 累積乗数（1年） | {{ "%.2f"|format(fiscal_multiplier.cumulative_multiplier_4q) }} |
| 累積乗数（2年） | {{ "%.2f"|format(fiscal_multiplier.cumulative_multiplier_8q) }} |
| 現在価値乗数 | {{ "%.2f"|format(fiscal_multiplier.present_value_multiplier) }} |
{% endif %}

### モデル診断

- Blanchard-Kahn条件: {{ "充足" if bk_satisfied else "未充足" }}
- 安定固有値: {{ n_stable }}個
- 不安定固有値: {{ n_unstable }}個

{% if policy_recommendations %}
## 政策提言

{% for rec in policy_recommendations %}
- {{ rec }}
{% endfor %}
{% endif %}

{% if caveats %}
## 注意事項・限界

{% for caveat in caveats %}
- {{ caveat }}
{% endfor %}
{% endif %}

---
*このレポートは日本財政政策DSGEシミュレーターによって自動生成されました。*
"""


class ReportGenerator:
    """分析レポートの生成"""

    def __init__(
        self,
        output_dir: Path | str | None = None,
        template: str | None = None,
    ) -> None:
        """
        Args:
            output_dir: 出力ディレクトリ
            template: カスタムテンプレート文字列（Noneの場合はデフォルトを使用）
        """
        self.output_dir = Path(output_dir) if output_dir else Path("./output/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.env = Environment(loader=BaseLoader())
        self.template = self.env.from_string(template or DEFAULT_REPORT_TEMPLATE)

    def generate_simulation_report(
        self,
        result: SimulationResult,
        title: str | None = None,
        policy_recommendations: list[str] | None = None,
        caveats: list[str] | None = None,
        save_path: Path | str | None = None,
    ) -> str:
        """シミュレーション結果のレポートを生成

        Args:
            result: シミュレーション結果
            title: レポートタイトル
            policy_recommendations: 政策提言リスト
            caveats: 注意事項リスト
            save_path: 保存先パス

        Returns:
            生成されたMarkdownレポート
        """
        if title is None:
            title = f"財政政策シミュレーションレポート: {result.scenario.name}"

        if caveats is None:
            caveats = self._generate_default_caveats()

        # インパルス応答のサマリー
        irf_summary = {}
        for var_name, var_data in result.impulse_response.variables.items():
            values = var_data.values
            if len(values) > 0:
                peak_idx = int(abs(max(values, key=abs)) == max(values, key=abs))
                peak_idx = values.index(max(values, key=abs)) if values else 0
                irf_summary[var_name] = {
                    "description": var_data.description,
                    "peak": values[peak_idx] if values else 0,
                    "peak_period": peak_idx,
                    "cumulative_4q": sum(values[:4]) if len(values) >= 4 else sum(values),
                }

        # エグゼクティブサマリーの生成
        executive_summary = self._generate_executive_summary(result)

        # テンプレートに渡すデータ
        context = {
            "title": title,
            "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "executive_summary": executive_summary,
            "params": result.parameters,
            "scenario": result.scenario,
            "steady_state": result.steady_state,
            "impulse_response": irf_summary,
            "fiscal_multiplier": result.fiscal_multiplier,
            "bk_satisfied": result.blanchard_kahn_satisfied,
            "n_stable": result.num_stable_eigenvalues,
            "n_unstable": result.num_unstable_eigenvalues,
            "policy_recommendations": policy_recommendations or [],
            "caveats": caveats,
        }

        report = self.template.render(**context)

        if save_path:
            Path(save_path).write_text(report, encoding="utf-8")
        elif self.output_dir:
            filename = f"report_{result.scenario.name}_{datetime.now():%Y%m%d_%H%M%S}.md"
            (self.output_dir / filename).write_text(report, encoding="utf-8")

        return report

    def _generate_executive_summary(self, result: SimulationResult) -> str:
        """エグゼクティブサマリーを生成"""
        scenario = result.scenario
        summary_parts = []

        summary_parts.append(
            f"本レポートは「{scenario.name}」政策のマクロ経済効果を分析しています。"
        )

        # 主要な効果を記述
        if result.fiscal_multiplier:
            mult = result.fiscal_multiplier
            if mult.impact_multiplier > 1:
                summary_parts.append(
                    f"インパクト乗数は{mult.impact_multiplier:.2f}であり、"
                    "短期的に政策効果が増幅されることを示しています。"
                )
            elif mult.impact_multiplier > 0:
                summary_parts.append(
                    f"インパクト乗数は{mult.impact_multiplier:.2f}であり、"
                    "政策は正の効果を持ちますが、完全なクラウディングアウトには至りません。"
                )
            else:
                summary_parts.append(
                    f"インパクト乗数は{mult.impact_multiplier:.2f}であり、"
                    "クラウディングアウト効果が支配的です。"
                )

        if not result.blanchard_kahn_satisfied:
            summary_parts.append(
                "**注意**: Blanchard-Kahn条件が満たされておらず、"
                "モデルの解の一意性・安定性が保証されません。"
            )

        return " ".join(summary_parts)

    def _generate_default_caveats(self) -> list[str]:
        """デフォルトの注意事項"""
        return [
            "本モデルは代表的家計・企業を仮定しており、異質性の効果は捕捉できません。",
            "開放経済の側面（為替レート、国際資本移動）は考慮されていません。",
            "パラメータの不確実性により、結果には相当の幅があり得ます。",
            "財政の持続可能性（債務動学）は長期シミュレーションで確認が必要です。",
            "モデルの線形近似は大きなショックに対して精度が低下する可能性があります。",
        ]

    def generate_comparison_report(
        self,
        comparison: ComparisonResult,
        title: str | None = None,
        save_path: Path | str | None = None,
    ) -> str:
        """シナリオ比較レポートを生成"""
        if title is None:
            title = "財政政策シナリオ比較レポート"

        lines = [
            f"# {title}",
            "",
            f"生成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## シナリオ比較サマリー",
            "",
            comparison.summary,
            "",
            "## 詳細比較",
            "",
            "| シナリオ | 産出効果 | 消費効果 | 投資効果 | インフレ効果 | 債務効果 | 財政乗数 |",
            "|---------|---------|---------|---------|-------------|---------|---------|",
        ]

        for comp in comparison.comparisons:
            lines.append(
                f"| {comp.scenario_name} "
                f"| {comp.impact_on_output:+.2f}% "
                f"| {comp.impact_on_consumption:+.2f}% "
                f"| {comp.impact_on_investment:+.2f}% "
                f"| {comp.impact_on_inflation:+.3f}%pt "
                f"| {comp.impact_on_debt:+.2f}% "
                f"| {comp.fiscal_multiplier:.2f} |"
            )

        report = "\n".join(lines)

        if save_path:
            Path(save_path).write_text(report, encoding="utf-8")

        return report
