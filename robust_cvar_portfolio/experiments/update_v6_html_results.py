"""Patch V6equity_only_next_steps_plan.html §13.2 with Random30 results."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
HTML_PATH = PROJECT / "V6equity_only_next_steps_plan.html"
R30_SUM = PROJECT / "robust_cvar_portfolio" / "outputs" / "equity_only" / "random30" / "random30_summary.csv"
R30_PARAMS = PROJECT / "robust_cvar_portfolio" / "outputs" / "equity_only" / "random30" / "random30_selected_params.csv"
BOOT_PATH = PROJECT / "robust_cvar_portfolio" / "outputs" / "equity_only" / "bootstrap" / "random30_bootstrap_summary.csv"


def _interpret(win_rate: float) -> str:
    if win_rate > 0.70:
        return "很强：随机中等规模个股池具有稳健改善"
    if win_rate >= 0.55:
        return "中等支持：有一定稳健性，效果依赖股票池结构"
    return "支持不足：SP30 可能是较有利样本，主文应降调"


def build_random30_html() -> str:
    if not R30_SUM.exists():
        return "<p><i>Random30 结果文件尚未生成。</i></p>"

    s = pd.read_csv(R30_SUM).iloc[0]
    win_a = float(s["win_rate_A"])
    win_b = float(s["win_rate_B"])
    mean_da = float(s["mean_delta_A"]) * 100
    med_da = float(s["median_delta_A"]) * 100
    worst_q = float(s.get("worst_quartile_delta_A", float("nan"))) * 100
    n = int(s["n_universes"])
    interp = _interpret(win_a)

    boot_line = ""
    if BOOT_PATH.exists():
        b = pd.read_csv(BOOT_PATH).iloc[0]
        boot_line = f"<tr><td>Random30 pooled vs A</td><td>{float(b['win_rate_A']):.1%}</td><td>跨 universe 胜率</td></tr>"

    return f"""
<h3>13.2 Random30 稳健性（Task 3，已完成）</h3>

<div class="good">
<b>状态：已完成。</b> {n} 个随机 30 股 universe（seed=42，从 SP100 池抽样）；每 universe 独立 validation 选参。
<br>输出：<code>outputs/equity_only/random30/random30_summary.csv</code>
</div>

<table>
<tr><th>汇总指标</th><th>数值</th><th>论文解读</th></tr>
<tr><td>WinRate vs A</td><td><b>{win_a:.1%}</b></td><td>{interp}</td></tr>
<tr><td>WinRate vs B</td><td>{win_b:.1%}</td><td>C_stable 相对固定 κ baseline</td></tr>
<tr><td>mean(Δ_A) [pp]</td><td>{mean_da:.2f}</td><td>平均 CVaR 改善（A − C_stable）</td></tr>
<tr><td>median(Δ_A) [pp]</td><td>{med_da:.2f}</td><td>中位数改善</td></tr>
<tr><td>worst quartile Δ_A [pp]</td><td>{worst_q:.2f}</td><td>最差 25% universe 的改善幅度</td></tr>
{boot_line}
</table>

<p>图表：<code>outputs/equity_only/random30/figures/fig_random30_cvar_improvement_hist.png</code>、
<code>fig_random30_winrate.png</code></p>
<p>论文 Table 2：<code>outputs/equity_only/paper_tables/table2_random30_robustness.csv</code></p>
"""


def update_v6_html() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    block = build_random30_html()
    pattern = r"<h3>13\.2 Random30 稳健性.*?(?=<h3>13\.3)"
    replacement = block.strip() + "\n\n"
    new_html, n = re.subn(pattern, replacement, html, count=1, flags=re.DOTALL)
    if n == 0:
        raise RuntimeError("Could not find §13.2 section to patch in V6 HTML")
    HTML_PATH.write_text(new_html, encoding="utf-8")
    print(f"Updated {HTML_PATH} §13.2 Random30", flush=True)


if __name__ == "__main__":
    update_v6_html()
