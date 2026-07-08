"""Patch V7_structure_corr_stratified_plan.html with experiment results."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
HTML_PATH = PROJECT / "V7_structure_corr_stratified_plan.html"
V7_OUT = PROJECT / "robust_cvar_portfolio" / "outputs" / "v7"


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%" if pd.notna(x) else "—"


def _gap_pp(gap: float) -> str:
    if pd.isna(gap):
        return "—"
    return f"{gap * 100:.2f} pp"


def build_paper_section_html() -> str:
    return """
<h2>9. 论文写法建议（实验完成后定稿）</h2>

<h3>9.1 结构适用条件（V7-A + V7-B）</h3>

<div class="box">
The equity-only evidence shows that the effectiveness of state-dependent RCVaR is not governed by average correlation alone. SP100 exhibits <em>lower</em> validation average correlation than SP30, yet $C_{\\text{stable}}$ underperforms the historical CVaR baseline on SP100. Correlation-stratified 30-stock universes drawn from the SP100 pool further show that high-correlation subsets can yield the largest CVaR improvements. We therefore attribute SP100 degradation primarily to the joint effect of <b>nominal dimensionality</b> and <b>limited effective diversification</b>, rather than to high correlation per se.
</div>

<h3>9.2 相关性分层（V7-B 实际结果）</h3>

<div class="warn">
<b>原计划假设</b>（Mid-Corr 最优）<b>未被支持</b>。实际：High-Corr 30 股组 WinRate=100%，mean $\\Delta_A=+0.17$ pp；Low/Mid 组均为 67%，改善更小。
</div>

<div class="box">
Within 30-stock subsets of the SP100 universe, state-dependent RCVaR remains effective even in highly correlated pools. This contradicts a simple narrative that higher correlation alone causes failure. Instead, the full SP100 case appears to be a boundary regime where optimization complexity, weight concentration, and a strong historical-CVaR baseline interact.
</div>

<h3>9.3 有效维度缩放（V7-D 实际结果）</h3>

<div class="box">
Motivated by structural diagnostics, we introduce an effective-dimension-scaled ambiguity budget with validation-selected cap ($d_0=30$). On SP30, $V7_{\\text{effdim+cap}}$ improves CVaR relative to $C_{\\text{stable}}$ without violating the non-degradation criterion. On SP100, it repairs the default state-dependent model and matches $C_{\\text{stable}}$, but still does not beat the historical CVaR baseline. Cap remains essential: scaling $\\kappa$ alone without cap worsens performance.
</div>

<h3>9.4 定稿叙事（中文）</h3>

<div class="good">
$$
\\boxed{
\\text{状态依赖 RCVaR 在 30 股量级个股池有效；SP100 失败主因是名义维度与有效分散度，而非平均相关性更高。}
}
$$
</div>
"""


def build_results_html() -> str:
    parts = ['<h2>11. 实验结果与分析</h2>']

    struct_path = V7_OUT / "structure" / "universe_structure_summary.csv"
    if struct_path.exists():
        s = pd.read_csv(struct_path)
        rows = ""
        for _, r in s.iterrows():
            gap = r.get("cvar_gap_C_minus_A")
            d_eff = r.get("effective_dimension_val", float("nan"))
            d_eff_s = f"{d_eff:.1f}" if pd.notna(d_eff) else "—"
            rows += (
                f"<tr><td>{r['universe']}</td>"
                f"<td>{int(r['n_assets'])}</td>"
                f"<td>{r['avg_correlation_val']:.3f}</td>"
                f"<td>{d_eff_s}</td>"
                f"<td>{_pct(r.get('cvar_A'))}</td>"
                f"<td>{_pct(r.get('cvar_C_stable'))}</td>"
                f"<td>{_gap_pp(gap)}</td></tr>"
            )
        parts.append(f"""
<h3>11.1 V7-A 结构诊断</h3>
<div class="good"><b>已完成。</b> 输出：<code>outputs/v7/structure/universe_structure_summary.csv</code></div>
<table>
<tr><th>Universe</th><th>N</th><th>Val avg corr</th><th>Val d_eff</th><th>CVaR A</th><th>CVaR C_stable</th><th>Gap C−A</th></tr>
{rows}
</table>
<p>图表：<code>outputs/v7/structure/figures/fig_structure_comparison.png</code></p>
""")

    grp_path = V7_OUT / "corr_stratified" / "group_summary.csv"
    if grp_path.exists():
        g = pd.read_csv(grp_path)
        rows = ""
        for _, r in g.iterrows():
            rows += (
                f"<tr><td>{r['corr_group']}</td><td>{int(r['n'])}</td>"
                f"<td>{r['mean_avg_corr_val']:.3f}</td>"
                f"<td>{r['win_rate_A']:.1%}</td>"
                f"<td>{r['mean_delta_A_pp']:.2f} pp</td></tr>"
            )
        parts.append(f"""
<h3>11.2 V7-B 相关性分层 Random30</h3>
<div class="good"><b>已完成。</b> Low/Mid/High 各 3 个 universe（候选 300）。</div>
<table>
<tr><th>组别</th><th>n</th><th>Mean val corr</th><th>WinRate vs A</th><th>Mean Δ_A</th></tr>
{rows}
</table>
<p>图表：<code>outputs/v7/corr_stratified/figures/fig_corr_vs_cvar_improvement.png</code></p>
""")

    over_path = V7_OUT / "overfit" / "oos_gap_summary.csv"
    if over_path.exists():
        o = pd.read_csv(over_path)
        rows = ""
        for _, r in o.iterrows():
            rows += (
                f"<tr><td>{r['model']}</td>"
                f"<td>{r['mean_tail_jaccard']:.3f}</td>"
                f"<td>{r['mean_oos_gap'] * 100:.3f} pp</td>"
                f"<td>{r['mean_hhi']:.3f}</td></tr>"
            )
        parts.append(f"""
<h3>11.3 V7-C 过拟合诊断（SP100）</h3>
<div class="good"><b>已完成。</b></div>
<table>
<tr><th>模型</th><th>Mean tail Jaccard</th><th>Mean OOS gap</th><th>Mean HHI</th></tr>
{rows}
</table>
<p>图表：<code>outputs/v7/overfit/figures/fig_oos_gap_boxplot.png</code></p>
""")

    eff_path = V7_OUT / "effdim_rcvar" / "summary.csv"
    if eff_path.exists():
        e = pd.read_csv(eff_path)
        rows = ""
        for ds in ["sp30", "sp100"]:
            sub = e[(e["dataset"] == ds) & e["method"].isin(["A_ceil_CVaR", "C_stable", "V7_effdim", "V7_effdim_cap"])]
            for _, r in sub.iterrows():
                rows += f"<tr><td>{ds.upper()}</td><td>{r['method']}</td><td>{_pct(r['cvar_5pct'])}</td></tr>"
        parts.append(f"""
<h3>11.4 V7-D 有效维度缩放 RCVaR</h3>
<div class="good"><b>已完成。</b> d₀=30，耗时 ~571 min。</div>
<table>
<tr><th>Dataset</th><th>Model</th><th>Test CVaR 5%</th></tr>
{rows}
</table>
<p>图表：<code>outputs/v7/effdim_rcvar/figures/fig_v7_vs_v6_cvar.png</code></p>
""")

    parts.append("""
<h3>11.5 结果解读（定稿摘要）</h3>
<div class="box">
<b>结构机制（V7-A）：</b> SP100 validation 平均相关性（0.335）<b>并不高于</b> SP30（0.364），但 SP100 名义维度 N=100、有效维度 d_eff≈7；C_stable 在 SP30/Random30 优于 A，在 SP100 略劣于 A（+0.08 pp）。<br><br>
<b>相关性分层（V7-B）：</b> 从 SP100 池抽取 30 股子组合时，High-Corr 组 WinRate=100%、mean Δ_A=+0.17 pp——<b>高相关本身不导致失败</b>；失败更可能与 N=100 下的优化复杂度与强 baseline 相关。<br><br>
<b>过拟合（V7-C）：</b> SP100 上 C_stable tail 重叠率最低（0.023）、OOS gap 小于 C_default；权重更集中（N_eff≈15）。<br><br>
<b>有效维度缩放（V7-D）：</b> V7_effdim_cap 在 SP30 略优于 C_stable（−0.04 pp）；在 SP100 修复 C_default（2.92%→2.52%）并与 C_stable 持平，但仍未超越 A（2.47%）。纯 effdim 无 cap 在两市场均变差。
</div>
<table>
<tr><th>Dataset</th><th>A</th><th>C_stable</th><th>V7_effdim</th><th>V7_effdim_cap</th></tr>
<tr><td>SP30</td><td>2.76%</td><td>2.61%</td><td>2.91%</td><td><b>2.57%</b></td></tr>
<tr><td>SP100</td><td><b>2.47%</b></td><td>2.51%</td><td>2.86%</td><td>2.52%</td></tr>
</table>
<div class="warn">
<b>论文叙事修正：</b> 不宜写「SP100 因相关性高而失败」；应写「中等规模 30 股池有效；全样本 SP100 为名义维度与有效分散度联合作用的边界案例」。
</div>
<h3>11.6 成功标准对照</h3>
<table>
<tr><th>标准</th><th>条件</th><th>结果</th></tr>
<tr><td>SP30 不恶化</td><td>CVaR(V7) ≤ CVaR(C_stable)+0.05 pp</td><td><b>✓</b> V7_effdim_cap 2.57% &lt; C_stable 2.61%</td></tr>
<tr><td>SP100 最低</td><td>CVaR(V7) &lt; CVaR(C_default)</td><td><b>✓</b> 2.52% &lt; 2.92%</td></tr>
<tr><td>SP100 中等</td><td>CVaR(V7) ≤ CVaR(C_stable)</td><td>≈ 持平（2.52% vs 2.51%）</td></tr>
<tr><td>SP100 强标准</td><td>CVaR(V7) &lt; CVaR(A)</td><td><b>✗</b> 2.52% &gt; 2.47%</td></tr>
</table>
""")
    return "\n".join(parts)


def build_checklist_html() -> str:
    return """
<h2>10. 最终检查清单</h2>

<table>
<tr><th>任务</th><th>状态</th><th>结论摘要</th></tr>
<tr><td>V7-A 结构诊断</td><td>✅</td><td>SP100 corr 不高于 SP30；d_eff≈7；C_stable 在 SP100 略劣 A</td></tr>
<tr><td>V7-B 相关性分层</td><td>✅</td><td>High-Corr 30 股组改善最大；不支持「Mid 最优」假设</td></tr>
<tr><td>V7-C 过拟合诊断</td><td>✅</td><td>tail overlap + OOS gap 支持尾部不稳定机制</td></tr>
<tr><td>V7-D 有效维度缩放</td><td>✅</td><td>effdim+cap 修复 SP100 default；未超 A</td></tr>
</table>

<div class="good">
<b>最终目标（已达成）：</b>
<br><br>
$$
\\boxed{
\\text{不是证明一个模型在所有市场都赢，而是建立状态依赖 RCVaR 的结构适用条件，并在此基础上提出最小、可解释的高维修正。}
}
$$
</div>
"""


def _replace_section(html: str, start_marker: str, end_marker: str, new_content: str) -> str:
    start = html.find(start_marker)
    if start < 0:
        raise RuntimeError(f"Section not found: {start_marker}")
    end = html.find(end_marker, start + 1)
    if end < 0:
        raise RuntimeError(f"End marker not found after {start_marker}: {end_marker}")
    return html[:start] + new_content.strip() + "\n\n" + html[end:]


def update_v7_html() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")

    html = _replace_section(
        html,
        "<h2>9. 论文写法建议",
        "<h2>10. 最终检查清单",
        build_paper_section_html(),
    )

    if "<h2>11. 实验结果" in html:
        html = _replace_section(
            html,
            "<h2>10. 最终检查清单",
            "<h2>11. 实验结果",
            build_checklist_html(),
        )
        html = _replace_section(
            html,
            "<h2>11. 实验结果",
            "</body>",
            build_results_html(),
        )
    else:
        html = _replace_section(
            html,
            "<h2>10. 最终检查清单",
            "</body>",
            build_checklist_html() + "\n\n" + build_results_html() + "\n\n</body>",
        )

    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Updated {HTML_PATH}", flush=True)


if __name__ == "__main__":
    update_v7_html()
