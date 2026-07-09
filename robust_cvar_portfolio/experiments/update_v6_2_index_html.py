"""Update V6_2_index_full_run_plan.html with experiment results + figures."""

from __future__ import annotations

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT = Path(__file__).resolve().parents[2]
HTML_PATH = PROJECT / "V6_2_index_full_run_plan.html"
OUT = PROJECT / "robust_cvar_portfolio" / "outputs" / "v6_index_precheck"
FIG = OUT / "figures"


def _pct(x: float) -> str:
    return f"{x * 100:.2f}%" if pd.notna(x) else "—"


def _load_summary(name: str) -> dict | None:
    p = OUT / name / "index_summary.csv"
    if not p.exists():
        return None
    return pd.read_csv(p).iloc[0].to_dict()


def _load_table(name: str) -> pd.DataFrame | None:
    p = OUT / name / "table_main.csv"
    return pd.read_csv(p) if p.exists() else None


def _verdict(win_a: bool, win_idx: bool, c_sharpe: float, a_sharpe: float) -> str:
    if win_a and win_idx:
        return "非常好：V6 在该指数池具有较强普适性"
    if win_a and not win_idx:
        return "较好：优于 Historical CVaR，但未跑赢指数 ETF"
    if not win_a and c_sharpe < a_sharpe:
        return "尾部风险 tradeoff：CVaR 未赢但需看 Sharpe/回撤结构"
    return "需谨慎：该股票池上 C_stable 未显示明确优势"


def build_results_html() -> str:
    parts = ['<h2>13. V6 完整补充实验结果（自动更新）</h2>']
    parts.append(
        '<div class="box ok"><p><b>主模型：V6 C_stable。</b> '
        "Point-in-time 成分股来源：unliftedq/index-constitution CSV；"
        f"输出目录：<code>outputs/v6_index_precheck/</code></p></div>"
    )

    all_summaries = []
    for name, label in [("dji30", "DJI30"), ("ndx100", "NDX100"), ("sp500", "SP500")]:
        s = _load_summary(name)
        tbl = _load_table(name)
        if s is None or tbl is None:
            parts.append(f"<h3>{label}</h3><p><i>尚未完成或未找到结果。</i></p>")
            continue
        all_summaries.append(s)
        c_row = tbl[tbl["method"] == "C_stable"].iloc[0]
        a_row = tbl[tbl["method"] == "A_ceil_CVaR"].iloc[0]
        verdict = _verdict(
            bool(s.get("win_C_vs_A")),
            bool(s.get("win_C_vs_Index")),
            float(c_row.get("sharpe_ratio", np.nan)),
            float(a_row.get("sharpe_ratio", np.nan)),
        )
        parts.append(f"<h3>{label}</h3>")
        parts.append(
            f'<div class="box warn"><p>{verdict}</p>'
            f"<p>耗时：{float(s.get('elapsed_sec', 0))/60:.1f} min；"
            f"d_eff/N≈{float(s.get('d_eff_over_N', np.nan)):.3f}</p></div>"
        )
        parts.append("<table><tr><th>模型</th><th>CVaR 5%</th><th>Sharpe</th><th>MaxDD</th><th>Ann.Return</th></tr>")
        for _, r in tbl.iterrows():
            parts.append(
                f"<tr><td>{r['method']}</td>"
                f"<td>{_pct(r['cvar_5pct'])}</td>"
                f"<td>{float(r['sharpe_ratio']):.2f}</td>"
                f"<td>{_pct(r['max_drawdown'])}</td>"
                f"<td>{_pct(r['annualized_return'])}</td></tr>"
            )
        parts.append("</table>")

    if len(all_summaries) >= 2:
        parts.append("<h3>横向对比：C_stable vs A（ΔCVaR，pp）</h3><table>")
        parts.append("<tr><th>股票池</th><th>Δ(A−C)</th><th>赢 A</th><th>赢 Index</th></tr>")
        for s in all_summaries:
            parts.append(
                f"<tr><td>{s['index'].upper()}</td>"
                f"<td>{float(s['delta_C_vs_A'])*100:.2f}</td>"
                f"<td>{'✓' if s.get('win_C_vs_A') else '✗'}</td>"
                f"<td>{'✓' if s.get('win_C_vs_Index') else '✗'}</td></tr>"
            )
        parts.append("</table>")

    parts.append(
        "<p>图表：<code>outputs/v6_index_precheck/figures/</code> "
        "（DJI30/NDX100 柱状图、结构诊断、CVaR improvement、NAV/Drawdown）</p>"
    )
    return "\n".join(parts)


def generate_figures() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    methods = ["Index_ETF", "Equal_Weight", "A_ceil_CVaR", "B_fixed_kappa", "C_stable"]
    colors = ["#6b7280", "#9ca3af", "#ef4444", "#f59e0b", "#10b981"]

    for name, title in [("dji30", "DJI30"), ("ndx100", "NDX100")]:
        tbl = _load_table(name)
        if tbl is None:
            continue
        sub = tbl[tbl["method"].isin(methods)]
        plt.figure(figsize=(9, 4))
        x = np.arange(len(sub))
        plt.bar(x, sub["cvar_5pct"] * 100, color=colors[: len(sub)])
        plt.xticks(x, sub["method"], rotation=20, ha="right")
        plt.ylabel("CVaR 5% (%)")
        plt.title(f"{title} Test CVaR (lower is better)")
        plt.tight_layout()
        plt.savefig(FIG / f"fig_{name}_cvar_bar.png", dpi=150)
        plt.close()

    struct_rows = []
    for name, n in [("dji30", 30), ("ndx100", 100), ("sp500", 500)]:
        s = _load_summary(name)
        if s:
            struct_rows.append(
                {
                    "pool": name.upper(),
                    "N": n,
                    "d_eff": float(s.get("d_eff", np.nan)),
                    "d_eff_over_N": float(s.get("d_eff_over_N", np.nan)),
                    "avg_corr": float(s.get("avg_correlation", np.nan)),
                }
            )
    if struct_rows:
        sdf = pd.DataFrame(struct_rows)
        fig, ax1 = plt.subplots(figsize=(8, 4))
        ax1.bar(sdf["pool"], sdf["d_eff_over_N"], color="#6366f1", alpha=0.8)
        ax1.set_ylabel("d_eff / N")
        ax1.set_title("Structure: effective dimension ratio")
        plt.tight_layout()
        plt.savefig(FIG / "fig_structure_deff_ratio.png", dpi=150)
        plt.close()

    imp_rows = []
    for name in ["sp30", "dji30", "ndx100"]:
        if name == "sp30":
            p = PROJECT / "robust_cvar_portfolio" / "outputs" / "equity_only" / "sp30" / "table_main.csv"
        else:
            p = OUT / name / "table_main.csv"
        if not p.exists():
            continue
        t = pd.read_csv(p)
        a = t[t["method"].isin(["A_ceil_CVaR", "A_Historical_CVaR"])]["cvar_5pct"].iloc[0]
        c = t[t["method"] == "C_stable"]["cvar_5pct"].iloc[0]
        imp_rows.append({"pool": name.upper(), "delta_pp": (a - c) * 100})
    if imp_rows:
        idf = pd.DataFrame(imp_rows)
        plt.figure(figsize=(7, 4))
        plt.bar(idf["pool"], idf["delta_pp"], color="#10b981")
        plt.axhline(0, color="red", ls="--")
        plt.ylabel("ΔCVaR vs A (pp)")
        plt.title("C_stable CVaR improvement vs Historical CVaR")
        plt.tight_layout()
        plt.savefig(FIG / "fig_cvar_improvement_cross_pool.png", dpi=150)
        plt.close()

    for name, title in [("dji30", "DJI30"), ("ndx100", "NDX100")]:
        fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True, gridspec_kw={"height_ratios": [2, 1]})
        has_any = False
        for method, color in [("C_stable", "#10b981"), ("A_ceil_CVaR", "#ef4444"), ("Index_ETF", "#6b7280")]:
            p = OUT / name / f"rolling_{method}.csv"
            if not p.exists():
                continue
            has_any = True
            df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
            nav = (1 + df["net_return"]).cumprod()
            axes[0].plot(nav.index, nav.values, label=method, color=color, lw=1.2)
            dd = nav / nav.cummax() - 1
            axes[1].fill_between(dd.index, dd.values, 0, alpha=0.15, color=color)
        if not has_any:
            plt.close()
            continue
        axes[0].set_title(f"{title} NAV")
        axes[0].legend(fontsize=8)
        axes[0].set_ylabel("NAV")
        axes[1].set_ylabel("Drawdown")
        plt.tight_layout()
        plt.savefig(FIG / f"fig_{name}_nav_drawdown.png", dpi=150)
        plt.close()


def update_all() -> None:
    html = HTML_PATH.read_text(encoding="utf-8")
    block = build_results_html()
    marker = "<p class=\"small\">Generated on"
    if "<h2>13. V6 完整补充实验结果" in html:
        html = re.sub(
            r"<h2>13\. V6 完整补充实验结果.*?(?=<p class=\"small\">Generated on)",
            block + "\n\n",
            html,
            count=1,
            flags=re.DOTALL,
        )
    else:
        html = html.replace(
            marker,
            block + "\n\n" + marker,
        )
    HTML_PATH.write_text(html, encoding="utf-8")
    print(f"Updated {HTML_PATH}", flush=True)


if __name__ == "__main__":
    generate_figures()
    update_all()
