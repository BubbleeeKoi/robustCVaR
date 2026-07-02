"""SP100 V4 diagnostic experiments (see sp100_diagnostic_experiment_v4plan.html)."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.data.loader import build_state_matrix
from robust_cvar_portfolio.data.sp100_universe import load_sp100_universe
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params, _make_engine
from robust_cvar_portfolio.portfolio.optimizer import loss_samples
from robust_cvar_portfolio.portfolio.rolling import run_rolling
from robust_cvar_portfolio.portfolio.weight_export import export_rebalance_weights
from robust_cvar_portfolio.risk.kappa import KappaParams
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, summarize_backtest
from robust_cvar_portfolio.src.robust_cvar_layer import robust_cvar_weights

V3_DIR = ROOT / "outputs" / "v3" / "sp100"
OUT = ROOT / "outputs" / "v3" / "sp100_diagnostics"
FIG = OUT / "figures"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _ensure_dirs() -> None:
    for sub in [
        "kappa_diagnostics", "weights_diagnostics", "turnover_diagnostics",
        "sector_diagnostics", "kappa_sensitivity", "q_weight_diagnostics", "window_sensitivity", "figures",
    ]:
        (OUT / sub).mkdir(parents=True, exist_ok=True)


def _load_bundle() -> tuple[dict, pd.DataFrame, pd.DataFrame, KappaParams]:
    config_path = ROOT / "configs" / "sp100.yaml"
    data_dir = ROOT / "data" / "processed" / "sp100"
    bundle = load_sp100_universe(config_path, data_dir, target_n=100, force=False)
    config = bundle["config"]
    returns = bundle["returns"]
    states = build_state_matrix(returns)
    learned = _learn_params(returns, states, config)
    return config, returns, states, learned


# ---------- Diag 1: kappa ----------
def diagnose_kappa(config: dict) -> pd.Series:
    kappa = pd.read_csv(V3_DIR / "kappa_series.csv", index_col=0, parse_dates=True).squeeze("columns")
    states = pd.read_csv(V3_DIR / "state_matrix.csv", index_col=0, parse_dates=True)
    test_start, test_end = config["splits"]["test"]
    kappa = kappa.loc[test_start:test_end]
    states = states.loc[test_start:test_end]
    df = pd.concat([kappa.rename("kappa"), states[["Vol_z", "DD_z", "Mom_z", "Corr_z"]]], axis=1).dropna()

    crisis_2020 = (df.index >= "2020-02-01") & (df.index <= "2020-06-30")
    crisis_2022 = (df.index >= "2022-01-01") & (df.index <= "2022-12-31")
    normal = ~(crisis_2020 | crisis_2022)

    summary = {
        "kappa_mean": df["kappa"].mean(),
        "kappa_min": df["kappa"].min(),
        "kappa_max": df["kappa"].max(),
        "kappa_std": df["kappa"].std(),
        "corr_kappa_vol": df["kappa"].corr(df["Vol_z"]),
        "corr_kappa_dd": df["kappa"].corr(df["DD_z"]),
        "corr_kappa_mom": df["kappa"].corr(df["Mom_z"]),
        "corr_kappa_corr": df["kappa"].corr(df["Corr_z"]),
        "kappa_2020_mean": df.loc[crisis_2020, "kappa"].mean(),
        "kappa_2022_mean": df.loc[crisis_2022, "kappa"].mean(),
        "kappa_normal_mean": df.loc[normal, "kappa"].mean(),
        "pct_kappa_above_1.8": float((df["kappa"] > 1.8).mean()),
        "pct_kappa_below_1.2": float((df["kappa"] < 1.2).mean()),
    }
    s = pd.Series(summary, name="value")
    s.to_csv(OUT / "kappa_diagnostics" / "kappa_summary.csv", header=["value"])

    corr = pd.Series(
        {
            "Vol_z": summary["corr_kappa_vol"],
            "DD_z": summary["corr_kappa_dd"],
            "Mom_z": summary["corr_kappa_mom"],
            "Corr_z": summary["corr_kappa_corr"],
        },
        name="corr_with_kappa",
    )
    corr.to_csv(OUT / "kappa_diagnostics" / "kappa_state_correlation.csv", header=["corr_with_kappa"])

    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df.index, df["kappa"], lw=1.0)
    ax.axvspan(pd.Timestamp("2020-02-01"), pd.Timestamp("2020-06-30"), alpha=0.15, color="red")
    ax.axvspan(pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31"), alpha=0.15, color="orange")
    ax.set_ylabel(r"$\kappa_t$")
    ax.set_title("SP100 kappa(s) time series (test)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_kappa_timeseries.png", dpi=150)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(df["Vol_z"], df["kappa"], alpha=0.35, s=10)
    axes[0].set_xlabel("Vol_z"); axes[0].set_ylabel("kappa")
    axes[1].scatter(df["DD_z"], df["kappa"], alpha=0.35, s=10, color="#d62728")
    axes[1].set_xlabel("DD_z"); axes[1].set_ylabel("kappa")
    fig.tight_layout()
    fig.savefig(FIG / "fig_kappa_vs_vol_dd.png", dpi=150)
    plt.close(fig)

    return s


# ---------- Diag 2-3: weights & turnover ----------
def _weight_panel(w_export: pd.DataFrame) -> pd.DataFrame:
    tickers = [c for c in w_export.columns if c not in {"kappa", "turnover"}]
    return w_export[tickers]


def weight_concentration(weights_dict: dict[str, pd.DataFrame]) -> pd.DataFrame:
    records = []
    for name, wexp in weights_dict.items():
        w = _weight_panel(wexp)
        hhi = (w ** 2).sum(axis=1)
        tmp = pd.DataFrame(
            {
                "method": name,
                "date": w.index,
                "hhi": hhi.values,
                "effective_n": (1.0 / hhi).values,
                "max_weight": w.max(axis=1).values,
            }
        )
        records.append(tmp)
    out = pd.concat(records, ignore_index=True)
    out.to_csv(OUT / "weights_diagnostics" / "weight_concentration.csv", index=False)
    return out


def pairwise_weight_distance(w1: pd.DataFrame, w2: pd.DataFrame) -> pd.Series:
    w1p, w2p = _weight_panel(w1), _weight_panel(w2)
    common = w1p.index.intersection(w2p.index)
    cols = w1p.columns.intersection(w2p.columns)
    return (w1p.loc[common, cols] - w2p.loc[common, cols]).abs().sum(axis=1)


def turnover_from_export(weights_dict: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    records = []
    for name, wexp in weights_dict.items():
        turn = wexp["turnover"] if "turnover" in wexp.columns else pd.Series(0.0, index=wexp.index)
        tmp = pd.DataFrame({"method": name, "date": turn.index, "turnover": turn.values})
        records.append(tmp)
    by_method = pd.concat(records, ignore_index=True)
    by_method.to_csv(OUT / "turnover_diagnostics" / "turnover_by_method.csv", index=False)
    summary = by_method.groupby("method")["turnover"].agg(["mean", "sum", "max"]).reset_index()
    summary.columns = ["method", "avg_turnover", "cum_turnover", "max_turnover"]
    summary.to_csv(OUT / "turnover_diagnostics" / "turnover_summary.csv", index=False)
    return by_method, summary


# ---------- Diag 4: sector ----------
def sector_exposure(w: pd.DataFrame, universe: pd.DataFrame) -> pd.DataFrame:
    sectors = universe.set_index("ticker")["sector"]
    tickers = w.columns.intersection(sectors.index)
    w = w[tickers]
    sectors = sectors.loc[tickers]
    exposures = {}
    for sec in sectors.unique():
        cols = sectors[sectors == sec].index
        exposures[sec] = w[cols].sum(axis=1)
    return pd.DataFrame(exposures, index=w.index)


def diagnose_sector(weights_dict: dict[str, pd.DataFrame], universe: pd.DataFrame) -> pd.DataFrame:
    summaries = []
    for name in ["C_manual_kappa", "B_fixed_kappa", "Historical_CVaR"]:
        if name not in weights_dict:
            continue
        w = _weight_panel(weights_dict[name])
        exp = sector_exposure(w, universe)
        exp.to_csv(OUT / "sector_diagnostics" / f"sector_exposure_{name}.csv")
        sec_hhi = (exp ** 2).sum(axis=1)
        summaries.append(
            {
                "method": name,
                "sector_hhi_mean": sec_hhi.mean(),
                "max_sector_exposure_mean": exp.max(axis=1).mean(),
                "max_sector_exposure_max": exp.max(axis=1).max(),
                "top_sector_at_max": exp.max(axis=1).idxmax() if len(exp) else "",
            }
        )
    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "sector_diagnostics" / "sector_concentration_summary.csv", index=False)

    if "C_manual_kappa" in weights_dict:
        exp_c = sector_exposure(_weight_panel(weights_dict["C_manual_kappa"]), universe)
        fig, ax = plt.subplots(figsize=(12, 5))
        im = ax.imshow(exp_c.T.values, aspect="auto", cmap="YlOrRd")
        ax.set_yticks(range(len(exp_c.columns)))
        ax.set_yticklabels(exp_c.columns, fontsize=7)
        ax.set_xlabel("Rebalance index")
        ax.set_title("C_manual sector exposure heatmap")
        plt.colorbar(im, ax=ax)
        fig.tight_layout()
        fig.savefig(FIG / "fig_sector_exposure_heatmap.png", dpi=150)
        plt.close(fig)
    return summary


# ---------- Diag 6: q weights ----------
def diagnose_q_weights(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    weights_dict: dict[str, pd.DataFrame],
    config: dict,
    engines: dict[str, RiskEngine],
) -> pd.DataFrame:
    test_start, test_end = config["splits"]["test"]
    alpha = config.get("alpha", 0.05)
    window = config.get("estimation_window", 252)
    cost = config.get("cost_rate", 0.001)
    idx_map = {d: i for i, d in enumerate(returns.index)}

    all_q_rows = []
    summaries = []

    for method in ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa"]:
        if method not in weights_dict or method not in engines:
            continue
        wexp = weights_dict[method]
        engine = engines[method]
        q_hhis, top1, top3, top5 = [], [], [], []

        for date, row in wexp.iterrows():
            loc = idx_map[date]
            if loc < window:
                continue
            hist = returns.iloc[loc - window : loc]
            feat = states.iloc[loc - window : loc]
            tickers = [c for c in wexp.columns if c not in {"kappa", "turnover"}]
            w = row[tickers].values.astype(float)
            w_prev = np.full(len(tickers), 1.0 / len(tickers))
            losses = loss_samples(w, hist.values, w_prev, cost)
            kappa = engine.kappa_vector(feat, w)
            q = robust_cvar_weights(losses, kappa, alpha)
            q_hhi = float(np.sum(q ** 2))
            q_sorted = np.sort(q)[::-1]
            q_hhis.append(q_hhi)
            top1.append(float(q_sorted[0]) if len(q_sorted) else 0)
            top3.append(float(q_sorted[:3].sum()))
            top5.append(float(q_sorted[:5].sum()))
            for j, (d_hist, loss_val, q_val, k_val) in enumerate(
                zip(hist.index, losses, q, kappa if hasattr(kappa, "__len__") else np.full(len(losses), kappa))
            ):
                if q_val > 1e-6:
                    all_q_rows.append(
                        {
                            "rebalance_date": date,
                            "scenario_date": d_hist,
                            "method": method,
                            "loss": loss_val,
                            "q": q_val,
                            "kappa": float(k_val),
                        }
                    )

        summaries.append(
            {
                "method": method,
                "q_hhi_mean": np.mean(q_hhis),
                "top1_q_mean": np.mean(top1),
                "top3_q_mean": np.mean(top3),
                "top5_q_mean": np.mean(top5),
            }
        )

    q_df = pd.DataFrame(all_q_rows)
    q_df.to_csv(OUT / "q_weight_diagnostics" / "q_weights_all.csv", index=False)
    for method in q_df["method"].unique():
        q_df[q_df["method"] == method].to_csv(OUT / "q_weight_diagnostics" / f"q_weights_{method.split('_')[0]}.csv", index=False)

    top_scenarios = (
        q_df.sort_values("q", ascending=False)
        .groupby(["method", "rebalance_date"])
        .head(5)
    )
    top_scenarios.to_csv(OUT / "q_weight_diagnostics" / "top_q_scenarios.csv", index=False)

    summary = pd.DataFrame(summaries)
    summary.to_csv(OUT / "q_weight_diagnostics" / "q_weight_summary.csv", index=False)

    if not summary.empty:
        fig, ax = plt.subplots(figsize=(8, 4))
        x = np.arange(len(summary))
        ax.bar(x - 0.2, summary["q_hhi_mean"], width=0.4, label="q HHI")
        ax.bar(x + 0.2, summary["top3_q_mean"], width=0.4, label="top-3 q share")
        ax.set_xticks(x)
        ax.set_xticklabels(summary["method"], rotation=15)
        ax.legend()
        ax.set_title("Worst-case q concentration by method")
        fig.tight_layout()
        fig.savefig(FIG / "fig_q_concentration.png", dpi=150)
        plt.close(fig)
    return summary


# ---------- Diag 5 & 7: sensitivity ----------
def _run_c_metrics(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    config: dict,
    learned: KappaParams,
    kappa_max: float | None = None,
    window: int | None = None,
) -> dict:
    cfg = dict(config)
    test_start, test_end = cfg["splits"]["test"]
    win = window or cfg.get("estimation_window", 252)
    params = KappaParams(
        kappa_max=kappa_max if kappa_max is not None else cfg.get("kappa_max", 1.0),
        beta_vol=cfg.get("beta_vol", 1.0),
        beta_dd=cfg.get("beta_dd", 1.0),
        beta_mom=cfg.get("beta_mom", 0.5),
        beta_corr=cfg.get("beta_corr", 0.5),
        beta_conc=cfg.get("beta_conc", 0.5),
        theta=np.array([0.0, 0.0, 0.0, 0.0]),
    )
    engine = RiskEngine(alpha=cfg.get("alpha", 0.05), kappa_mode="manual", params=params)
    frame = run_rolling(
        returns, states, engine, test_start, test_end,
        cfg.get("cost_rate", 0.001), win, min(cfg.get("optimizer_maxiter", 35), 30),
    )
    m = summarize_backtest(frame["net_return"].values, frame["loss"].values, frame["turnover"].values, cfg.get("alpha", 0.05))
    m["kappa_max"] = params.kappa_max
    m["window"] = win
    return m


def kappa_max_sensitivity(returns, states, config, learned) -> pd.DataFrame:
    rows = []
    for km in [0.50, 1.00, 1.50]:
        _log(f"  kappa_max={km} ...")
        rows.append(_run_c_metrics(returns, states, config, learned, kappa_max=km))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "kappa_sensitivity" / "kappa_max_sensitivity_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["kappa_max"], df["cvar_5pct"], marker="o", label="C_manual CVaR")
    ax.axhline(0.028601, color="gray", ls="--", label="A CVaR (V3)")
    ax.axhline(0.026579, color="green", ls="--", label="B CVaR (V3)")
    ax.set_xlabel("kappa_max"); ax.set_ylabel("CVaR 5%"); ax.legend(fontsize=8)
    ax.set_title("kappa_max sensitivity (C_manual, test)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_kappa_max_sensitivity.png", dpi=150)
    plt.close(fig)
    return df


def window_sensitivity(returns, states, config, learned) -> pd.DataFrame:
    rows = []
    for w in [252, 504]:
        _log(f"  window M={w} ...")
        rows.append(_run_c_metrics(returns, states, config, learned, window=w))
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "window_sensitivity" / "window_sensitivity_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(df["window"], df["cvar_5pct"], marker="o")
    ax.axhline(0.028601, color="gray", ls="--", label="A CVaR")
    ax.set_xlabel("estimation window M"); ax.set_ylabel("CVaR 5%"); ax.legend()
    ax.set_title("Window sensitivity (C_manual, test)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_window_sensitivity.png", dpi=150)
    plt.close(fig)
    return df


def build_diagnostic_summary(
    kappa_s: pd.Series,
    wt_conc: pd.DataFrame,
    turn_sum: pd.DataFrame,
    sector_sum: pd.DataFrame,
    q_sum: pd.DataFrame,
    km_df: pd.DataFrame,
    win_df: pd.DataFrame,
) -> pd.DataFrame:
    def agg(method: str, col: str) -> float:
        sub = wt_conc[wt_conc["method"] == method]
        return float(sub[col].mean()) if len(sub) else np.nan

    rows = [
        {
            "diagnostic_item": "kappa_response",
            "main_result": f"mean={kappa_s['kappa_mean']:.3f}, corr(vol)={kappa_s['corr_kappa_vol']:.3f}, corr(dd)={kappa_s['corr_kappa_dd']:.3f}, crisis2020={kappa_s['kappa_2020_mean']:.3f} vs normal={kappa_s['kappa_normal_mean']:.3f}",
            "supports_failure_reason": "no" if kappa_s["corr_kappa_vol"] > 0 and kappa_s["corr_kappa_dd"] > 0 else "yes",
            "evidence_file": "kappa_diagnostics/kappa_summary.csv",
            "next_action": "kappa 方向正确；若 mean 过高则降低 kappa_max 或平滑 kappa_t",
        },
        {
            "diagnostic_item": "weight_concentration",
            "main_result": f"C eff_n={agg('C_manual_kappa','effective_n'):.1f} vs B {agg('B_fixed_kappa','effective_n'):.1f} vs A {agg('A_no_kappa','effective_n'):.1f}; C max_w={agg('C_manual_kappa','max_weight'):.3f}",
            "supports_failure_reason": "yes" if agg("C_manual_kappa", "effective_n") < agg("B_fixed_kappa", "effective_n") else "no",
            "evidence_file": "weights_diagnostics/weight_concentration.csv",
            "next_action": "若 C 更集中：加 HHI penalty 或单资产上限",
        },
        {
            "diagnostic_item": "turnover",
            "main_result": turn_sum.set_index("method")["avg_turnover"].to_dict().__str__(),
            "supports_failure_reason": "yes" if turn_sum.loc[turn_sum["method"] == "C_manual_kappa", "avg_turnover"].iloc[0] > turn_sum.loc[turn_sum["method"] == "B_fixed_kappa", "avg_turnover"].iloc[0] else "no",
            "evidence_file": "turnover_diagnostics/turnover_summary.csv",
            "next_action": "若 C 换手更高：平滑 kappa 或降低调仓频率",
        },
        {
            "diagnostic_item": "sector_exposure",
            "main_result": sector_sum.to_dict(orient="records").__str__(),
            "supports_failure_reason": "unclear",
            "evidence_file": "sector_diagnostics/sector_concentration_summary.csv",
            "next_action": "若行业 HHI 高：加 sector cap",
        },
        {
            "diagnostic_item": "q_concentration",
            "main_result": q_sum.to_dict(orient="records").__str__(),
            "supports_failure_reason": "yes" if len(q_sum) and q_sum.loc[q_sum["method"] == "C_manual_kappa", "q_hhi_mean"].iloc[0] > q_sum.loc[q_sum["method"] == "B_fixed_kappa", "q_hhi_mean"].iloc[0] else "unclear",
            "evidence_file": "q_weight_diagnostics/q_weight_summary.csv",
            "next_action": "若 q 更集中：增大 M 或降低 kappa_max",
        },
        {
            "diagnostic_item": "kappa_max_sensitivity",
            "main_result": f"best CVaR={km_df['cvar_5pct'].min():.4f} at kappa_max={km_df.loc[km_df['cvar_5pct'].idxmin(), 'kappa_max']}",
            "supports_failure_reason": "yes" if km_df["cvar_5pct"].min() < 0.028601 else "no",
            "evidence_file": "kappa_sensitivity/kappa_max_sensitivity_summary.csv",
            "next_action": "用 val 选择更小的 kappa_max" if km_df.loc[km_df["cvar_5pct"].idxmin(), "kappa_max"] < 1.0 else "状态依赖机制可能无效",
        },
        {
            "diagnostic_item": "window_sensitivity",
            "main_result": f"best CVaR={win_df['cvar_5pct'].min():.4f} at M={win_df.loc[win_df['cvar_5pct'].idxmin(), 'window']}",
            "supports_failure_reason": "yes" if win_df["cvar_5pct"].min() < 0.029171 else "no",
            "evidence_file": "window_sensitivity/window_sensitivity_summary.csv",
            "next_action": "增大估计窗口 M" if win_df.loc[win_df["cvar_5pct"].idxmin(), "window"] > 252 else "窗口不是主因",
        },
    ]
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "diagnostic_summary.csv", index=False)
    md = ["# SP100 Diagnostic Summary\n"] + [f"## {r['diagnostic_item']}\n- {r['main_result']}\n- supports: {r['supports_failure_reason']}\n- next: {r['next_action']}\n" for _, r in df.iterrows()]
    (OUT / "diagnostic_summary.md").write_text("\n".join(md), encoding="utf-8")
    return df


def export_all_weights(
    returns, states, config, learned, maxiter: int = 40,
) -> dict[str, pd.DataFrame]:
    test_start, test_end = config["splits"]["test"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    out: dict[str, pd.DataFrame] = {}

    model_keys = ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa", "D_state_action"]
    for key in model_keys:
        ckpt = OUT / "weights_diagnostics" / f"weights_{key}.csv"
        if ckpt.exists():
            _log(f"  weights {key} loaded checkpoint")
            out[key] = pd.read_csv(ckpt, index_col=0, parse_dates=True)
            continue
        _log(f"  weights {key} exporting ...")
        engine = _make_engine(key, config, learned)
        wexp = export_rebalance_weights(
            returns, states, engine, test_start, test_end, cost, window, maxiter,
        )
        wexp.to_csv(ckpt)
        out[key] = wexp

    hist_ckpt = OUT / "weights_diagnostics" / "weights_Historical_CVaR.csv"
    if hist_ckpt.exists():
        out["Historical_CVaR"] = pd.read_csv(hist_ckpt, index_col=0, parse_dates=True)
    else:
        _log("  weights Historical_CVaR exporting ...")
        hist_engine = RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="plain")
        out["Historical_CVaR"] = export_rebalance_weights(
            returns, states, hist_engine, test_start, test_end, cost, window, maxiter,
        )
        out["Historical_CVaR"].to_csv(hist_ckpt)

    # copy to v3 dir names per plan
    mapping = {
        "A_no_kappa": V3_DIR / "weights_A.csv",
        "B_fixed_kappa": V3_DIR / "weights_B.csv",
        "C_manual_kappa": V3_DIR / "weights_C_manual.csv",
        "D_state_action": V3_DIR / "weights_D.csv",
        "Historical_CVaR": V3_DIR / "weights_HistoricalCVaR.csv",
    }
    for k, path in mapping.items():
        if k in out:
            _weight_panel(out[k]).to_csv(path, index_label="date")

    return out


def plot_weight_concentration(wt_conc: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, col, title in zip(axes, ["effective_n", "hhi", "max_weight"], ["Effective N", "HHI", "Max weight"]):
        for method in wt_conc["method"].unique():
            sub = wt_conc[wt_conc["method"] == method]
            ax.plot(sub["date"], sub[col], label=method, alpha=0.8)
        ax.set_title(title)
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(FIG / "fig_weight_concentration.png", dpi=150)
    plt.close(fig)


def plot_turnover(turn_df: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(10, 4))
    for method in turn_df["method"].unique():
        sub = turn_df[turn_df["method"] == method]
        ax.plot(sub["date"], sub["turnover"], label=method, alpha=0.8)
    ax.legend(fontsize=7)
    ax.set_title("Turnover at rebalance (by method)")
    fig.tight_layout()
    fig.savefig(FIG / "fig_turnover_comparison.png", dpi=150)
    plt.close(fig)


def run_all(skip_sensitivity: bool = False) -> dict:
    _ensure_dirs()
    t0 = time.time()
    _log("=== V4 SP100 Diagnostics ===")
    config, returns, states, learned = _load_bundle()

    _log("\n[1/7] Kappa diagnostics")
    kappa_s = diagnose_kappa(config)

    _log("\n[2/7] Export weights (A/B/C/D/Hist)")
    weights_dict = export_all_weights(returns, states, config, learned, maxiter=40)

    _log("\n[3/7] Weight concentration")
    wt_conc = weight_concentration(weights_dict)
    plot_weight_concentration(wt_conc)
    if "C_manual_kappa" in weights_dict and "B_fixed_kappa" in weights_dict:
        d_cb = pairwise_weight_distance(weights_dict["C_manual_kappa"], weights_dict["B_fixed_kappa"])
        d_ca = pairwise_weight_distance(weights_dict["C_manual_kappa"], weights_dict["A_no_kappa"])
        d_cb.to_csv(OUT / "weights_diagnostics" / "weight_distance_C_vs_B.csv")
        d_ca.to_csv(OUT / "weights_diagnostics" / "weight_distance_C_vs_A.csv")

    _log("\n[4/7] Turnover")
    turn_df, turn_sum = turnover_from_export(weights_dict)
    plot_turnover(turn_df)

    _log("\n[5/7] Sector exposure")
    universe = pd.read_csv(ROOT / "data" / "processed" / "sp100" / "universe.csv")
    sector_sum = diagnose_sector(weights_dict, universe)

    _log("\n[6/7] q-weight diagnostics")
    engines = {k: _make_engine(k, config, learned) for k in ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa"]}
    q_sum = diagnose_q_weights(returns, states, weights_dict, config, engines)

    km_df = pd.DataFrame()
    win_df = pd.DataFrame()
    if not skip_sensitivity:
        _log("\n[7a/7] kappa_max sensitivity (6 runs, ~slow)")
        km_df = kappa_max_sensitivity(returns, states, config, learned)
        _log("\n[7b/7] window sensitivity (3 runs)")
        win_df = window_sensitivity(returns, states, config, learned)
    else:
        _log("\n[7/7] sensitivity skipped")

    summary = build_diagnostic_summary(kappa_s, wt_conc, turn_sum, sector_sum, q_sum, km_df, win_df)
    elapsed = time.time() - t0
    _log(f"\nDone in {elapsed/60:.1f} min. Output: {OUT}")
    return {
        "kappa": kappa_s.to_dict(),
        "summary": summary,
        "elapsed_sec": elapsed,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-sensitivity", action="store_true")
    parser.add_argument("--sensitivity-only", action="store_true", help="only run diag 7 + summary from existing outputs")
    args = parser.parse_args()
    if args.sensitivity_only:
        run_sensitivity_and_summary()
    else:
        run_all(skip_sensitivity=args.skip_sensitivity)


def run_sensitivity_and_summary() -> None:
    """Complete diag 7 and summary using cached diag 1-6 outputs."""
    _ensure_dirs()
    config, returns, states, learned = _load_bundle()
    kappa_s = pd.read_csv(OUT / "kappa_diagnostics" / "kappa_summary.csv", index_col=0).squeeze("columns")
    wt_conc = pd.read_csv(OUT / "weights_diagnostics" / "weight_concentration.csv")
    turn_sum = pd.read_csv(OUT / "turnover_diagnostics" / "turnover_summary.csv")
    sector_sum = pd.read_csv(OUT / "sector_diagnostics" / "sector_concentration_summary.csv")
    q_sum = pd.read_csv(OUT / "q_weight_diagnostics" / "q_weight_summary.csv")
    _log("[7a/7] kappa_max sensitivity")
    km_df = kappa_max_sensitivity(returns, states, config, learned)
    _log("[7b/7] window sensitivity")
    win_df = window_sensitivity(returns, states, config, learned)
    build_diagnostic_summary(kappa_s, wt_conc, turn_sum, sector_sum, q_sum, km_df, win_df)
    _log(f"Summary written to {OUT / 'diagnostic_summary.csv'}")


if __name__ == "__main__":
    main()
