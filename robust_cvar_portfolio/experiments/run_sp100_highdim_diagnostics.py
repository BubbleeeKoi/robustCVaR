"""V6 Task 4: SP100 high-dimensional failure diagnostics."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.equity_common import (
    cross_sectional_metrics,
    equity_dir,
    rolling_cvar_gap,
)
from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle, v5_out_dir
from robust_cvar_portfolio.portfolio.rolling import monthly_rebalance_dates
from robust_cvar_portfolio.portfolio.weight_export import export_rebalance_weights
from robust_cvar_portfolio.experiments.v5_common import (
    engine_historical,
    engine_manual,
    load_selected_params,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_sp100_highdim_diagnostics() -> None:
    out = equity_dir("sp100_diagnostics")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    _log("=== V6 SP100 High-Dimensional Diagnostics ===")

    bundle = load_v5_bundle("sp100")
    returns = bundle["returns"]
    states = bundle["states"]
    config = bundle["config"]
    learned = bundle["learned"]
    test_start, test_end = config["splits"]["test"]

    v5_test = v5_out_dir("sp100") / "test"
    frame_a = pd.read_csv(v5_test / "rolling_A_Historical_CVaR.csv", parse_dates=["date"])
    frame_c_def = pd.read_csv(v5_test / "rolling_C_default.csv", parse_dates=["date"])
    frame_c_st = pd.read_csv(v5_test / "rolling_C_stable.csv", parse_dates=["date"])

    sel = load_selected_params(v5_out_dir("sp100") / "validation" / "selected_params.json")

    # Cross-sectional structure (full SP100 panel, test period)
    xs = cross_sectional_metrics(returns.loc[test_start:test_end], window=60)
    xs.to_csv(out / "cross_sectional_structure.csv")

    # Weight diagnostics at rebalance dates
    w_a_path = v5_test / "weights_A_Historical_CVaR.csv"
    if not w_a_path.exists():
        export_rebalance_weights(
            returns, states, engine_historical(config), test_start, test_end,
            config.get("cost_rate", 0.001), config.get("estimation_window", 252),
            config.get("optimizer_maxiter", 50),
        )
    w_c_path = v5_test / "weights_C_stable.csv"
    if not w_c_path.exists():
        engine_c = engine_manual(config, learned, sel["kappa_max"])
        export_rebalance_weights(
            returns, states, engine_c, test_start, test_end,
            config.get("cost_rate", 0.001), config.get("estimation_window", 252),
            config.get("optimizer_maxiter", 50),
            weight_cap=sel.get("weight_cap"), kappa_rho=sel.get("rho"),
        )

    tickers = list(returns.columns)
    rebalance = set(monthly_rebalance_dates(returns.index))
    test_idx = returns.index[(returns.index >= test_start) & (returns.index <= test_end)]

    rows = []
    if w_c_path.exists():
        w_c = pd.read_csv(w_c_path, index_col=0, parse_dates=True)
        w_a = pd.read_csv(w_a_path, index_col=0, parse_dates=True) if w_a_path.exists() else None
        for date in w_c.index:
            if date not in rebalance:
                continue
            wc = w_c.loc[date, tickers].values
            hhi_c = float(np.sum(wc**2))
            row = {
                "date": date,
                "hhi_C_stable": hhi_c,
                "max_weight_C_stable": float(wc.max()),
                "n_eff_C_stable": 1.0 / hhi_c if hhi_c > 0 else np.nan,
            }
            if w_a is not None and date in w_a.index:
                wa = w_a.loc[date, tickers].values
                row["hhi_A"] = float(np.sum(wa**2))
                row["max_weight_A"] = float(wa.max())
            if date in frame_c_st["date"].values:
                to = frame_c_st.loc[frame_c_st["date"] == date, "turnover"]
                row["turnover_C_stable"] = float(to.iloc[0]) if len(to) else np.nan
            rows.append(row)
    weight_diag = pd.DataFrame(rows).set_index("date") if rows else pd.DataFrame()
    if not weight_diag.empty:
        weight_diag.to_csv(out / "weight_diagnostics_rebalance.csv")

    # CVaR gap C_default vs A (rolling)
    gap_def = rolling_cvar_gap(frame_a, frame_c_def, window=60)
    gap_st = rolling_cvar_gap(frame_a, frame_c_st, window=60)
    det = pd.DataFrame({"gap_C_default": gap_def, "gap_C_stable": gap_st})
    det.to_csv(out / "c_vs_a_deterioration.csv")

    # Align diagnostics for correlation
    aligned = det.join(xs, how="inner")
    if not weight_diag.empty:
        aligned = aligned.join(weight_diag, how="inner")

    corr_rows = []
    for gap_col in ["gap_C_default", "gap_C_stable"]:
        for feat in ["hhi_C_stable", "max_weight_C_stable", "turnover_C_stable", "avg_correlation", "pc1_share", "effective_dimension", "vol_dispersion"]:
            if gap_col not in aligned.columns or feat not in aligned.columns:
                continue
            sub = aligned[[gap_col, feat]].dropna()
            if len(sub) < 10:
                continue
            corr_rows.append(
                {
                    "gap": gap_col,
                    "feature": feat,
                    "correlation": float(sub[gap_col].corr(sub[feat])),
                    "n": len(sub),
                }
            )
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(out / "deterioration_correlation_table.csv", index=False)

    from robust_cvar_portfolio.experiments.equity_common import EQUITY_OUT

    paper = EQUITY_OUT / "paper_tables"
    paper.mkdir(parents=True, exist_ok=True)
    corr_df.to_csv(paper / "table4_sp100_diagnostics.csv", index=False)

    def _scatter(x, y, xlabel, ylabel, path, title):
        sub = aligned[[x, y]].dropna()
        if sub.empty:
            return
        plt.figure(figsize=(6, 4))
        plt.scatter(sub[x], sub[y], alpha=0.5, s=20)
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.tight_layout()
        plt.savefig(path, dpi=150)
        plt.close()

    if "hhi_C_stable" in aligned.columns:
        _scatter("hhi_C_stable", "gap_C_default", "HHI (C_stable)", "CVaR gap (C-A)",
                 fig_dir / "fig_cvar_gap_vs_hhi.png", "SP100: CVaR gap vs weight HHI")
    if "turnover_C_stable" in aligned.columns:
        _scatter("turnover_C_stable", "gap_C_default", "Turnover", "CVaR gap (C-A)",
                 fig_dir / "fig_cvar_gap_vs_turnover.png", "SP100: CVaR gap vs turnover")

    if "effective_dimension" in aligned.columns:
        plt.figure(figsize=(10, 4))
        aligned["effective_dimension"].plot(lw=0.8)
        plt.title("SP100 effective dimension (60d window)")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_effective_dimension.png", dpi=150)
        plt.close()

    if "pc1_share" in aligned.columns:
        plt.figure(figsize=(10, 4))
        aligned["pc1_share"].plot(lw=0.8)
        plt.title("SP100 PC1 variance share (60d window)")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_pc1_share.png", dpi=150)
        plt.close()

    _log("\n=== Correlation highlights (gap_C_default) ===")
    sub = corr_df[corr_df["gap"] == "gap_C_default"].sort_values("correlation", key=abs, ascending=False)
    _log(sub.head(8).to_string(index=False))
    _log(f"\nOutput: {out}")


if __name__ == "__main__":
    run_sp100_highdim_diagnostics()
