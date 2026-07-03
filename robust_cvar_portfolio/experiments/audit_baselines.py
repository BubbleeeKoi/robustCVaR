"""Step 0: Baseline audit — compare A_Historical vs Historical CVaR weights and returns."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.v5_common import (
    ROOT,
    engine_historical,
    export_test_weights,
    load_v5_bundle,
    metrics_row,
    run_historical_test,
    run_test_backtest,
    weight_diff_max,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def audit_baselines(dataset: str) -> None:
    out = ROOT / "outputs" / "v5" / "audit"
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(f"=== V5 Baseline Audit ({dataset}) ===")

    bundle = load_v5_bundle(dataset)
    config = bundle["config"]
    returns = bundle["returns"]
    states = bundle["states"]
    tickers = list(returns.columns)
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)
    test_start, test_end = config["splits"]["test"]

    a_engine = engine_historical(config)
    a_ckpt = out / f"rolling_A_no_kappa_recomputed_{dataset}.csv"
    if a_ckpt.exists():
        a_frame = pd.read_csv(a_ckpt, parse_dates=["date"])
    else:
        _log("  recompute A_Historical (plain_ceil) ...")
        a_frame = run_test_backtest(returns, states, a_engine, config)
        a_frame.to_csv(a_ckpt, index=False)

    hist_ckpt = out / f"rolling_Historical_CVaR_recomputed_{dataset}.csv"
    if hist_ckpt.exists():
        hist_frame = pd.read_csv(hist_ckpt, parse_dates=["date"])
    else:
        _log("  recompute Historical_CVaR ...")
        hist_frame = run_historical_test(returns, config)
        hist_frame.to_csv(hist_ckpt, index=False)

    w_a_path = out / f"weights_A_{dataset}.csv"
    w_h_path = out / f"weights_Historical_{dataset}.csv"
    if not w_a_path.exists():
        export_test_weights(returns, states, a_engine, config, w_a_path)
    if not w_h_path.exists():
        export_test_weights(returns, states, engine_historical(config), config, w_h_path)

    w_a = pd.read_csv(w_a_path, index_col=0, parse_dates=True)
    w_h = pd.read_csv(w_h_path, index_col=0, parse_dates=True)
    max_diff = weight_diff_max(w_a, w_h, tickers)

    compare_rows = []
    for date in w_a.index.intersection(w_h.index):
        row = {"date": date, "max_abs_diff": float(np.max(np.abs(w_a.loc[date, tickers] - w_h.loc[date, tickers])))}
        compare_rows.append(row)
    compare_df = pd.DataFrame(compare_rows)
    compare_df.to_csv(out / f"compare_A_vs_Historical_weights_{dataset}.csv", index=False)

    m_a = metrics_row(a_frame, config, "A_Historical_CVaR")
    m_h = metrics_row(hist_frame, config, "Historical_CVaR")
    summary = pd.DataFrame([m_a, m_h])
    summary["max_weight_diff"] = max_diff
    summary["weights_identical"] = max_diff < 1e-6
    summary["net_return_corr"] = float(a_frame["net_return"].corr(hist_frame["net_return"]))
    summary.to_csv(out / f"baseline_audit_summary_{dataset}.csv", index=False)

    _log(f"  max |w_A - w_H| = {max_diff:.2e}")
    _log(f"  net_return corr = {summary['net_return_corr'].iloc[0]:.6f}")
    _log(f"  A CVaR={m_a['cvar_5pct']:.4f}, Hist CVaR={m_h['cvar_5pct']:.4f}")
    _log(f"Output: {out} ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sp100")
    args = parser.parse_args()
    audit_baselines(args.dataset)
