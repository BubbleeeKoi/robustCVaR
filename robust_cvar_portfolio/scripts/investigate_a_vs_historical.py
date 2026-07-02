"""Investigate A_no_kappa vs Historical_CVaR discrepancy."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.data.loader import build_state_matrix
from robust_cvar_portfolio.data.sp100_universe import load_sp100_universe
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params, _make_engine
from robust_cvar_portfolio.portfolio.optimizer import loss_samples, optimize_portfolio, softmax
from robust_cvar_portfolio.portfolio.rolling import monthly_rebalance_dates
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.baselines import _historical_cvar_objective
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, summarize_backtest, turnover
from robust_cvar_portfolio.src.robust_cvar_layer import robust_cvar, verify_degeneracy


def compare_risk_measures(n: int = 252, alpha: float = 0.05, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for _ in range(500):
        z = rng.normal(0, 0.02, n)
        plain = cvar_alpha(z, alpha)
        rcvar = robust_cvar(z, 1.0, alpha)
        rows.append({"plain": plain, "rcvar_k1": rcvar, "diff": rcvar - plain, "rel_diff": (rcvar - plain) / max(plain, 1e-12)})
    return pd.DataFrame(rows)


def objective_a(logits, hist, w_prev, cost, engine, feat):
    w = softmax(logits)
    losses = loss_samples(w, hist, w_prev, cost)
    risk, _ = engine.portfolio_risk(losses, feat, w)
    return risk


def compare_single_rebalance(hist: np.ndarray, w_prev: np.ndarray, feat: pd.DataFrame, cost: float, alpha: float, maxiter: int):
    engine = RiskEngine(alpha=alpha, kappa_mode="plain")
    n = hist.shape[1]

    res_a = minimize(objective_a, np.zeros(n), args=(hist.values, w_prev, cost, engine, feat), method="L-BFGS-B", options={"maxiter": maxiter})
    w_a = softmax(res_a.x)

    res_h = minimize(_historical_cvar_objective, np.zeros(n), args=(hist.values, w_prev, cost, alpha), method="L-BFGS-B", options={"maxiter": maxiter})
    w_h = softmax(res_h.x)

    losses_a = loss_samples(w_a, hist.values, w_prev, cost)
    losses_h = loss_samples(w_h, hist.values, w_prev, cost)

    return {
        "w_l1_dist": float(np.abs(w_a - w_h).sum()),
        "obj_a_rcvar": float(robust_cvar(losses_a, 1.0, alpha)),
        "obj_a_cvar": float(cvar_alpha(losses_a, alpha)),
        "obj_h_cvar": float(cvar_alpha(losses_h, alpha)),
        "obj_h_on_wa": float(cvar_alpha(losses_a, alpha)),
        "opt_a_success": res_a.success,
        "opt_h_success": res_h.success,
    }


def historical_backtest_fixed_cost(returns, start, end, window, cost_rate, alpha, maxiter):
    """Historical CVaR with rebalance-day cost applied like run_rolling."""
    from robust_cvar_portfolio.src.baselines import _softmax

    rebal = set(monthly_rebalance_dates(returns.index))
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    n = returns.shape[1]
    w_prev = np.full(n, 1.0 / n)
    current_w = w_prev.copy()
    rows = []
    for date in returns.index[mask]:
        loc = idx_map[date]
        day_to = 0.0
        if date in rebal and loc >= window:
            hist = returns.iloc[loc - window : loc].values
            old_w = w_prev.copy()
            res = minimize(_historical_cvar_objective, np.zeros(n), args=(hist, old_w, cost_rate, alpha), method="L-BFGS-B", options={"maxiter": maxiter})
            current_w = _softmax(res.x if res.success else np.zeros(n))
            day_to = turnover(current_w, old_w)
            w_prev = current_w.copy()
        net = float(current_w @ returns.loc[date].values - cost_rate * day_to)
        rows.append({"net_return": net, "loss": -net, "turnover": day_to})
    return pd.DataFrame(rows)


def main() -> None:
    alpha = 0.05
    print("=== 1. RCVaR(kappa=1) vs CVaR on random loss vectors (n=252) ===")
    df = compare_risk_measures()
    print(df[["diff", "rel_diff"]].describe().to_string())
    print(f"mean rel diff: {df['rel_diff'].mean():.6f}")

    sample = verify_degeneracy(np.random.randn(252) * 0.02, alpha)
    print("verify_degeneracy sample:", sample)

    print("\n=== 2. Load SP100, compare rebalance optimizations ===")
    cfg_path = ROOT / "configs" / "sp100.yaml"
    bundle = load_sp100_universe(cfg_path, ROOT / "data" / "processed" / "sp100")
    returns = bundle["returns"]
    states = build_state_matrix(returns)
    learned = _learn_params(returns, states, bundle["config"])
    test_start, test_end = bundle["config"]["splits"]["test"]
    window = 252
    cost = 0.001
    maxiter = 50

    rebal = [d for d in monthly_rebalance_dates(returns.index) if test_start <= str(d.date()) <= test_end]
    idx_map = {d: i for i, d in enumerate(returns.index)}
    w_prev = np.full(returns.shape[1], 1.0 / returns.shape[1])
    comp_rows = []
    for date in rebal:
        loc = idx_map[date]
        if loc < window:
            continue
        hist = returns.iloc[loc - window : loc]
        feat = states.iloc[loc - window : loc]
        row = compare_single_rebalance(hist, w_prev, feat, cost, alpha, maxiter)
        row["date"] = date
        comp_rows.append(row)
        engine = _make_engine("A_no_kappa", bundle["config"], learned)
        w_a, _, _ = optimize_portfolio(hist, feat, w_prev, engine, cost, maxiter)
        w_prev = w_a.copy()
    comp = pd.DataFrame(comp_rows)
    print(f"rebalance dates compared: {len(comp)}")
    print(f"mean w L1 dist (A vs Hist): {comp['w_l1_dist'].mean():.4f}")
    print(f"mean obj_a_rcvar: {comp['obj_a_rcvar'].mean():.6f}")
    print(f"mean obj_a_cvar:  {comp['obj_a_cvar'].mean():.6f}")
    print(f"mean obj_h_cvar:  {comp['obj_h_cvar'].mean():.6f}")

    print("\n=== 3. Full test metrics: V3 results vs fixed Historical ===")
    from robust_cvar_portfolio.src.baselines import historical_cvar_backtest
    from robust_cvar_portfolio.portfolio.rolling import run_rolling

    engine = _make_engine("A_no_kappa", bundle["config"], learned)
    frame_a = run_rolling(returns, states, engine, test_start, test_end, cost, window, maxiter)
    frame_h = historical_cvar_backtest(returns, test_start, test_end, window, cost, alpha, maxiter)
    frame_h_fixed = historical_backtest_fixed_cost(returns, test_start, test_end, window, cost, alpha, maxiter)

    m_a = summarize_backtest(frame_a["net_return"], frame_a["loss"], frame_a["turnover"], alpha)
    m_h = summarize_backtest(frame_h["net_return"], frame_h["loss"], frame_h["turnover"], alpha)
    m_hf = summarize_backtest(frame_h_fixed["net_return"], frame_h_fixed["loss"], frame_h_fixed["turnover"], alpha)

    summary = pd.DataFrame([m_a, m_h, m_hf], index=["A_rolling", "Hist_baseline_buggy_cost", "Hist_fixed_rebal_cost"])
    print(summary[["cvar_5pct", "max_drawdown", "sharpe_ratio", "avg_turnover"]].to_string())

    out = ROOT / "outputs" / "v3" / "sp100_diagnostics" / "a_vs_historical_investigation.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out)
    comp.to_csv(out.parent / "a_vs_historical_rebalance_compare.csv", index=False)
    df.describe().to_csv(out.parent / "rcvar_vs_cvar_measure_diff.csv")
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
