"""Baseline audit: unify CVaR definitions and fair SP100 comparison."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.data.loader import build_state_matrix
from robust_cvar_portfolio.data.sp100_universe import load_sp100_universe
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params
from robust_cvar_portfolio.portfolio.rolling import run_rolling
from robust_cvar_portfolio.portfolio.weight_export import export_rebalance_weights
from robust_cvar_portfolio.risk.kappa import KappaParams
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.backtest import crisis_loss
from robust_cvar_portfolio.src.baselines import historical_cvar_backtest
from robust_cvar_portfolio.src.risk_metrics import summarize_backtest

OUT = ROOT / "outputs" / "v3" / "sp100_baseline_audit"


def _log(msg: str) -> None:
    print(msg, flush=True)


def _engine_frac(config: dict, learned: KappaParams) -> RiskEngine:
    return RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="plain", params=KappaParams())


def _engine_ceil(config: dict) -> RiskEngine:
    return RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="plain_ceil", params=KappaParams())


def _engine_fixed(config: dict) -> RiskEngine:
    return RiskEngine(
        alpha=config.get("alpha", 0.05),
        kappa_mode="fixed",
        params=KappaParams(),
        fixed_k=config.get("fixed_kappa", 2.0),
    )


def _engine_manual(config: dict, learned: KappaParams, kappa_max: float) -> RiskEngine:
    params = KappaParams(
        kappa_max=kappa_max,
        beta_vol=config.get("beta_vol", 1.0),
        beta_dd=config.get("beta_dd", 1.0),
        beta_mom=config.get("beta_mom", 0.5),
        beta_corr=config.get("beta_corr", 0.5),
        beta_conc=config.get("beta_conc", 0.5),
        theta=learned.theta,
    )
    params.theta = [0.0, 0.0, 0.0, 0.0]
    return RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="manual", params=params)


def _metrics(frame: pd.DataFrame, config: dict, name: str, group: str) -> dict:
    m = summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], config.get("alpha", 0.05))
    m["method"] = name
    m["group"] = group
    idx = frame.set_index("date") if "date" in frame.columns else frame
    m["crisis_2020"] = crisis_loss(idx["net_return"], "2020-02-01", "2020-04-30")
    m["crisis_2022"] = crisis_loss(idx["net_return"], "2022-01-01", "2022-12-31")
    return m


def _select_kappa_max(returns, states, config, learned, grid: list[float]) -> tuple[float, pd.DataFrame]:
    val_start, val_end = config["splits"]["val"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)
    rows = []
    best_km, best_cvar = grid[0], float("inf")
    for km in grid:
        _log(f"  val sweep kappa_max={km} ...")
        engine = _engine_manual(config, learned, km)
        frame = run_rolling(returns, states, engine, val_start, val_end, cost, window, maxiter)
        m = summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], config.get("alpha", 0.05))
        cvar = m["cvar_5pct"]
        rows.append({"kappa_max": km, **m})
        if cvar < best_cvar:
            best_cvar = cvar
            best_km = km
    return best_km, pd.DataFrame(rows)


def run_baseline_audit() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log("=== SP100 Baseline Audit (CVaR definition unification) ===")

    cfg_path = ROOT / "configs" / "sp100.yaml"
    bundle = load_sp100_universe(cfg_path, ROOT / "data" / "processed" / "sp100")
    config = bundle["config"]
    returns = bundle["returns"]
    states = build_state_matrix(returns)
    learned = _learn_params(returns, states, config)

    test_start, test_end = config["splits"]["test"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)

    km_grid = [0.5, 0.75, 1.0, 1.25, 1.5]
    _log("\n[1] Validation select kappa_max for C_calibrated")
    best_km, val_df = _select_kappa_max(returns, states, config, learned, km_grid)
    val_df.to_csv(OUT / "validation_kappa_max_sweep.csv", index=False)
    _log(f"  selected kappa_max={best_km}")

    models: list[tuple[str, str, object]] = [
        ("A_frac_CVaR", "internal_ablation", _engine_frac(config, learned)),
        ("A_ceil_CVaR", "internal_ablation", _engine_ceil(config)),
        ("B_fixed_kappa", "internal_ablation", _engine_fixed(config)),
        ("C_default", "internal_ablation", _engine_manual(config, learned, config.get("kappa_max", 1.0))),
        ("C_calibrated", "internal_ablation", _engine_manual(config, learned, best_km)),
    ]

    results: dict[str, pd.DataFrame] = {}
    metrics_rows = []

    for name, group, engine in models:
        ckpt = OUT / f"rolling_{name}.csv"
        if ckpt.exists():
            _log(f"  load checkpoint {name}")
            frame = pd.read_csv(ckpt, parse_dates=["date"])
        else:
            _log(f"  run {name} ...")
            frame = run_rolling(returns, states, engine, test_start, test_end, cost, window, maxiter)
            frame.to_csv(ckpt, index=False)
        results[name] = frame
        metrics_rows.append(_metrics(frame, config, name, group))

    _log("\n[2] Historical_CVaR_fixed")
    hist_ckpt = OUT / "rolling_Historical_CVaR_fixed.csv"
    if hist_ckpt.exists():
        hist = pd.read_csv(hist_ckpt, parse_dates=["date"])
    else:
        hist = historical_cvar_backtest(returns, test_start, test_end, window, cost, config.get("alpha", 0.05), maxiter)
        hist = hist.reset_index().rename(columns={"index": "date"})
        hist.to_csv(hist_ckpt, index=False)
    results["Historical_CVaR_fixed"] = hist
    metrics_rows.append(_metrics(hist, config, "Historical_CVaR_fixed", "external_baseline"))

    w_ckpt = OUT / "weights_Historical_CVaR_fixed.csv"
    if not w_ckpt.exists():
        _log("  export Historical weights ...")
        hist_engine = RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="plain_ceil")
        wexp = export_rebalance_weights(returns, states, hist_engine, test_start, test_end, cost, window, maxiter)
        tickers = [c for c in wexp.columns if c not in {"kappa", "turnover"}]
        wexp[tickers].to_csv(w_ckpt, index_label="date")

    table = pd.DataFrame(metrics_rows)
    table.to_csv(OUT / "fair_comparison_table.csv", index=False)

    summary = {
        "selected_kappa_max_val": best_km,
        "A_frac_cvar": float(table.loc[table["method"] == "A_frac_CVaR", "cvar_5pct"].iloc[0]),
        "A_ceil_cvar": float(table.loc[table["method"] == "A_ceil_CVaR", "cvar_5pct"].iloc[0]),
        "Historical_fixed_cvar": float(table.loc[table["method"] == "Historical_CVaR_fixed", "cvar_5pct"].iloc[0]),
        "C_default_cvar": float(table.loc[table["method"] == "C_default", "cvar_5pct"].iloc[0]),
        "C_calibrated_cvar": float(table.loc[table["method"] == "C_calibrated", "cvar_5pct"].iloc[0]),
        "C_calibrated_beats_A_frac": bool(
            table.loc[table["method"] == "C_calibrated", "cvar_5pct"].iloc[0]
            < table.loc[table["method"] == "A_frac_CVaR", "cvar_5pct"].iloc[0]
        ),
        "elapsed_sec": time.time() - t0,
    }
    with (OUT / "audit_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _log("\n=== Fair Comparison (Test CVaR 5%) ===")
    _log(table[["method", "group", "cvar_5pct", "max_drawdown", "avg_turnover", "sharpe_ratio"]].to_string(index=False))
    _log(f"\nOutput: {OUT}")
    _log(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    run_baseline_audit()
