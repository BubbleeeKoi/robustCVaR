"""V6 index supplement experiment shared utilities."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_cvar_portfolio.data.index_universe import build_index_panel, rebalance_constituent_log
from robust_cvar_portfolio.experiments.equity_common import (
    calibrate_c_stable,
    cross_sectional_metrics,
    effective_n,
    load_v6_config,
    v6_objective,
)
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params
from robust_cvar_portfolio.experiments.v5_common import (
    ROOT,
    avg_weight_hhi,
    engine_fixed,
    engine_historical,
    engine_manual,
    metrics_row,
    parse_cap,
    parse_rho,
)
from robust_cvar_portfolio.portfolio.rolling_pit import run_equal_weight_pit, run_rolling_pit
from robust_cvar_portfolio.src.baselines import spy_benchmark_returns
from robust_cvar_portfolio.src.risk_metrics import summarize_backtest

INDEX_OUT = ROOT / "outputs" / "v6_index_precheck"

INDEX_MODELS = [
    ("Index_ETF", "benchmark"),
    ("Equal_Weight", "benchmark"),
    ("A_ceil_CVaR", "baseline"),
    ("B_fixed_kappa", "baseline"),
    ("C_stable", "proposed"),
]


def index_dir(name: str) -> Path:
    return INDEX_OUT / name


def _log(out: Path, msg: str) -> None:
    print(msg, flush=True)
    log_path = out / "run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def run_pit_val_backtest(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    engine,
    config: dict,
    constituents_at,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    val_start, val_end = config["splits"]["val"]
    result = run_rolling_pit(
        returns,
        states,
        engine,
        constituents_at,
        val_start,
        val_end,
        config.get("cost_rate", 0.001),
        config.get("estimation_window", 252),
        config.get("optimizer_maxiter", 50),
        weight_cap=weight_cap,
        kappa_rho=kappa_rho,
        record_weights=True,
        min_data_frac=config.get("min_data_frac", 0.8),
    )
    frame, weights = result  # type: ignore[misc]
    tickers = [c for c in weights.columns if c not in {"date", "kappa", "turnover", "n_used"}]
    m = summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], config.get("alpha", 0.05))
    m["avg_hhi"] = avg_weight_hhi(weights, tickers) if tickers else 0.0
    m["avg_n_used"] = float(weights["n_used"].mean()) if "n_used" in weights.columns else np.nan
    return frame, weights, m


def run_pit_test_backtest(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    engine,
    config: dict,
    constituents_at,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
    record_diagnostics: bool = False,
    record_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, ...]:
    test_start, test_end = config["splits"]["test"]
    return run_rolling_pit(
        returns,
        states,
        engine,
        constituents_at,
        test_start,
        test_end,
        config.get("cost_rate", 0.001),
        config.get("estimation_window", 252),
        config.get("optimizer_maxiter", 50),
        weight_cap=weight_cap,
        kappa_rho=kappa_rho,
        record_diagnostics=record_diagnostics,
        record_weights=record_weights,
        min_data_frac=config.get("min_data_frac", 0.8),
    )


def calibrate_c_stable_pit(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    config: dict,
    learned,
    v6_cfg: dict,
    constituents_at,
) -> dict[str, Any]:
    km_grid = v6_cfg["kappa_max_grid"]
    cap_grid = [parse_cap(x) for x in v6_cfg["weight_cap_grid"]]
    rho_grid = [parse_rho(x) for x in v6_cfg["rho_grid"]]

    best_km, best_j = km_grid[0], float("inf")
    km_rows = []
    for km in km_grid:
        print(f"    val kappa_max={km}...", flush=True)
        engine = engine_manual(config, learned, km)
        _, _, m = run_pit_val_backtest(returns, states, engine, config, constituents_at)
        j = v6_objective(m, m["avg_hhi"], v6_cfg)
        km_rows.append({"kappa_max": km, "J_val": j, **m})
        if j < best_j:
            best_j, best_km = j, km

    best_cap, best_j_cap = None, float("inf")
    cap_rows = []
    for cap in cap_grid:
        engine = engine_manual(config, learned, best_km)
        _, _, m = run_pit_val_backtest(
            returns, states, engine, config, constituents_at, weight_cap=cap
        )
        j = v6_objective(m, m["avg_hhi"], v6_cfg)
        cap_rows.append({"kappa_max": best_km, "weight_cap": cap, "J_val": j, **m})
        if j < best_j_cap:
            best_j_cap, best_cap = j, cap

    best_rho, best_j_rho = None, float("inf")
    rho_rows = []
    for rho in rho_grid:
        engine = engine_manual(config, learned, best_km)
        _, _, m = run_pit_val_backtest(
            returns,
            states,
            engine,
            config,
            constituents_at,
            weight_cap=best_cap,
            kappa_rho=rho,
        )
        j = v6_objective(m, m["avg_hhi"], v6_cfg)
        rho_rows.append(
            {"kappa_max": best_km, "weight_cap": best_cap, "rho": rho, "J_val": j, **m}
        )
        if j < best_j_rho:
            best_j_rho, best_rho = j, rho

    return {
        "kappa_max": best_km,
        "weight_cap": best_cap,
        "rho": best_rho,
        "kappa_grid": pd.DataFrame(km_rows),
        "cap_grid": pd.DataFrame(cap_rows),
        "rho_grid": pd.DataFrame(rho_rows),
    }


def run_index_benchmark(config: dict) -> pd.DataFrame:
    test_start, test_end = config["splits"]["test"]
    ticker = config["benchmark_ticker"]
    rets = spy_benchmark_returns(ticker, test_start, test_end)
    frame = pd.DataFrame(
        {
            "date": rets.index,
            "net_return": rets.values,
            "loss": -rets.values,
            "turnover": 0.0,
        }
    )
    return frame


def weight_stats_from_pit(weights_path: Path) -> dict[str, float]:
    if not weights_path.exists():
        return {"avg_hhi": np.nan, "effective_holdings": np.nan, "max_weight": np.nan}
    w = pd.read_csv(weights_path)
    tickers = [c for c in w.columns if c not in {"date", "kappa", "turnover", "n_used"}]
    if not tickers:
        return {"avg_hhi": np.nan, "effective_holdings": np.nan, "max_weight": np.nan}
    arr = w[tickers].values
    hhi = float(np.mean(np.sum(arr**2, axis=1)))
    return {
        "avg_hhi": hhi,
        "effective_holdings": effective_n(hhi),
        "max_weight": float(np.max(arr)),
    }


def run_index_experiment(name: str, force_data: bool = False) -> dict[str, Any]:
    out = index_dir(name)
    out.mkdir(parents=True, exist_ok=True)
    done_path = out / "done.json"
    if done_path.exists() and not force_data:
        with done_path.open(encoding="utf-8") as f:
            return json.load(f)

    t0 = time.time()
    v6_cfg = load_v6_config()
    _log(out, f"=== V6 Index {name.upper()} ===")

    bundle = build_index_panel(name, force=force_data)
    config = bundle["config"]
    returns = bundle["returns"]
    states = bundle["states"]
    history = bundle["history"]
    constituents_at = bundle["constituents_at"]

    proxy_returns = bundle["proxy_returns"]
    learned = _learn_params(proxy_returns, states, config)

    const_log = rebalance_constituent_log(
        returns,
        history,
        config["splits"]["test"][0],
        config["splits"]["test"][1],
        config.get("estimation_window", 252),
        config.get("min_data_frac", 0.8),
    )
    const_log["index_name"] = name.upper()
    const_log["benchmark_ticker"] = config["benchmark_ticker"]
    const_log.to_csv(out / "constituent_log.csv", index=False)

    params_path = out / "selected_params.json"
    if params_path.exists():
        _log(out, "  skip calibration (selected_params.json exists)")
        with params_path.open(encoding="utf-8") as f:
            stable = json.load(f)
    else:
        _log(out, "  calibrating C_stable on validation...")
        sel = calibrate_c_stable_pit(returns, states, config, learned, v6_cfg, constituents_at)
        sel["kappa_grid"].to_csv(out / "kappa_grid.csv", index=False)
        sel["cap_grid"].to_csv(out / "cap_grid.csv", index=False)
        sel["rho_grid"].to_csv(out / "rho_grid.csv", index=False)
        stable = {k: sel[k] for k in ("kappa_max", "weight_cap", "rho")}
        with params_path.open("w", encoding="utf-8") as f:
            json.dump(stable, f, indent=2)

    specs = [
        ("Index_ETF", None, None, None, False),
        ("Equal_Weight", None, None, None, False),
        ("A_ceil_CVaR", engine_historical(config), None, None, True),
        ("B_fixed_kappa", engine_fixed(config), None, None, True),
        (
            "C_stable",
            engine_manual(config, learned, stable["kappa_max"]),
            stable.get("weight_cap"),
            stable.get("rho"),
            True,
        ),
    ]

    rows = []
    record_diag = name == "sp500"
    for method, engine, cap, rho, is_opt in specs:
        ckpt = out / f"rolling_{method}.csv"
        if ckpt.exists():
            _log(out, f"  skip test (checkpoint): {method}")
            frame = pd.read_csv(ckpt, parse_dates=["date"])
        elif method == "Index_ETF":
            frame = run_index_benchmark(config)
            frame.to_csv(ckpt, index=False)
        elif method == "Equal_Weight":
            frame = run_equal_weight_pit(
                returns,
                constituents_at,
                config["splits"]["test"][0],
                config["splits"]["test"][1],
                config.get("cost_rate", 0.001),
                config.get("estimation_window", 252),
                config.get("min_data_frac", 0.8),
            )
            frame.to_csv(ckpt, index=False)
        else:
            _log(out, f"  test backtest: {method}...")
            t_m = time.time()
            result = run_pit_test_backtest(
                returns,
                states,
                engine,
                config,
                constituents_at,
                weight_cap=cap,
                kappa_rho=rho,
                record_diagnostics=record_diag and method == "C_stable",
                record_weights=record_diag and method == "C_stable",
            )
            if record_diag and method == "C_stable":
                frame, weights, diag = result  # type: ignore[misc]
                diag.to_csv(out / "solve_diagnostics.csv", index=False)
                weights.to_csv(out / f"weights_{method}.csv", index=False)
            else:
                frame = result  # type: ignore[assignment]
            frame.to_csv(ckpt, index=False)
            _log(out, f"  finished {method} in {(time.time()-t_m)/60:.1f} min")

        row = metrics_row(frame, config, method, group="index_v6")
        if method == "C_stable":
            ws = weight_stats_from_pit(out / f"weights_{method}.csv")
            row.update(ws)
        rows.append(row)

    table = pd.DataFrame(rows)
    table.to_csv(out / "table_main.csv", index=False)

    mid = pd.Timestamp(config["splits"]["test"][0]) + (
        pd.Timestamp(config["splits"]["test"][1]) - pd.Timestamp(config["splits"]["test"][0])
    ) / 2
    mid_tickers = constituents_at(mid)
    cols = [c for c in mid_tickers if c in returns.columns][: min(60, config.get("target_n", 100))]
    sub = returns.loc[config["splits"]["test"][0] : config["splits"]["test"][1], cols].copy()
    sub = sub.loc[:, sub.notna().mean() >= 0.8].fillna(0.0)
    if sub.shape[1] >= 5 and len(sub) > 60:
        struct = cross_sectional_metrics(sub, window=60)
    else:
        struct = pd.DataFrame()
    struct_summary = {
        "avg_correlation": float(struct["avg_correlation"].mean()) if not struct.empty else np.nan,
        "pc1_share": float(struct["pc1_share"].mean()) if not struct.empty else np.nan,
        "d_eff": float(struct["effective_dimension"].mean()) if not struct.empty else np.nan,
        "d_eff_over_N": (
            float(struct["effective_dimension"].mean()) / max(len(cols), 1)
            if not struct.empty
            else np.nan
        ),
        "universe_turnover_mean": float(const_log["universe_turnover"].mean()),
    }
    pd.DataFrame([struct_summary]).to_csv(out / "structure_summary.csv", index=False)

    a_cvar = float(table.loc[table["method"] == "A_ceil_CVaR", "cvar_5pct"].iloc[0])
    c_cvar = float(table.loc[table["method"] == "C_stable", "cvar_5pct"].iloc[0])
    idx_cvar = float(table.loc[table["method"] == "Index_ETF", "cvar_5pct"].iloc[0])

    summary = {
        "index": name,
        "elapsed_sec": time.time() - t0,
        "A_cvar": a_cvar,
        "C_stable_cvar": c_cvar,
        "Index_cvar": idx_cvar,
        "delta_C_vs_A": a_cvar - c_cvar,
        "win_C_vs_A": c_cvar < a_cvar,
        "win_C_vs_Index": c_cvar < idx_cvar,
        **stable,
        **struct_summary,
    }
    pd.DataFrame([summary]).to_csv(out / "index_summary.csv", index=False)

    with done_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=str)

    _log(out, f"  Done {name} in {(time.time()-t0)/60:.1f} min")
    return summary
