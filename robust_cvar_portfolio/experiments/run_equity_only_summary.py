"""V6 Task 1-2: Organize SP30/SP100 V5 results + paper tables + SP30 Sharpe check."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.equity_common import (
    EQUITY_OUT,
    MODELS_EQUITY,
    copy_v5_equity_results,
    effective_n,
    equity_dir,
    load_v6_config,
)
from robust_cvar_portfolio.experiments.v5_common import v5_out_dir


def _log(msg: str) -> None:
    print(msg, flush=True)


def _fmt_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    out = df[cols].copy()
    if "cvar_5pct" in out.columns:
        out["cvar_5pct_pct"] = (out["cvar_5pct"] * 100).round(2)
    if "sharpe_ratio" in out.columns:
        out["sharpe_ratio"] = out["sharpe_ratio"].round(3)
    if "annualized_return" in out.columns:
        out["ann_return_pct"] = (out["annualized_return"] * 100).round(2)
    if "max_drawdown" in out.columns:
        out["max_drawdown_pct"] = (out["max_drawdown"] * 100).round(1)
    if "avg_turnover" in out.columns:
        out["avg_turnover"] = out["avg_turnover"].round(4)
    return out


def run_equity_only_summary() -> None:
    v6_cfg = load_v6_config()
    _log("=== V6 Equity-Only Summary ===")

    copy_v5_equity_results(v6_cfg["equity_datasets"])

    paper = EQUITY_OUT / "paper_tables"
    paper.mkdir(parents=True, exist_ok=True)

    # Table 1: SP30 main
    sp30 = pd.read_csv(equity_dir("sp30") / "table_main.csv")
    t1_cols = [
        "method",
        "cvar_5pct",
        "max_drawdown",
        "sharpe_ratio",
        "annualized_return",
        "avg_turnover",
    ]
    t1 = _fmt_table(sp30, t1_cols)
    t1.to_csv(paper / "table1_sp30_main.csv", index=False)
    _log("\n[Table 1] SP30 main results:")
    _log(t1.to_string(index=False))

    # SP30 Sharpe / return trade-off (Task 4 in plan = section 6)
    diag_dir = equity_dir("sp30") / "diagnostics"
    diag_dir.mkdir(exist_ok=True)
    a_row = sp30[sp30["method"] == "A_ceil_CVaR"].iloc[0]
    cs_row = sp30[sp30["method"] == "C_stable"].iloc[0]
    b_row = sp30[sp30["method"] == "B_fixed_kappa"].iloc[0]
    tradeoff = pd.DataFrame(
        [
            {
                "comparison": "C_stable vs A",
                "delta_cvar_pct": (cs_row["cvar_5pct"] - a_row["cvar_5pct"]) * 100,
                "delta_sharpe": cs_row["sharpe_ratio"] - a_row["sharpe_ratio"],
                "delta_ann_return_pct": (cs_row["annualized_return"] - a_row["annualized_return"]) * 100,
                "delta_maxdd_pct": (cs_row["max_drawdown"] - a_row["max_drawdown"]) * 100,
                "delta_turnover": cs_row["avg_turnover"] - a_row["avg_turnover"],
                "interpretation": "CVaR improved; Sharpe higher than A but lower than B",
            },
            {
                "comparison": "C_stable vs B",
                "delta_cvar_pct": (cs_row["cvar_5pct"] - b_row["cvar_5pct"]) * 100,
                "delta_sharpe": cs_row["sharpe_ratio"] - b_row["sharpe_ratio"],
                "delta_ann_return_pct": (cs_row["annualized_return"] - b_row["annualized_return"]) * 100,
                "delta_maxdd_pct": (cs_row["max_drawdown"] - b_row["max_drawdown"]) * 100,
                "delta_turnover": cs_row["avg_turnover"] - b_row["avg_turnover"],
                "interpretation": "C_stable beats B on CVaR; Sharpe lower than B",
            },
        ]
    )
    tradeoff.to_csv(diag_dir / "sp30_sharpe_return_tradeoff.csv", index=False)
    _log("\n[SP30 Sharpe trade-off]")
    _log(tradeoff.to_string(index=False))

    # Table 3: SP100 high-dimensional stress test
    sp100_full_path = equity_dir("sp100") / "table_main_full.csv"
    if not sp100_full_path.exists():
        sp100_full_path = v5_out_dir("sp100") / "test" / "table_main.csv"
    sp100_full = pd.read_csv(sp100_full_path)
    sp100_full["method"] = sp100_full["method"].replace({"A_Historical_CVaR": "A_ceil_CVaR"})
    sp100_show = sp100_full[
        sp100_full["method"].isin(
            ["A_ceil_CVaR", "C_default", "C_cap", "C_stable", "B_fixed_kappa"]
        )
    ].copy()
    t3 = _fmt_table(
        sp100_show,
        ["method", "cvar_5pct", "max_drawdown", "sharpe_ratio", "annualized_return", "avg_turnover"],
    )
    t3.to_csv(paper / "table3_sp100_high_dimensional.csv", index=False)
    _log("\n[Table 3] SP100 high-dimensional:")
    _log(t3.to_string(index=False))

    # Copy selected params
    for ds in v6_cfg["equity_datasets"]:
        src = v5_out_dir(ds) / "validation" / "selected_params.json"
        if src.exists():
            import shutil

            shutil.copy(src, equity_dir(ds) / "selected_params.json")

    summary = {
        "sp30_c_stable_cvar": float(cs_row["cvar_5pct"]),
        "sp30_a_cvar": float(a_row["cvar_5pct"]),
        "sp30_c_stable_sharpe": float(cs_row["sharpe_ratio"]),
        "sp30_a_sharpe": float(a_row["sharpe_ratio"]),
        "sp100_c_stable_cvar": float(
            sp100_full.loc[sp100_full["method"] == "C_stable", "cvar_5pct"].iloc[0]
        ),
        "sp100_a_cvar": float(
            sp100_full.loc[sp100_full["method"] == "A_ceil_CVaR", "cvar_5pct"].iloc[0]
        ),
    }
    pd.DataFrame([summary]).to_csv(EQUITY_OUT / "equity_summary.csv", index=False)
    _log(f"\nOutput: {EQUITY_OUT}")


if __name__ == "__main__":
    run_equity_only_summary()
