"""V7-B step 2: Backtest correlation-stratified universes."""

from __future__ import annotations

import argparse
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
from robust_cvar_portfolio.experiments.equity_common import calibrate_c_stable, load_v6_config, run_equity_models_test
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params
from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle
from robust_cvar_portfolio.experiments.v7_common import paper_dir, v7_dir


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_corr_stratified_backtest(n_per_group: int = 3) -> None:
    out = v7_dir("corr_stratified")
    sel_path = out / "selected_low_mid_high_universes.csv"
    if not sel_path.exists():
        raise FileNotFoundError(f"Run generate first: {sel_path}")

    selected = pd.read_csv(sel_path)
    v6_cfg = load_v6_config()
    bundle = load_v5_bundle("sp100")
    full_returns = bundle["returns"]
    config = bundle["config"].copy()
    config["optimizer_maxiter"] = min(config.get("optimizer_maxiter", 50), 50)

    t0 = time.time()
    _log(f"=== V7-B Corr-stratified backtest ({len(selected)} universes) ===")

    all_rows = []
    param_rows = []

    for _, row in selected.iterrows():
        uid = int(row["universe_id"])
        group = row["corr_group"]
        udir = out / f"universe_{group}_{row['group_rank']:02d}"
        ckpt = udir / "done.json"
        tickers = row["tickers"].split(",")

        if ckpt.exists():
            _log(f"  skip {group}/{row['group_rank']} (checkpoint)")
            with ckpt.open(encoding="utf-8") as f:
                rec = json.load(f)
            all_rows.extend(rec["metrics"])
            param_rows.append(rec["params"])
            continue

        _log(f"  {group} rank {row['group_rank']}: corr={row['avg_corr_val']:.3f}")
        udir.mkdir(parents=True, exist_ok=True)
        returns = full_returns[tickers].copy()
        states = build_state_matrix(returns)
        learned = _learn_params(returns, states, config)

        sel = calibrate_c_stable(returns, states, config, learned, v6_cfg)
        stable = {k: sel[k] for k in ("kappa_max", "weight_cap", "rho")}
        with (udir / "selected_params.json").open("w", encoding="utf-8") as f:
            json.dump(stable, f, indent=2)

        table = run_equity_models_test(returns, states, config, learned, stable, udir)
        table["universe_id"] = uid
        table["corr_group"] = group
        table["avg_corr_val"] = float(row["avg_corr_val"])
        all_rows.extend(table.to_dict("records"))

        c_row = table[table["method"] == "C_stable"].iloc[0]
        a_row = table[table["method"] == "A_ceil_CVaR"].iloc[0]
        rec = {
            "universe_id": uid,
            "corr_group": group,
            "avg_corr_val": float(row["avg_corr_val"]),
            "A_cvar": float(a_row["cvar_5pct"]),
            "C_stable_cvar": float(c_row["cvar_5pct"]),
            "delta_A": float(a_row["cvar_5pct"] - c_row["cvar_5pct"]),
            "win_vs_A": bool(c_row["cvar_5pct"] < a_row["cvar_5pct"]),
            **stable,
        }
        param_rows.append(rec)
        with ckpt.open("w", encoding="utf-8") as f:
            json.dump({"params": rec, "metrics": table.to_dict("records")}, f, indent=2)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out / "results_all.csv", index=False)
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(out / "selected_params.csv", index=False)

    group_rows = []
    for g in ["low", "mid", "high"]:
        sub = param_df[param_df["corr_group"] == g]
        if sub.empty:
            continue
        group_rows.append(
            {
                "corr_group": g,
                "n": len(sub),
                "mean_avg_corr_val": float(sub["avg_corr_val"].mean()),
                "win_rate_A": float(sub["win_vs_A"].mean()),
                "mean_delta_A_pp": float(sub["delta_A"].mean() * 100),
                "median_delta_A_pp": float(sub["delta_A"].median() * 100),
            }
        )
    group_df = pd.DataFrame(group_rows)
    group_df.to_csv(out / "group_summary.csv", index=False)
    group_df.to_csv(paper_dir() / "table_v7_corr_stratified.csv", index=False)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)

    if not param_df.empty:
        plt.figure(figsize=(7, 5))
        plt.scatter(param_df["avg_corr_val"], param_df["delta_A"] * 100, c=param_df["corr_group"].map(
            {"low": "#10b981", "mid": "#2563eb", "high": "#dc2626"}
        ), s=80)
        plt.axhline(0, color="gray", ls="--")
        plt.xlabel("Validation avg correlation")
        plt.ylabel("Δ CVaR vs A (pp)")
        plt.title("V7-B: Correlation vs C_stable improvement")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_corr_vs_cvar_improvement.png", dpi=150)
        plt.close()

        plt.figure(figsize=(6, 4))
        plt.bar(group_df["corr_group"], group_df["mean_delta_A_pp"])
        plt.axhline(0, color="red", ls="--")
        plt.ylabel("Mean Δ_A (pp)")
        plt.title("V7-B: Group mean CVaR improvement")
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_group_cvar_comparison.png", dpi=150)
        plt.close()

    sp30_val_corr = None
    struct_path = v7_dir("structure") / "universe_structure_summary.csv"
    if struct_path.exists():
        st = pd.read_csv(struct_path)
        sp30_row = st[st["universe"] == "SP30"]
        if not sp30_row.empty:
            sp30_val_corr = float(sp30_row["avg_correlation_val"].iloc[0])
            plt.figure(figsize=(7, 4))
            plt.hist(param_df["avg_corr_val"], bins=12, alpha=0.5, label="Stratified universes")
            plt.axvline(sp30_val_corr, color="blue", ls="--", label=f"SP30 ({sp30_val_corr:.3f})")
            plt.xlabel("Validation avg correlation")
            plt.ylabel("Count")
            plt.title("SP30 position in correlation space")
            plt.legend()
            plt.tight_layout()
            plt.savefig(fig_dir / "fig_sp30_position_in_corr_space.png", dpi=150)
            plt.close()

    _log("\n=== Group Summary ===")
    _log(group_df.to_string(index=False))
    _log(f"Done in {(time.time()-t0)/60:.1f} min")
    _log(f"Output: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-per-group", type=int, default=3)
    args = parser.parse_args()
    run_corr_stratified_backtest(args.n_per_group)
