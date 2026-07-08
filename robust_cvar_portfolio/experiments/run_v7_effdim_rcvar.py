"""V7-D: Effective-dimension-scaled RCVaR."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.data.loader import build_state_matrix
from robust_cvar_portfolio.experiments.equity_common import calibrate_c_stable, load_v6_config
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params
from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle
from robust_cvar_portfolio.experiments.v7_common import (
    load_v7_config,
    paper_dir,
    run_v7_models_test,
    v7_dir,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_effdim_rcvar(d0: float = 30.0) -> None:
    v7_cfg = load_v7_config()
    d0 = d0 or v7_cfg.get("effdim", {}).get("d0", 30)
    v6_cfg = load_v6_config()
    out = v7_dir("effdim_rcvar")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    t0 = time.time()
    _log(f"=== V7-D Effdim-scaled RCVaR (d0={d0}) ===")

    all_results = []

    for ds in ["sp30", "sp100"]:
        _log(f"  dataset: {ds}")
        bundle = load_v5_bundle(ds)
        returns = bundle["returns"]
        states = bundle["states"]
        config = bundle["config"].copy()
        learned = bundle["learned"]
        config["optimizer_maxiter"] = min(config.get("optimizer_maxiter", 50), 50)
        ds_out = out / ds
        ds_out.mkdir(parents=True, exist_ok=True)
        sel = calibrate_c_stable(returns, states, config, learned, v6_cfg)
        stable = {k: sel[k] for k in ("kappa_max", "weight_cap", "rho")}
        with (ds_out / "selected_params.json").open("w", encoding="utf-8") as f:
            json.dump({**stable, "d0": d0}, f, indent=2)

        table = run_v7_models_test(
            returns, states, config, learned, stable, ds_out, d0,
            models=["A_ceil_CVaR", "B_fixed_kappa", "C_default", "C_stable", "V7_effdim", "V7_effdim_cap"],
        )
        table["dataset"] = ds
        table.to_csv(out / f"{ds}_results.csv", index=False)
        all_results.extend(table.to_dict("records"))

    sel_path = v7_dir("corr_stratified") / "selected_low_mid_high_universes.csv"
    strat_rows = []
    if sel_path.exists():
        selected = pd.read_csv(sel_path)
        bundle = load_v5_bundle("sp100")
        full_returns = bundle["returns"]
        config = bundle["config"].copy()
        config["optimizer_maxiter"] = min(config.get("optimizer_maxiter", 50), 50)

        for _, row in selected.iterrows():
            group = row["corr_group"]
            rank = int(row["group_rank"])
            _log(f"  strat {group}/{rank}: corr={row['avg_corr_val']:.3f}")
            udir = out / f"strat_{group}_{rank:02d}"
            udir.mkdir(parents=True, exist_ok=True)
            tickers = row["tickers"].split(",")
            returns = full_returns[tickers].copy()
            states = build_state_matrix(returns)
            learned = _learn_params(returns, states, config)
            sel = calibrate_c_stable(returns, states, config, learned, v6_cfg)
            stable = {k: sel[k] for k in ("kappa_max", "weight_cap", "rho")}

            table = run_v7_models_test(
                returns, states, config, learned, stable, udir, d0,
                models=["C_stable", "V7_effdim", "V7_effdim_cap", "A_ceil_CVaR"],
            )
            for rec in table.to_dict("records"):
                rec["corr_group"] = group
                rec["avg_corr_val"] = float(row["avg_corr_val"])
                strat_rows.append(rec)

    strat_df = pd.DataFrame(strat_rows)
    if not strat_df.empty:
        strat_df.to_csv(out / "corr_stratified_results.csv", index=False)

    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(out / "summary.csv", index=False)
    summary_df.to_csv(paper_dir() / "table_v7_effdim_results.csv", index=False)

    if not summary_df.empty:
        focus = summary_df[summary_df["method"].isin(["A_ceil_CVaR", "C_stable", "V7_effdim", "V7_effdim_cap"])]
        pivot = focus.pivot(index="dataset", columns="method", values="cvar_5pct") * 100
        pivot.plot(kind="bar", figsize=(8, 4))
        plt.ylabel("Test CVaR 5% (%)")
        plt.title("V7-D vs V6 on SP30 / SP100")
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_v7_vs_v6_cvar.png", dpi=150)
        plt.close()

    _log("\n=== SP30 / SP100 Summary ===")
    sub = summary_df[summary_df["method"].isin(["C_stable", "V7_effdim", "V7_effdim_cap", "A_ceil_CVaR"])]
    _log(sub[["dataset", "method", "cvar_5pct"]].to_string(index=False))

    sp30 = summary_df[(summary_df["dataset"] == "sp30") & (summary_df["method"] == "C_stable")]
    sp30_v7 = summary_df[(summary_df["dataset"] == "sp30") & (summary_df["method"] == "V7_effdim_cap")]
    if not sp30.empty and not sp30_v7.empty:
        delta_pp = (float(sp30_v7["cvar_5pct"].iloc[0]) - float(sp30["cvar_5pct"].iloc[0])) * 100
        _log(f"  SP30 V7_effdim_cap - C_stable: {delta_pp:+.2f} pp (limit +0.05 pp)")

    sp100_a = summary_df[(summary_df["dataset"] == "sp100") & (summary_df["method"] == "A_ceil_CVaR")]
    for m in ["V7_effdim", "V7_effdim_cap"]:
        sp100_v = summary_df[(summary_df["dataset"] == "sp100") & (summary_df["method"] == m)]
        if not sp100_a.empty and not sp100_v.empty:
            beats_a = float(sp100_v["cvar_5pct"].iloc[0]) < float(sp100_a["cvar_5pct"].iloc[0])
            _log(f"  SP100 {m} beats A: {beats_a}")

    _log(f"Done in {(time.time()-t0)/60:.1f} min")
    _log(f"Output: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--d0", type=float, default=30)
    args = parser.parse_args()
    run_effdim_rcvar(args.d0)
