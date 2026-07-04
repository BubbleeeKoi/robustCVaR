"""V6 Task 3: Random 30-stock universes from SP100 pool."""

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
from robust_cvar_portfolio.experiments.equity_common import (
    EQUITY_OUT,
    calibrate_c_stable,
    equity_dir,
    load_v6_config,
    run_equity_models_test,
)
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params
from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle


def _log(msg: str) -> None:
    print(msg, flush=True)
    log_path = EQUITY_OUT / "random30" / "run_log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as f:
        f.write(msg + "\n")


def generate_universes(tickers: list[str], n: int, seeds: list[int]) -> list[list[str]]:
    """One universe per fixed seed (reproducible)."""
    out: list[list[str]] = []
    for s in seeds:
        rng = np.random.default_rng(s)
        out.append(sorted(rng.choice(tickers, size=n, replace=False).tolist()))
    return out


def run_random30(
    n_universes: int | None = None,
    n_assets: int = 30,
    seeds: list[int] | None = None,
) -> None:
    v6_cfg = load_v6_config()
    r30_cfg = v6_cfg["random30"]
    seed_list = seeds or r30_cfg.get("seeds") or [r30_cfg.get("seed", 42)]
    k = n_universes or len(seed_list) or r30_cfg["n_universes"]
    seed_list = list(seed_list)[:k]
    while len(seed_list) < k:
        seed_list.append(int(seed_list[-1]) + 1)
    n_assets = n_assets or r30_cfg["n_assets"]

    out = equity_dir("random30")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(f"=== V6 Random30 ({k} universes × {n_assets} stocks, seeds={seed_list}) ===")

    bundle = load_v5_bundle("sp100")
    full_returns = bundle["returns"]
    config = bundle["config"].copy()
    tickers = list(full_returns.columns)

    universes = generate_universes(tickers, n_assets, seed_list)
    uni_df = pd.DataFrame(
        [
            {"universe_id": i, "seed": seed_list[i], "tickers": ",".join(u)}
            for i, u in enumerate(universes)
        ]
    )
    uni_df.to_csv(out / "random_universe_list.csv", index=False)

    all_rows = []
    param_rows = []

    for uid, uni_tickers in enumerate(universes):
        udir = out / f"universe_{uid:03d}"
        ckpt_summary = udir / "done.json"
        if ckpt_summary.exists():
            _log(f"  skip universe {uid} (checkpoint)")
            with ckpt_summary.open(encoding="utf-8") as f:
                rec = json.load(f)
            all_rows.append(rec["metrics"])
            param_rows.append(rec["params"])
            continue

        _log(f"  universe {uid}/{k-1}: {uni_tickers[:5]}... ({len(uni_tickers)} stocks)")
        udir.mkdir(parents=True, exist_ok=True)
        returns = full_returns[uni_tickers].copy()
        states = build_state_matrix(returns)
        learned = _learn_params(returns, states, config)
        cfg = config.copy()
        cfg["optimizer_maxiter"] = min(cfg.get("optimizer_maxiter", 50), 50)

        sel = calibrate_c_stable(returns, states, cfg, learned, v6_cfg)
        sel["kappa_grid"].to_csv(udir / "kappa_grid.csv", index=False)
        sel["cap_grid"].to_csv(udir / "cap_grid.csv", index=False)
        sel["rho_grid"].to_csv(udir / "rho_grid.csv", index=False)
        stable = {key: sel[key] for key in ("kappa_max", "weight_cap", "rho")}
        with (udir / "selected_params.json").open("w", encoding="utf-8") as f:
            json.dump(stable, f, indent=2)

        table = run_equity_models_test(returns, states, cfg, learned, stable, udir)
        table["universe_id"] = uid
        all_rows.extend(table.to_dict("records"))

        row = table[table["method"] == "C_stable"].iloc[0]
        a_row = table[table["method"] == "A_ceil_CVaR"].iloc[0]
        b_row = table[table["method"] == "B_fixed_kappa"].iloc[0]
        rec = {
            "universe_id": uid,
            "seed": seed_list[uid],
            "A_cvar": float(a_row["cvar_5pct"]),
            "B_cvar": float(b_row["cvar_5pct"]),
            "C_stable_cvar": float(row["cvar_5pct"]),
            "delta_A": float(a_row["cvar_5pct"] - row["cvar_5pct"]),
            "delta_B": float(b_row["cvar_5pct"] - row["cvar_5pct"]),
            "win_vs_A": bool(row["cvar_5pct"] < a_row["cvar_5pct"]),
            "win_vs_B": bool(row["cvar_5pct"] < b_row["cvar_5pct"]),
            **stable,
        }
        param_rows.append({**rec, "tickers": ",".join(uni_tickers)})
        with ckpt_summary.open("w", encoding="utf-8") as f:
            json.dump({"params": param_rows[-1], "metrics": table.to_dict("records")}, f, indent=2)

    all_df = pd.DataFrame(all_rows)
    all_df.to_csv(out / "random30_all_results.csv", index=False)
    param_df = pd.DataFrame(param_rows)
    param_df.to_csv(out / "random30_selected_params.csv", index=False)

    summary = {
        "n_universes": k,
        "n_assets": n_assets,
        "seeds": ",".join(str(s) for s in seed_list),
        "win_rate_A": float(param_df["win_vs_A"].mean()),
        "win_rate_B": float(param_df["win_vs_B"].mean()),
        "mean_delta_A": float(param_df["delta_A"].mean()),
        "median_delta_A": float(param_df["delta_A"].median()),
        "mean_delta_B": float(param_df["delta_B"].mean()),
        "median_delta_B": float(param_df["delta_B"].median()),
        "worst_quartile_delta_A": float(param_df["delta_A"].quantile(0.25)),
        "elapsed_sec": time.time() - t0,
    }
    pd.DataFrame([summary]).to_csv(out / "random30_summary.csv", index=False)

    paper = EQUITY_OUT / "paper_tables"
    paper.mkdir(parents=True, exist_ok=True)
    paper_tbl = pd.DataFrame([summary])
    paper_tbl.to_csv(paper / "table2_random30_robustness.csv", index=False)

    fig_dir = out / "figures"
    fig_dir.mkdir(exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.hist(param_df["delta_A"] * 100, bins=15, edgecolor="black", alpha=0.7)
    plt.axvline(0, color="red", ls="--")
    plt.xlabel("CVaR improvement vs A (pp)")
    plt.ylabel("Count")
    plt.title(f"Random30: ΔCVaR(C_stable - A), WinRate={summary['win_rate_A']:.1%}")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_random30_cvar_improvement_hist.png", dpi=150)
    plt.close()

    plt.figure(figsize=(5, 4))
    plt.bar(["vs A", "vs B"], [summary["win_rate_A"], summary["win_rate_B"]])
    plt.ylim(0, 1)
    plt.ylabel("Win rate")
    plt.title("Random30 C_stable win rate")
    plt.tight_layout()
    plt.savefig(fig_dir / "fig_random30_winrate.png", dpi=150)
    plt.close()

    _log("\n=== Random30 Summary ===")
    _log(f"  WinRate vs A: {summary['win_rate_A']:.1%}")
    _log(f"  WinRate vs B: {summary['win_rate_B']:.1%}")
    _log(f"  mean Δ_A: {summary['mean_delta_A']*100:.2f} pp")
    _log(f"  median Δ_A: {summary['median_delta_A']*100:.2f} pp")
    _log(f"Output: {out}")
    _log(f"Done in {(time.time()-t0)/60:.1f} min")

    from robust_cvar_portfolio.experiments.run_equity_bootstrap import run_equity_bootstrap
    from robust_cvar_portfolio.experiments.update_v6_html_results import update_v6_html

    run_equity_bootstrap()
    update_v6_html()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-universes", type=int, default=None)
    parser.add_argument("--n-assets", type=int, default=30)
    parser.add_argument("--seeds", type=str, default=None, help="comma-separated, e.g. 42,123,456")
    args = parser.parse_args()
    seed_list = [int(x) for x in args.seeds.split(",")] if args.seeds else None
    run_random30(args.n_universes, args.n_assets, seed_list)
