"""V7-B step 1: Generate correlation-stratified Random30 universes."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle
from robust_cvar_portfolio.experiments.v7_common import (
    load_v7_config,
    pc1_share_from_block,
    effective_dimension_from_block,
    universe_avg_corr_val,
    v7_dir,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def generate_candidates(
    tickers: list[str],
    full_returns: pd.DataFrame,
    config: dict,
    n_candidates: int,
    n_assets: int,
    base_seed: int,
    window: int,
) -> pd.DataFrame:
    val_start, val_end = config["splits"]["val"]
    val_sub = full_returns.loc[val_start:val_end]
    rows = []
    for k in range(n_candidates):
        rng = np.random.default_rng(base_seed + k)
        uni = sorted(rng.choice(tickers, size=n_assets, replace=False).tolist())
        sub = full_returns[uni].loc[val_start:val_end]
        block = sub.values[-window:] if len(sub) > window else sub.values
        rows.append(
            {
                "universe_id": k,
                "seed": base_seed + k,
                "tickers": ",".join(uni),
                "n_assets": n_assets,
                "avg_corr_val": universe_avg_corr_val(full_returns[uni], config, window),
                "pc1_share_val": pc1_share_from_block(block),
                "effdim_val": effective_dimension_from_block(block),
            }
        )
    return pd.DataFrame(rows).sort_values("avg_corr_val").reset_index(drop=True)


def select_groups(candidates: pd.DataFrame, n_per_group: int) -> pd.DataFrame:
    n = len(candidates)
    mid = n // 2
    low = candidates.head(n_per_group).copy()
    low["corr_group"] = "low"
    high = candidates.tail(n_per_group).copy()
    high["corr_group"] = "high"
    start = max(0, mid - n_per_group // 2)
    mid_grp = candidates.iloc[start : start + n_per_group].copy()
    mid_grp["corr_group"] = "mid"
    selected = pd.concat([low, mid_grp, high], ignore_index=True)
    selected["group_rank"] = selected.groupby("corr_group").cumcount()
    return selected


def run_generate(
    n_candidates: int = 300,
    n_assets: int = 30,
    n_per_group: int = 3,
    source: str = "sp100",
) -> None:
    v7_cfg = load_v7_config()
    cs = v7_cfg.get("corr_stratified", {})
    n_candidates = n_candidates or cs.get("n_candidates", 300)
    n_assets = n_assets or cs.get("n_assets", 30)
    n_per_group = n_per_group or cs.get("n_per_group", 3)
    base_seed = cs.get("base_seed", 42)
    window = v7_cfg.get("structure_window", 60)

    out = v7_dir("corr_stratified")
    out.mkdir(parents=True, exist_ok=True)
    _log(f"=== V7-B Generate corr-stratified universes (K={n_candidates}, {n_per_group}×3) ===")

    bundle = load_v5_bundle(source)
    full_returns = bundle["returns"]
    config = bundle["config"]
    tickers = list(full_returns.columns)

    candidates = generate_candidates(
        tickers, full_returns, config, n_candidates, n_assets, base_seed, window
    )
    candidates.to_csv(out / f"candidate_{n_candidates}_universes.csv", index=False)

    selected = select_groups(candidates, n_per_group)
    selected.to_csv(out / "selected_low_mid_high_universes.csv", index=False)

    _log(f"  avg_corr range: [{candidates['avg_corr_val'].min():.3f}, {candidates['avg_corr_val'].max():.3f}]")
    for g in ["low", "mid", "high"]:
        sub = selected[selected["corr_group"] == g]
        _log(f"  {g}: mean corr={sub['avg_corr_val'].mean():.3f}, ids={sub['universe_id'].tolist()}")
    _log(f"Output: {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-candidates", type=int, default=300)
    parser.add_argument("--n-assets", type=int, default=30)
    parser.add_argument("--n-per-group", type=int, default=3)
    parser.add_argument("--source", type=str, default="sp100")
    args = parser.parse_args()
    run_generate(args.n_candidates, args.n_assets, args.n_per_group, args.source)
