"""V5 Step 1–3: validation calibration (kappa_max, cap, rho) — staged grid search."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.v5_common import (
    engine_manual,
    load_selected_params,
    load_v5_bundle,
    load_v5_config,
    parse_cap,
    parse_rho,
    save_selected_params,
    validation_objective,
    v5_out_dir,
    run_val_backtest,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _select_min(rows: list[dict], key: str = "J_val") -> dict:
    return min(rows, key=lambda r: r[key])


def calibrate_dataset(dataset: str, skip_cap: bool = False, skip_rho: bool = False) -> dict:
    v5_cfg = load_v5_config()
    bundle = load_v5_bundle(dataset)
    config = bundle["config"]
    returns = bundle["returns"]
    states = bundle["states"]
    learned = bundle["learned"]

    val_dir = v5_out_dir(dataset) / "validation"
    val_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    _log(f"=== V5 Calibration ({dataset}) ===")

    # Stage 1: kappa_max
    km_rows = []
    best_km = v5_cfg["kappa_max_grid"][0]
    best_j = float("inf")
    for km in v5_cfg["kappa_max_grid"]:
        _log(f"  [stage1] kappa_max={km} ...")
        engine = engine_manual(config, learned, km)
        _, _, m = run_val_backtest(returns, states, engine, config)
        j = validation_objective(m, m["avg_hhi"], v5_cfg)
        row = {"kappa_max": km, "J_val": j, **m}
        km_rows.append(row)
        if j < best_j:
            best_j = j
            best_km = km
    km_df = pd.DataFrame(km_rows)
    km_df.to_csv(val_dir / "kappa_max_grid.csv", index=False)
    _log(f"  selected kappa_max={best_km} (J_val={best_j:.6f})")

    # Stage 2: weight cap
    cap_rows = []
    best_cap = None
    best_j_cap = float("inf")
    if not skip_cap:
        for cap_raw in v5_cfg["weight_cap_grid"]:
            cap = parse_cap(cap_raw)
            label = "none" if cap is None else f"{cap:.2f}"
            _log(f"  [stage2] weight_cap={label} ...")
            engine = engine_manual(config, learned, best_km)
            _, _, m = run_val_backtest(returns, states, engine, config, weight_cap=cap)
            j = validation_objective(m, m["avg_hhi"], v5_cfg)
            row = {"kappa_max": best_km, "weight_cap": cap, "J_val": j, **m}
            cap_rows.append(row)
            if j < best_j_cap:
                best_j_cap = j
                best_cap = cap
        cap_df = pd.DataFrame(cap_rows)
        cap_df.to_csv(val_dir / "cap_grid.csv", index=False)
        _log(f"  selected weight_cap={best_cap} (J_val={best_j_cap:.6f})")
    else:
        best_cap = None

    # Stage 3: kappa smoothing rho
    rho_rows = []
    best_rho = None
    best_j_rho = float("inf")
    if not skip_rho:
        for rho_raw in v5_cfg["rho_grid"]:
            rho = parse_rho(rho_raw)
            label = "none" if rho is None else f"{rho:.1f}"
            _log(f"  [stage3] rho={label} ...")
            engine = engine_manual(config, learned, best_km)
            _, _, m = run_val_backtest(
                returns, states, engine, config, weight_cap=best_cap, kappa_rho=rho
            )
            j = validation_objective(m, m["avg_hhi"], v5_cfg)
            row = {
                "kappa_max": best_km,
                "weight_cap": best_cap,
                "rho": rho,
                "J_val": j,
                **m,
            }
            rho_rows.append(row)
            if j < best_j_rho:
                best_j_rho = j
                best_rho = rho
        rho_df = pd.DataFrame(rho_rows)
        rho_df.to_csv(val_dir / "smoothing_grid.csv", index=False)
        _log(f"  selected rho={best_rho} (J_val={best_j_rho:.6f})")
    else:
        best_rho = None

    selected = {
        "dataset": dataset,
        "kappa_max": best_km,
        "weight_cap": best_cap,
        "rho": best_rho,
        "default_kappa_max": config.get("kappa_max", 1.0),
        "selection_objective": v5_cfg["selection_objective"],
        "elapsed_sec": time.time() - t0,
    }
    save_selected_params(val_dir / "selected_params.json", selected)
    _log(f"  saved {val_dir / 'selected_params.json'}")
    _log(f"Done in {(time.time()-t0)/60:.1f} min")
    return selected


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sp100")
    parser.add_argument("--skip-cap", action="store_true")
    parser.add_argument("--skip-rho", action="store_true")
    args = parser.parse_args()
    calibrate_dataset(args.dataset, skip_cap=args.skip_cap, skip_rho=args.skip_rho)
