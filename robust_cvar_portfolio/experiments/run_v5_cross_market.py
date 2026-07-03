"""V5 cross-market: audit + calibration + test for all datasets."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.audit_baselines import audit_baselines
from robust_cvar_portfolio.experiments.run_v5_calibration import calibrate_dataset
from robust_cvar_portfolio.experiments.run_v5_test import run_v5_test
from robust_cvar_portfolio.experiments.v5_common import ROOT as PKG_ROOT, load_v5_config, v5_out_dir


def _log(msg: str) -> None:
    print(msg, flush=True)


def run_cross_market(datasets: list[str] | None = None, skip_audit: bool = False) -> None:
    v5_cfg = load_v5_config()
    datasets = datasets or v5_cfg["datasets"]
    cross_dir = PKG_ROOT / "outputs" / "v5" / "cross_market"
    cross_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    for ds in datasets:
        _log(f"\n{'='*60}\nDataset: {ds}\n{'='*60}")
        if not skip_audit:
            audit_baselines(ds)
        calibrate_dataset(ds)
        run_v5_test(ds)

    param_rows = []
    cvar_rows = []
    turn_rows = []
    paper_rows = []

    for ds in datasets:
        sel_path = v5_out_dir(ds) / "validation" / "selected_params.json"
        test_path = v5_out_dir(ds) / "test" / "table_main.csv"
        if not sel_path.exists() or not test_path.exists():
            continue
        import json

        with sel_path.open(encoding="utf-8") as f:
            sel = json.load(f)
        table = pd.read_csv(test_path)

        from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle

        bundle = load_v5_bundle(ds)
        n = bundle["n_assets"]

        param_rows.append(
            {
                "dataset": ds,
                "N": n,
                "kappa_max": sel["kappa_max"],
                "weight_cap": sel.get("weight_cap"),
                "rho": sel.get("rho"),
            }
        )

        def _cvar(method: str) -> float | None:
            sub = table[table["method"] == method]
            return float(sub["cvar_5pct"].iloc[0]) if len(sub) else None

        a = _cvar("A_Historical_CVaR")
        b = _cvar("B_fixed_kappa")
        cd = _cvar("C_default")
        cc = _cvar("C_calibrated")
        cs = _cvar("C_stable")
        best = min(x for x in [a, b, cc, cs] if x is not None)

        cvar_rows.append({"dataset": ds, "A": a, "B": b, "C_default": cd, "C_calibrated": cc, "C_stable": cs})
        turn_rows.append(
            {
                "dataset": ds,
                "A_turnover": table.loc[table["method"] == "A_Historical_CVaR", "avg_turnover"].iloc[0],
                "C_stable_turnover": table.loc[table["method"] == "C_stable", "avg_turnover"].iloc[0],
            }
        )
        paper_rows.append(
            {
                "dataset": ds,
                "N": n,
                "selected_kappa_max": sel["kappa_max"],
                "selected_cap": sel.get("weight_cap"),
                "selected_rho": sel.get("rho"),
                "A_CVaR": a,
                "B_CVaR": b,
                "C_default_CVaR": cd,
                "C_calibrated_CVaR": cc,
                "C_stable_CVaR": cs,
                "best_among_all": cs == best if cs is not None else False,
            }
        )

    pd.DataFrame(param_rows).to_csv(cross_dir / "selected_params_by_dataset.csv", index=False)
    pd.DataFrame(cvar_rows).to_csv(cross_dir / "test_cvar_by_dataset.csv", index=False)
    pd.DataFrame(turn_rows).to_csv(cross_dir / "test_turnover_by_dataset.csv", index=False)
    pd.DataFrame(paper_rows).to_csv(cross_dir / "final_paper_table.csv", index=False)

    _log(f"\nCross-market summary: {cross_dir}")
    _log(f"Total elapsed: {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="*", default=None)
    parser.add_argument("--skip-audit", action="store_true")
    args = parser.parse_args()
    run_cross_market(args.datasets, skip_audit=args.skip_audit)
