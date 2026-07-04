"""V6 Task 5: Block bootstrap significance for equity-only results."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.equity_common import EQUITY_OUT, equity_dir, load_v6_config
from robust_cvar_portfolio.experiments.v3_analysis import paired_cvar_test
from robust_cvar_portfolio.experiments.v5_common import v5_out_dir


def _log(msg: str) -> None:
    print(msg, flush=True)


def _bootstrap_pair(
    frame_a: pd.DataFrame,
    frame_b: pd.DataFrame,
    label: str,
    n_boot: int,
    block_size: int,
) -> dict:
    a = frame_a.set_index("date") if "date" in frame_a.columns else frame_a
    b = frame_b.set_index("date") if "date" in frame_b.columns else frame_b
    aligned = a.join(b, lsuffix="_a", rsuffix="_b", how="inner")
    res = paired_cvar_test(
        aligned["loss_a"].values,
        aligned["loss_b"].values,
        n_boot=n_boot,
        block_size=block_size,
    )
    return {
        "comparison": label,
        "mean_cvar_diff_b_minus_a": res["mean_cvar_diff"],
        "ci_lo": res["ci_lo"],
        "ci_hi": res["ci_hi"],
        "prob_b_lt_a": res["prob_cvar_c_lt_a"],
        "interpretation": "prob_b_lt_a = P(CVaR_B < CVaR_A)",
    }


def run_equity_bootstrap() -> None:
    v6_cfg = load_v6_config()
    boot_cfg = v6_cfg["bootstrap"]
    n_boot = boot_cfg["n_boot"]
    block_size = boot_cfg["block_size"]

    out = EQUITY_OUT / "bootstrap"
    out.mkdir(parents=True, exist_ok=True)
    paper = EQUITY_OUT / "paper_tables"
    paper.mkdir(parents=True, exist_ok=True)

    _log("=== V6 Equity Bootstrap ===")
    rows = []

    # SP30: C_stable vs A, C_stable vs B
    sp30_dir = equity_dir("sp30")
    if not (sp30_dir / "rolling_A_ceil_CVaR.csv").exists():
        sp30_v5 = v5_out_dir("sp30") / "test"
        fa = pd.read_csv(sp30_v5 / "rolling_A_Historical_CVaR.csv", parse_dates=["date"])
        fb = pd.read_csv(sp30_v5 / "rolling_B_fixed_kappa.csv", parse_dates=["date"])
        fc = pd.read_csv(sp30_v5 / "rolling_C_stable.csv", parse_dates=["date"])
    else:
        fa = pd.read_csv(sp30_dir / "rolling_A_ceil_CVaR.csv", parse_dates=["date"])
        fb = pd.read_csv(sp30_dir / "rolling_B_fixed_kappa.csv", parse_dates=["date"])
        fc = pd.read_csv(sp30_dir / "rolling_C_stable.csv", parse_dates=["date"])

    r1 = _bootstrap_pair(fa, fc, "SP30: C_stable vs A", n_boot, block_size)
    r1["dataset"] = "sp30"
    r2 = _bootstrap_pair(fb, fc, "SP30: C_stable vs B", n_boot, block_size)
    r2["dataset"] = "sp30"
    rows.extend([r1, r2])
    pd.DataFrame([r1, r2]).to_csv(out / "sp30_bootstrap_summary.csv", index=False)
    _log(f"  SP30 C_stable vs A: P(C<A)={r1['prob_b_lt_a']:.1%}, Δ={-r1['mean_cvar_diff_b_minus_a']*100:.2f}pp")
    _log(f"  SP30 C_stable vs B: P(C<B)={r2['prob_b_lt_a']:.1%}")

    # SP100: C_stable vs C_default, C_stable vs A
    sp100_v5 = v5_out_dir("sp100") / "test"
    fa = pd.read_csv(sp100_v5 / "rolling_A_Historical_CVaR.csv", parse_dates=["date"])
    fdef = pd.read_csv(sp100_v5 / "rolling_C_default.csv", parse_dates=["date"])
    fc = pd.read_csv(sp100_v5 / "rolling_C_stable.csv", parse_dates=["date"])

    r3 = _bootstrap_pair(fdef, fc, "SP100: C_stable vs C_default", n_boot, block_size)
    r3["dataset"] = "sp100"
    r4 = _bootstrap_pair(fa, fc, "SP100: C_stable vs A", n_boot, block_size)
    r4["dataset"] = "sp100"
    rows.extend([r3, r4])
    pd.DataFrame([r3, r4]).to_csv(out / "sp100_bootstrap_summary.csv", index=False)
    _log(f"  SP100 C_stable vs C_default: P(C_stable<C_def)={r3['prob_b_lt_a']:.1%}")
    _log(f"  SP100 C_stable vs A: P(C_stable<A)={r4['prob_b_lt_a']:.1%}")

    # Random30 pooled (if available)
    r30_summary = equity_dir("random30") / "random30_summary.csv"
    r30_boot_path = out / "random30_bootstrap_summary.csv"
    if r30_summary.exists():
        param_df = pd.read_csv(equity_dir("random30") / "random30_selected_params.csv")
        pooled = {
            "comparison": "Random30 pooled: C_stable vs A",
            "dataset": "random30",
            "n_universes": len(param_df),
            "win_rate_A": float(param_df["win_vs_A"].mean()),
            "mean_delta_A_pp": float(param_df["delta_A"].mean() * 100),
            "median_delta_A_pp": float(param_df["delta_A"].median() * 100),
            "prob_b_lt_a": float(param_df["win_vs_A"].mean()),
            "interpretation": "win_rate as pooled P(CVaR_C < CVaR_A) across universes",
        }
        pd.DataFrame([pooled]).to_csv(r30_boot_path, index=False)
        rows.append(pooled)
        _log(f"  Random30 pooled win rate vs A: {pooled['win_rate_A']:.1%}")

    all_boot = pd.DataFrame(rows)
    all_boot.to_csv(paper / "table5_bootstrap.csv", index=False)
    all_boot.to_csv(out / "bootstrap_all_summary.csv", index=False)
    _log(f"\nOutput: {out}")


if __name__ == "__main__":
    run_equity_bootstrap()
