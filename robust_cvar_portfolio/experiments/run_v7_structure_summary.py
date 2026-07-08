"""V7-A: Structure diagnostics summary for SP30, Random30, SP100."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.equity_common import EQUITY_OUT, effective_n
from robust_cvar_portfolio.experiments.v5_common import load_v5_bundle
from robust_cvar_portfolio.experiments.v7_common import (
    cvar_gap_c_minus_a,
    load_v7_config,
    paper_dir,
    test_structure_summary,
    validation_structure_summary,
    v7_dir,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _load_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path)


def run_v7_structure_summary() -> None:
    v7_cfg = load_v7_config()
    window = v7_cfg.get("structure_window", 60)
    out = v7_dir("structure")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    _log("=== V7-A Structure Summary ===")

    rows = []

    for ds in ["sp30", "sp100"]:
        bundle = load_v5_bundle(ds)
        returns = bundle["returns"]
        config = bundle["config"]
        val_s, val_e = config["splits"]["val"]
        test_s, test_e = config["splits"]["test"]
        val_struct = validation_structure_summary(returns, val_s, val_e, window)
        test_struct = test_structure_summary(returns, test_s, test_e, window)

        tbl = _load_table(EQUITY_OUT / ds / "table_main.csv")
        a_cvar = c_cvar = c_def = None
        hhi_c = turnover_c = None
        if not tbl.empty:
            a_cvar = float(tbl.loc[tbl["method"] == "A_ceil_CVaR", "cvar_5pct"].iloc[0])
            c_cvar = float(tbl.loc[tbl["method"] == "C_stable", "cvar_5pct"].iloc[0])
            if "C_default" in tbl["method"].values:
                c_def = float(tbl.loc[tbl["method"] == "C_default", "cvar_5pct"].iloc[0])
            turnover_c = float(tbl.loc[tbl["method"] == "C_stable", "avg_turnover"].iloc[0])

        rows.append(
            {
                "universe": ds.upper(),
                "n_assets": bundle["n_assets"],
                **val_struct,
                **test_struct,
                "cvar_A": a_cvar,
                "cvar_C_stable": c_cvar,
                "cvar_C_default": c_def,
                "cvar_gap_C_minus_A": cvar_gap_c_minus_a(c_cvar, a_cvar) if c_cvar and a_cvar else None,
                "avg_turnover_C_stable": turnover_c,
            }
        )

    r30_params = _load_table(EQUITY_OUT / "random30" / "random30_selected_params.csv")
    bundle = load_v5_bundle("sp100")
    full_returns = bundle["returns"]
    config = bundle["config"]
    val_s, val_e = config["splits"]["val"]
    test_s, test_e = config["splits"]["test"]

    if not r30_params.empty:
        val_corrs, test_corrs = [], []
        for _, row in r30_params.iterrows():
            tickers = row["tickers"].split(",")
            sub = full_returns[tickers]
            val_corrs.append(validation_structure_summary(sub, val_s, val_e, window)["avg_correlation_val"])
            test_corrs.append(test_structure_summary(sub, test_s, test_e, window)["avg_correlation_test"])

        c_rows = r30_params.copy()
        rows.append(
            {
                "universe": "Random30 (n=3)",
                "n_assets": 30,
                "avg_correlation_val": float(pd.Series(val_corrs).mean()),
                "pc1_share_val": float("nan"),
                "effective_dimension_val": float("nan"),
                "avg_correlation_test": float(pd.Series(test_corrs).mean()),
                "pc1_share_test": float("nan"),
                "effective_dimension_test": float("nan"),
                "cvar_A": float(r30_params["A_cvar"].mean()),
                "cvar_C_stable": float(r30_params["C_stable_cvar"].mean()),
                "cvar_C_default": None,
                "cvar_gap_C_minus_A": float((r30_params["C_stable_cvar"] - r30_params["A_cvar"]).mean()),
                "avg_turnover_C_stable": None,
            }
        )

    summary = pd.DataFrame(rows)
    summary.to_csv(out / "universe_structure_summary.csv", index=False)
    summary.to_csv(paper_dir() / "table_v7_structure_summary.csv", index=False)

    sub = summary.dropna(subset=["avg_correlation_val"])
    if len(sub) >= 2:
        plt.figure(figsize=(7, 5))
        colors = {"SP30": "#2563eb", "SP100": "#dc2626", "Random30 (n=3)": "#10b981"}
        for _, row in sub.iterrows():
            c = colors.get(row["universe"], "#666")
            plt.scatter(
                row["avg_correlation_val"],
                row.get("cvar_gap_C_minus_A", 0) * 100 if pd.notna(row.get("cvar_gap_C_minus_A")) else 0,
                s=120,
                label=row["universe"],
                c=c,
            )
        plt.axhline(0, color="gray", ls="--", lw=0.8)
        plt.xlabel("Validation avg correlation")
        plt.ylabel("Test CVaR gap C_stable - A (pp)")
        plt.title("V7-A: Correlation structure vs C-A gap")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_structure_comparison.png", dpi=150)
        plt.close()

        plt.figure(figsize=(7, 5))
        labels = sub["universe"].tolist()
        x = range(len(labels))
        plt.bar([i - 0.2 for i in x], sub["avg_correlation_val"], width=0.4, label="Val avg corr")
        if "avg_correlation_test" in sub.columns:
            plt.bar([i + 0.2 for i in x], sub["avg_correlation_test"], width=0.4, label="Test avg corr")
        plt.xticks(list(x), labels, rotation=15)
        plt.ylabel("Average correlation")
        plt.title("SP30 / Random30 / SP100 correlation structure")
        plt.legend()
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_sp30_random30_sp100_corr_space.png", dpi=150)
        plt.close()

    _log(summary.to_string(index=False))
    _log(f"\nOutput: {out}")


if __name__ == "__main__":
    run_v7_structure_summary()
