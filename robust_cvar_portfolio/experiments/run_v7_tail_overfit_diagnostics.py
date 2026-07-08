"""V7-C: Tail-sample overfitting diagnostics on SP100."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.v5_common import (
    engine_fixed,
    engine_historical,
    engine_manual,
    load_selected_params,
    load_v5_bundle,
    v5_out_dir,
)
from robust_cvar_portfolio.experiments.v7_common import paper_dir, v7_dir
from robust_cvar_portfolio.portfolio.rolling import monthly_rebalance_dates
from robust_cvar_portfolio.portfolio.weight_export import export_rebalance_weights
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha


def _log(msg: str) -> None:
    print(msg, flush=True)


def worst_tail_indices(losses: np.ndarray, alpha: float = 0.05) -> set[int]:
    n = len(losses)
    k = max(1, int(np.ceil(alpha * n)))
    order = np.argsort(losses)[::-1]
    return set(order[:k].tolist())


def jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return np.nan
    union = a | b
    if not union:
        return np.nan
    return len(a & b) / len(union)


def run_tail_overfit_diagnostics() -> None:
    out = v7_dir("overfit")
    fig_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(exist_ok=True)
    _log("=== V7-C Tail Overfit Diagnostics (SP100) ===")

    bundle = load_v5_bundle("sp100")
    returns = bundle["returns"]
    states = bundle["states"]
    config = bundle["config"]
    learned = bundle["learned"]
    test_start, test_end = config["splits"]["test"]
    alpha = config.get("alpha", 0.05)
    window = config.get("estimation_window", 252)
    cost = config.get("cost_rate", 0.001)
    maxiter = config.get("optimizer_maxiter", 50)

    sel = load_selected_params(v5_out_dir("sp100") / "validation" / "selected_params.json")
    km_def = config.get("kappa_max", 1.0)

    models = {
        "A_ceil_CVaR": (engine_historical(config), None, None),
        "B_fixed_kappa": (engine_fixed(config), None, None),
        "C_default": (engine_manual(config, learned, km_def), None, None),
        "C_stable": (
            engine_manual(config, learned, sel["kappa_max"]),
            sel.get("weight_cap"),
            sel.get("rho"),
        ),
    }

    overlap_rows = []
    oos_rows = []
    q_rows = []

    for name, (engine, cap, rho) in models.items():
        wexp = export_rebalance_weights(
            returns, states, engine, test_start, test_end,
            cost, window, maxiter, weight_cap=cap, kappa_rho=rho,
        )
        tickers = [c for c in wexp.columns if c not in {"kappa", "turnover"}]
        reb_dates = [d for d in wexp.index if d in set(monthly_rebalance_dates(returns.index))]
        prev_tail: set[int] | None = None

        for i, date in enumerate(reb_dates):
            loc = returns.index.get_loc(date)
            if loc < window:
                continue
            hist = returns.iloc[loc - window : loc]
            w = wexp.loc[date, tickers].values.astype(float)
            gross = hist.values @ w
            to = float(wexp.loc[date, "turnover"]) if "turnover" in wexp.columns else 0.0
            losses = -gross + cost * to
            tail = worst_tail_indices(losses, alpha)
            j_t = jaccard(tail, prev_tail) if prev_tail is not None else np.nan
            overlap_rows.append({"date": date, "model": name, "tail_jaccard": j_t, "tail_size": len(tail)})
            prev_tail = tail

            rcvar_train, _ = engine.portfolio_risk(losses, states.iloc[loc - window : loc], w)
            if i + 1 < len(reb_dates):
                nxt = reb_dates[i + 1]
                hold = returns.loc[date:nxt].iloc[1:]
            else:
                hold = returns.loc[date:test_end].iloc[1:]
            if len(hold) > 5:
                hold_loss = -(hold.values @ w)
                cvar_hold = cvar_alpha(hold_loss, alpha)
                oos_rows.append(
                    {
                        "date": date,
                        "model": name,
                        "rcvar_train": float(rcvar_train),
                        "cvar_holdout": float(cvar_hold),
                        "oos_gap": float(cvar_hold - rcvar_train),
                    }
                )

            hhi = float(np.sum(w**2))
            q_rows.append(
                {
                    "date": date,
                    "model": name,
                    "hhi": hhi,
                    "max_weight": float(w.max()),
                    "n_eff": 1.0 / hhi if hhi > 0 else np.nan,
                }
            )

    overlap_df = pd.DataFrame(overlap_rows)
    oos_df = pd.DataFrame(oos_rows)
    q_df = pd.DataFrame(q_rows)
    overlap_df.to_csv(out / "tail_set_overlap.csv", index=False)
    oos_df.to_csv(out / "oos_gap_by_rebalance.csv", index=False)
    q_df.to_csv(out / "q_weight_concentration.csv", index=False)

    summary_rows = []
    for name in models:
        sub_o = overlap_df[(overlap_df["model"] == name) & overlap_df["tail_jaccard"].notna()]
        sub_g = oos_df[oos_df["model"] == name]
        sub_q = q_df[q_df["model"] == name]
        summary_rows.append(
            {
                "model": name,
                "mean_tail_jaccard": float(sub_o["tail_jaccard"].mean()) if len(sub_o) else np.nan,
                "mean_oos_gap": float(sub_g["oos_gap"].mean()) if len(sub_g) else np.nan,
                "median_oos_gap": float(sub_g["oos_gap"].median()) if len(sub_g) else np.nan,
                "mean_hhi": float(sub_q["hhi"].mean()) if len(sub_q) else np.nan,
                "mean_n_eff": float(sub_q["n_eff"].mean()) if len(sub_q) else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "oos_gap_summary.csv", index=False)
    summary.to_csv(paper_dir() / "table_v7_overfit_diagnostics.csv", index=False)

    if not overlap_df.empty:
        plt.figure(figsize=(8, 4))
        for name in models:
            sub = overlap_df[overlap_df["model"] == name]
            plt.plot(sub["date"], sub["tail_jaccard"], label=name, alpha=0.8)
        plt.ylabel("Tail set Jaccard overlap")
        plt.title("SP100: consecutive rebalance tail overlap")
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_tail_overlap.png", dpi=150)
        plt.close()

    if not oos_df.empty:
        data = [oos_df[oos_df["model"] == m]["oos_gap"].values * 100 for m in models]
        plt.figure(figsize=(7, 4))
        plt.boxplot(data, tick_labels=list(models.keys()))
        plt.axhline(0, color="gray", ls="--")
        plt.ylabel("OOS gap (pp)")
        plt.title("SP100: in-sample RCVaR vs holdout CVaR gap")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_oos_gap_boxplot.png", dpi=150)
        plt.close()

    if not q_df.empty:
        plt.figure(figsize=(7, 4))
        means = q_df.groupby("model")["hhi"].mean()
        plt.bar(means.index, means.values)
        plt.ylabel("Mean weight HHI")
        plt.title("SP100: weight concentration by model")
        plt.xticks(rotation=20)
        plt.tight_layout()
        plt.savefig(fig_dir / "fig_q_hhi.png", dpi=150)
        plt.close()

    _log(summary.to_string(index=False))
    _log(f"\nOutput: {out}")


if __name__ == "__main__":
    run_tail_overfit_diagnostics()
