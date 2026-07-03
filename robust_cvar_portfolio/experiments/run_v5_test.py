"""V5 Step 4–7: fixed validation params → test evaluation + figures."""

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

from robust_cvar_portfolio.experiments.v5_common import (
    avg_weight_hhi,
    engine_fixed,
    engine_historical,
    engine_manual,
    export_test_weights,
    load_selected_params,
    load_v5_bundle,
    metrics_row,
    run_test_backtest,
    v5_out_dir,
)


def _log(msg: str) -> None:
    print(msg, flush=True)


def _plot_nav(results: dict[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, df in results.items():
        nav = (1.0 + df["net_return"]).cumprod()
        dates = df["date"] if "date" in df.columns else df.index
        plt.plot(dates, nav, label=name)
    plt.legend(fontsize=7)
    plt.title("NAV (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_drawdown(results: dict[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, df in results.items():
        nav = (1.0 + df["net_return"]).cumprod()
        dd = 1.0 - nav / nav.cummax()
        dates = df["date"] if "date" in df.columns else df.index
        plt.plot(dates, dd, label=name)
    plt.legend(fontsize=7)
    plt.title("Drawdown (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_kappa(frame: pd.DataFrame, path: Path) -> None:
    sub = frame.dropna(subset=["kappa"])
    if sub.empty:
        return
    plt.figure(figsize=(10, 4))
    plt.plot(sub["date"], sub["kappa"], marker="o", markersize=3)
    plt.title("Kappa at rebalance (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_turnover(results: dict[str, pd.DataFrame], path: Path) -> None:
    rows = []
    for name, df in results.items():
        rows.append({"method": name, "avg_turnover": df["turnover"].mean()})
    tb = pd.DataFrame(rows)
    plt.figure(figsize=(8, 4))
    plt.bar(tb["method"], tb["avg_turnover"])
    plt.xticks(rotation=45, ha="right")
    plt.title("Average turnover (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_v5_test(dataset: str) -> None:
    bundle = load_v5_bundle(dataset)
    config = bundle["config"]
    returns = bundle["returns"]
    states = bundle["states"]
    learned = bundle["learned"]
    tickers = list(returns.columns)

    val_dir = v5_out_dir(dataset) / "validation"
    test_dir = v5_out_dir(dataset) / "test"
    fig_dir = v5_out_dir(dataset) / "figures"
    test_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    params_path = val_dir / "selected_params.json"
    if not params_path.exists():
        raise FileNotFoundError(f"Run calibration first: {params_path}")
    sel = load_selected_params(params_path)
    km_star = sel["kappa_max"]
    cap_star = sel.get("weight_cap")
    rho_star = sel.get("rho")

    t0 = time.time()
    _log(f"=== V5 Test ({dataset}) kappa_max={km_star}, cap={cap_star}, rho={rho_star} ===")

    models: list[tuple[str, object, float | None, float | None]] = [
        ("A_Historical_CVaR", engine_historical(config), None, None),
        ("B_fixed_kappa", engine_fixed(config), None, None),
        ("C_default", engine_manual(config, learned, config.get("kappa_max", 1.0)), None, None),
        ("C_calibrated", engine_manual(config, learned, km_star), None, None),
        ("C_cap", engine_manual(config, learned, km_star), cap_star, None),
        ("C_smooth", engine_manual(config, learned, km_star), None, rho_star),
        (
            "C_stable",
            engine_manual(config, learned, km_star),
            cap_star,
            rho_star,
        ),
    ]

    results: dict[str, pd.DataFrame] = {}
    metrics_rows = []

    for name, engine, cap, rho in models:
        ckpt = test_dir / f"rolling_{name}.csv"
        if ckpt.exists():
            _log(f"  load {name}")
            frame = pd.read_csv(ckpt, parse_dates=["date"])
        else:
            _log(f"  run {name} ...")
            frame = run_test_backtest(returns, states, engine, config, weight_cap=cap, kappa_rho=rho)
            frame.to_csv(ckpt, index=False)
        results[name] = frame
        m = metrics_row(frame, config, name)
        m["kappa_max"] = km_star if name.startswith("C_") and name != "C_default" else config.get("kappa_max", 1.0)
        m["weight_cap"] = cap
        m["rho"] = rho
        metrics_rows.append(m)

        w_ckpt = test_dir / f"weights_{name}.csv"
        if not w_ckpt.exists() and name in {
            "A_Historical_CVaR",
            "B_fixed_kappa",
            "C_calibrated",
            "C_stable",
        }:
            export_test_weights(returns, states, engine, config, w_ckpt, weight_cap=cap, kappa_rho=rho)

    table = pd.DataFrame(metrics_rows)
    table.to_csv(test_dir / "table_main.csv", index=False)
    ablation = table[table["method"].str.startswith("C_")].copy()
    ablation.to_csv(test_dir / "table_ablation.csv", index=False)

    diag_rows = []
    for name in ["C_calibrated", "C_stable", "A_Historical_CVaR"]:
        wpath = test_dir / f"weights_{name}.csv"
        if wpath.exists():
            w = pd.read_csv(wpath, index_col=0, parse_dates=True)
            hhi = avg_weight_hhi(w, tickers)
            max_w = float(w[tickers].max(axis=1).mean())
            diag_rows.append({"method": name, "avg_hhi": hhi, "avg_max_weight": max_w})
    if diag_rows:
        pd.DataFrame(diag_rows).to_csv(test_dir / "stability_diagnostics.csv", index=False)

    _plot_nav(results, fig_dir / "nav_comparison.png")
    _plot_drawdown(results, fig_dir / "drawdown_comparison.png")
    _plot_kappa(results["C_calibrated"], fig_dir / "kappa_series.png")
    _plot_turnover(results, fig_dir / "turnover_comparison.png")

    if (val_dir / "kappa_max_grid.csv").exists():
        kg = pd.read_csv(val_dir / "kappa_max_grid.csv")
        plt.figure(figsize=(8, 4))
        plt.plot(kg["kappa_max"], kg["J_val"], marker="o")
        plt.xlabel("kappa_max")
        plt.ylabel("J_val")
        plt.title("Validation kappa_max grid")
        plt.tight_layout()
        plt.savefig(fig_dir / "validation_kappa_grid.png", dpi=150)
        plt.close()

    a_cvar = float(table.loc[table["method"] == "A_Historical_CVaR", "cvar_5pct"].iloc[0])
    c_cal = float(table.loc[table["method"] == "C_calibrated", "cvar_5pct"].iloc[0])
    c_stable = float(table.loc[table["method"] == "C_stable", "cvar_5pct"].iloc[0])
    b_cvar = float(table.loc[table["method"] == "B_fixed_kappa", "cvar_5pct"].iloc[0])

    summary = {
        "dataset": dataset,
        "selected_params": sel,
        "A_cvar": a_cvar,
        "B_cvar": b_cvar,
        "C_calibrated_cvar": c_cal,
        "C_stable_cvar": c_stable,
        "C_calibrated_beats_A": c_cal < a_cvar,
        "C_stable_beats_min_AB": c_stable < min(a_cvar, b_cvar),
        "elapsed_sec": time.time() - t0,
    }
    with (test_dir / "test_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    _log("\n=== Test CVaR 5% ===")
    _log(table[["method", "cvar_5pct", "max_drawdown", "avg_turnover", "sharpe_ratio"]].to_string(index=False))
    _log(f"\nC_calibrated beats A: {summary['C_calibrated_beats_A']}")
    _log(f"C_stable beats min(A,B): {summary['C_stable_beats_min_AB']}")
    _log(f"Output: {test_dir}")
    _log(f"Done in {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="sp100")
    args = parser.parse_args()
    run_v5_test(args.dataset)
