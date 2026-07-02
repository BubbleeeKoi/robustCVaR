"""V3 SP100 experiment: main ablation + SPY benchmark + baselines + bootstrap."""

from __future__ import annotations

import json
import shutil
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
from robust_cvar_portfolio.data.sp100_universe import load_sp100_universe
from robust_cvar_portfolio.experiments.run_v2_experiment import (
    MODELS,
    _learn_params,
    _make_engine,
    _metrics_from_rolling,
    _plot_drawdown,
    _plot_nav,
)
from robust_cvar_portfolio.experiments.v3_analysis import (
    block_bootstrap_cvar,
    crisis_subsample_metrics,
    paired_cvar_test,
    plot_cvar_bootstrap,
    plot_kappa_time_series,
    plot_kappa_vs_state,
    plot_nav_vs_benchmark,
)
from robust_cvar_portfolio.portfolio.rolling import run_rolling
from robust_cvar_portfolio.risk.kappa import kappa_series_from_states
from robust_cvar_portfolio.src.backtest import crisis_loss
from robust_cvar_portfolio.src.baselines import (
    benchmark_metrics,
    buy_and_hold_backtest,
    equal_weight_backtest,
    historical_cvar_backtest,
    max_diversification_backtest,
    mean_variance_backtest,
    metrics_from_frame,
    min_variance_backtest,
    risk_parity_backtest,
    spy_benchmark_returns,
)
from robust_cvar_portfolio.src.risk_metrics import summarize_backtest

BASELINES = {
    "Equal_Weight": equal_weight_backtest,
    "Buy_and_Hold": buy_and_hold_backtest,
    "Mean_Variance": mean_variance_backtest,
    "Min_Variance": min_variance_backtest,
    "Historical_CVaR": historical_cvar_backtest,
    "Risk_Parity": risk_parity_backtest,
    "Max_Diversification": max_diversification_backtest,
}

CRISIS_PERIODS = {
    "covid_2020": ("2020-02-01", "2020-04-30"),
    "volatility_2022": ("2022-01-01", "2022-12-31"),
}


def _log(msg: str) -> None:
    print(msg, flush=True)


def _checkpoint_path(out_dir: Path, model_key: str) -> Path:
    return out_dir / f"rolling_{model_key}.csv"


def _load_checkpoint(out_dir: Path, model_key: str) -> pd.DataFrame | None:
    path = _checkpoint_path(out_dir, model_key)
    if not path.exists():
        return None
    frame = pd.read_csv(path, parse_dates=["date"])
    return frame


def _save_checkpoint(out_dir: Path, model_key: str, frame: pd.DataFrame) -> None:
    frame.to_csv(_checkpoint_path(out_dir, model_key), index=False)


def _frame_to_series(frame: pd.DataFrame) -> pd.Series:
    if "date" in frame.columns:
        return frame.set_index("date")["net_return"]
    return frame["net_return"]


def _run_baselines(returns: pd.DataFrame, config: dict, test_start: str, test_end: str) -> dict[str, pd.DataFrame]:
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    alpha = config.get("alpha", 0.05)
    maxiter = config.get("optimizer_maxiter", 50)
    ra = config.get("mv_risk_aversion", 2.0)
    out: dict[str, pd.DataFrame] = {}
    for name, fn in BASELINES.items():
        print(f"  baseline {name} ...")
        kwargs: dict = {"returns": returns, "start": test_start, "end": test_end, "cost_rate": cost}
        if name in {"Mean_Variance", "Min_Variance", "Historical_CVaR", "Risk_Parity", "Max_Diversification"}:
            kwargs["window"] = window
        if name == "Mean_Variance":
            kwargs["risk_aversion"] = ra
        if name == "Historical_CVaR":
            kwargs["alpha"] = alpha
            kwargs["maxiter"] = maxiter
        out[name] = fn(**kwargs)
    return out


def _metrics_row(name: str, frame: pd.DataFrame, config: dict, group: str) -> dict:
    m = metrics_from_frame(frame, config.get("alpha", 0.05))
    m["method"] = name
    m["group"] = group
    m["crisis_2020"] = crisis_loss(frame["net_return"], "2020-02-01", "2020-04-30")
    m["crisis_2022"] = crisis_loss(frame["net_return"], "2022-01-01", "2022-12-31")
    return m


def run_v3(force_data: bool = False, skip_models: list[str] | None = None, resume: bool = True) -> None:
    t0 = time.time()
    config_path = ROOT / "configs" / "sp100.yaml"
    data_dir = ROOT / "data" / "processed" / "sp100"
    out_dir = ROOT / "outputs" / "v3" / "sp100"
    final_dir = ROOT / "outputs" / "v3" / "sp100_final"
    out_dir.mkdir(parents=True, exist_ok=True)

    _log("=" * 60)
    _log("V3 SP100 Experiment")
    _log("=" * 60)

    bundle = load_sp100_universe(config_path, data_dir, target_n=100, force=force_data)
    config = bundle["config"]
    returns = bundle["returns"]
    _log(f"Data: {returns.shape[1]} assets, {len(returns)} days, {returns.index.min().date()} ~ {returns.index.max().date()}")
    states = build_state_matrix(returns)
    learned_params = _learn_params(returns, states, config)

    test_start, test_end = config["splits"]["test"]
    alpha = config.get("alpha", 0.05)
    skip = set(skip_models or [])

    np.save(out_dir / "state_matrix.npy", states[states.columns[:4]].values)
    states.to_csv(out_dir / "state_matrix.csv")
    bundle["universe"].to_csv(out_dir / "universe.csv", index=False)

    # --- Main ablation models ---
    model_results: dict[str, pd.DataFrame] = {}
    model_metrics: list[dict] = []
    for model_key in MODELS:
        if model_key in skip:
            continue
        cached = _load_checkpoint(out_dir, model_key) if resume else None
        if cached is not None:
            _log(f"  model {model_key} ... loaded checkpoint")
            frame = cached
        else:
            _log(f"  model {model_key} ... running")
            t_model = time.time()
            engine = _make_engine(model_key, config, learned_params)
            frame = run_rolling(
                returns,
                states,
                engine,
                test_start,
                test_end,
                config.get("cost_rate", 0.001),
                config.get("estimation_window", 252),
                config.get("optimizer_maxiter", 50),
            )
            frame["date"] = pd.to_datetime(frame["date"])
            _save_checkpoint(out_dir, model_key, frame)
            _log(f"  model {model_key} done in {time.time() - t_model:.1f}s")
        model_results[model_key] = frame
        model_metrics.append(_metrics_row(model_key, frame.set_index("date"), config, "ablation"))

    kappa_s = kappa_series_from_states(states.loc[test_start:test_end], "manual", learned_params)
    kappa_s.to_csv(out_dir / "kappa_series.csv", header=["kappa"])

    pd.concat({k: v.assign(model=k) for k, v in model_results.items()}).to_csv(
        out_dir / "rolling_results.csv", index=False
    )

    ablation_df = pd.DataFrame(model_metrics)
    ablation_df.to_csv(out_dir / "cvar_table.csv", index=False)
    ablation_df.to_csv(out_dir / "table1_main_results.csv", index=False)
    ablation_df.to_csv(out_dir / "table2_ablation_results.csv", index=False)

    _plot_nav(
        {k: model_results[k] for k in ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa", "C_learned_kappa"] if k in model_results},
        out_dir / "nav_curve.png",
    )
    _plot_drawdown(
        {k: model_results[k] for k in ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa", "C_learned_kappa"] if k in model_results},
        out_dir / "drawdown.png",
    )

    # --- Baselines ---
    _log("\n--- Baselines ---")
    baseline_results = _run_baselines(returns, config, test_start, test_end)
    baseline_metrics = [_metrics_row(k, v, config, "baseline") for k, v in baseline_results.items()]
    baseline_df = pd.DataFrame(baseline_metrics)
    baseline_df.to_csv(out_dir / "baseline_metrics.csv", index=False)

    all_metrics = ablation_df.copy()
    all_metrics = pd.concat([all_metrics, baseline_df], ignore_index=True)
    all_metrics.to_csv(out_dir / "all_methods_table.csv", index=False)
    all_metrics.to_csv(out_dir / "table3_baseline_comparison.csv", index=False)

    # --- SPY benchmark ---
    _log("\n--- SPY Benchmark ---")
    spy_ret = spy_benchmark_returns(config.get("benchmark_ticker", "SPY"), test_start, test_end)
    spy_frame = pd.DataFrame({"net_return": spy_ret, "loss": -spy_ret, "turnover": 0.0})
    spy_metrics = _metrics_row("SPY", spy_frame, config, "benchmark")
    benchmark_rows = []
    for model_key in ["C_manual_kappa", "A_no_kappa"]:
        if model_key not in model_results:
            continue
        port = _frame_to_series(model_results[model_key])
        bm = benchmark_metrics(port, spy_ret, alpha=alpha)
        bm["method"] = model_key
        benchmark_rows.append(bm)
    spy_row = {k: v for k, v in spy_metrics.items() if k not in {"method", "group"}}
    spy_row["method"] = "SPY"
    benchmark_rows.append(spy_row)
    benchmark_df = pd.DataFrame(benchmark_rows)
    benchmark_df.to_csv(out_dir / "benchmark_comparison.csv", index=False)

    nav_dict = {"SPY": (1 + spy_ret).cumprod()}
    if "C_manual_kappa" in model_results:
        c_nav = (1 + _frame_to_series(model_results["C_manual_kappa"])).cumprod()
        nav_dict["C_manual"] = c_nav
    plot_nav_vs_benchmark(nav_dict, out_dir / "nav_vs_spy.png")
    plot_nav_vs_benchmark(nav_dict, out_dir / "fig1_nav_vs_benchmark.png")

    # drawdown vs spy
    plt.figure(figsize=(10, 5))
    for name, series in nav_dict.items():
        dd = 1 - series / series.cummax()
        plt.plot(dd.index, dd.values, label=name)
    plt.legend()
    plt.title("Drawdown vs SPY")
    plt.tight_layout()
    plt.savefig(out_dir / "drawdown_vs_spy.png", dpi=150)
    plt.savefig(out_dir / "fig2_drawdown_vs_benchmark.png", dpi=150)
    plt.close()

    # all methods nav
    plt.figure(figsize=(11, 5))
    for name, frame in {**model_results, **baseline_results}.items():
        nav = (1 + _frame_to_series(frame)).cumprod()
        plt.plot(nav.index, nav.values, label=name, alpha=0.8)
    plt.legend(fontsize=6, ncol=2)
    plt.title("All Methods NAV (Test)")
    plt.tight_layout()
    plt.savefig(out_dir / "all_methods_nav.png", dpi=150)
    plt.close()

    # --- Kappa interpretability ---
    _log("\n--- Kappa interpretability ---")
    plot_kappa_time_series(kappa_s, out_dir / "fig3_kappa_time_series.png")
    plot_kappa_vs_state(kappa_s, states.loc[test_start:test_end], out_dir / "fig4_kappa_vs_vol_dd.png")

    # --- Bootstrap & crisis ---
    _log("\n--- Bootstrap & crisis ---")
    if "A_no_kappa" in model_results and "C_manual_kappa" in model_results:
        loss_a = model_results["A_no_kappa"]["loss"].values
        loss_c = model_results["C_manual_kappa"]["loss"].values
        boot_a = block_bootstrap_cvar(loss_a, config.get("bootstrap_n", 500), config.get("bootstrap_block", 20), alpha)
        boot_c = block_bootstrap_cvar(loss_c, config.get("bootstrap_n", 500), config.get("bootstrap_block", 20), alpha)
        boot_df = pd.DataFrame(
            [
                {"method": "A_no_kappa", **boot_a},
                {"method": "C_manual_kappa", **boot_c},
            ]
        )
        boot_df.to_csv(out_dir / "bootstrap_cvar_ci.csv", index=False)
        boot_df.to_csv(out_dir / "table5_bootstrap_ci.csv", index=False)

        diff = paired_cvar_test(loss_a, loss_c, config.get("bootstrap_n", 500), config.get("bootstrap_block", 20), alpha)
        pd.DataFrame([diff]).to_csv(out_dir / "cvar_difference_test.csv", index=False)
        plot_cvar_bootstrap(diff, out_dir / "fig5_cvar_difference_bootstrap.png")

    crisis_rows = []
    for name, frame in {**model_results, **baseline_results}.items():
        sub = crisis_subsample_metrics(frame, CRISIS_PERIODS, alpha)
        sub["method"] = name
        crisis_rows.append(sub)
    crisis_df = pd.concat(crisis_rows, ignore_index=True)
    crisis_df.to_csv(out_dir / "crisis_subsample_table.csv", index=False)
    crisis_df.to_csv(out_dir / "table4_crisis_subsample.csv", index=False)

    # weights export
    if "C_manual_kappa" in model_results:
        model_results["C_manual_kappa"].to_csv(out_dir / "weights_C_manual.csv", index=False)

    # rolling all methods
    all_rolling = []
    for name, frame in {**model_results, **baseline_results}.items():
        tmp = frame.reset_index() if frame.index.name == "date" else frame.copy()
        if "date" not in tmp.columns and frame.index.name == "date":
            tmp["date"] = frame.index
        tmp["method"] = name
        all_rolling.append(tmp)
    pd.concat(all_rolling, ignore_index=True).to_csv(out_dir / "rolling_results_all_methods.csv", index=False)

    # summary
    c_cvar = ablation_df.loc[ablation_df["method"] == "C_manual_kappa", "cvar_5pct"]
    a_cvar = ablation_df.loc[ablation_df["method"] == "A_no_kappa", "cvar_5pct"]
    c_mdd = ablation_df.loc[ablation_df["method"] == "C_manual_kappa", "max_drawdown"]
    a_mdd = ablation_df.loc[ablation_df["method"] == "A_no_kappa", "max_drawdown"]
    summary = {
        "n_assets": int(returns.shape[1]),
        "test_period": [test_start, test_end],
        "C_manual_cvar": float(c_cvar.iloc[0]) if len(c_cvar) else None,
        "A_cvar": float(a_cvar.iloc[0]) if len(a_cvar) else None,
        "cvar_improvement_pct": float((a_cvar.iloc[0] - c_cvar.iloc[0]) / a_cvar.iloc[0] * 100) if len(a_cvar) and len(c_cvar) else None,
        "C_mdd": float(c_mdd.iloc[0]) if len(c_mdd) else None,
        "A_mdd": float(a_mdd.iloc[0]) if len(a_mdd) else None,
        "success_cvar": bool(c_cvar.iloc[0] < a_cvar.iloc[0]) if len(a_cvar) and len(c_cvar) else False,
        "success_mdd": bool(c_mdd.iloc[0] < a_mdd.iloc[0]) if len(a_mdd) and len(c_mdd) else False,
        "elapsed_sec": time.time() - t0,
        "note": "Version A universe; survivorship bias possible",
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # copy to sp100_final
    final_dir.mkdir(parents=True, exist_ok=True)
    for fname in [
        "table1_main_results.csv", "table2_ablation_results.csv", "table3_baseline_comparison.csv",
        "table4_crisis_subsample.csv", "table5_bootstrap_ci.csv",
        "fig1_nav_vs_benchmark.png", "fig2_drawdown_vs_benchmark.png",
        "fig3_kappa_time_series.png", "fig4_kappa_vs_vol_dd.png", "fig5_cvar_difference_bootstrap.png",
        "weights_C_manual.csv", "kappa_series.csv", "rolling_results_all_methods.csv", "summary.json",
    ]:
        src = out_dir / fname
        if src.exists():
            shutil.copy(src, final_dir / fname)

    _log("\n=== V3 Summary ===")
    _log(ablation_df[["method", "cvar_5pct", "max_drawdown", "sharpe_ratio"]].to_string(index=False))
    _log(f"\nC vs A CVaR improvement: {summary.get('cvar_improvement_pct', 'N/A'):.2f}%")
    _log(f"Success (CVaR): {summary['success_cvar']}, Success (MDD): {summary['success_mdd']}")
    _log(f"Elapsed: {summary['elapsed_sec']:.1f}s")
    _log(f"Outputs: {out_dir}")
    _log(f"Final:   {final_dir}")


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--no-resume", action="store_true", help="ignore model checkpoints")
    parser.add_argument("--skip-model", action="append", default=[], help="skip model e.g. D_state_action")
    args = parser.parse_args()
    run_v3(force_data=args.force_data, skip_models=args.skip_model or None, resume=not args.no_resume)


if __name__ == "__main__":
    main()
