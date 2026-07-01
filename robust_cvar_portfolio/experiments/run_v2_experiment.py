"""V2 ablation and cross-market experiment runner."""

from __future__ import annotations

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

from robust_cvar_portfolio.data.loader import build_state_matrix, load_dataset
from robust_cvar_portfolio.portfolio.rolling import run_rolling
from robust_cvar_portfolio.risk.kappa import KappaParams, fit_kappa_theta, kappa_series_from_states
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.backtest import crisis_loss, metrics_from_result
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha


DATASETS = ["etf10", "etf20", "sp30"]

MODELS = {
    "A_no_kappa": {"mode": "plain", "label": "CVaR baseline (w/o κ)"},
    "B_fixed_kappa": {"mode": "fixed", "label": "Fixed robust κ=K"},
    "C_manual_kappa": {"mode": "manual", "label": "Manual κ(s)"},
    "C_learned_kappa": {"mode": "learned", "label": "Learned κ_θ(s)"},
    "D_state_action": {"mode": "state_action", "label": "State-action κ(s,w) [ablation]"},
}


def _stress_target(returns: pd.DataFrame, start: str, end: str, window: int = 20) -> pd.Series:
    ew = returns.mean(axis=1)
    loss = -ew
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = loss.loc[mask]
    return sub.rolling(window).apply(lambda x: cvar_alpha(x.values, 0.05), raw=False).bfill()


def _learn_params(returns: pd.DataFrame, states: pd.DataFrame, config: dict) -> KappaParams:
    train_end = config["splits"]["val"][1]
    val_start, val_end = config["splits"]["val"]
    stress = _stress_target(returns, val_start, val_end)
    val_states = states.loc[val_start:val_end]
    stress = stress.reindex(val_states.index).fillna(stress.median())
    theta = fit_kappa_theta(val_states, stress, config.get("kappa_max", 1.0))
    return KappaParams(
        kappa_max=config.get("kappa_max", 1.0),
        beta_vol=config.get("beta_vol", 1.0),
        beta_dd=config.get("beta_dd", 1.0),
        beta_mom=config.get("beta_mom", 0.5),
        beta_corr=config.get("beta_corr", 0.5),
        beta_conc=config.get("beta_conc", 0.5),
        theta=theta,
    )


def _make_engine(model_key: str, config: dict, learned_params: KappaParams) -> RiskEngine:
    spec = MODELS[model_key]
    params = learned_params if spec["mode"] in {"manual", "learned", "state_action"} else KappaParams()
    if spec["mode"] == "manual":
        params.theta = np.array([0.0, 0.0, 0.0, 0.0])  # use beta_* not theta
    return RiskEngine(
        alpha=config.get("alpha", 0.05),
        kappa_mode=spec["mode"],
        params=params,
        fixed_k=config.get("fixed_kappa", 2.0),
    )


def _metrics_from_rolling(frame: pd.DataFrame, alpha: float) -> dict:
    fake = frame.copy()
    fake["event"] = "daily"
    return metrics_from_result(fake, alpha)


def _plot_nav(results: dict[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, df in results.items():
        nav = (1.0 + df["net_return"]).cumprod()
        plt.plot(df["date"], nav, label=MODELS[name]["label"])
    plt.legend(fontsize=8)
    plt.title("NAV Curve (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _plot_drawdown(results: dict[str, pd.DataFrame], path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, df in results.items():
        nav = (1.0 + df["net_return"]).cumprod()
        dd = 1.0 - nav / nav.cummax()
        plt.plot(df["date"], dd, label=MODELS[name]["label"])
    plt.legend(fontsize=8)
    plt.title("Drawdown (Test)")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def run_single_dataset(dataset_name: str, config_dir: Path, out_root: Path, force_data: bool = False) -> pd.DataFrame:
    print(f"\n{'='*60}\nDataset: {dataset_name}\n{'='*60}")
    t0 = time.time()
    cfg_path = config_dir / f"{dataset_name}.yaml"
    data_root = ROOT / "data" / "processed"
    bundle = load_dataset(cfg_path, data_root, force=force_data)
    config = bundle["config"]
    returns = bundle["returns"]
    states = build_state_matrix(returns)
    learned_params = _learn_params(returns, states, config)

    out_dir = out_root / dataset_name
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "state_matrix.npy", states[states.columns[:4]].values)
    states.to_csv(out_dir / "state_matrix.csv")

    test_start, test_end = config["splits"]["test"]
    results: dict[str, pd.DataFrame] = {}
    metrics_rows = []

    for model_key in MODELS:
        print(f"  running {model_key} ...")
        engine = _make_engine(model_key, config, learned_params)
        frame = run_rolling(
            returns,
            states,
            engine,
            test_start,
            test_end,
            config.get("cost_rate", 0.001),
            config.get("estimation_window", 252),
            config.get("optimizer_maxiter", 150),
        )
        frame["date"] = pd.to_datetime(frame["date"])
        results[model_key] = frame
        m = _metrics_from_rolling(frame, config.get("alpha", 0.05))
        m["model"] = model_key
        m["dataset"] = dataset_name
        m["crisis_2020"] = crisis_loss(frame.set_index("date")["net_return"], "2020-02-01", "2020-04-30")
        m["crisis_2022"] = crisis_loss(frame.set_index("date")["net_return"], "2022-01-01", "2022-12-31")
        metrics_rows.append(m)

    # κ series for C_learned
    kappa_s = kappa_series_from_states(states.loc[test_start:test_end], "learned", learned_params)
    kappa_s.to_csv(out_dir / "kappa_series.csv")
    pd.concat(results.values()).to_csv(out_dir / "rolling_results.csv", index=False)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(out_dir / "cvar_table.csv", index=False)

    _plot_nav({k: results[k] for k in ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa", "C_learned_kappa"]}, out_dir / "nav_curve.png")
    _plot_drawdown({k: results[k] for k in ["A_no_kappa", "B_fixed_kappa", "C_manual_kappa", "C_learned_kappa"]}, out_dir / "drawdown.png")

    # D ablation plot separately
    _plot_nav({"C_learned_kappa": results["C_learned_kappa"], "D_state_action": results["D_state_action"]}, out_dir / "d_ablation_nav.png")

    summary = {
        "dataset": dataset_name,
        "learned_theta": learned_params.theta.tolist(),
        "best_cvar_model": metrics_df.sort_values("cvar_5pct").iloc[0]["model"],
        "C_manual_cvar": float(metrics_df.loc[metrics_df["model"] == "C_manual_kappa", "cvar_5pct"].iloc[0]),
        "C_learned_cvar": float(metrics_df.loc[metrics_df["model"] == "C_learned_kappa", "cvar_5pct"].iloc[0]),
        "A_cvar": float(metrics_df.loc[metrics_df["model"] == "A_no_kappa", "cvar_5pct"].iloc[0]),
        "C_success_cvar_lowest": bool(
            metrics_df.loc[metrics_df["model"] == "C_manual_kappa", "cvar_5pct"].iloc[0]
            <= metrics_df["cvar_5pct"].min() + 1e-9
        ),
        "elapsed_sec": time.time() - t0,
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(metrics_df[["model", "cvar_5pct", "max_drawdown", "crisis_2020"]].to_string(index=False))
    print(f"  done in {time.time()-t0:.1f}s")
    return metrics_df


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="", help="single dataset e.g. sp30")
    parser.add_argument("--force-data", action="store_true", help="force re-download")
    args = parser.parse_args()

    config_dir = ROOT / "configs"
    out_root = ROOT / "outputs" / "v2"
    out_root.mkdir(parents=True, exist_ok=True)

    datasets = [args.dataset] if args.dataset else DATASETS
    all_metrics = []
    for ds in datasets:
        try:
            all_metrics.append(run_single_dataset(ds, config_dir, out_root, force_data=args.force_data))
        except Exception as exc:  # noqa: BLE001
            print(f"  FAILED {ds}: {exc}")

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        existing = out_root / "cross_market_cvar_table.csv"
        if args.dataset and existing.exists():
            prev = pd.read_csv(existing)
            prev = prev[prev["dataset"] != args.dataset]
            combined = pd.concat([prev, combined], ignore_index=True)
        combined.to_csv(out_root / "cross_market_cvar_table.csv", index=False)
        ablation = combined.pivot_table(index="model", values="cvar_5pct", columns="dataset")
        ablation.to_csv(out_root / "ablation_summary.csv")
        print("\n=== Cross-market ablation (CVaR 5%) ===")
        print(ablation.to_string())
        print(f"\nAll V2 outputs: {out_root}")


if __name__ == "__main__":
    main()
