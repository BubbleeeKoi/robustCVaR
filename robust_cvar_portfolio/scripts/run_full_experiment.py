"""End-to-end experiment runner."""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = ROOT.parent
sys.path.insert(0, str(PROJECT_ROOT))

from robust_cvar_portfolio.src.agents.ppo import TrainConfig, evaluate_policy, train_agent
from robust_cvar_portfolio.src.backtest import build_metrics_table, plot_drawdown_comparison, plot_nav_comparison
from robust_cvar_portfolio.src.baselines import equal_weight_backtest, metrics_from_frame, min_variance_backtest
from robust_cvar_portfolio.src.data_loader import load_config, run_data_pipeline
from robust_cvar_portfolio.src.env_portfolio import PortfolioEnv, PortfolioEnvConfig
from robust_cvar_portfolio.src.evaluation import combine_all_metrics, save_json
from robust_cvar_portfolio.src.features import build_market_features
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha
from robust_cvar_portfolio.src.robust_cvar_layer import verify_degeneracy
from robust_cvar_portfolio.src.rolling_rcvar import run_all_variants


def main() -> None:
    config_path = ROOT / "configs" / "etf10.yaml"
    data_dir = ROOT / "data" / "processed"
    out_dir = ROOT / "outputs"
    tables_dir = out_dir / "tables"
    figures_dir = out_dir / "figures"
    rolling_dir = out_dir / "rolling_portfolio"

    for path in [tables_dir, figures_dir, rolling_dir]:
        path.mkdir(parents=True, exist_ok=True)

    print("=== Step 1: Data pipeline (akshare) ===")
    t0 = time.time()
    returns_path = data_dir / "returns.csv"
    config = load_config(config_path)
    if returns_path.exists():
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        print(f"Data loaded from cache: {returns.shape}, elapsed {time.time()-t0:.1f}s")
    else:
        paths = run_data_pipeline(config_path, data_dir)
        returns = pd.read_csv(paths["returns"], index_col=0, parse_dates=True)
        print(f"Data ready: {returns.shape}, elapsed {time.time()-t0:.1f}s")

    print("=== Step 2: Robust CVaR layer sanity checks ===")
    sample_losses = (-returns.iloc[:252].mean(axis=1).values)
    degeneracy = verify_degeneracy(sample_losses, alpha=config["alpha"], k_fixed=config["fixed_kappa"])
    save_json(degeneracy, tables_dir / "rcvar_degeneracy_check.json")
    print(degeneracy)

    print("=== Step 3: Rolling robust CVaR portfolios (A/B/C/D) on test split ===")
    rolling_metrics_path = rolling_dir / "rolling_metrics.csv"
    if rolling_metrics_path.exists():
        rolling_metrics = pd.read_csv(rolling_metrics_path)
        rolling_results = {}
        for name in rolling_metrics["strategy"]:
            rolling_results[name] = pd.read_csv(rolling_dir / f"rolling_{name}.csv")
        print("Rolling results loaded from cache:")
        print(rolling_metrics.to_string(index=False))
    else:
        t0 = time.time()
        rolling_results = run_all_variants(returns, config, split_name="test")
        rolling_metrics = build_metrics_table(rolling_results, config)
        rolling_metrics.to_csv(rolling_dir / "rolling_metrics.csv", index=False)
        for name, frame in rolling_results.items():
            frame.to_csv(rolling_dir / f"rolling_{name}.csv", index=False)
        plot_nav_comparison(rolling_results, rolling_dir / "rolling_nav_comparison.png")
        plot_drawdown_comparison(rolling_results, rolling_dir / "rolling_drawdown_comparison.png")
        print(rolling_metrics.to_string(index=False))
        print(f"Rolling done in {time.time()-t0:.1f}s")

    print("=== Step 4: Traditional baselines on test split ===")
    test_start, test_end = config["splits"]["test"]
    baseline_frames = {
        "equal_weight": equal_weight_backtest(returns, test_start, test_end, config["cost_rate"]),
        "min_variance": min_variance_backtest(returns, test_start, test_end, config["estimation_window"], config["cost_rate"]),
    }
    baseline_metrics = pd.DataFrame(
        [{"method": k, **metrics_from_frame(v, config["alpha"])} for k, v in baseline_frames.items()]
    )
    baseline_metrics.to_csv(tables_dir / "baseline_metrics.csv", index=False)
    print(baseline_metrics.to_string(index=False))

    print("=== Step 5: PortfolioEnv + RL agents (train on train, eval on test) ===")
    features = build_market_features(returns)
    train_start, train_end = config["splits"]["train"]
    env_cfg = PortfolioEnvConfig(
        lookback=config.get("lookback_L", 20),
        cost_rate=config["cost_rate"],
        kappa_max=config["kappa_max"],
        beta_vol=config["beta_vol"],
        beta_dd=config["beta_dd"],
        beta_conc=config["beta_conc"],
    )
    train_env = PortfolioEnv(returns, features, train_start, train_end, env_cfg)
    test_env = PortfolioEnv(returns, features, test_start, test_end, env_cfg)

    rl_specs = {
        "ppo": TrainConfig(objective="return", train_iters=30, rollout_steps=64),
        "cvar_ppo": TrainConfig(objective="cvar", train_iters=30, rollout_steps=64, alpha=config["alpha"]),
        "fixed_robust_ppo": TrainConfig(
            objective="fixed_robust",
            train_iters=30,
            rollout_steps=64,
            alpha=config["alpha"],
            fixed_kappa=config["fixed_kappa"],
        ),
        "state_robust_ppo": TrainConfig(objective="state_robust", train_iters=30, rollout_steps=64, alpha=config["alpha"]),
        "sad_robust_ppo": TrainConfig(
            objective="sad_robust",
            train_iters=30,
            rollout_steps=64,
            alpha=config["alpha"],
            kappa_max=config["kappa_max"],
        ),
    }
    rl_results = {}
    for name, tcfg in rl_specs.items():
        print(f"  training {name} ...")
        _, policy_fn = train_agent(train_env, tcfg)
        rl_results[name] = evaluate_policy(test_env, policy_fn)

    rl_metrics = pd.DataFrame(
        [
            {
                "method": name,
                "cvar_5pct": cvar_alpha(v["losses"], config["alpha"]),
                "annualized_return": float((1 + v["net_returns"]).prod() ** (252 / max(len(v["net_returns"]), 1)) - 1),
            }
            for name, v in rl_results.items()
        ]
    )
    rl_metrics.to_csv(tables_dir / "rl_metrics.csv", index=False)
    print(rl_metrics.to_string(index=False))

    print("=== Step 6: Final paper table ===")
    final_table = combine_all_metrics(rolling_results, rl_results, baseline_frames, config)
    final_table = final_table.sort_values("cvar_5pct")
    final_table.to_csv(tables_dir / "final_paper_metrics.csv", index=False)
    save_json(
        {
            "primary_metric": "cvar_5pct",
            "best_rolling": rolling_metrics.sort_values("cvar_5pct").iloc[0].to_dict(),
            "best_overall": final_table.iloc[0].to_dict(),
            "success_check": {
                "D_cvar_lt_A": bool(
                    rolling_metrics.loc[rolling_metrics["strategy"] == "D_state_action_robust", "cvar_5pct"].iloc[0]
                    < rolling_metrics.loc[rolling_metrics["strategy"] == "A_plain_cvar", "cvar_5pct"].iloc[0]
                ),
            },
        },
        tables_dir / "final_summary.json",
    )
    print(final_table.to_string(index=False))
    print(f"\nAll outputs saved under: {out_dir}")


if __name__ == "__main__":
    main()
