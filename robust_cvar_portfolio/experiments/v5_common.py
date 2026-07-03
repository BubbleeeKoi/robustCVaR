"""V5 shared utilities: data loading, engines, validation objective, metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_cvar_portfolio.data.loader import build_state_matrix, load_dataset
from robust_cvar_portfolio.data.sp100_universe import load_sp100_universe
from robust_cvar_portfolio.experiments.run_v2_experiment import _learn_params
from robust_cvar_portfolio.portfolio.rolling import run_rolling
from robust_cvar_portfolio.portfolio.weight_export import export_rebalance_weights
from robust_cvar_portfolio.risk.kappa import KappaParams
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.backtest import crisis_loss
from robust_cvar_portfolio.src.baselines import historical_cvar_backtest
from robust_cvar_portfolio.src.risk_metrics import summarize_backtest

ROOT = Path(__file__).resolve().parents[1]
V5_CFG_PATH = ROOT / "configs" / "v5_common.yaml"
DATA_ROOT = ROOT / "data" / "processed"

DATASET_YAML = {
    "etf10": "etf10.yaml",
    "etf20": "etf20.yaml",
    "sp30": "sp30.yaml",
    "sp100": "sp100.yaml",
}


def load_v5_config() -> dict[str, Any]:
    with V5_CFG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def v5_out_dir(dataset: str) -> Path:
    return ROOT / "outputs" / "v5" / dataset


def load_v5_bundle(dataset: str) -> dict[str, Any]:
    if dataset not in DATASET_YAML:
        raise ValueError(f"unknown dataset: {dataset}")
    cfg_path = ROOT / "configs" / DATASET_YAML[dataset]
    if dataset == "sp100":
        bundle = load_sp100_universe(cfg_path, DATA_ROOT / "sp100")
    else:
        bundle = load_dataset(cfg_path, DATA_ROOT)
    states = build_state_matrix(bundle["returns"])
    learned = _learn_params(bundle["returns"], states, bundle["config"])
    bundle["states"] = states
    bundle["learned"] = learned
    bundle["n_assets"] = bundle["returns"].shape[1]
    return bundle


def engine_historical(config: dict) -> RiskEngine:
    return RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="plain_ceil", params=KappaParams())


def engine_fixed(config: dict) -> RiskEngine:
    return RiskEngine(
        alpha=config.get("alpha", 0.05),
        kappa_mode="fixed",
        params=KappaParams(),
        fixed_k=config.get("fixed_kappa", 2.0),
    )


def engine_manual(config: dict, learned: KappaParams, kappa_max: float) -> RiskEngine:
    params = KappaParams(
        kappa_max=kappa_max,
        beta_vol=config.get("beta_vol", 1.0),
        beta_dd=config.get("beta_dd", 1.0),
        beta_mom=config.get("beta_mom", 0.5),
        beta_corr=config.get("beta_corr", 0.5),
        beta_conc=config.get("beta_conc", 0.5),
        theta=learned.theta,
    )
    params.theta = np.array([0.0, 0.0, 0.0, 0.0])
    return RiskEngine(alpha=config.get("alpha", 0.05), kappa_mode="manual", params=params)


def avg_weight_hhi(weights: pd.DataFrame, tickers: list[str]) -> float:
    if weights.empty:
        return 0.0
    w = weights[tickers].values
    return float(np.mean(np.sum(w**2, axis=1)))


def validation_objective(
    metrics: dict,
    avg_hhi: float,
    v5_cfg: dict,
) -> float:
    sel = v5_cfg["selection_objective"]
    return (
        metrics["cvar_5pct"]
        + sel["lambda_turnover"] * metrics["avg_turnover"]
        + sel["lambda_hhi"] * avg_hhi
    )


def run_val_backtest(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    engine: RiskEngine,
    config: dict,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    val_start, val_end = config["splits"]["val"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)
    frame, weights = run_rolling(
        returns,
        states,
        engine,
        val_start,
        val_end,
        cost,
        window,
        maxiter,
        weight_cap=weight_cap,
        kappa_rho=kappa_rho,
        record_weights=True,
    )
    tickers = list(returns.columns)
    m = summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], config.get("alpha", 0.05))
    hhi = avg_weight_hhi(weights, tickers)
    m["avg_hhi"] = hhi
    return frame, weights, m


def run_test_backtest(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    engine: RiskEngine,
    config: dict,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
) -> pd.DataFrame:
    test_start, test_end = config["splits"]["test"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)
    frame = run_rolling(
        returns,
        states,
        engine,
        test_start,
        test_end,
        cost,
        window,
        maxiter,
        weight_cap=weight_cap,
        kappa_rho=kappa_rho,
    )
    return frame


def metrics_row(frame: pd.DataFrame, config: dict, name: str, group: str = "v5") -> dict:
    m = summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], config.get("alpha", 0.05))
    m["method"] = name
    m["group"] = group
    idx = frame.set_index("date") if "date" in frame.columns else frame
    m["crisis_2020"] = crisis_loss(idx["net_return"], "2020-02-01", "2020-04-30")
    m["crisis_2022"] = crisis_loss(idx["net_return"], "2022-01-01", "2022-12-31")
    return m


def run_historical_test(returns: pd.DataFrame, config: dict) -> pd.DataFrame:
    test_start, test_end = config["splits"]["test"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)
    hist = historical_cvar_backtest(
        returns, test_start, test_end, window, cost, config.get("alpha", 0.05), maxiter
    )
    return hist.reset_index().rename(columns={"index": "date"})


def save_selected_params(path: Path, params: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)


def load_selected_params(path: Path) -> dict:
    with path.open(encoding="utf-8") as f:
        return json.load(f)


def parse_cap(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.lower() == "null"):
        return None
    return float(value)


def parse_rho(value: Any) -> float | None:
    if value is None or (isinstance(value, str) and value.lower() == "null"):
        return None
    return float(value)


def weight_diff_max(w_a: pd.DataFrame, w_b: pd.DataFrame, tickers: list[str]) -> float:
    common = w_a.index.intersection(w_b.index)
    if len(common) == 0:
        return float("nan")
    a = w_a.loc[common, tickers].values
    b = w_b.loc[common, tickers].values
    return float(np.max(np.abs(a - b)))


def export_test_weights(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    engine: RiskEngine,
    config: dict,
    out_path: Path,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
) -> pd.DataFrame:
    test_start, test_end = config["splits"]["test"]
    cost = config.get("cost_rate", 0.001)
    window = config.get("estimation_window", 252)
    maxiter = config.get("optimizer_maxiter", 50)
    wexp = export_rebalance_weights(
        returns,
        states,
        engine,
        test_start,
        test_end,
        cost,
        window,
        maxiter,
        weight_cap=weight_cap,
        kappa_rho=kappa_rho,
    )
    tickers = [c for c in wexp.columns if c not in {"kappa", "turnover"}]
    wexp[tickers].to_csv(out_path, index_label="date")
    return wexp
