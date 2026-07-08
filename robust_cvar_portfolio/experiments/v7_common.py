"""V7 shared utilities: structure metrics, paths, calibration."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_cvar_portfolio.experiments.equity_common import (
    calibrate_c_stable,
    cross_sectional_metrics,
    effective_n,
    load_v6_config,
)
from robust_cvar_portfolio.experiments.v5_common import (
    ROOT,
    engine_manual,
    load_v5_bundle,
    metrics_row,
    run_test_backtest,
    run_val_backtest,
)

V7_OUT = ROOT / "outputs" / "v7"
V7_CFG_PATH = ROOT / "configs" / "v7_common.yaml"
EQUITY_OUT = ROOT / "outputs" / "equity_only"

MODELS_V7 = ["A_ceil_CVaR", "B_fixed_kappa", "C_default", "C_stable"]


def load_v7_config() -> dict[str, Any]:
    with V7_CFG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def v7_dir(name: str) -> Path:
    return V7_OUT / name


def paper_dir() -> Path:
    d = V7_OUT / "paper_tables"
    d.mkdir(parents=True, exist_ok=True)
    return d


def effective_dimension_from_block(block: np.ndarray) -> float:
    if block.shape[0] < 5:
        return float(block.shape[1])
    cov = np.cov(block.T)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    denom = float(np.sum(eig**2))
    if denom < 1e-12:
        return float(block.shape[1])
    return float(np.sum(eig) ** 2 / denom)


def avg_correlation_from_block(block: np.ndarray) -> float:
    corr = np.corrcoef(block.T)
    n = corr.shape[0]
    off = corr[np.triu_indices(n, k=1)]
    return float(np.nanmean(off))


def pc1_share_from_block(block: np.ndarray) -> float:
    cov = np.cov(block.T)
    eig = np.maximum(np.linalg.eigvalsh(cov), 0.0)
    total = float(np.sum(eig))
    if total < 1e-12:
        return np.nan
    return float(eig[-1] / total)


def validation_structure_summary(
    returns: pd.DataFrame,
    val_start: str,
    val_end: str,
    window: int = 60,
) -> dict[str, float]:
    sub = returns.loc[val_start:val_end]
    xs = cross_sectional_metrics(sub, window=window)
    if xs.empty:
        block = sub.values
        return {
            "avg_correlation_val": avg_correlation_from_block(block),
            "pc1_share_val": pc1_share_from_block(block),
            "effective_dimension_val": effective_dimension_from_block(block),
        }
    return {
        "avg_correlation_val": float(xs["avg_correlation"].mean()),
        "pc1_share_val": float(xs["pc1_share"].mean()),
        "effective_dimension_val": float(xs["effective_dimension"].mean()),
    }


def test_structure_summary(
    returns: pd.DataFrame,
    test_start: str,
    test_end: str,
    window: int = 60,
) -> dict[str, float]:
    sub = returns.loc[test_start:test_end]
    xs = cross_sectional_metrics(sub, window=window)
    if xs.empty:
        block = sub.values
        return {
            "avg_correlation_test": avg_correlation_from_block(block),
            "pc1_share_test": pc1_share_from_block(block),
            "effective_dimension_test": effective_dimension_from_block(block),
        }
    return {
        "avg_correlation_test": float(xs["avg_correlation"].mean()),
        "pc1_share_test": float(xs["pc1_share"].mean()),
        "effective_dimension_test": float(xs["effective_dimension"].mean()),
    }


def universe_avg_corr_val(returns: pd.DataFrame, config: dict, window: int = 60) -> float:
    val_start, val_end = config["splits"]["val"]
    return validation_structure_summary(returns, val_start, val_end, window)["avg_correlation_val"]


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)


def engine_effdim(config: dict, learned, kappa_max: float, d0: float):
    return engine_manual(config, learned, kappa_max), d0


def calibrate_v7_effdim(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    config: dict,
    learned,
    v7_cfg: dict,
    d0: float,
    use_cap: bool = False,
) -> dict[str, Any]:
    """Validation selection for V7_effdim or V7_effdim_cap."""
    v6_cfg = load_v6_config()
    sel = calibrate_c_stable(returns, states, config, learned, v6_cfg)
    if not use_cap:
        sel["weight_cap"] = None
        sel["rho"] = None
    sel["d0"] = d0
    sel["model"] = "V7_effdim_cap" if use_cap else "V7_effdim"
    return sel


def run_v7_models_test(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    config: dict,
    learned,
    stable_params: dict,
    out_dir: Path,
    d0: float,
    models: list[str] | None = None,
) -> pd.DataFrame:
    from robust_cvar_portfolio.experiments.v5_common import engine_fixed, engine_historical

    out_dir.mkdir(parents=True, exist_ok=True)
    km_def = config.get("kappa_max", 1.0)
    sp = stable_params
    models = models or ["C_stable", "V7_effdim", "V7_effdim_cap"]

    rows = []
    all_specs: list[tuple[str, object, float | None, float | None, float | None]] = [
        ("A_ceil_CVaR", engine_historical(config), None, None, None),
        ("B_fixed_kappa", engine_fixed(config), None, None, None),
        ("C_default", engine_manual(config, learned, km_def), None, None, None),
        (
            "C_stable",
            engine_manual(config, learned, sp["kappa_max"]),
            sp.get("weight_cap"),
            sp.get("rho"),
            None,
        ),
        (
            "V7_effdim",
            engine_manual(config, learned, sp["kappa_max"]),
            None,
            None,
            d0,
        ),
        (
            "V7_effdim_cap",
            engine_manual(config, learned, sp["kappa_max"]),
            sp.get("weight_cap"),
            sp.get("rho"),
            d0,
        ),
    ]
    spec_map = {name: spec for spec in all_specs for name in [spec[0]]}

    for name in models:
        if name not in spec_map:
            continue
        _, engine, cap, rho, ed0 = spec_map[name]
        ckpt = out_dir / f"rolling_{name}.csv"
        if ckpt.exists():
            frame = pd.read_csv(ckpt, parse_dates=["date"])
        else:
            frame = run_test_backtest(
                returns, states, engine, config,
                weight_cap=cap, kappa_rho=rho, effdim_d0=ed0,
            )
            frame.to_csv(ckpt, index=False)
        rows.append(metrics_row(frame, config, name, group="v7"))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / "table_main.csv", index=False)
    return table


def cvar_gap_c_minus_a(c_cvar: float, a_cvar: float) -> float:
    return float(c_cvar - a_cvar)
