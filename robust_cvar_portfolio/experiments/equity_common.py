"""V6 equity-only shared utilities."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_cvar_portfolio.experiments.v5_common import (
    ROOT,
    avg_weight_hhi,
    engine_fixed,
    engine_historical,
    engine_manual,
    load_v5_bundle,
    metrics_row,
    parse_cap,
    parse_rho,
    run_test_backtest,
    run_val_backtest,
    v5_out_dir,
)
from robust_cvar_portfolio.portfolio.rolling import monthly_rebalance_dates

EQUITY_OUT = ROOT / "outputs" / "equity_only"
V6_CFG_PATH = ROOT / "configs" / "v6_equity_common.yaml"

MODELS_EQUITY = [
    ("A_ceil_CVaR", "baseline"),
    ("B_fixed_kappa", "baseline"),
    ("C_default", "proposed"),
    ("C_stable", "proposed"),
]


def load_v6_config() -> dict[str, Any]:
    with V6_CFG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def equity_dir(name: str) -> Path:
    return EQUITY_OUT / name


def v6_objective(metrics: dict, avg_hhi: float, v6_cfg: dict) -> float:
    sel = v6_cfg["selection_objective"]
    return (
        metrics["cvar_5pct"]
        + sel["lambda_turnover"] * metrics["avg_turnover"]
        + sel["lambda_hhi"] * avg_hhi
    )


def calibrate_c_stable(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    config: dict,
    learned,
    v6_cfg: dict,
) -> dict[str, Any]:
    """Staged validation selection for C_stable (kappa -> cap -> rho)."""
    km_grid = v6_cfg["kappa_max_grid"]
    cap_grid = [parse_cap(x) for x in v6_cfg["weight_cap_grid"]]
    rho_grid = [parse_rho(x) for x in v6_cfg["rho_grid"]]

    best_km, best_j = km_grid[0], float("inf")
    km_rows = []
    for km in km_grid:
        engine = engine_manual(config, learned, km)
        _, _, m = run_val_backtest(returns, states, engine, config)
        j = v6_objective(m, m["avg_hhi"], v6_cfg)
        km_rows.append({"kappa_max": km, "J_val": j, **m})
        if j < best_j:
            best_j, best_km = j, km

    best_cap, best_j_cap = None, float("inf")
    cap_rows = []
    for cap in cap_grid:
        engine = engine_manual(config, learned, best_km)
        _, _, m = run_val_backtest(returns, states, engine, config, weight_cap=cap)
        j = v6_objective(m, m["avg_hhi"], v6_cfg)
        cap_rows.append({"kappa_max": best_km, "weight_cap": cap, "J_val": j, **m})
        if j < best_j_cap:
            best_j_cap, best_cap = j, cap

    best_rho, best_j_rho = None, float("inf")
    rho_rows = []
    for rho in rho_grid:
        engine = engine_manual(config, learned, best_km)
        _, _, m = run_val_backtest(
            returns, states, engine, config, weight_cap=best_cap, kappa_rho=rho
        )
        j = v6_objective(m, m["avg_hhi"], v6_cfg)
        rho_rows.append(
            {
                "kappa_max": best_km,
                "weight_cap": best_cap,
                "rho": rho,
                "J_val": j,
                **m,
            }
        )
        if j < best_j_rho:
            best_j_rho, best_rho = j, rho

    return {
        "kappa_max": best_km,
        "weight_cap": best_cap,
        "rho": best_rho,
        "kappa_grid": pd.DataFrame(km_rows),
        "cap_grid": pd.DataFrame(cap_rows),
        "rho_grid": pd.DataFrame(rho_rows),
    }


def run_equity_models_test(
    returns: pd.DataFrame,
    states: pd.DataFrame,
    config: dict,
    learned,
    stable_params: dict,
    out_dir: Path,
    prefix: str = "",
) -> pd.DataFrame:
    """Run 4 equity models on test; use checkpoints."""
    out_dir.mkdir(parents=True, exist_ok=True)
    km_def = config.get("kappa_max", 1.0)
    sp = stable_params

    specs: list[tuple[str, object, float | None, float | None]] = [
        ("A_ceil_CVaR", engine_historical(config), None, None),
        ("B_fixed_kappa", engine_fixed(config), None, None),
        ("C_default", engine_manual(config, learned, km_def), None, None),
        (
            "C_stable",
            engine_manual(config, learned, sp["kappa_max"]),
            sp.get("weight_cap"),
            sp.get("rho"),
        ),
    ]

    rows = []
    for name, engine, cap, rho in specs:
        tag = f"{prefix}{name}" if prefix else name
        ckpt = out_dir / f"rolling_{tag}.csv"
        if ckpt.exists():
            frame = pd.read_csv(ckpt, parse_dates=["date"])
        else:
            frame = run_test_backtest(returns, states, engine, config, weight_cap=cap, kappa_rho=rho)
            frame.to_csv(ckpt, index=False)
        rows.append(metrics_row(frame, config, name, group="equity_v6"))

    table = pd.DataFrame(rows)
    table.to_csv(out_dir / f"{prefix}table_main.csv" if prefix else out_dir / "table_main.csv", index=False)
    return table


def copy_v5_equity_results(datasets: list[str] | None = None) -> None:
    """Copy SP30/SP100 V5 outputs into equity_only layout."""
    datasets = datasets or ["sp30", "sp100"]
    rename = {
        "A_Historical_CVaR": "A_ceil_CVaR",
    }
    for ds in datasets:
        src = v5_out_dir(ds)
        dst = equity_dir(ds)
        dst.mkdir(parents=True, exist_ok=True)
        test_src = src / "test"
        if not test_src.exists():
            continue
        for f in test_src.glob("rolling_*.csv"):
            name = f.stem.replace("rolling_", "")
            new_name = rename.get(name, name)
            shutil.copy(f, dst / f"rolling_{new_name}.csv")
        if (test_src / "table_main.csv").exists():
            tbl = pd.read_csv(test_src / "table_main.csv")
            tbl["method"] = tbl["method"].replace(rename)
            tbl.to_csv(dst / "table_main_full.csv", index=False)
            main = tbl[tbl["method"].isin([m[0] for m in MODELS_EQUITY])].copy()
            main.to_csv(dst / "table_main.csv", index=False)
        fig_src = src / "figures"
        if fig_src.exists():
            fig_dst = dst / "figures"
            fig_dst.mkdir(exist_ok=True)
            for fig in fig_src.glob("*.png"):
                shutil.copy(fig, fig_dst / fig.name)
        val_src = src / "validation" / "selected_params.json"
        if val_src.exists():
            shutil.copy(val_src, dst / "selected_params.json")


def effective_n(hhi: float) -> float:
    return 1.0 / hhi if hhi > 1e-12 else float("nan")


def cross_sectional_metrics(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    """Daily cross-sectional structure metrics."""
    arr = returns.values
    idx = returns.index
    rows = []
    for t in range(window, len(returns)):
        block = arr[t - window : t]
        corr = np.corrcoef(block.T)
        if corr.ndim < 2:
            continue
        n = corr.shape[0]
        off = corr[np.triu_indices(n, k=1)]
        avg_corr = float(np.nanmean(off))
        cov = np.cov(block.T)
        eig = np.linalg.eigvalsh(cov)
        eig = np.maximum(eig, 0)
        total = eig.sum()
        pc1 = float(eig[-1] / total) if total > 1e-12 else np.nan
        d_eff = float(total**2 / np.sum(eig**2)) if np.sum(eig**2) > 1e-12 else np.nan
        vols = block.std(axis=0)
        vol_disp = float(np.std(vols))
        rows.append(
            {
                "date": idx[t],
                "avg_correlation": avg_corr,
                "pc1_share": pc1,
                "effective_dimension": d_eff,
                "vol_dispersion": vol_disp,
            }
        )
    return pd.DataFrame(rows).set_index("date")


def rolling_cvar_gap(frame_a: pd.DataFrame, frame_c: pd.DataFrame, window: int = 60) -> pd.Series:
    from robust_cvar_portfolio.src.risk_metrics import cvar_alpha

    a = frame_a.set_index("date") if "date" in frame_a.columns else frame_a
    c = frame_c.set_index("date") if "date" in frame_c.columns else frame_c
    aligned = a.join(c, lsuffix="_a", rsuffix="_c", how="inner")
    gaps = []
    dates = []
    for i in range(window, len(aligned)):
        sub_a = aligned["loss_a"].iloc[i - window : i].values
        sub_c = aligned["loss_c"].iloc[i - window : i].values
        gaps.append(cvar_alpha(sub_c, 0.05) - cvar_alpha(sub_a, 0.05))
        dates.append(aligned.index[i])
    return pd.Series(gaps, index=dates, name="cvar_gap_c_minus_a")


def save_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, default=str)
