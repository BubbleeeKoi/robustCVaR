"""Multi-dataset loader (V2)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from robust_cvar_portfolio.src.data_loader import load_config, run_data_pipeline


def load_dataset(config_path: Path, data_root: Path, force: bool = False) -> dict[str, Any]:
    dataset_name = load_config(config_path)["dataset"]
    out_dir = data_root / dataset_name
    returns_path = out_dir / "returns.csv"
    legacy_path = data_root / "returns.csv"
    summary_path = out_dir / "dataset_summary.csv"

    need_download = force or not returns_path.exists()
    if summary_path.exists() and not need_download:
        summary = pd.read_csv(summary_path)
        train_row = summary[summary["split"] == "train"]
        if not train_row.empty and int(train_row.iloc[0]["n_days"]) == 0:
            need_download = True

    if need_download:
        out_dir.mkdir(parents=True, exist_ok=True)
        if not force and dataset_name == "etf10" and legacy_path.exists() and returns_path.exists():
            pass
        elif not force and dataset_name == "etf10" and legacy_path.exists():
            import shutil
            for fname in ["returns.csv", "prices.csv", "splits.json", "dataset_summary.csv"]:
                src = data_root / fname
                if src.exists():
                    shutil.copy(src, out_dir / fname)
        else:
            run_data_pipeline(config_path, out_dir)

    config = load_config(config_path)
    returns = pd.read_csv(out_dir / "returns.csv", index_col=0, parse_dates=True)
    prices = pd.read_csv(out_dir / "prices.csv", index_col=0, parse_dates=True)
    returns.to_csv(out_dir / "raw_data.csv", index_label="date")
    return {"config": config, "returns": returns, "prices": prices, "dataset": dataset_name, "dir": out_dir}


def build_state_matrix(returns: pd.DataFrame, z_window: int = 252) -> pd.DataFrame:
    from robust_cvar_portfolio.data.state import build_v2_state

    states = build_v2_state(returns, z_window)
    return states
