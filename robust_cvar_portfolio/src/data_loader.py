"""Load ETF price data via akshare and persist processed returns."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import akshare as ak
import numpy as np
import pandas as pd
import yaml


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def _fetch_us_daily(symbol: str, start: str, end: str) -> pd.Series:
    """Download one US ticker via akshare and return close prices."""
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            frame = ak.stock_us_daily(symbol=symbol, adjust="")
            if frame is None or frame.empty:
                raise ValueError(f"empty data for {symbol}")
            date_col = "date" if "date" in frame.columns else frame.columns[0]
            close_col = "close" if "close" in frame.columns else "Close"
            series = frame[[date_col, close_col]].copy()
            series[date_col] = pd.to_datetime(series[date_col])
            series = series.set_index(date_col)[close_col].astype(float)
            series = series.sort_index()
            series = series.loc[start:end]
            if series.empty:
                raise ValueError(f"no rows in [{start}, {end}] for {symbol}")
            return series.rename(symbol)
        except Exception as exc:  # noqa: BLE001 - retry wrapper
            last_error = exc
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"failed to download {symbol}: {last_error}")


def download_prices(
    tickers: list[str],
    start_date: str,
    end_date: str,
    sleep_sec: float = 0.3,
) -> pd.DataFrame:
    frames: list[pd.Series] = []
    for ticker in tickers:
        frames.append(_fetch_us_daily(ticker, start_date, end_date))
        time.sleep(sleep_sec)
    prices = pd.concat(frames, axis=1).sort_index()
    prices = prices.dropna(how="any")
    return prices


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna(how="any")
    return returns


def build_splits(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    return {name: tuple(bounds) for name, bounds in config["splits"].items()}


def summarize_dataset(prices: pd.DataFrame, returns: pd.DataFrame, splits: dict) -> pd.DataFrame:
    rows = []
    for name, (start, end) in splits.items():
        mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        sub = returns.loc[mask]
        rows.append(
            {
                "split": name,
                "start": start,
                "end": end,
                "n_days": len(sub),
                "n_assets": returns.shape[1],
            }
        )
    rows.append(
        {
            "split": "full",
            "start": str(prices.index.min().date()),
            "end": str(prices.index.max().date()),
            "n_days": len(returns),
            "n_assets": returns.shape[1],
        }
    )
    return pd.DataFrame(rows)


def run_data_pipeline(
    config_path: Path,
    output_dir: Path,
) -> dict[str, Path]:
    config = load_config(config_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    prices = download_prices(
        tickers=config["tickers"],
        start_date=config["start_date"],
        end_date=config["end_date"],
    )
    returns = compute_returns(prices)
    splits = build_splits(config)
    summary = summarize_dataset(prices, returns, splits)

    prices_path = output_dir / "prices.csv"
    returns_path = output_dir / "returns.csv"
    splits_path = output_dir / "splits.json"
    summary_path = output_dir / "dataset_summary.csv"

    prices.to_csv(prices_path, index_label="date")
    returns.to_csv(returns_path, index_label="date")
    with splits_path.open("w", encoding="utf-8") as handle:
        json.dump(splits, handle, indent=2, ensure_ascii=False)
    summary.to_csv(summary_path, index=False)

    return {
        "prices": prices_path,
        "returns": returns_path,
        "splits": splits_path,
        "summary": summary_path,
    }
