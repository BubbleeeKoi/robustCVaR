"""Point-in-time index constituents and price panels (DJI30 / NDX100 / SP500)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from robust_cvar_portfolio.data.loader import build_state_matrix
from robust_cvar_portfolio.portfolio.rolling import monthly_rebalance_dates
from robust_cvar_portfolio.src.data_loader import compute_returns, download_prices

ROOT = Path(__file__).resolve().parents[1]
RAW_CONST = ROOT / "data" / "raw" / "constituents"
PROC = ROOT / "data" / "processed"

INDEX_FILES = {
    "dji30": "dow30.csv",
    "ndx100": "nasdaq100.csv",
    "sp500": "sp500.csv",
}


def load_index_config(name: str) -> dict[str, Any]:
    path = ROOT / "configs" / f"{name}.yaml"
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_index_history(name: str) -> pd.DataFrame:
    fname = INDEX_FILES.get(name)
    if fname is None:
        raise ValueError(f"unknown index: {name}")
    path = RAW_CONST / fname
    if not path.exists():
        raise FileNotFoundError(f"missing constituents file: {path}")
    df = pd.read_csv(path, encoding="utf-8-sig")
    df["opt-in"] = pd.to_datetime(df["opt-in"])
    df["opt-out"] = pd.to_datetime(df["opt-out"], errors="coerce")
    return df


def is_member(row: pd.Series, date: pd.Timestamp) -> bool:
    d = pd.Timestamp(date).normalize()
    if row["opt-in"] > d:
        return False
    if pd.notna(row["opt-out"]) and row["opt-out"] <= d:
        return False
    return True


def constituents_at(history: pd.DataFrame, date: pd.Timestamp) -> list[str]:
    active: set[str] = set()
    for _, row in history.iterrows():
        if is_member(row, date):
            active.add(str(row["symbol"]).strip())
    return sorted(active)


def union_symbols(history: pd.DataFrame, start: str, end: str) -> list[str]:
    start_d, end_d = pd.Timestamp(start), pd.Timestamp(end)
    syms: set[str] = set()
    for _, row in history.iterrows():
        opt_in = row["opt-in"]
        opt_out = row["opt-out"] if pd.notna(row["opt-out"]) else pd.Timestamp("2099-12-31")
        if opt_in <= end_d and opt_out >= start_d:
            syms.add(str(row["symbol"]).strip())
    return sorted(syms)


def build_proxy_returns(prices: pd.DataFrame, history: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight daily returns of PIT active members with available prices."""
    vals: dict[pd.Timestamp, float] = {}
    for i, date in enumerate(prices.index):
        if i == 0:
            continue
        prev_date = prices.index[i - 1]
        members = constituents_at(history, date)
        cols = [c for c in members if c in prices.columns]
        prev = prices.loc[prev_date, cols]
        curr = prices.loc[date, cols]
        ok = prev.notna() & curr.notna() & (prev > 0)
        if ok.sum() == 0:
            continue
        vals[date] = float((curr[ok] / prev[ok] - 1.0).mean())
    if not vals:
        raise RuntimeError("empty proxy returns")
    return pd.DataFrame({"proxy": pd.Series(vals).sort_index()})


def filter_available(
    tickers: list[str],
    returns: pd.DataFrame,
    date: pd.Timestamp,
    window: int,
    min_frac: float = 0.8,
) -> tuple[list[str], list[str]]:
    loc = returns.index.get_loc(date)
    if loc < window:
        return [], tickers
    hist = returns.iloc[loc - window : loc]
    used, dropped = [], []
    for t in tickers:
        if t not in returns.columns:
            dropped.append(f"{t}:missing_col")
            continue
        col = hist[t]
        valid = col.notna().sum()
        if valid >= window * min_frac:
            used.append(t)
        else:
            dropped.append(f"{t}:insufficient({valid}/{window})")
    return used, dropped


def build_index_panel(
    name: str,
    force: bool = False,
    sleep_sec: float = 0.15,
) -> dict[str, Any]:
    config = load_index_config(name)
    out_dir = PROC / name
    out_dir.mkdir(parents=True, exist_ok=True)
    prices_path = out_dir / "prices.csv"
    returns_path = out_dir / "returns.csv"
    meta_path = out_dir / "index_meta.json"

    history = load_index_history(name)
    if prices_path.exists() and returns_path.exists() and not force:
        prices = pd.read_csv(prices_path, index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
    else:
        tickers = union_symbols(history, config["start_date"], config["end_date"])
        print(f"  [{name}] downloading {len(tickers)} tickers...", flush=True)
        prices = download_prices(
            tickers,
            config["start_date"],
            config["end_date"],
            sleep_sec=sleep_sec,
            min_days=60,
            require_full_overlap=False,
        )
        returns = prices.pct_change(fill_method=None)
        prices.to_csv(prices_path, index_label="date")
        returns.to_csv(returns_path, index_label="date")

    proxy_returns = build_proxy_returns(prices, history)
    states = build_state_matrix(proxy_returns)

    meta = {
        "index_name": name,
        "benchmark_ticker": config.get("benchmark_ticker"),
        "point_in_time_flag": True,
        "n_union_tickers": int(prices.shape[1]),
        "data_source": "unliftedq/index-constitution CSV",
    }
    with meta_path.open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    splits_path = out_dir / "splits.json"
    if not splits_path.exists():
        with splits_path.open("w", encoding="utf-8") as f:
            json.dump(config["splits"], f, indent=2)

    return {
        "config": config,
        "history": history,
        "prices": prices,
        "returns": returns,
        "proxy_returns": proxy_returns,
        "states": states,
        "dir": out_dir,
        "constituents_at": lambda d: constituents_at(history, d),
    }


def rebalance_constituent_log(
    returns: pd.DataFrame,
    history: pd.DataFrame,
    start: str,
    end: str,
    window: int = 252,
    min_frac: float = 0.8,
) -> pd.DataFrame:
    rebal = monthly_rebalance_dates(returns.index)
    rows = []
    prev_set: set[str] = set()
    for date in rebal:
        if date < pd.Timestamp(start) or date > pd.Timestamp(end):
            continue
        loc = returns.index.get_loc(date)
        if loc < window:
            continue
        raw = constituents_at(history, date)
        used, dropped = filter_available(raw, returns, date, window, min_frac)
        curr = set(used)
        turnover = 1 - len(curr & prev_set) / max(len(curr | prev_set), 1) if prev_set else 0.0
        rows.append(
            {
                "rebalance_date": date,
                "index_name": "",
                "n_constituents_raw": len(raw),
                "n_available": len(used),
                "n_used": len(used),
                "dropped_tickers": ";".join(dropped[:20]),
                "point_in_time_flag": True,
                "universe_turnover": turnover,
                "constituents_at_t": ",".join(used),
            }
        )
        prev_set = curr
    return pd.DataFrame(rows)
