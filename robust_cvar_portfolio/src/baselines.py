"""Traditional portfolio baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .risk_metrics import net_portfolio_return, summarize_backtest, turnover


def equal_weight_backtest(
    returns: pd.DataFrame,
    start: str,
    end: str,
    cost_rate: float = 0.001,
    rebalance: str = "monthly",
) -> pd.DataFrame:
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = returns.loc[mask]
    n = sub.shape[1]
    w = np.full(n, 1.0 / n)
    w_prev = w.copy()
    rows = []
    for date, row in sub.iterrows():
        if rebalance == "monthly" and date.is_month_end:
            w_prev = w.copy()
        net_ret = net_portfolio_return(w, row.values, w_prev, cost_rate)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": turnover(w, w_prev)})
    return pd.DataFrame(rows).set_index("date")


def min_variance_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    inv = np.linalg.pinv(cov)
    raw = inv @ np.ones(n)
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 1e-12:
        return np.full(n, 1.0 / n)
    return raw / raw.sum()


def min_variance_backtest(
    returns: pd.DataFrame,
    start: str,
    end: str,
    window: int = 252,
    cost_rate: float = 0.001,
) -> pd.DataFrame:
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = returns.loc[mask]
    n = sub.shape[1]
    w = np.full(n, 1.0 / n)
    w_prev = w.copy()
    rows = []
    for i, (date, row) in enumerate(sub.iterrows()):
        full_loc = returns.index.get_loc(date)
        if date.is_month_end and full_loc >= window:
            hist = returns.iloc[full_loc - window : full_loc]
            w = min_variance_weights(hist.cov().values)
            w_prev = w.copy()
        net_ret = net_portfolio_return(w, row.values, w_prev, cost_rate)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": turnover(w, w_prev)})
    return pd.DataFrame(rows).set_index("date")


def metrics_from_frame(frame: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    return summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], alpha=alpha)
