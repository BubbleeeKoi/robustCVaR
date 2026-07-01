"""Rolling backtest engine (V2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_cvar_portfolio.portfolio.optimizer import optimize_portfolio
from robust_cvar_portfolio.risk.kappa import KappaParams
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, turnover


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    groups = index.to_series().groupby([index.year, index.month])
    return [idx.max() for _, idx in groups]


def run_rolling(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    engine: RiskEngine,
    start: str,
    end: str,
    cost_rate: float = 0.001,
    estimation_window: int = 252,
    optimizer_maxiter: int = 150,
) -> pd.DataFrame:
    rebalance_dates = set(monthly_rebalance_dates(returns.index))
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    period_index = returns.index[mask]
    n_assets = returns.shape[1]
    w_prev = np.full(n_assets, 1.0 / n_assets)
    current_w = w_prev.copy()
    rows = []

    for date in period_index:
        loc = idx_map[date]
        day_to = 0.0
        kappa_t = np.nan
        if date in rebalance_dates and loc >= estimation_window:
            hist = returns.iloc[loc - estimation_window : loc]
            feat = features.iloc[loc - estimation_window : loc]
            old_w = w_prev.copy()
            current_w, _, kappa_t = optimize_portfolio(
                hist, feat, old_w, engine, cost_rate, maxiter=optimizer_maxiter
            )
            day_to = turnover(current_w, old_w)
            w_prev = current_w.copy()

        net_ret = float(current_w @ returns.loc[date].values - cost_rate * day_to)
        rows.append(
            {
                "date": date,
                "net_return": net_ret,
                "loss": -net_ret,
                "turnover": day_to,
                "kappa": kappa_t,
            }
        )

    frame = pd.DataFrame(rows)
    frame["rolling_cvar_60"] = frame["loss"].rolling(60, min_periods=20).apply(
        lambda x: cvar_alpha(x.values, 0.05), raw=False
    )
    return frame
