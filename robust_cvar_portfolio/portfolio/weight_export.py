"""Export portfolio weights at monthly rebalance dates."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_cvar_portfolio.portfolio.optimizer import optimize_portfolio
from robust_cvar_portfolio.portfolio.rolling import _engine_for_rebalance, monthly_rebalance_dates
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.risk_metrics import turnover


def export_rebalance_weights(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    engine: RiskEngine,
    start: str,
    end: str,
    cost_rate: float = 0.001,
    estimation_window: int = 252,
    optimizer_maxiter: int = 50,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
    hhi_penalty: float = 0.0,
) -> pd.DataFrame:
    """Return weight matrix (rows=rebalance dates, cols=tickers)."""
    rebalance_dates = set(monthly_rebalance_dates(returns.index))
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    period_index = returns.index[mask]
    n_assets = returns.shape[1]
    tickers = list(returns.columns)
    w_prev = np.full(n_assets, 1.0 / n_assets)
    kappa_bar_prev = 1.0
    rows: list[dict] = []

    for date in period_index:
        loc = idx_map[date]
        if date in rebalance_dates and loc >= estimation_window:
            hist = returns.iloc[loc - estimation_window : loc]
            feat = features.iloc[loc - estimation_window : loc]
            old_w = w_prev.copy()
            opt_engine, kappa_bar_prev = _engine_for_rebalance(
                engine, features, loc, kappa_rho, kappa_bar_prev
            )
            w, _, kappa_t = optimize_portfolio(
                hist,
                feat,
                old_w,
                opt_engine,
                cost_rate,
                maxiter=optimizer_maxiter,
                weight_cap=weight_cap,
                hhi_penalty=hhi_penalty,
            )
            w_prev = w.copy()
            row = {"date": date, "kappa": kappa_t, "turnover": turnover(w, old_w)}
            for i, t in enumerate(tickers):
                row[t] = w[i]
            rows.append(row)

    if not rows:
        return pd.DataFrame()
    frame = pd.DataFrame(rows).set_index("date")
    return frame
