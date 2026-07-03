"""Rolling backtest engine (V2)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_cvar_portfolio.portfolio.optimizer import optimize_portfolio
from robust_cvar_portfolio.risk.kappa import KappaParams, kappa_manual
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, turnover


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    groups = index.to_series().groupby([index.year, index.month])
    return [idx.max() for _, idx in groups]


def _engine_for_rebalance(
    engine: RiskEngine,
    features: pd.DataFrame,
    loc: int,
    kappa_rho: float | None,
    kappa_bar_prev: float,
) -> tuple[RiskEngine, float]:
    if kappa_rho is None or engine.kappa_mode != "manual":
        return engine, kappa_bar_prev
    state_row = features.iloc[loc]
    kappa_raw = kappa_manual(state_row, engine.params)
    kappa_bar = kappa_rho * kappa_bar_prev + (1.0 - kappa_rho) * kappa_raw
    smooth_engine = RiskEngine(
        alpha=engine.alpha,
        kappa_mode="fixed",
        params=KappaParams(),
        fixed_k=kappa_bar,
    )
    return smooth_engine, kappa_bar


def run_rolling(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    engine: RiskEngine,
    start: str,
    end: str,
    cost_rate: float = 0.001,
    estimation_window: int = 252,
    optimizer_maxiter: int = 150,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
    hhi_penalty: float = 0.0,
    record_weights: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    rebalance_dates = set(monthly_rebalance_dates(returns.index))
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    period_index = returns.index[mask]
    n_assets = returns.shape[1]
    tickers = list(returns.columns)
    w_prev = np.full(n_assets, 1.0 / n_assets)
    current_w = w_prev.copy()
    kappa_bar_prev = 1.0
    rows = []
    weight_rows: list[dict] = []

    for date in period_index:
        loc = idx_map[date]
        day_to = 0.0
        kappa_t = np.nan
        if date in rebalance_dates and loc >= estimation_window:
            hist = returns.iloc[loc - estimation_window : loc]
            feat = features.iloc[loc - estimation_window : loc]
            old_w = w_prev.copy()
            opt_engine, kappa_bar_prev = _engine_for_rebalance(
                engine, features, loc, kappa_rho, kappa_bar_prev
            )
            current_w, _, kappa_t = optimize_portfolio(
                hist,
                feat,
                old_w,
                opt_engine,
                cost_rate,
                maxiter=optimizer_maxiter,
                weight_cap=weight_cap,
                hhi_penalty=hhi_penalty,
            )
            day_to = turnover(current_w, old_w)
            w_prev = current_w.copy()
            if record_weights:
                row = {"date": date, "kappa": kappa_t, "turnover": day_to}
                for i, t in enumerate(tickers):
                    row[t] = current_w[i]
                weight_rows.append(row)

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
    if record_weights:
        weights = pd.DataFrame(weight_rows)
        return frame, weights
    return frame
