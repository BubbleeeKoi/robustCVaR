"""Traditional portfolio baselines."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk_metrics import cvar_alpha, net_portfolio_return, summarize_backtest, turnover


def _monthly_rebalance_mask(index: pd.DatetimeIndex) -> set[pd.Timestamp]:
    groups = index.to_series().groupby([index.year, index.month])
    return {idx.max() for _, idx in groups}


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


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


def buy_and_hold_backtest(
    returns: pd.DataFrame,
    start: str,
    end: str,
    cost_rate: float = 0.001,
) -> pd.DataFrame:
    """Initial equal weight, no rebalancing."""
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = returns.loc[mask]
    n = sub.shape[1]
    w = np.full(n, 1.0 / n)
    w_prev = np.zeros(n)
    rows = []
    for i, (date, row) in enumerate(sub.iterrows()):
        if i == 0:
            w_prev = np.zeros(n)
        else:
            w_prev = w.copy()
        net_ret = net_portfolio_return(w, row.values, w_prev, cost_rate if i == 0 else 0.0)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": turnover(w, w_prev)})
    return pd.DataFrame(rows).set_index("date")


def mean_variance_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float = 2.0) -> np.ndarray:
    n = len(mu)
    inv = np.linalg.pinv(cov)
    raw = inv @ mu / max(risk_aversion, 1e-6)
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 1e-12:
        return np.full(n, 1.0 / n)
    return raw / raw.sum()


def mean_variance_backtest(
    returns: pd.DataFrame,
    start: str,
    end: str,
    window: int = 252,
    cost_rate: float = 0.001,
    risk_aversion: float = 2.0,
) -> pd.DataFrame:
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = returns.loc[mask]
    n = sub.shape[1]
    w = np.full(n, 1.0 / n)
    w_prev = w.copy()
    rebal = _monthly_rebalance_mask(returns.index)
    rows = []
    for date, row in sub.iterrows():
        loc = returns.index.get_loc(date)
        if date in rebal and loc >= window:
            hist = returns.iloc[loc - window : loc]
            w_new = mean_variance_weights(hist.mean().values, hist.cov().values, risk_aversion)
            w_prev = w.copy()
            w = w_new
        net_ret = net_portfolio_return(w, row.values, w_prev, cost_rate)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": turnover(w, w_prev)})
    return pd.DataFrame(rows).set_index("date")


def _historical_cvar_objective(logits: np.ndarray, hist: np.ndarray, w_prev: np.ndarray, cost: float, alpha: float) -> float:
    w = _softmax(logits)
    gross = hist @ w
    losses = -gross + cost * np.sum(np.abs(w - w_prev))
    return cvar_alpha(losses, alpha)


def historical_cvar_backtest(
    returns: pd.DataFrame,
    start: str,
    end: str,
    window: int = 252,
    cost_rate: float = 0.001,
    alpha: float = 0.05,
    maxiter: int = 80,
) -> pd.DataFrame:
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    sub = returns.loc[mask]
    n = sub.shape[1]
    w = np.full(n, 1.0 / n)
    w_prev = w.copy()
    rebal = _monthly_rebalance_mask(returns.index)
    rows = []
    for date, row in sub.iterrows():
        loc = returns.index.get_loc(date)
        day_to = 0.0
        if date in rebal and loc >= window:
            hist = returns.iloc[loc - window : loc].values
            old_w = w_prev.copy()
            res = minimize(
                _historical_cvar_objective,
                np.zeros(n),
                args=(hist, old_w, cost_rate, alpha),
                method="L-BFGS-B",
                options={"maxiter": maxiter},
            )
            w = _softmax(res.x if res.success else np.zeros(n))
            day_to = turnover(w, old_w)
            w_prev = w.copy()
        net_ret = float(w @ row.values - cost_rate * day_to)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": day_to})
    return pd.DataFrame(rows).set_index("date")


def risk_parity_weights(cov: np.ndarray) -> np.ndarray:
    n = cov.shape[0]
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    inv_vol = 1.0 / vol
    w = inv_vol / inv_vol.sum()
    return w


def risk_parity_backtest(
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
    rebal = _monthly_rebalance_mask(returns.index)
    rows = []
    for date, row in sub.iterrows():
        loc = returns.index.get_loc(date)
        if date in rebal and loc >= window:
            hist = returns.iloc[loc - window : loc]
            w = risk_parity_weights(hist.cov().values)
            w_prev = w.copy()
        net_ret = net_portfolio_return(w, row.values, w_prev, cost_rate)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": turnover(w, w_prev)})
    return pd.DataFrame(rows).set_index("date")


def max_diversification_weights(cov: np.ndarray) -> np.ndarray:
    vol = np.sqrt(np.maximum(np.diag(cov), 1e-12))
    inv = np.linalg.pinv(cov)
    raw = inv @ vol
    raw = np.maximum(raw, 0.0)
    if raw.sum() <= 1e-12:
        n = len(vol)
        return np.full(n, 1.0 / n)
    return raw / raw.sum()


def max_diversification_backtest(
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
    rebal = _monthly_rebalance_mask(returns.index)
    rows = []
    for date, row in sub.iterrows():
        loc = returns.index.get_loc(date)
        if date in rebal and loc >= window:
            hist = returns.iloc[loc - window : loc]
            w = max_diversification_weights(hist.cov().values)
            w_prev = w.copy()
        net_ret = net_portfolio_return(w, row.values, w_prev, cost_rate)
        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": turnover(w, w_prev)})
    return pd.DataFrame(rows).set_index("date")


def spy_benchmark_returns(
    benchmark_ticker: str,
    start: str,
    end: str,
    sleep_sec: float = 0.2,
) -> pd.Series:
    from robust_cvar_portfolio.src.data_loader import _fetch_us_daily

    prices = _fetch_us_daily(benchmark_ticker, start, end)
    return prices.pct_change().dropna().rename(benchmark_ticker)


def benchmark_metrics(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
    alpha: float = 0.05,
) -> dict[str, float]:
    aligned = pd.concat([portfolio_returns, benchmark_returns], axis=1, join="inner").dropna()
    if aligned.empty:
        return {}
    p = aligned.iloc[:, 0]
    b = aligned.iloc[:, 1]
    excess = p - b
    te = float(excess.std(ddof=1) * np.sqrt(252))
    beta = float(np.cov(p, b, ddof=1)[0, 1] / np.var(b, ddof=1)) if np.var(b) > 1e-12 else 0.0
    ir = float(excess.mean() * 252 / te) if te > 1e-12 else 0.0
    base = summarize_backtest(p.values, (-p).values, np.zeros(len(p)), alpha=alpha)
    base["beta_to_spy"] = beta
    base["tracking_error"] = te
    base["information_ratio"] = ir
    return base


def metrics_from_frame(frame: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    return summarize_backtest(frame["net_return"], frame["loss"], frame["turnover"], alpha=alpha)
