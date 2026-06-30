"""Portfolio risk and performance metrics."""

from __future__ import annotations

import numpy as np
import pandas as pd


def annualized_return(daily_returns: np.ndarray | pd.Series, periods: int = 252) -> float:
    r = np.asarray(daily_returns, dtype=float)
    if len(r) == 0:
        return 0.0
    cumulative = float(np.prod(1.0 + r))
    years = len(r) / periods
    if years <= 0:
        return 0.0
    return cumulative ** (1.0 / years) - 1.0


def annualized_volatility(daily_returns: np.ndarray | pd.Series, periods: int = 252) -> float:
    r = np.asarray(daily_returns, dtype=float)
    return float(np.std(r, ddof=1) * np.sqrt(periods))


def sharpe_ratio(daily_returns: np.ndarray | pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    vol = annualized_volatility(daily_returns, periods)
    if vol <= 1e-12:
        return 0.0
    excess = annualized_return(daily_returns, periods) - rf
    return float(excess / vol)


def sortino_ratio(daily_returns: np.ndarray | pd.Series, periods: int = 252, rf: float = 0.0) -> float:
    r = np.asarray(daily_returns, dtype=float)
    downside = r[r < 0]
    if len(downside) == 0:
        return 0.0
    downside_vol = float(np.std(downside, ddof=1) * np.sqrt(periods))
    if downside_vol <= 1e-12:
        return 0.0
    return float((annualized_return(r, periods) - rf) / downside_vol)


def nav_from_returns(daily_returns: np.ndarray | pd.Series) -> np.ndarray:
    r = np.asarray(daily_returns, dtype=float)
    return np.cumprod(1.0 + r)


def max_drawdown(daily_returns: np.ndarray | pd.Series) -> float:
    nav = nav_from_returns(daily_returns)
    peak = np.maximum.accumulate(nav)
    dd = 1.0 - nav / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def var_alpha(losses: np.ndarray | pd.Series, alpha: float = 0.05) -> float:
    x = np.asarray(losses, dtype=float)
    if len(x) == 0:
        return 0.0
    return float(np.quantile(x, 1.0 - alpha))


def cvar_alpha(losses: np.ndarray | pd.Series, alpha: float = 0.05) -> float:
    x = np.sort(np.asarray(losses, dtype=float))[::-1]
    n = len(x)
    if n == 0:
        return 0.0
    upper = max(1, int(np.ceil(alpha * n)))
    return float(np.mean(x[:upper]))


def turnover(weights: np.ndarray, w_prev: np.ndarray) -> float:
    return float(np.sum(np.abs(weights - w_prev)))


def transaction_cost(weights: np.ndarray, w_prev: np.ndarray, cost_rate: float) -> float:
    return float(cost_rate * turnover(weights, w_prev))


def calmar_ratio(daily_returns: np.ndarray | pd.Series, periods: int = 252) -> float:
    mdd = max_drawdown(daily_returns)
    if mdd <= 1e-12:
        return 0.0
    return float(annualized_return(daily_returns, periods) / mdd)


def net_portfolio_return(weights: np.ndarray, asset_returns: np.ndarray, w_prev: np.ndarray, cost_rate: float) -> float:
    gross = float(weights @ asset_returns)
    cost = transaction_cost(weights, w_prev, cost_rate)
    return gross - cost


def net_portfolio_loss(weights: np.ndarray, asset_returns: np.ndarray, w_prev: np.ndarray, cost_rate: float) -> float:
    return -net_portfolio_return(weights, asset_returns, w_prev, cost_rate)


def summarize_backtest(
    net_returns: np.ndarray | pd.Series,
    losses: np.ndarray | pd.Series,
    turnovers: np.ndarray | pd.Series,
    alpha: float = 0.05,
) -> dict[str, float]:
    net_returns = np.asarray(net_returns, dtype=float)
    losses = np.asarray(losses, dtype=float)
    turnovers = np.asarray(turnovers, dtype=float)
    return {
        "annualized_return": annualized_return(net_returns),
        "annualized_volatility": annualized_volatility(net_returns),
        "sharpe_ratio": sharpe_ratio(net_returns),
        "sortino_ratio": sortino_ratio(net_returns),
        "max_drawdown": max_drawdown(net_returns),
        "var_5pct": var_alpha(losses, alpha),
        "cvar_5pct": cvar_alpha(losses, alpha),
        "avg_turnover": float(np.mean(turnovers)) if len(turnovers) else 0.0,
        "total_transaction_cost": float(np.sum(turnovers) * 0.0),
        "calmar_ratio": calmar_ratio(net_returns),
    }
