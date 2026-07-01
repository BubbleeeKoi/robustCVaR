"""Portfolio weight optimizer minimizing RCVaR."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.src.risk_metrics import turnover


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


def loss_samples(
    weights: np.ndarray,
    hist_returns: np.ndarray,
    w_prev: np.ndarray,
    cost_rate: float,
) -> np.ndarray:
    gross = hist_returns @ weights
    to = np.sum(np.abs(weights - w_prev))
    return -gross + cost_rate * to


def optimize_portfolio(
    hist_returns: pd.DataFrame,
    features: pd.DataFrame,
    w_prev: np.ndarray,
    engine: RiskEngine,
    cost_rate: float = 0.001,
    maxiter: int = 150,
) -> tuple[np.ndarray, float, float]:
    n = hist_returns.shape[1]
    equal = np.full(n, 1.0 / n)

    def objective(logits: np.ndarray) -> float:
        w = softmax(logits)
        losses = loss_samples(w, hist_returns.values, w_prev, cost_rate)
        risk, _ = engine.portfolio_risk(losses, features, w)
        return risk

    result = minimize(objective, np.zeros(n), method="L-BFGS-B", options={"maxiter": maxiter})
    w_opt = softmax(result.x if result.success else np.zeros(n))
    if not result.success:
        w_opt = equal
    losses = loss_samples(w_opt, hist_returns.values, w_prev, cost_rate)
    risk, kappa = engine.portfolio_risk(losses, features, w_opt)
    return w_opt, risk, float(kappa[-1])
