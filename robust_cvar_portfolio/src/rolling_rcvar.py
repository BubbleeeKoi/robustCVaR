"""Non-RL rolling robust CVaR portfolio optimization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .features import build_market_features, kappa_vector_for_losses
from .risk_metrics import net_portfolio_loss, turnover
from .robust_cvar_layer import robust_cvar


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)


@dataclass
class RollingConfig:
    alpha: float = 0.05
    cost_rate: float = 0.001
    estimation_window: int = 252
    kappa_mode: str = "plain"  # plain | fixed | state | state_action
    kappa_max: float = 1.0
    fixed_kappa: float = 2.0
    beta_vol: float = 1.0
    beta_dd: float = 1.0
    beta_conc: float = 0.5


def _loss_samples(
    weights: np.ndarray,
    hist_returns: np.ndarray,
    w_prev: np.ndarray,
    cost_rate: float,
) -> np.ndarray:
    gross = hist_returns @ weights
    to = np.sum(np.abs(weights - w_prev))
    return -gross + cost_rate * to


def optimize_weights(
    hist_returns: pd.DataFrame,
    features: pd.DataFrame,
    w_prev: np.ndarray,
    cfg: RollingConfig,
) -> tuple[np.ndarray, float, float]:
    n = hist_returns.shape[1]
    equal = np.full(n, 1.0 / n)

    def objective(logits: np.ndarray) -> float:
        w = softmax(logits)
        losses = _loss_samples(w, hist_returns.values, w_prev, cfg.cost_rate)
        kappa = kappa_vector_for_losses(
            mode=cfg.kappa_mode,
            features=features,
            weights=w,
            w_prev=w_prev,
            kappa_max=cfg.kappa_max,
            fixed_k=cfg.fixed_kappa,
            beta_vol=cfg.beta_vol,
            beta_dd=cfg.beta_dd,
            beta_conc=cfg.beta_conc,
        )
        return robust_cvar(losses, kappa, cfg.alpha)

    x0 = np.zeros(n)
    result = minimize(objective, x0, method="L-BFGS-B", options={"maxiter": 200})
    w_opt = softmax(result.x if result.success else x0)
    losses = _loss_samples(w_opt, hist_returns.values, w_prev, cfg.cost_rate)
    kappa = kappa_vector_for_losses(
        mode=cfg.kappa_mode,
        features=features,
        weights=w_opt,
        w_prev=w_prev,
        kappa_max=cfg.kappa_max,
        fixed_k=cfg.fixed_kappa,
        beta_vol=cfg.beta_vol,
        beta_dd=cfg.beta_dd,
        beta_conc=cfg.beta_conc,
    )
    rcvar = robust_cvar(losses, kappa, cfg.alpha)
    kappa_t = float(kappa[-1]) if len(kappa) else 1.0
    if not result.success:
        w_opt = equal
    return w_opt, rcvar, kappa_t


def monthly_rebalance_dates(index: pd.DatetimeIndex) -> list[pd.Timestamp]:
    groups = index.to_series().groupby([index.year, index.month])
    return [idx.max() for _, idx in groups]


def run_rolling_backtest(
    returns: pd.DataFrame,
    rebalance_dates: list[pd.Timestamp],
    cfg: RollingConfig,
    start: str,
    end: str,
) -> pd.DataFrame:
    features_all = build_market_features(returns)
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    period_index = returns.index[mask]
    n_assets = returns.shape[1]
    w_prev = np.full(n_assets, 1.0 / n_assets)

    rows = []
    rebalance_set = set(pd.Timestamp(d) for d in rebalance_dates)
    current_w = w_prev.copy()

    for date in period_index:
        loc = idx_map[date]
        day_turnover = 0.0
        kappa_t = np.nan
        if date in rebalance_set and loc >= cfg.estimation_window:
            hist = returns.iloc[loc - cfg.estimation_window : loc]
            feat = features_all.iloc[loc - cfg.estimation_window : loc]
            old_w = w_prev.copy()
            current_w, _, kappa_t = optimize_weights(hist, feat, old_w, cfg)
            day_turnover = turnover(current_w, old_w)
            w_prev = current_w.copy()

        asset_ret = returns.loc[date].values
        net_ret = float(current_w @ asset_ret - cfg.cost_rate * day_turnover)
        loss = -net_ret
        rows.append(
            {
                "date": date,
                "event": "daily",
                "net_return": net_ret,
                "loss": loss,
                "turnover": day_turnover,
                "kappa": kappa_t,
                **{f"w_{i}": current_w[i] for i in range(n_assets)},
            }
        )

    return pd.DataFrame(rows)


def run_all_variants(
    returns: pd.DataFrame,
    config: dict,
    split_name: str,
) -> dict[str, pd.DataFrame]:
    start, end = config["splits"][split_name]
    rebalance_dates = monthly_rebalance_dates(returns.index)
    variants = {
        "A_plain_cvar": RollingConfig(kappa_mode="plain"),
        "B_fixed_robust": RollingConfig(kappa_mode="fixed", fixed_kappa=config.get("fixed_kappa", 2.0)),
        "C_state_robust": RollingConfig(kappa_mode="state"),
        "D_state_action_robust": RollingConfig(kappa_mode="state_action"),
    }
    for cfg in variants.values():
        cfg.alpha = config.get("alpha", 0.05)
        cfg.cost_rate = config.get("cost_rate", 0.001)
        cfg.estimation_window = config.get("estimation_window", 252)
        cfg.kappa_max = config.get("kappa_max", 1.0)
        cfg.fixed_kappa = config.get("fixed_kappa", 2.0)
        cfg.beta_vol = config.get("beta_vol", 1.0)
        cfg.beta_dd = config.get("beta_dd", 1.0)
        cfg.beta_conc = config.get("beta_conc", 0.5)

    return {
        name: run_rolling_backtest(returns, rebalance_dates, cfg, start, end)
        for name, cfg in variants.items()
    }
