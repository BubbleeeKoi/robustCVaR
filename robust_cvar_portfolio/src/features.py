"""Market state features and kappa(s, w) design."""

from __future__ import annotations

import numpy as np
import pandas as pd


def sigmoid(x: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def rolling_zscore(series: pd.Series, window: int = 252, eps: float = 1e-8) -> pd.Series:
    mu = series.rolling(window, min_periods=max(20, window // 5)).mean()
    sigma = series.rolling(window, min_periods=max(20, window // 5)).std()
    return (series - mu) / (sigma + eps)


def market_volatility(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    ew = np.full(returns.shape[1], 1.0 / returns.shape[1])
    port_ret = returns.values @ ew
    return pd.Series(port_ret, index=returns.index).rolling(window).std()


def market_drawdown(returns: pd.DataFrame) -> pd.Series:
    ew = np.full(returns.shape[1], 1.0 / returns.shape[1])
    port_ret = returns.values @ ew
    nav = (1.0 + pd.Series(port_ret, index=returns.index)).cumprod()
    peak = nav.cummax()
    return 1.0 - nav / peak


def avg_correlation(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    if returns.shape[1] < 2:
        return pd.Series(0.0, index=returns.index)
    values = []
    arr = returns.values
    for idx in range(len(returns)):
        if idx + 1 < window:
            values.append(np.nan)
            continue
        block = arr[idx + 1 - window : idx + 1]
        corr = np.corrcoef(block.T)
        mask = ~np.eye(corr.shape[0], dtype=bool)
        values.append(float(np.nanmean(corr[mask])))
    return pd.Series(values, index=returns.index)


def tail_loss_intensity(returns: pd.DataFrame, window: int = 20, q: float = 0.05) -> pd.Series:
    ew = np.full(returns.shape[1], 1.0 / returns.shape[1])
    port_ret = returns.values @ ew
    port_loss = -port_ret
    return pd.Series(port_loss, index=returns.index).rolling(window).quantile(q)


def cross_sectional_dispersion(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    cs_std = returns.std(axis=1)
    return cs_std.rolling(window).mean()


def build_market_features(returns: pd.DataFrame, z_window: int = 252) -> pd.DataFrame:
    vol = market_volatility(returns)
    corr = avg_correlation(returns)
    dd = market_drawdown(returns)
    tail = tail_loss_intensity(returns)
    disp = cross_sectional_dispersion(returns)
    frame = pd.DataFrame(
        {
            "Vol": vol,
            "Corr": corr,
            "DD": dd,
            "Tail": tail,
            "Disp": disp,
        },
        index=returns.index,
    )
    for col in frame.columns:
        frame[f"{col}_z"] = rolling_zscore(frame[col], window=z_window)
    return frame.bfill().ffill().fillna(0.0)


def concentration(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=float)
    w = w / (w.sum() + 1e-12)
    return float(np.sum(w**2))


def kappa_state_only(
    features: pd.Series,
    kappa_max: float = 1.0,
    beta_vol: float = 1.0,
    beta_dd: float = 1.0,
) -> float:
    score = beta_vol * features.get("Vol_z", 0.0) + beta_dd * features.get("DD_z", 0.0)
    return float(1.0 + kappa_max * sigmoid(score))


def kappa_state_action(
    features: pd.Series,
    weights: np.ndarray,
    kappa_max: float = 1.0,
    beta_vol: float = 1.0,
    beta_dd: float = 1.0,
    beta_conc: float = 0.5,
) -> float:
    conc = concentration(weights)
    conc_z = (conc - 0.10) / 0.05
    score = (
        beta_vol * features.get("Vol_z", 0.0)
        + beta_dd * features.get("DD_z", 0.0)
        + beta_conc * conc_z
    )
    return float(1.0 + kappa_max * sigmoid(score))


def kappa_vector_for_losses(
    mode: str,
    features: pd.DataFrame,
    weights: np.ndarray,
    w_prev: np.ndarray,
    kappa_max: float,
    fixed_k: float,
    beta_vol: float,
    beta_dd: float,
    beta_conc: float,
) -> np.ndarray:
    if mode == "fixed":
        return np.full(len(features), fixed_k, dtype=float)
    if mode == "plain":
        return np.ones(len(features), dtype=float)
    if mode == "state":
        return np.array(
            [
                kappa_state_only(row, kappa_max, beta_vol, beta_dd)
                for _, row in features.iterrows()
            ],
            dtype=float,
        )
    if mode == "state_action":
        base = kappa_state_action(
            features.iloc[-1],
            weights,
            kappa_max,
            beta_vol,
            beta_dd,
            beta_conc,
        )
        vec = np.array(
            [
                kappa_state_action(row, weights, kappa_max, beta_vol, beta_dd, beta_conc)
                for _, row in features.iterrows()
            ],
            dtype=float,
        )
        vec[-1] = base
        return vec
    raise ValueError(f"unknown kappa mode: {mode}")
