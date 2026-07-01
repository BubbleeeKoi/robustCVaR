"""V2 state construction: volatility, drawdown, momentum, correlation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_cvar_portfolio.src.features import (
    avg_correlation,
    market_drawdown,
    market_volatility,
    rolling_zscore,
)


def market_momentum(returns: pd.DataFrame, window: int = 20) -> pd.Series:
    ew = np.full(returns.shape[1], 1.0 / returns.shape[1])
    port_ret = returns.values @ ew
    return pd.Series(port_ret, index=returns.index).rolling(window).sum()


def build_v2_state(returns: pd.DataFrame, z_window: int = 252) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "Vol": market_volatility(returns),
            "DD": market_drawdown(returns),
            "Mom": market_momentum(returns),
            "Corr": avg_correlation(returns),
        },
        index=returns.index,
    )
    for col in frame.columns:
        frame[f"{col}_z"] = rolling_zscore(frame[col], window=z_window)
    return frame.bfill().ffill().fillna(0.0)
