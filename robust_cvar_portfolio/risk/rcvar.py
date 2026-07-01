"""RCVaR computation (V2 wrapper)."""

from __future__ import annotations

import numpy as np

from robust_cvar_portfolio.src.robust_cvar_layer import robust_cvar, robust_cvar_weights


def compute_rcvar(
    losses: np.ndarray,
    kappa: np.ndarray | float,
    alpha: float = 0.05,
) -> float:
    return robust_cvar(losses, kappa, alpha)


def compute_rcvar_weights(
    losses: np.ndarray,
    kappa: np.ndarray | float,
    alpha: float = 0.05,
) -> np.ndarray:
    return robust_cvar_weights(losses, kappa, alpha)
