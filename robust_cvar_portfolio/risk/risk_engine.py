"""Unified risk engine: state → κ → RCVaR."""

from __future__ import annotations

import numpy as np
import pandas as pd

from robust_cvar_portfolio.risk.kappa import KappaParams, kappa_vector_for_losses_v2
from robust_cvar_portfolio.risk.rcvar import compute_rcvar


class RiskEngine:
    def __init__(
        self,
        alpha: float = 0.05,
        kappa_mode: str = "manual",
        params: KappaParams | None = None,
        fixed_k: float = 2.0,
    ) -> None:
        self.alpha = alpha
        self.kappa_mode = kappa_mode
        self.params = params or KappaParams()
        self.fixed_k = fixed_k

    def kappa_vector(
        self,
        features: pd.DataFrame,
        weights: np.ndarray,
    ) -> np.ndarray:
        return kappa_vector_for_losses_v2(
            self.kappa_mode,
            features,
            weights,
            self.params,
            self.fixed_k,
        )

    def portfolio_risk(
        self,
        losses: np.ndarray,
        features: pd.DataFrame,
        weights: np.ndarray,
    ) -> tuple[float, np.ndarray]:
        kappa = self.kappa_vector(features, weights)
        return compute_rcvar(losses, kappa, self.alpha), kappa
