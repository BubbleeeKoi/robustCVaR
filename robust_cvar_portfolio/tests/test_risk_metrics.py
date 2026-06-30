"""Tests for risk metrics."""

import numpy as np

from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, net_portfolio_loss


def test_cvar_on_losses():
    losses = np.array([0.01, 0.02, 0.03, 0.10, 0.20])
    val = cvar_alpha(losses, alpha=0.2)
    assert val >= 0.10


def test_net_loss_sign():
    w = np.array([0.5, 0.5])
    r = np.array([0.01, 0.02])
    loss = net_portfolio_loss(w, r, w, 0.0)
    assert loss < 0
