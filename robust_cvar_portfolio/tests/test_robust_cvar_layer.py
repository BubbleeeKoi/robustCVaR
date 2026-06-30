"""Tests for robust CVaR layer."""

import numpy as np

from robust_cvar_portfolio.src.risk_metrics import cvar_alpha
from robust_cvar_portfolio.src.robust_cvar_layer import robust_cvar


def test_kappa_one_equals_cvar():
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60])
    plain = cvar_alpha(losses, alpha=0.1)
    rcvar = robust_cvar(losses, 1.0, alpha=0.1)
    assert abs(plain - rcvar) < 1e-10


def test_fixed_kappa_not_less_than_plain():
    losses = np.array([0.01, 0.02, 0.03, 0.04, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60])
    plain = robust_cvar(losses, 1.0, alpha=0.1)
    robust = robust_cvar(losses, 2.0, alpha=0.1)
    assert robust >= plain - 1e-10
