"""Tests for fractional vs ceil CVaR definitions."""

import numpy as np

from robust_cvar_portfolio.src.risk_metrics import cvar_alpha_ceil, cvar_alpha_fractional
from robust_cvar_portfolio.src.robust_cvar_layer import robust_cvar


def test_fractional_equals_rcvar_kappa_one():
    rng = np.random.default_rng(42)
    for n in [10, 50, 252, 500]:
        losses = rng.normal(0, 0.02, n)
        frac = cvar_alpha_fractional(losses, alpha=0.05)
        rcvar = robust_cvar(losses, 1.0, alpha=0.05)
        assert abs(frac - rcvar) < 1e-10, f"n={n} diff={frac-rcvar}"


def test_ceil_differs_from_fractional_when_alpha_n_non_integer():
    losses = np.linspace(0.01, 0.30, 252)
    ceil = cvar_alpha_ceil(losses, alpha=0.05)
    frac = cvar_alpha_fractional(losses, alpha=0.05)
    assert abs(ceil - frac) > 1e-6


def test_ceil_equals_fractional_when_alpha_n_integer():
    losses = np.linspace(0.01, 0.30, 100)
    ceil = cvar_alpha_ceil(losses, alpha=0.05)
    frac = cvar_alpha_fractional(losses, alpha=0.05)
    assert abs(ceil - frac) < 1e-10
