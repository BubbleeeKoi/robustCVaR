"""Robust CVaR (RCVaR) risk layer."""

from __future__ import annotations

import numpy as np

from .risk_metrics import cvar_alpha


def robust_cvar(
    losses: np.ndarray,
    kappa: np.ndarray | float,
    alpha: float = 0.05,
) -> float:
    """Compute RCVaR by greedy tail allocation with per-sample caps."""
    z = np.asarray(losses, dtype=float)
    k = np.asarray(kappa, dtype=float)
    if z.ndim != 1:
        raise ValueError("losses must be one-dimensional")
    if k.ndim == 0:
        k = np.full_like(z, float(k))
    if len(k) != len(z):
        raise ValueError("kappa must match losses length")
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1]")

    n = len(z)
    order = np.argsort(-z)
    caps = k / (alpha * n)
    remaining = 1.0
    value = 0.0
    for idx in order:
        if remaining <= 1e-12:
            break
        weight = min(float(caps[idx]), remaining)
        value += weight * z[idx]
        remaining -= weight
    return float(value)


def robust_cvar_weights(
    losses: np.ndarray,
    kappa: np.ndarray | float,
    alpha: float = 0.05,
) -> np.ndarray:
    z = np.asarray(losses, dtype=float)
    k = np.asarray(kappa, dtype=float)
    if k.ndim == 0:
        k = np.full_like(z, float(k))
    n = len(z)
    order = np.argsort(-z)
    caps = k / (alpha * n)
    q = np.zeros(n, dtype=float)
    remaining = 1.0
    for idx in order:
        if remaining <= 1e-12:
            break
        weight = min(float(caps[idx]), remaining)
        q[idx] = weight
        remaining -= weight
    return q


def verify_degeneracy(losses: np.ndarray, alpha: float = 0.05, k_fixed: float = 2.0) -> dict[str, float]:
    z = np.asarray(losses, dtype=float)
    plain = cvar_alpha(z, alpha)
    rcvar_plain = robust_cvar(z, 1.0, alpha)
    rcvar_fixed = robust_cvar(z, k_fixed, alpha)
    rcvar_high = robust_cvar(z, np.where(z >= np.quantile(z, 0.9), 2.0, 1.0), alpha)
    return {
        "plain_cvar": plain,
        "rcvar_kappa_1": rcvar_plain,
        "rcvar_kappa_K": rcvar_fixed,
        "rcvar_high_tail_kappa": rcvar_high,
    }
