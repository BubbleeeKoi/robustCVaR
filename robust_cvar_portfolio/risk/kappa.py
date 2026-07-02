"""κ(s) and κ(s,w) — manual and learned risk budgets (V2)."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from robust_cvar_portfolio.src.features import concentration, sigmoid


STATE_COLS = ["Vol_z", "DD_z", "Mom_z", "Corr_z"]


@dataclass
class KappaParams:
    kappa_max: float = 1.0
    beta_vol: float = 1.0
    beta_dd: float = 1.0
    beta_mom: float = 0.5
    beta_corr: float = 0.5
    beta_conc: float = 0.5
    theta: np.ndarray = field(default_factory=lambda: np.array([1.0, 1.0, 0.5, 0.5]))


def f_theta(state_row: pd.Series | np.ndarray, theta: np.ndarray) -> float:
    if isinstance(state_row, pd.Series):
        x = np.array([state_row.get(c, 0.0) for c in STATE_COLS], dtype=float)
    else:
        x = np.asarray(state_row, dtype=float)
    return float(theta @ x)


def kappa_manual(state_row: pd.Series, params: KappaParams) -> float:
    score = (
        params.beta_vol * state_row.get("Vol_z", 0.0)
        + params.beta_dd * state_row.get("DD_z", 0.0)
        + params.beta_mom * state_row.get("Mom_z", 0.0)
        + params.beta_corr * state_row.get("Corr_z", 0.0)
    )
    return float(1.0 + params.kappa_max * sigmoid(score))


def kappa_learned(state_row: pd.Series, params: KappaParams) -> float:
    score = f_theta(state_row, params.theta)
    return float(1.0 + params.kappa_max * sigmoid(score))


def kappa_state_action(
    state_row: pd.Series,
    weights: np.ndarray,
    params: KappaParams,
) -> float:
    conc_z = (concentration(weights) - 0.10) / 0.05
    score = (
        params.beta_vol * state_row.get("Vol_z", 0.0)
        + params.beta_dd * state_row.get("DD_z", 0.0)
        + params.beta_mom * state_row.get("Mom_z", 0.0)
        + params.beta_corr * state_row.get("Corr_z", 0.0)
        + params.beta_conc * conc_z
    )
    return float(1.0 + params.kappa_max * sigmoid(score))


def kappa_series_from_states(states: pd.DataFrame, mode: str, params: KappaParams) -> pd.Series:
    if mode == "plain":
        return pd.Series(1.0, index=states.index)
    if mode == "fixed":
        return pd.Series(params.fixed_k if hasattr(params, "fixed_k") else 2.0, index=states.index)
    fn = kappa_learned if mode == "learned" else kappa_manual
    return pd.Series([fn(row, params) for _, row in states.iterrows()], index=states.index)


def kappa_vector_for_losses_v2(
    mode: str,
    features: pd.DataFrame,
    weights: np.ndarray,
    params: KappaParams,
    fixed_k: float = 2.0,
) -> np.ndarray:
    if mode in {"plain", "plain_frac", "plain_ceil"}:
        return np.ones(len(features), dtype=float)
    if mode == "fixed":
        return np.full(len(features), fixed_k, dtype=float)
    if mode == "manual":
        return np.array([kappa_manual(row, params) for _, row in features.iterrows()])
    if mode == "learned":
        return np.array([kappa_learned(row, params) for _, row in features.iterrows()])
    if mode == "state_action":
        return np.array([kappa_state_action(row, weights, params) for _, row in features.iterrows()])
    raise ValueError(f"unknown kappa mode: {mode}")


def fit_kappa_theta(
    states: pd.DataFrame,
    stress_target: pd.Series,
    kappa_max: float = 1.0,
) -> np.ndarray:
    """Learn θ: high vol/dd/corr → higher κ; negative momentum → higher κ."""
    aligned = states.join(stress_target.rename("stress"), how="inner").dropna()
    if len(aligned) < 20:
        return np.array([1.0, 1.0, -0.5, 0.5])
    x = aligned[STATE_COLS].values
    y = aligned["stress"].values
    y = (y - np.nanmin(y)) / (np.nanmax(y) - np.nanmin(y) + 1e-8)
    x_aug = np.column_stack([x, np.ones(len(x))])
    coef, _, _, _ = np.linalg.lstsq(x_aug, y, rcond=None)
    theta = coef[:4]
    if np.linalg.norm(theta) < 1e-6:
        theta = np.array([1.0, 1.0, -0.5, 0.5])
    return theta
