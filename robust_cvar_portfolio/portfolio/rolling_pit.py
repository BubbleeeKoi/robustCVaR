"""Rolling backtest with point-in-time changing universes."""

from __future__ import annotations

import time
from typing import Callable

import numpy as np
import pandas as pd

from robust_cvar_portfolio.portfolio.optimizer import optimize_portfolio
from robust_cvar_portfolio.portfolio.rolling import _engine_for_rebalance, monthly_rebalance_dates
from robust_cvar_portfolio.risk.risk_engine import RiskEngine
from robust_cvar_portfolio.data.index_universe import filter_available
from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, turnover


def _remap_weights(
    w_old: np.ndarray,
    tickers_old: list[str],
    tickers_new: list[str],
) -> np.ndarray:
    if not tickers_new:
        return np.array([])
    w = np.zeros(len(tickers_new))
    for i, t in enumerate(tickers_new):
        if t in tickers_old:
            w[i] = w_old[tickers_old.index(t)]
    if w.sum() > 1e-12:
        return w / w.sum()
    return np.full(len(tickers_new), 1.0 / len(tickers_new))


def run_rolling_pit(
    returns: pd.DataFrame,
    features: pd.DataFrame,
    engine: RiskEngine,
    constituents_at: Callable[[pd.Timestamp], list[str]],
    start: str,
    end: str,
    cost_rate: float = 0.001,
    estimation_window: int = 252,
    optimizer_maxiter: int = 50,
    weight_cap: float | None = None,
    kappa_rho: float | None = None,
    hhi_penalty: float = 0.0,
    record_weights: bool = False,
    record_diagnostics: bool = False,
    min_data_frac: float = 0.8,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rebalance_dates = set(monthly_rebalance_dates(returns.index))
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    period_index = returns.index[mask]

    tickers_cur: list[str] = []
    w_prev = np.array([])
    current_w = np.array([])
    kappa_bar_prev = 1.0
    rows = []
    weight_rows: list[dict] = []
    diag_rows: list[dict] = []

    for date in period_index:
        loc = idx_map[date]
        day_to = 0.0
        kappa_t = np.nan
        n_used = len(tickers_cur)

        if date in rebalance_dates and loc >= estimation_window:
            raw = constituents_at(date)
            used, dropped = filter_available(
                raw, returns, date, estimation_window, min_data_frac
            )
            if len(used) < 3:
                rows.append(
                    {
                        "date": date,
                        "net_return": 0.0,
                        "loss": 0.0,
                        "turnover": 0.0,
                        "kappa": np.nan,
                        "n_used": 0,
                        "solve_ok": False,
                    }
                )
                continue

            tickers_new = used
            hist = returns[tickers_new].iloc[loc - estimation_window : loc].fillna(0.0)
            feat = features.iloc[loc - estimation_window : loc]
            old_w = _remap_weights(w_prev, tickers_cur, tickers_new)
            opt_engine, kappa_bar_prev = _engine_for_rebalance(
                engine, features, loc, kappa_rho, kappa_bar_prev
            )

            t0 = time.perf_counter()
            current_w, _, kappa_t = optimize_portfolio(
                hist,
                feat,
                old_w,
                opt_engine,
                cost_rate,
                maxiter=optimizer_maxiter,
                weight_cap=weight_cap,
                hhi_penalty=hhi_penalty,
            )
            solve_sec = time.perf_counter() - t0
            day_to = turnover(current_w, old_w)
            w_prev = current_w.copy()
            tickers_cur = tickers_new
            n_used = len(tickers_cur)

            if record_weights:
                row = {"date": date, "kappa": kappa_t, "turnover": day_to, "n_used": n_used}
                for i, t in enumerate(tickers_cur):
                    row[t] = current_w[i]
                weight_rows.append(row)

            if record_diagnostics:
                tail_n = max(int(np.ceil(len(hist) * engine.alpha)), 1)
                diag_rows.append(
                    {
                        "date": date,
                        "n_constituents_raw": len(raw),
                        "n_used_t": n_used,
                        "solve_time": solve_sec,
                        "solve_ok": True,
                        "tail_sample_size": tail_n,
                        "top10_weight_sum": float(np.sum(np.sort(current_w)[-10:])),
                        "max_weight": float(current_w.max()),
                        "hhi": float(np.sum(current_w**2)),
                        "dropped_count": len(dropped),
                    }
                )

        if len(tickers_cur) == 0:
            net_ret = 0.0
        else:
            r = returns.loc[date, tickers_cur].fillna(0.0).values
            net_ret = float(current_w @ r - cost_rate * day_to)

        rows.append(
            {
                "date": date,
                "net_return": net_ret,
                "loss": -net_ret,
                "turnover": day_to,
                "kappa": kappa_t,
                "n_used": n_used,
            }
        )

    frame = pd.DataFrame(rows)
    frame["rolling_cvar_60"] = frame["loss"].rolling(60, min_periods=20).apply(
        lambda x: cvar_alpha(x.values, 0.05), raw=False
    )

    if record_weights and record_diagnostics:
        return frame, pd.DataFrame(weight_rows), pd.DataFrame(diag_rows)
    if record_weights:
        return frame, pd.DataFrame(weight_rows)
    if record_diagnostics:
        return frame, pd.DataFrame(diag_rows)
    return frame


def run_equal_weight_pit(
    returns: pd.DataFrame,
    constituents_at: Callable[[pd.Timestamp], list[str]],
    start: str,
    end: str,
    cost_rate: float = 0.001,
    estimation_window: int = 252,
    min_data_frac: float = 0.8,
) -> pd.DataFrame:
    rebalance_dates = set(monthly_rebalance_dates(returns.index))
    idx_map = {d: i for i, d in enumerate(returns.index)}
    mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
    period_index = returns.index[mask]

    tickers_cur: list[str] = []
    w_prev = np.array([])
    current_w = np.array([])
    rows = []

    for date in period_index:
        loc = idx_map[date]
        day_to = 0.0
        if date in rebalance_dates and loc >= estimation_window:
            raw = constituents_at(date)
            used, _ = filter_available(raw, returns, date, estimation_window, min_data_frac)
            if len(used) >= 1:
                tickers_new = used
                old_w = _remap_weights(w_prev, tickers_cur, tickers_new)
                current_w = np.full(len(tickers_new), 1.0 / len(tickers_new))
                day_to = turnover(current_w, old_w)
                w_prev = current_w.copy()
                tickers_cur = tickers_new

        if len(tickers_cur) == 0:
            net_ret = 0.0
        else:
            r = returns.loc[date, tickers_cur].fillna(0.0).values
            net_ret = float(current_w @ r - cost_rate * day_to)

        rows.append({"date": date, "net_return": net_ret, "loss": -net_ret, "turnover": day_to})
    return pd.DataFrame(rows)
