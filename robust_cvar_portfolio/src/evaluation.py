"""Evaluation helpers for RL and rolling experiments."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .backtest import build_metrics_table, crisis_loss, extract_daily_frame, metrics_from_result
from .baselines import equal_weight_backtest, metrics_from_frame, min_variance_backtest
from .risk_metrics import summarize_backtest


def evaluate_rollout(net_returns, losses, turnovers, alpha=0.05) -> dict[str, float]:
    return summarize_backtest(net_returns, losses, turnovers, alpha=alpha)


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, ensure_ascii=False)


def combine_all_metrics(
    rolling_results: dict[str, pd.DataFrame],
    rl_results: dict[str, dict],
    baseline_frames: dict[str, pd.DataFrame],
    config: dict,
) -> pd.DataFrame:
    rows = []
    for name, frame in rolling_results.items():
        m = metrics_from_result(frame, alpha=config.get("alpha", 0.05))
        daily = extract_daily_frame(frame)
        m["method"] = f"rolling_{name}"
        m["crisis_2020"] = crisis_loss(daily["net_return"], "2020-02-01", "2020-04-30")
        m["crisis_2022"] = crisis_loss(daily["net_return"], "2022-01-01", "2022-12-31")
        rows.append(m)
    for name, payload in rl_results.items():
        m = evaluate_rollout(payload["net_returns"], payload["losses"], payload.get("turnovers", payload["net_returns"] * 0))
        m["method"] = f"rl_{name}"
        rows.append(m)
    for name, frame in baseline_frames.items():
        m = metrics_from_frame(frame, alpha=config.get("alpha", 0.05))
        m["method"] = f"baseline_{name}"
        rows.append(m)
    return pd.DataFrame(rows)
