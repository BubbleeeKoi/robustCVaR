"""Backtest utilities and result aggregation."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .risk_metrics import summarize_backtest


def extract_daily_frame(result: pd.DataFrame) -> pd.DataFrame:
    daily = result[result["event"] == "daily"].copy()
    daily["date"] = pd.to_datetime(daily["date"])
    return daily.set_index("date")


def metrics_from_result(result: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    daily = extract_daily_frame(result)
    return summarize_backtest(
        net_returns=daily["net_return"].values,
        losses=daily["loss"].values,
        turnovers=daily["turnover"].values,
        alpha=alpha,
    )


def crisis_loss(net_returns: pd.Series, start: str, end: str) -> float:
    sub = net_returns.loc[start:end]
    if sub.empty:
        return 0.0
    nav = (1.0 + sub).cumprod()
    return float(1.0 - nav.iloc[-1] / nav.cummax().iloc[-1])


def build_metrics_table(results: dict[str, pd.DataFrame], config: dict) -> pd.DataFrame:
    rows = []
    for name, frame in results.items():
        daily = extract_daily_frame(frame)
        metrics = metrics_from_result(frame, alpha=config.get("alpha", 0.05))
        metrics["strategy"] = name
        metrics["crisis_2020"] = crisis_loss(daily["net_return"], "2020-02-01", "2020-04-30")
        metrics["crisis_2022"] = crisis_loss(daily["net_return"], "2022-01-01", "2022-12-31")
        rows.append(metrics)
    table = pd.DataFrame(rows)
    cols = ["strategy"] + [c for c in table.columns if c != "strategy"]
    return table[cols]


def plot_nav_comparison(results: dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, frame in results.items():
        daily = extract_daily_frame(frame)
        nav = (1.0 + daily["net_return"]).cumprod()
        plt.plot(nav.index, nav.values, label=name)
    plt.legend()
    plt.title("Rolling Portfolio NAV Comparison (Test)")
    plt.xlabel("Date")
    plt.ylabel("NAV")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_drawdown_comparison(results: dict[str, pd.DataFrame], output_path: Path) -> None:
    plt.figure(figsize=(10, 5))
    for name, frame in results.items():
        daily = extract_daily_frame(frame)
        nav = (1.0 + daily["net_return"]).cumprod()
        dd = 1.0 - nav / nav.cummax()
        plt.plot(dd.index, dd.values, label=name)
    plt.legend()
    plt.title("Rolling Portfolio Drawdown Comparison (Test)")
    plt.xlabel("Date")
    plt.ylabel("Drawdown")
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
