"""V3 analysis: bootstrap, benchmark plots, kappa interpretability."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from robust_cvar_portfolio.src.risk_metrics import cvar_alpha, max_drawdown


def block_bootstrap_cvar(
    losses: np.ndarray,
    n_boot: int = 500,
    block_size: int = 20,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = len(losses)
    if n < block_size:
        return {"cvar_mean": cvar_alpha(losses, alpha), "cvar_lo": np.nan, "cvar_hi": np.nan}
    samples = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            idx.extend(range(start, min(start + block_size, n)))
        idx = idx[:n]
        samples.append(cvar_alpha(losses[idx], alpha))
    arr = np.array(samples)
    return {
        "cvar_mean": float(np.mean(arr)),
        "cvar_lo": float(np.percentile(arr, 2.5)),
        "cvar_hi": float(np.percentile(arr, 97.5)),
    }


def paired_cvar_test(
    losses_a: np.ndarray,
    losses_c: np.ndarray,
    n_boot: int = 500,
    block_size: int = 20,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = min(len(losses_a), len(losses_c))
    a, c = losses_a[:n], losses_c[:n]
    diffs = []
    for _ in range(n_boot):
        idx = []
        while len(idx) < n:
            start = rng.integers(0, max(1, n - block_size + 1))
            idx.extend(range(start, min(start + block_size, n)))
        idx = idx[:n]
        diffs.append(cvar_alpha(c[idx], alpha) - cvar_alpha(a[idx], alpha))
    arr = np.array(diffs)
    prob_c_better = float(np.mean(arr < 0))
    return {
        "mean_cvar_diff": float(np.mean(arr)),
        "ci_lo": float(np.percentile(arr, 2.5)),
        "ci_hi": float(np.percentile(arr, 97.5)),
        "prob_cvar_c_lt_a": prob_c_better,
    }


def crisis_subsample_metrics(
    frame: pd.DataFrame,
    periods: dict[str, tuple[str, str]],
    alpha: float = 0.05,
) -> pd.DataFrame:
    rows = []
    for name, (start, end) in periods.items():
        sub = frame.loc[start:end]
        if sub.empty:
            continue
        rows.append(
            {
                "period": name,
                "start": start,
                "end": end,
                "n_days": len(sub),
                "cvar_5pct": cvar_alpha(sub["loss"].values, alpha),
                "max_drawdown": max_drawdown(sub["net_return"].values),
                "total_return": float((1 + sub["net_return"]).prod() - 1),
            }
        )
    return pd.DataFrame(rows)


def plot_kappa_time_series(kappa: pd.Series, path: Path) -> None:
    plt.figure(figsize=(11, 4))
    plt.plot(kappa.index, kappa.values, color="#1f77b4", lw=1.2)
    for label, start, end in [
        ("COVID-19", "2020-02-01", "2020-06-30"),
        ("2022 Vol", "2022-01-01", "2022-12-31"),
    ]:
        plt.axvspan(pd.Timestamp(start), pd.Timestamp(end), alpha=0.15, label=label)
    plt.ylabel(r"$\kappa_t$")
    plt.title(r"State-dependent $\kappa(s_t)$ over test period")
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_kappa_vs_state(
    kappa: pd.Series,
    states: pd.DataFrame,
    path: Path,
) -> None:
    aligned = pd.concat([kappa.rename("kappa"), states], axis=1).dropna()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    vol_col = [c for c in aligned.columns if "vol" in c.lower()][0]
    dd_col = [c for c in aligned.columns if "dd" in c.lower()][0]
    axes[0].scatter(aligned[vol_col], aligned["kappa"], alpha=0.4, s=12)
    axes[0].set_xlabel("Volatility (z-score)")
    axes[0].set_ylabel(r"$\kappa$")
    axes[0].set_title(r"$\kappa$ vs Volatility")
    axes[1].scatter(aligned[dd_col], aligned["kappa"], alpha=0.4, s=12, color="#d62728")
    axes[1].set_xlabel("Drawdown (z-score)")
    axes[1].set_ylabel(r"$\kappa$")
    axes[1].set_title(r"$\kappa$ vs Drawdown")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_nav_vs_benchmark(
    nav_dict: dict[str, pd.Series],
    path: Path,
    title: str = "NAV vs SPY",
) -> None:
    plt.figure(figsize=(10, 5))
    for name, nav in nav_dict.items():
        plt.plot(nav.index, nav.values, label=name)
    plt.legend(fontsize=8)
    plt.title(title)
    plt.ylabel("NAV")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_cvar_bootstrap(diff_stats: dict[str, float], path: Path) -> None:
    plt.figure(figsize=(6, 4))
    lo, hi = diff_stats["ci_lo"], diff_stats["ci_hi"]
    mean = diff_stats["mean_cvar_diff"]
    plt.barh(["CVaR(C)-CVaR(A)"], [mean], xerr=[[mean - lo], [hi - mean]], capsize=6)
    plt.axvline(0, color="k", lw=0.8)
    plt.title(f"P(C<A)={diff_stats['prob_cvar_c_lt_a']:.1%}")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
