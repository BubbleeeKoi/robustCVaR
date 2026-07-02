"""SP100 universe construction (Version A: current large-cap list, survivorship bias noted)."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from robust_cvar_portfolio.src.data_loader import compute_returns, download_prices, load_config

# Large-cap S&P names with listing history well before 2010 (excludes ABBV, NOW, META, TSLA, etc.)
SP100_CANDIDATES: list[str] = [
    "AAPL", "MSFT", "AMZN", "JNJ", "UNH", "XOM", "JPM", "V", "PG", "MA",
    "HD", "CVX", "MRK", "PEP", "KO", "WMT", "CSCO", "MCD", "ACN", "ABT",
    "DHR", "VZ", "ADBE", "TXN", "NKE", "PM", "ORCL", "DIS", "WFC", "CMCSA",
    "AMD", "QCOM", "IBM", "AMGN", "UPS", "CAT", "BA", "GE", "SBUX", "GS",
    "MS", "BLK", "INTU", "LOW", "DE", "T", "MDLZ", "GILD", "ADP", "TJX",
    "MMC", "CI", "SO", "DUK", "MO", "CB", "SYK", "PGR", "USB", "CL",
    "MMM", "EOG", "SLB", "GM", "F", "MET", "AIG", "TRV", "COP", "BIIB",
    "ADSK", "CRM", "SCHW", "PNC", "TGT", "LMT", "AXP", "C", "BAC", "INTC",
    "PFE", "NVDA", "NEE", "APD", "SHW", "EMR", "ITW", "BDX", "CSX", "NSC",
    "FDX", "KMB", "AON", "CME", "COF", "ALL", "YUM", "ROST", "BKNG", "AVGO",
    "COST", "SPGI", "ISRG", "ICE", "PSX", "MPC", "VLO", "REGN", "PLD", "HON",
    "RTX", "ELV", "ZTS", "EW", "TFC", "DG", "GOOG", "BRK-B", "MSFT", "JPM",
]

# dedupe while preserving order
_seen: set[str] = set()
SP100_CANDIDATES = [t for t in SP100_CANDIDATES if not (t in _seen or _seen.add(t))]

SECTOR_MAP: dict[str, str] = {
    "AAPL": "Technology", "MSFT": "Technology", "AMZN": "Consumer", "JNJ": "Healthcare",
    "UNH": "Healthcare", "XOM": "Energy", "JPM": "Financials", "V": "Financials",
    "PG": "Consumer", "MA": "Financials", "HD": "Consumer", "CVX": "Energy",
    "MRK": "Healthcare", "PEP": "Consumer", "KO": "Consumer", "AVGO": "Technology",
    "COST": "Consumer", "WMT": "Consumer", "CSCO": "Technology", "MCD": "Consumer",
    "ACN": "Technology", "ABT": "Healthcare", "DHR": "Healthcare", "VZ": "Communication",
    "ADBE": "Technology", "TXN": "Technology", "NKE": "Consumer", "PM": "Consumer",
    "ORCL": "Technology", "DIS": "Communication", "WFC": "Financials", "CMCSA": "Communication",
    "AMD": "Technology", "HON": "Industrials", "QCOM": "Technology", "IBM": "Technology",
    "AMGN": "Healthcare", "UPS": "Industrials", "CAT": "Industrials", "BA": "Industrials",
    "GE": "Industrials", "SBUX": "Consumer", "GS": "Financials", "MS": "Financials",
    "BLK": "Financials", "SPGI": "Financials", "INTU": "Technology", "LOW": "Consumer",
    "RTX": "Industrials", "DE": "Industrials", "ELV": "Healthcare", "T": "Communication",
    "MDLZ": "Consumer", "GILD": "Healthcare", "ADP": "Industrials", "ISRG": "Healthcare",
    "TJX": "Consumer", "MMC": "Financials", "CI": "Healthcare", "SO": "Utilities",
    "DUK": "Utilities", "MO": "Consumer", "ZTS": "Healthcare", "CB": "Financials",
    "SYK": "Healthcare", "PLD": "Real Estate", "PGR": "Financials", "USB": "Financials",
    "ICE": "Financials", "CL": "Consumer", "MMM": "Industrials", "EOG": "Energy",
    "SLB": "Energy", "GM": "Consumer", "F": "Consumer", "MET": "Financials",
    "AIG": "Financials", "TRV": "Financials", "COP": "Energy", "PSX": "Energy",
    "MPC": "Energy", "VLO": "Energy", "REGN": "Healthcare", "BIIB": "Healthcare",
    "ADSK": "Technology", "CRM": "Technology", "BKNG": "Consumer", "SCHW": "Financials",
    "PNC": "Financials", "TGT": "Consumer", "LMT": "Industrials", "AXP": "Financials",
    "C": "Financials", "BAC": "Financials", "INTC": "Technology", "PFE": "Healthcare",
    "NVDA": "Technology", "GOOG": "Communication", "NEE": "Utilities", "APD": "Materials",
    "SHW": "Materials", "EMR": "Industrials", "ITW": "Industrials", "BDX": "Healthcare",
    "CSX": "Industrials", "NSC": "Industrials", "FDX": "Industrials", "KMB": "Consumer",
    "EW": "Healthcare", "AON": "Financials", "CME": "Financials", "TFC": "Financials",
    "COF": "Financials", "ALL": "Financials", "YUM": "Consumer", "ROST": "Consumer",
    "DG": "Consumer", "BRK-B": "Financials",
}


def _panel_ok(returns: pd.DataFrame, start: str, min_days: int) -> bool:
    if returns.empty:
        return False
    sub = returns.loc[returns.index >= pd.Timestamp(start)]
    return len(sub) >= min_days


def _select_sp100_panel(
    raw: pd.DataFrame,
    candidates: list[str],
    start: str,
    target_n: int,
    min_panel_days: int,
) -> tuple[pd.DataFrame, list[str]]:
    """Pick target_n tickers maximizing overlapping history from start."""
    start_ts = pd.Timestamp(start)
    avail = [t for t in candidates if t in raw.columns]
    scores: list[tuple[str, int, pd.Timestamp]] = []
    for t in avail:
        s = raw[t].dropna()
        s = s.loc[start_ts:]
        if len(s) < min_panel_days:
            continue
        scores.append((t, len(s), s.index.min()))

    scores.sort(key=lambda x: (-x[1], x[2], avail.index(x[0]) if x[0] in avail else 999))
    pool = [t for t, _, _ in scores]
    if len(pool) < target_n:
        raise RuntimeError(f"only {len(pool)} tickers meet min_panel_days={min_panel_days}")

    selected = pool[:target_n]
    panel = raw[selected].loc[start_ts:].dropna(how="any")

    while len(panel) < min_panel_days and len(selected) > target_n // 2:
        first_dates = {t: raw[t].dropna().index.min() for t in selected}
        worst = max(first_dates, key=lambda t: first_dates[t])
        selected.remove(worst)
        panel = raw[selected].loc[start_ts:].dropna(how="any")

    if len(panel) < min_panel_days:
        raise RuntimeError(
            f"panel too short after selection: {len(panel)} days (need {min_panel_days})"
        )
    return panel, selected


def load_sp100_universe(
    config_path: Path,
    output_dir: Path,
    target_n: int = 100,
    force: bool = False,
) -> dict[str, pd.DataFrame | dict | list[str]]:
    """Build SP100 panel with full 2010–2024 overlap."""
    output_dir.mkdir(parents=True, exist_ok=True)
    returns_path = output_dir / "returns.csv"
    universe_path = output_dir / "universe.csv"
    config = load_config(config_path)
    start = config["start_date"]
    min_panel_days = config.get("min_history_days", 1500)

    if returns_path.exists() and universe_path.exists() and not force:
        prices = pd.read_csv(output_dir / "prices.csv", index_col=0, parse_dates=True)
        returns = pd.read_csv(returns_path, index_col=0, parse_dates=True)
        if _panel_ok(returns, start, min_panel_days):
            universe = pd.read_csv(universe_path)
            return {
                "config": config,
                "prices": prices,
                "returns": returns,
                "universe": universe,
                "tickers": universe["ticker"].tolist(),
            }
        print("  cached SP100 panel too short; rebuilding ...")

    tickers = [t for t in (config.get("tickers") or SP100_CANDIDATES) if t]
    end = config["end_date"]

    print(f"  downloading {len(tickers)} SP100 candidates ...")
    raw = download_prices(
        tickers, start, end, sleep_sec=0.12, min_days=min_panel_days, require_full_overlap=False
    )
    prices, selected = _select_sp100_panel(raw, tickers, start, target_n, min_panel_days)
    returns = compute_returns(prices)

    universe_rows = []
    for rank, ticker in enumerate(selected, start=1):
        universe_rows.append(
            {
                "rank": rank,
                "ticker": ticker,
                "sector": SECTOR_MAP.get(ticker, "Unknown"),
                "note": "Version A: current large-cap list; survivorship bias possible",
            }
        )
    universe = pd.DataFrame(universe_rows)

    prices.to_csv(output_dir / "prices.csv", index_label="date")
    returns.to_csv(output_dir / "returns.csv", index_label="date")
    universe.to_csv(universe_path, index=False)

    import json

    with (output_dir / "splits.json").open("w", encoding="utf-8") as f:
        json.dump(config["splits"], f, indent=2, ensure_ascii=False)

    print(
        f"  SP100 universe: {len(selected)} assets, {len(returns)} return days, "
        f"{returns.index.min().date()} ~ {returns.index.max().date()}"
    )
    return {
        "config": config,
        "prices": prices,
        "returns": returns,
        "universe": universe,
        "tickers": selected,
    }
