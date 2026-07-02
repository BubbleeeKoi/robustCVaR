"""Download SP100 universe data."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.data.sp100_universe import load_sp100_universe


def main() -> None:
    config_path = ROOT / "configs" / "sp100.yaml"
    data_dir = ROOT / "data" / "processed" / "sp100"
    load_sp100_universe(config_path, data_dir, target_n=100, force=True)  # always rebuild


if __name__ == "__main__":
    main()
