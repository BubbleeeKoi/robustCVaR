"""One-shot V3 pipeline: rebuild SP100 data if needed, then run full experiment."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.run_v3_experiment import run_v3


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--force-data", action="store_true")
    args = parser.parse_args()
    run_v3(force_data=args.force_data)


if __name__ == "__main__":
    main()
