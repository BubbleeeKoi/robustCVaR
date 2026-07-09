"""V6 index supplement experiments: DJI30, NDX100, SP500 (point-in-time)."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = ROOT.parent
sys.path.insert(0, str(PROJECT))

from robust_cvar_portfolio.experiments.index_common import INDEX_OUT, run_index_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="V6 index PIT backtests")
    parser.add_argument(
        "--index",
        choices=["dji30", "ndx100", "sp500", "all"],
        default="all",
    )
    parser.add_argument("--force-data", action="store_true")
    parser.add_argument("--force-run", action="store_true", help="delete done.json only")
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="delete all checkpoints and re-run from scratch",
    )
    args = parser.parse_args()

    indices = ["dji30", "ndx100", "sp500"] if args.index == "all" else [args.index]
    t0 = time.time()
    summaries = []
    for name in indices:
        out = INDEX_OUT / name
        if args.fresh and out.exists():
            import shutil
            shutil.rmtree(out)
        if args.force_run and (out / "done.json").exists():
            (out / "done.json").unlink()
        summaries.append(run_index_experiment(name, force_data=args.force_data))

    from robust_cvar_portfolio.experiments.update_v6_2_index_html import (
        generate_figures,
        update_all,
    )
    generate_figures()
    update_all()

    print(f"\nAll done in {(time.time()-t0)/60:.1f} min", flush=True)
    for s in summaries:
        print(
            f"  {s['index']}: C_stable CVaR={s['C_stable_cvar']*100:.2f}% "
            f"win vs A={s['win_C_vs_A']} ({(time.time()-t0)/60:.0f} min total)",
            flush=True,
        )


if __name__ == "__main__":
    main()
