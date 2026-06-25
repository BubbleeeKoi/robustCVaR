"""Compare reproduced paths against paths extracted from the paper figure.

The script reads CSV paths under outputs/, reports point counts, and computes
the mean nearest Manhattan grid distance from each reproduced path to the
corresponding PDF-extracted reference path.
"""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUTS = ROOT / "outputs"


PAIRS = {
    "a_no_uncertainty": (
        "paper_extracted_path_a_no_uncertainty.csv",
        "path_a_strict_cvar_no_uncertainty.csv",
    ),
    "b_rn_k_2": (
        "paper_extracted_path_b_rn_k_2.csv",
        "path_b_strict_cvar_rn_k_2.csv",
    ),
    "c_kl_k_2": (
        "paper_extracted_path_c_kl_k_2.csv",
        "path_c_strict_kl_evar_alpha_0_03.csv",
    ),
    "d_unfix_kappa": (
        "paper_extracted_path_d_unfix_kappa.csv",
        "path_d_strict_ncvar_decision_kappa.csv",
    ),
}


def main() -> None:
    for name, (paper_file, reproduced_file) in PAIRS.items():
        paper = read_xy(OUTPUTS / paper_file)
        reproduced = read_xy(OUTPUTS / reproduced_file)
        print(name)
        print(f"  paper_points={len(paper)} reproduced_points={len(reproduced)}")
        print(f"  mean_nearest_grid_distance={mean_nearest_distance(reproduced, paper):.3f}")


def read_xy(path: Path) -> list[tuple[int, int]]:
    rows: list[tuple[int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        next(f)
        for line in f:
            parts = line.strip().split(",")
            if len(parts) == 3:
                _, x, y = parts
            else:
                x, y = parts
            rows.append((int(x), int(y)))
    return rows


def mean_nearest_distance(a: list[tuple[int, int]], b: list[tuple[int, int]]) -> float:
    if not a or not b:
        return float("inf")
    total = 0.0
    for x, y in a:
        total += min(abs(x - bx) + abs(y - by) for bx, by in b)
    return total / len(a)


if __name__ == "__main__":
    main()
