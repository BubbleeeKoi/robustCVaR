"""Run the strict Figure 1 reproduction pipeline.

This script builds the GridWorld, solves the CVaR/NCVaR and KL/EVaR value
iterations for the four Figure 1 panels, saves value/policy/path artifacts,
and renders the 2 x 2 reproduction figures.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from robust_cvar_repro.gridworld import GridWorld, GridWorldConfig
from robust_cvar_repro.evaluation import (
    evaluate_augmented_policy,
    evaluate_markov_policy,
)
from robust_cvar_repro.render import render_comparison, render_paper_style_comparison
from robust_cvar_repro.value_iteration import (
    SingleAlphaResult,
    SolverConfig,
    confidence_grid,
    evar_confidence_from_kl_budget,
    extract_path,
    extract_single_alpha_path,
    interpolate_value_surface,
    kl_radius_from_kl_budget,
    save_path,
    solve_kl_robust_value_iteration_vectorized,
    solve_robust_cvar_pwl,
)


OUTPUTS = ROOT / "outputs"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--episodes",
        type=int,
        default=5_000,
        help="Monte Carlo episodes per panel (default: 5000).",
    )
    parser.add_argument(
        "--cold-start",
        action="store_true",
        help="Ignore saved value surfaces and initialize from risk-neutral values.",
    )
    parser.add_argument(
        "--reuse-evaluation",
        action="store_true",
        help="Keep existing evaluation JSON files instead of rerunning simulations.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = GridWorld(GridWorldConfig())
    OUTPUTS.mkdir(exist_ok=True)
    env.export_obstacles(OUTPUTS / "paper_figure_obstacles.csv")

    alpha = 0.48
    kl_kappa = 2.0
    alpha_evar_equiv = evar_confidence_from_kl_budget(alpha, kl_kappa)
    y_grid = confidence_grid(alpha=alpha, points=21)
    config = SolverConfig(gamma=0.95, tolerance=1e-4, max_iterations=500)

    panels = []

    print("running Algorithm 2 style CVaR: alpha=0.48, no uncertainty", flush=True)
    base_surface_path = OUTPUTS / "value_surface_a_strict_cvar_no_uncertainty.npy"
    base_initial = (
        np.load(base_surface_path)
        if base_surface_path.exists() and not args.cold_start
        else None
    )
    if base_initial is not None:
        print(f"  warm start={base_surface_path.name}", flush=True)
    base_cvar_result = solve_robust_cvar_pwl(
        env,
        y_grid,
        1.0,
        config,
        initial_values=base_initial,
    )
    cvar_outputs = [
        (
            "a_strict_cvar_no_uncertainty",
            "alpha=0.48, no uncertainty",
            base_cvar_result,
            alpha,
        ),
        (
            "b_strict_cvar_rn_k_2",
            "alpha=0.48, RN K=2",
            base_cvar_result,
            alpha / 2.0,
        ),
    ]

    for slug, title, result, start_y in cvar_outputs:
        displayed_values = interpolate_value_surface(result.values, y_grid, start_y)
        path = extract_path(
            env,
            result,
            y_grid,
            start_y=start_y,
            kappa=1.0,
            gamma=config.gamma,
            update_risk_state=True,
            extrapolation=config.perspective_extrapolation,
        )
        evaluation_path = OUTPUTS / f"evaluation_{slug}.json"
        np.save(OUTPUTS / f"value_{slug}.npy", displayed_values)
        np.save(OUTPUTS / f"value_surface_{slug}.npy", result.values)
        np.save(OUTPUTS / f"policy_surface_{slug}.npy", result.policy)
        save_path(path, OUTPUTS / f"path_{slug}.csv")
        if not (args.reuse_evaluation and evaluation_path.exists()):
            evaluate_augmented_policy(
                env,
                result,
                y_grid,
                start_y,
                1.0,
                gamma=config.gamma,
                alpha=start_y,
                episodes=args.episodes,
                extrapolation=config.perspective_extrapolation,
            ).save(evaluation_path)
        panels.append((title, displayed_values, path, start_y))
        print(
            f"  iterations={result.iterations}, residual={result.residual:.6f}, "
            f"path_steps={len(path) - 1}",
            flush=True,
        )

    print("running Algorithm 2 style NCVaR with documented assumed kappa(x,a) in [1,2]", flush=True)
    ncvar_surface_path = OUTPUTS / "value_surface_d_strict_ncvar_decision_kappa.npy"
    ncvar_initial = (
        np.load(ncvar_surface_path)
        if ncvar_surface_path.exists() and not args.cold_start
        else base_cvar_result.values
    )
    if ncvar_surface_path.exists() and not args.cold_start:
        print(f"  warm start={ncvar_surface_path.name}", flush=True)
    ncvar_result = solve_robust_cvar_pwl(
        env,
        y_grid,
        env.kappa_decision_dependent,
        config,
        initial_values=ncvar_initial,
    )
    ncvar_values = interpolate_value_surface(ncvar_result.values, y_grid, alpha)
    ncvar_path = extract_path(
        env,
        ncvar_result,
        y_grid,
        start_y=alpha,
        kappa=env.kappa_decision_dependent,
        gamma=config.gamma,
        update_risk_state=True,
        extrapolation=config.perspective_extrapolation,
    )
    np.save(OUTPUTS / "value_d_strict_ncvar_decision_kappa.npy", ncvar_values)
    np.save(OUTPUTS / "value_surface_d_strict_ncvar_decision_kappa.npy", ncvar_result.values)
    np.save(OUTPUTS / "policy_surface_d_strict_ncvar_decision_kappa.npy", ncvar_result.policy)
    save_path(ncvar_path, OUTPUTS / "path_d_strict_ncvar_decision_kappa.csv")
    ncvar_evaluation_path = OUTPUTS / "evaluation_d_strict_ncvar_decision_kappa.json"
    if not (args.reuse_evaluation and ncvar_evaluation_path.exists()):
        evaluate_augmented_policy(
            env,
            ncvar_result,
            y_grid,
            alpha,
            env.kappa_decision_dependent,
            gamma=config.gamma,
            alpha=alpha,
            episodes=args.episodes,
            extrapolation=config.perspective_extrapolation,
        ).save(ncvar_evaluation_path)
    panels.append(("alpha=0.48, assumed kappa(x,a) in [1,2]", ncvar_values, ncvar_path, alpha))
    print(
        f"  iterations={ncvar_result.iterations}, residual={ncvar_result.residual:.6f}, "
        f"path_steps={len(ncvar_path) - 1}",
        flush=True,
    )

    print("running calibrated Figure 1c: local KL radius=0.03", flush=True)
    calibrated_kl_radius = 0.03
    kl_result = solve_kl_robust_value_iteration_vectorized(
        env,
        calibrated_kl_radius,
        config,
    )
    kl_path = extract_single_alpha_path(
        env,
        SingleAlphaResult(
            values=kl_result.values,
            policy=kl_result.policy,
            iterations=kl_result.iterations,
            residual=kl_result.residual,
        ),
    )
    kl_slug = "c_strict_kl_evar_alpha_0_03"
    np.save(OUTPUTS / f"value_{kl_slug}.npy", kl_result.values)
    save_path(kl_path, OUTPUTS / f"path_{kl_slug}.csv")
    kl_evaluation_path = OUTPUTS / f"evaluation_{kl_slug}.json"
    if not (args.reuse_evaluation and kl_evaluation_path.exists()):
        evaluate_markov_policy(
            env,
            kl_result,
            gamma=config.gamma,
            alpha=0.03,
            episodes=args.episodes,
        ).save(kl_evaluation_path)
    panels.insert(
        2,
        (
            "alpha=0.48, calibrated KL/EVaR proxy radius=0.03",
            kl_result.values,
            kl_path,
            0.03,
        ),
    )
    print(
        f"  iterations={kl_result.iterations}, residual={kl_result.residual:.6f}, "
        f"path_steps={len(kl_path) - 1}",
        flush=True,
    )

    strict_kl_radius = kl_radius_from_kl_budget(alpha, kl_kappa)
    print(
        "running separate KL formula diagnostic: kappa=2, "
        f"radius={strict_kl_radius:.4f}",
        flush=True,
    )
    strict_kl_result = solve_kl_robust_value_iteration_vectorized(
        env,
        strict_kl_radius,
        config,
    )
    strict_kl_path = extract_single_alpha_path(env, strict_kl_result)
    np.save(
        OUTPUTS / "diagnostic_value_c_kl_kappa_2.npy",
        strict_kl_result.values,
    )
    save_path(
        strict_kl_path,
        OUTPUTS / "diagnostic_path_c_kl_kappa_2.csv",
    )
    strict_evaluation_path = OUTPUTS / "diagnostic_evaluation_c_kl_kappa_2.json"
    if not (args.reuse_evaluation and strict_evaluation_path.exists()):
        evaluate_markov_policy(
            env,
            strict_kl_result,
            gamma=config.gamma,
            alpha=alpha_evar_equiv,
            episodes=args.episodes,
        ).save(strict_evaluation_path)

    render_comparison(env, panels, OUTPUTS / "strict_reproduction_gridworld.png")
    render_paper_style_comparison(env, panels, OUTPUTS / "strict_reproduction_gridworld_paper_style.png")
    manifest = {
        "paper": {
            "alpha": alpha,
            "alpha_cvar_rn": alpha / 2.0,
            "alpha_evar_reported_in_figure_caption": 0.03,
            "alpha_evar_note": "Figure caption reports 0.03; Section III.B with kappa=2 gives the value below.",
            "alpha_evar_from_section_3b_kappa_2": alpha_evar_equiv,
            "grid": [env.width, env.height],
            "obstacles": len(env.obstacles),
        },
        "implementation": {
            "gamma_assumption": config.gamma,
            "confidence_points": len(y_grid),
            "theta_assumption": 2.067,
            "collision_terminates_assumption": env.config.obstacle_terminates,
            "wall_collision_stays_assumption": env.config.wall_collision_stays,
            "ncvar_perspective_extension": config.perspective_extrapolation,
            "ncvar_risk_state_projection": "min(y, 1)",
            "kl_kappa": kl_kappa,
            "figure_1c_calibrated_kl_radius": calibrated_kl_radius,
            "diagnostic_kl_radius_from_section_3b": strict_kl_radius,
            "kl_radius_formula": "-ln(alpha) + (1/alpha) * ln(kappa)",
            "decision_kappa_source": "normalized local expected collision cost; not published by paper",
            "monte_carlo_episodes": args.episodes,
        },
    }
    (OUTPUTS / "reproduction_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {OUTPUTS / 'strict_reproduction_gridworld.png'}", flush=True)
    print(f"wrote {OUTPUTS / 'strict_reproduction_gridworld_paper_style.png'}", flush=True)


if __name__ == "__main__":
    main()
