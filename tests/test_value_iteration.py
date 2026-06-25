"""Regression tests for the paper reproduction solvers."""

from __future__ import annotations

import math
import unittest

import numpy as np

from robust_cvar_repro.gridworld import GridWorld, GridWorldConfig
from robust_cvar_repro.value_iteration import (
    SolverConfig,
    _robust_continuation_pwl,
    confidence_grid,
    evar_confidence_from_kl_budget,
    evar_kl_radius,
    kl_radius_from_kl_budget,
    solve_robust_cvar_pwl,
)


class ValueIterationTests(unittest.TestCase):
    def test_evar_confidence_maps_to_kl_radius(self) -> None:
        self.assertAlmostEqual(evar_kl_radius(0.03), -math.log(0.03))
        self.assertEqual(evar_kl_radius(1.0), 0.0)

    def test_kl_budget_kappa_two_maps_to_section_3b_radius(self) -> None:
        alpha = 0.48
        kappa = 2.0
        alpha_prime = evar_confidence_from_kl_budget(alpha, kappa)
        radius = kl_radius_from_kl_budget(alpha, kappa)
        self.assertAlmostEqual(alpha_prime, alpha / (2.0 ** (1.0 / alpha)))
        self.assertAlmostEqual(radius, -math.log(alpha) + (1.0 / alpha) * math.log(kappa))
        self.assertAlmostEqual(radius, evar_kl_radius(alpha_prime))
        self.assertGreater(alpha_prime, 0.03)
        self.assertLess(radius, -math.log(0.03))

    def test_kappa_above_one_changes_robust_continuation(self) -> None:
        values = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 4.0, 8.0],
            ]
        )
        y_grid = np.array([0.0, 0.5, 1.0])
        states = np.array([0, 1], dtype=np.int32)
        probs = np.array([0.9, 0.1])
        costs = np.zeros(2)

        base = _robust_continuation_pwl(
            values, y_grid, states, probs, costs, 0.5, 1.0, 1.0
        )
        decision_dependent = _robust_continuation_pwl(
            values, y_grid, states, probs, costs, 0.5, 2.0, 1.0
        )
        self.assertGreater(decision_dependent, base)

    def test_returned_allocation_is_feasible(self) -> None:
        values = np.array(
            [
                [0.0, 1.0, 2.0],
                [0.0, 4.0, 8.0],
            ]
        )
        y_grid = np.array([0.0, 0.5, 1.0])
        states = np.array([0, 1], dtype=np.int32)
        probs = np.array([0.9, 0.1])
        costs = np.zeros(2)
        _, allocation = _robust_continuation_pwl(
            values,
            y_grid,
            states,
            probs,
            costs,
            0.5,
            2.0,
            1.0,
            return_allocation=True,
        )
        self.assertAlmostEqual(float(probs @ allocation), 0.5)
        self.assertTrue(np.all(allocation >= 0.0))
        self.assertTrue(np.all(allocation <= 2.0 + 1e-12))

    def test_decision_dependent_kappa_no_longer_equals_kappa_one(self) -> None:
        env = GridWorld(
            GridWorldConfig(
                width=8,
                height=7,
                start=(6, 6),
                goal=(6, 0),
                obstacle_count=4,
                use_paper_figure_obstacles=False,
                obstacle_seed=7,
            )
        )
        y_grid = confidence_grid(points=7, theta=1.8)
        config = SolverConfig(gamma=0.9, tolerance=1e-3, max_iterations=80)
        base = solve_robust_cvar_pwl(env, y_grid, 1.0, config)
        decision = solve_robust_cvar_pwl(
            env, y_grid, env.kappa_decision_dependent, config
        )
        self.assertGreater(
            float(np.max(np.abs(base.values - decision.values))),
            1e-6,
        )

    def test_warm_start_shape_is_validated(self) -> None:
        env = GridWorld(
            GridWorldConfig(
                width=5,
                height=5,
                start=(4, 4),
                goal=(4, 0),
                obstacle_count=1,
                use_paper_figure_obstacles=False,
                obstacle_seed=3,
            )
        )
        y_grid = confidence_grid(points=5, theta=2.0)
        with self.assertRaises(ValueError):
            solve_robust_cvar_pwl(
                env,
                y_grid,
                1.0,
                SolverConfig(max_iterations=1),
                initial_values=np.zeros((2, 2)),
            )


if __name__ == "__main__":
    unittest.main()
