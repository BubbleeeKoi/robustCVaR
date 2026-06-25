"""Tests for Monte Carlo policy evaluation."""

from __future__ import annotations

import unittest

import numpy as np

from robust_cvar_repro.evaluation import _summarize


class EvaluationTests(unittest.TestCase):
    def test_cvar_uses_largest_cost_tail(self) -> None:
        result = _summarize(
            costs=np.array([1.0, 2.0, 3.0, 10.0]),
            steps=np.ones(4, dtype=np.int32),
            successes=np.ones(4, dtype=bool),
            collisions=np.zeros(4, dtype=bool),
            alpha=0.5,
        )
        self.assertEqual(result.var_cost, 3.0)
        self.assertEqual(result.cvar_cost, 6.5)
        self.assertEqual(result.success_rate, 1.0)


if __name__ == "__main__":
    unittest.main()
