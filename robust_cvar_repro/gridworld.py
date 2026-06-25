"""GridWorld environment for the paper's Figure 1 reproduction.

This module defines the 64 x 53 stochastic GridWorld, the obstacle set
extracted from the PDF figure, transition probabilities, one-step costs,
terminal-state handling, and the documented assumed decision-dependent
kappa(x,a) used for the NCVaR panel.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np


Action = tuple[str, int, int]


ACTIONS: tuple[Action, ...] = (
    ("east", 1, 0),
    ("south", 0, 1),
    ("west", -1, 0),
    ("north", 0, -1),
)


PAPER_FIGURE_OBSTACLES: frozenset[tuple[int, int]] = frozenset(
    {
        (4, 34),
        (6, 42),
        (7, 13),
        (8, 23),
        (11, 32),
        (13, 20),
        (14, 38),
        (15, 25),
        (17, 13),
        (18, 17),
        (20, 20),
        (23, 33),
        (25, 13),
        (25, 18),
        (25, 26),
        (28, 18),
        (28, 23),
        (28, 31),
        (29, 11),
        (31, 32),
        (31, 40),
        (32, 14),
        (34, 21),
        (35, 38),
        (36, 18),
        (36, 27),
        (37, 21),
        (37, 31),
        (39, 26),
        (40, 36),
        (42, 16),
        (42, 40),
        (43, 12),
        (43, 19),
        (43, 22),
        (43, 31),
        (44, 26),
        (44, 36),
        (45, 32),
        (46, 17),
        (46, 26),
        (46, 29),
        (47, 15),
        (47, 19),
        (47, 38),
        (48, 23),
        (48, 29),
        (49, 18),
        (50, 11),
        (50, 20),
        (50, 35),
        (50, 41),
        (51, 16),
        (51, 22),
        (51, 27),
        (51, 30),
        (51, 37),
        (52, 19),
        (53, 16),
        (54, 26),
        (55, 13),
        (55, 20),
        (55, 22),
        (55, 32),
        (56, 18),
        (56, 28),
        (56, 35),
        (56, 41),
        (58, 16),
        (58, 19),
        (59, 11),
        (59, 22),
        (59, 29),
        (59, 36),
        (60, 16),
        (60, 26),
        (60, 32),
        (60, 41),
        (61, 18),
        (61, 29),
    }
)


@dataclass(frozen=True)
class GridWorldConfig:
    width: int = 64
    height: int = 53
    start: tuple[int, int] = (60, 50)
    goal: tuple[int, int] = (60, 2)
    obstacle_count: int = 80
    use_paper_figure_obstacles: bool = True
    obstacle_seed: int = 20240527
    intended_prob: float = 0.95
    safe_cost: float = 1.0
    obstacle_cost: float = 40.0
    # The paper reports a collision cost but not the post-collision state
    # dynamics. Terminal collisions match the cited CVaR GridWorld convention.
    obstacle_terminates: bool = True
    # Boundary behavior is also not specified in the paper text.
    wall_collision_stays: bool = True


class GridWorld:
    def __init__(self, config: GridWorldConfig):
        self.config = config
        self.width = config.width
        self.height = config.height
        self.n_states = self.width * self.height
        self.n_actions = len(ACTIONS)
        self.start_state = self.to_state(config.start)
        self.goal_state = self.to_state(config.goal)
        self.obstacles = self._generate_obstacles()
        self.terminal_mask = self._build_terminal_mask()
        self.transition_states, self.transition_probs, self.transition_costs, self.expected_costs = (
            self._build_transitions()
        )
        self.next_states_padded, self.next_probs_padded, self.next_costs_padded = (
            self._build_padded_transition_arrays()
        )
        self.kappa_decision_dependent = self._build_decision_dependent_kappa()

    def to_state(self, xy: tuple[int, int]) -> int:
        x, y = xy
        return y * self.width + x

    def to_xy(self, state: int) -> tuple[int, int]:
        return state % self.width, state // self.width

    def is_inside(self, xy: tuple[int, int]) -> bool:
        x, y = xy
        return 0 <= x < self.width and 0 <= y < self.height

    def is_blocked(self, xy: tuple[int, int]) -> bool:
        return xy in self.obstacles

    def is_terminal_state(self, state: int) -> bool:
        return bool(self.terminal_mask[state])

    def export_obstacles(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            f.write("x,y\n")
            for x, y in sorted(self.obstacles):
                f.write(f"{x},{y}\n")

    def _generate_obstacles(self) -> set[tuple[int, int]]:
        if self.config.use_paper_figure_obstacles:
            return set(PAPER_FIGURE_OBSTACLES)

        rng = np.random.default_rng(self.config.obstacle_seed)
        reserved = {self.config.start, self.config.goal}
        candidates = [
            (x, y)
            for y in range(self.height)
            for x in range(self.width)
            if (x, y) not in reserved
        ]

        for _ in range(10_000):
            idx = rng.choice(len(candidates), size=self.config.obstacle_count, replace=False)
            obstacles = {candidates[int(i)] for i in idx}
            if self._has_free_path(obstacles):
                return obstacles

        raise RuntimeError("Could not generate a connected obstacle map.")

    def _build_terminal_mask(self) -> np.ndarray:
        terminal = np.zeros(self.n_states, dtype=bool)
        terminal[self.goal_state] = True
        if self.config.obstacle_terminates:
            for xy in self.obstacles:
                terminal[self.to_state(xy)] = True
        return terminal

    def _has_free_path(self, obstacles: set[tuple[int, int]]) -> bool:
        q: deque[tuple[int, int]] = deque([self.config.start])
        seen = {self.config.start}
        while q:
            xy = q.popleft()
            if xy == self.config.goal:
                return True
            x, y = xy
            for _, dx, dy in ACTIONS:
                nxt = (x + dx, y + dy)
                if self.is_inside(nxt) and nxt not in obstacles and nxt not in seen:
                    seen.add(nxt)
                    q.append(nxt)
        return False

    def _attempt_move(
        self, state: int, action_index: int
    ) -> tuple[int, float, bool]:
        if self.is_terminal_state(state):
            return self.goal_state, 0.0, False

        x, y = self.to_xy(state)
        _, dx, dy = ACTIONS[action_index]
        target = (x + dx, y + dy)
        if not self.is_inside(target):
            next_state = state if self.config.wall_collision_stays else self.goal_state
            return next_state, self.config.obstacle_cost, True
        if self.is_blocked(target):
            next_state = self.to_state(target)
            return next_state, self.config.obstacle_cost, True
        return self.to_state(target), self.config.safe_cost, False

    def _build_transitions(self):
        other_prob = (1.0 - self.config.intended_prob) / 3.0
        action_attempt_probs = np.full((self.n_actions, self.n_actions), other_prob)
        np.fill_diagonal(action_attempt_probs, self.config.intended_prob)

        transition_states: list[list[np.ndarray]] = []
        transition_probs: list[list[np.ndarray]] = []
        transition_costs: list[list[np.ndarray]] = []
        expected_costs = np.zeros((self.n_states, self.n_actions), dtype=float)

        for state in range(self.n_states):
            state_rows: list[np.ndarray] = []
            prob_rows: list[np.ndarray] = []
            cost_rows: list[np.ndarray] = []
            for action in range(self.n_actions):
                accum: dict[int, float] = {}
                cost_accum: dict[int, float] = {}
                expected_cost = 0.0
                for attempted_action, prob in enumerate(action_attempt_probs[action]):
                    next_state, cost, _ = self._attempt_move(state, attempted_action)
                    accum[next_state] = accum.get(next_state, 0.0) + float(prob)
                    cost_accum[next_state] = cost_accum.get(next_state, 0.0) + float(prob) * cost
                    expected_cost += float(prob) * cost

                states = np.array(list(accum.keys()), dtype=np.int32)
                probs = np.array([accum[int(s)] for s in states], dtype=float)
                costs = np.array([cost_accum[int(s)] / accum[int(s)] for s in states], dtype=float)
                state_rows.append(states)
                prob_rows.append(probs)
                cost_rows.append(costs)
                expected_costs[state, action] = expected_cost
            transition_states.append(state_rows)
            transition_probs.append(prob_rows)
            transition_costs.append(cost_rows)

        return transition_states, transition_probs, transition_costs, expected_costs

    def _build_padded_transition_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        max_support = max(
            len(self.transition_states[state][action])
            for state in range(self.n_states)
            for action in range(self.n_actions)
        )
        next_states = np.zeros((self.n_states, self.n_actions, max_support), dtype=np.int32)
        next_probs = np.zeros((self.n_states, self.n_actions, max_support), dtype=float)
        next_costs = np.zeros((self.n_states, self.n_actions, max_support), dtype=float)
        for state in range(self.n_states):
            for action in range(self.n_actions):
                states = self.transition_states[state][action]
                probs = self.transition_probs[state][action]
                costs = self.transition_costs[state][action]
                next_states[state, action, : len(states)] = states
                next_probs[state, action, : len(probs)] = probs
                next_costs[state, action, : len(costs)] = costs
        return next_states, next_probs, next_costs

    def _build_decision_dependent_kappa(self) -> np.ndarray:
        kappa = np.ones((self.n_states, self.n_actions), dtype=float)
        for state in range(self.n_states):
            if state == self.goal_state:
                continue
            for action in range(self.n_actions):
                risk_cost = max(self.expected_costs[state, action] - self.config.safe_cost, 0.0)
                risk_score = min(risk_cost / (self.config.obstacle_cost - self.config.safe_cost), 1.0)
                kappa[state, action] = 1.0 + risk_score
        return kappa
