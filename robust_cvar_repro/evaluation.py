"""Monte Carlo evaluation for reproduced risk-sensitive policies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np

from .gridworld import GridWorld
from .value_iteration import (
    SingleAlphaResult,
    SolverResult,
    greedy_action_and_allocation_at_y,
)


@dataclass(frozen=True)
class EvaluationResult:
    episodes: int
    mean_discounted_cost: float
    std_discounted_cost: float
    var_cost: float
    cvar_cost: float
    success_rate: float
    collision_rate: float
    timeout_rate: float
    mean_steps: float
    risk_state_projection_rate: float

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(asdict(self), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )


def evaluate_augmented_policy(
    env: GridWorld,
    result: SolverResult,
    y_grid: np.ndarray,
    start_y: float,
    kappa: float | np.ndarray,
    *,
    gamma: float,
    alpha: float,
    episodes: int = 5_000,
    max_steps: int = 1_000,
    seed: int = 20240624,
    extrapolation: str = "risk_neutral",
) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    costs = np.zeros(episodes, dtype=float)
    steps = np.zeros(episodes, dtype=np.int32)
    successes = np.zeros(episodes, dtype=bool)
    collisions = np.zeros(episodes, dtype=bool)
    projection_count = 0
    transition_count = 0

    for episode in range(episodes):
        state = env.start_state
        y = float(start_y)
        discount = 1.0
        for step in range(max_steps):
            action, allocation = greedy_action_and_allocation_at_y(
                env,
                result.values,
                y_grid,
                state,
                y,
                kappa,
                gamma,
                extrapolation=extrapolation,
            )
            probs = env.transition_probs[state][action]
            support_index = int(rng.choice(len(probs), p=probs))
            next_state = int(env.transition_states[state][action][support_index])
            outcome_cost = float(env.transition_costs[state][action][support_index])
            costs[episode] += discount * outcome_cost
            steps[episode] = step + 1
            discount *= gamma
            raw_next_y = max(float(allocation[support_index]), 1e-12)
            if raw_next_y > y_grid[-1]:
                projection_count += 1
            transition_count += 1
            y = min(raw_next_y, float(y_grid[-1]))
            state = next_state

            if state == env.goal_state:
                successes[episode] = True
                break
            if env.is_terminal_state(state):
                collisions[episode] = True
                break

    return _summarize(
        costs,
        steps,
        successes,
        collisions,
        alpha,
        risk_state_projection_rate=projection_count / max(transition_count, 1),
    )


def evaluate_markov_policy(
    env: GridWorld,
    result: SingleAlphaResult,
    *,
    gamma: float,
    alpha: float,
    episodes: int = 5_000,
    max_steps: int = 1_000,
    seed: int = 20240624,
) -> EvaluationResult:
    rng = np.random.default_rng(seed)
    costs = np.zeros(episodes, dtype=float)
    steps = np.zeros(episodes, dtype=np.int32)
    successes = np.zeros(episodes, dtype=bool)
    collisions = np.zeros(episodes, dtype=bool)

    for episode in range(episodes):
        state = env.start_state
        discount = 1.0
        for step in range(max_steps):
            action = int(result.policy[state])
            probs = env.transition_probs[state][action]
            support_index = int(rng.choice(len(probs), p=probs))
            next_state = int(env.transition_states[state][action][support_index])
            outcome_cost = float(env.transition_costs[state][action][support_index])
            costs[episode] += discount * outcome_cost
            steps[episode] = step + 1
            discount *= gamma
            state = next_state

            if state == env.goal_state:
                successes[episode] = True
                break
            if env.is_terminal_state(state):
                collisions[episode] = True
                break

    return _summarize(
        costs,
        steps,
        successes,
        collisions,
        alpha,
        risk_state_projection_rate=0.0,
    )


def _summarize(
    costs: np.ndarray,
    steps: np.ndarray,
    successes: np.ndarray,
    collisions: np.ndarray,
    alpha: float,
    risk_state_projection_rate: float = 0.0,
) -> EvaluationResult:
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    tail_count = max(1, int(np.ceil(alpha * len(costs))))
    ordered = np.sort(costs)
    tail = ordered[-tail_count:]
    return EvaluationResult(
        episodes=len(costs),
        mean_discounted_cost=float(np.mean(costs)),
        std_discounted_cost=float(np.std(costs)),
        var_cost=float(tail[0]),
        cvar_cost=float(np.mean(tail)),
        success_rate=float(np.mean(successes)),
        collision_rate=float(np.mean(collisions)),
        timeout_rate=float(np.mean(~successes & ~collisions)),
        mean_steps=float(np.mean(steps)),
        risk_state_projection_rate=float(risk_state_projection_rate),
    )
