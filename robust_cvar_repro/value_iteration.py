"""Value-iteration algorithms for robust CVaR, NCVaR, and KL/EVaR panels.

This module contains the numerical core of the reproduction: confidence-grid
construction, Algorithm 2-style piecewise-linear CVaR/NCVaR updates, KL robust
Bellman updates, interpolation on y * V(x,y), and deterministic display-path
extraction for plotting and path comparison.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from .gridworld import GridWorld


@dataclass(frozen=True)
class SolverConfig:
    gamma: float = 0.95
    tolerance: float = 1e-4
    max_iterations: int = 500
    perspective_extrapolation: str = "risk_neutral"


@dataclass
class SolverResult:
    values: np.ndarray
    policy: np.ndarray
    iterations: int
    residual: float


@dataclass
class SingleAlphaResult:
    values: np.ndarray
    policy: np.ndarray
    iterations: int
    residual: float


def confidence_grid(
    alpha: float = 0.48,
    alpha_evar_proxy: float = 0.03,
    points: int = 21,
    theta: float = 2.067,
) -> np.ndarray:
    """Return the geometric confidence grid used by the reproduction.

    The paper states that the experiment uses 21 points with
    y_{i+1} = theta * y_i, but it does not publish theta. The default
    value is the project assumption inherited from the cited GridWorld
    setup, not a directly recoverable paper parameter.
    """
    del alpha, alpha_evar_proxy
    if points < 2:
        raise ValueError("The confidence grid requires at least two points.")
    if theta <= 1.0:
        raise ValueError("theta must be greater than one for a geometric grid.")
    positive = theta ** np.arange(-(points - 2), 1, dtype=float)
    return np.concatenate([np.array([0.0]), positive])


def evar_kl_radius(alpha: float) -> float:
    """Return the KL radius in the EVaR dual representation.

    EVaR_alpha(Z) = sup_{Q: D_KL(Q||P) <= -log(alpha)} E_Q[Z].
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("EVaR confidence level alpha must be in (0, 1].")
    return -math.log(alpha)


def evar_confidence_from_kl_budget(alpha: float, kappa: float) -> float:
    """Return the equivalent EVaR confidence level under a fixed KL budget.

    Section III.B sets K = ln(kappa) and shows robust CVaR with KL
    uncertainty reduces to EVaR with alpha' = alpha / kappa^(1/alpha).
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    if kappa < 1.0:
        raise ValueError("kappa must be at least 1.")
    return alpha / (kappa ** (1.0 / alpha))


def kl_radius_from_kl_budget(alpha: float, kappa: float) -> float:
    """Return the combined KL radius for robust CVaR under K = ln(kappa).

    D_KL(Q, P) <= -ln(alpha) + (1/alpha) * ln(kappa) = -ln(alpha').
    """
    if not 0.0 < alpha <= 1.0:
        raise ValueError("alpha must be in (0, 1].")
    if kappa < 1.0:
        raise ValueError("kappa must be at least 1.")
    return -math.log(alpha) + (1.0 / alpha) * math.log(kappa)


def interpolate_value_surface(values: np.ndarray, y_grid: np.ndarray, y: float) -> np.ndarray:
    if y <= y_grid[0]:
        return values[:, 0]
    if y >= y_grid[-1]:
        return values[:, -1]
    right = int(np.searchsorted(y_grid, y))
    left = right - 1
    span = y_grid[right] - y_grid[left]
    weight = (y - y_grid[left]) / span
    interpolated_y_value = (
        (1.0 - weight) * y_grid[left] * values[:, left]
        + weight * y_grid[right] * values[:, right]
    )
    return interpolated_y_value / max(y, 1e-12)


def solve_risk_neutral_value_iteration(env: GridWorld, config: SolverConfig) -> SingleAlphaResult:
    values = np.zeros(env.n_states, dtype=float)
    policy = np.zeros(env.n_states, dtype=np.int8)
    non_terminal = ~env.terminal_mask
    for iteration in range(1, config.max_iterations + 1):
        old = values
        action_values = np.empty((env.n_states, env.n_actions), dtype=float)
        for action in range(env.n_actions):
            next_values = old[env.next_states_padded[:, action, :]]
            continuation = env.next_costs_padded[:, action, :] + config.gamma * next_values
            action_values[:, action] = np.sum(env.next_probs_padded[:, action, :] * continuation, axis=1)
        best_actions = np.argmin(action_values, axis=1)
        values = action_values[np.arange(env.n_states), best_actions]
        values[env.terminal_mask] = 0.0
        policy = best_actions.astype(np.int8)
        policy[env.terminal_mask] = 0
        residual = float(np.max(np.abs(values[non_terminal] - old[non_terminal])))
        if residual < config.tolerance:
            return SingleAlphaResult(values=values, policy=policy, iterations=iteration, residual=residual)
    return SingleAlphaResult(values=values, policy=policy, iterations=config.max_iterations, residual=residual)


def solve_robust_cvar_pwl(
    env: GridWorld,
    y_grid: np.ndarray,
    kappa: float | np.ndarray,
    config: SolverConfig,
    initial_values: np.ndarray | None = None,
) -> SolverResult:
    if initial_values is None:
        neutral = solve_risk_neutral_value_iteration(env, config).values
        values = np.repeat(neutral[:, None], len(y_grid), axis=1)
    else:
        expected_shape = (env.n_states, len(y_grid))
        if initial_values.shape != expected_shape:
            raise ValueError(
                f"initial_values has shape {initial_values.shape}; "
                f"expected {expected_shape}."
            )
        values = np.array(initial_values, dtype=float, copy=True)
    policy = np.zeros((env.n_states, len(y_grid)), dtype=np.int8)
    non_goal = ~env.terminal_mask

    for iteration in range(1, config.max_iterations + 1):
        old = values
        new = np.zeros_like(old)

        for state in range(env.n_states):
            if env.is_terminal_state(state):
                continue

            for y_index, y in enumerate(y_grid):
                best_value = float("inf")
                best_action = 0
                for action in range(env.n_actions):
                    local_kappa = (
                        float(kappa[state, action])
                        if isinstance(kappa, np.ndarray)
                        else float(kappa)
                    )
                    states = env.transition_states[state][action]
                    probs = env.transition_probs[state][action]
                    costs = env.transition_costs[state][action]
                    candidate = _robust_continuation_pwl(
                        old,
                        y_grid,
                        states,
                        probs,
                        costs,
                        float(y),
                        local_kappa,
                        config.gamma,
                        extrapolation=config.perspective_extrapolation,
                    )
                    if candidate < best_value:
                        best_value = candidate
                        best_action = action
                new[state, y_index] = best_value
                policy[state, y_index] = best_action

        residual = float(np.max(np.abs(new[non_goal] - old[non_goal])))
        values = new
        if iteration % 25 == 0:
            alpha_index = int(np.argmin(np.abs(y_grid - 0.48)))
            alpha_residual = float(np.max(np.abs(new[non_goal, alpha_index] - old[non_goal, alpha_index])))
            print(
                f"    iter={iteration} residual={residual:.6f} alpha_residual={alpha_residual:.6f}",
                flush=True,
            )
        if residual < config.tolerance:
            return SolverResult(values=values, policy=policy, iterations=iteration, residual=residual)

    return SolverResult(values=values, policy=policy, iterations=config.max_iterations, residual=residual)


def solve_kl_robust_value_iteration_vectorized(
    env: GridWorld,
    kl_radius: float,
    config: SolverConfig,
) -> SingleAlphaResult:
    values = np.zeros(env.n_states, dtype=float)
    policy = np.zeros(env.n_states, dtype=np.int8)
    non_goal = ~env.terminal_mask

    for iteration in range(1, config.max_iterations + 1):
        old = values
        action_values = np.empty((env.n_states, env.n_actions), dtype=float)

        for action in range(env.n_actions):
            states = env.next_states_padded[:, action, :]
            probs = env.next_probs_padded[:, action, :]
            scores = env.next_costs_padded[:, action, :] + config.gamma * old[states]
            action_values[:, action] = _kl_worst_case_expectation_vectorized(scores, probs, kl_radius)

        best_actions = np.argmin(action_values, axis=1)
        new = action_values[np.arange(env.n_states), best_actions]
        policy = best_actions.astype(np.int8)
        new[env.terminal_mask] = 0.0
        policy[env.terminal_mask] = 0

        residual = float(np.max(np.abs(new[non_goal] - old[non_goal])))
        values = new
        if residual < config.tolerance:
            return SingleAlphaResult(values=values, policy=policy, iterations=iteration, residual=residual)

    return SingleAlphaResult(values=values, policy=policy, iterations=config.max_iterations, residual=residual)


def extract_path(
    env: GridWorld,
    result: SolverResult,
    y_grid: np.ndarray,
    start_y: float,
    kappa: float | np.ndarray = 1.0,
    gamma: float = 0.95,
    max_steps: int = 1_000,
    update_risk_state: bool = False,
    extrapolation: str = "risk_neutral",
) -> list[tuple[int, int]]:
    state = env.start_state
    y = float(start_y)
    path = [env.to_xy(state)]
    seen: dict[tuple[int, int], int] = {}

    for step in range(max_steps):
        if env.is_terminal_state(state):
            break
        xy = env.to_xy(state)
        seen[xy] = seen.get(xy, 0) + 1
        if seen[xy] > 8:
            break

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
        states = env.transition_states[state][action]
        probs = env.transition_probs[state][action]
        next_index = int(np.argmax(probs))
        next_state = int(states[next_index])
        if update_risk_state:
            y = min(max(float(allocation[next_index]), 1e-12), float(y_grid[-1]))
        state = next_state
        path.append(env.to_xy(state))

    return path


def greedy_action_at_y(
    env: GridWorld,
    values: np.ndarray,
    y_grid: np.ndarray,
    state: int,
    y: float,
    kappa: float | np.ndarray,
    gamma: float = 0.95,
) -> int:
    action, _ = greedy_action_and_allocation_at_y(
        env,
        values,
        y_grid,
        state,
        y,
        kappa,
        gamma,
    )
    return action


def greedy_action_and_allocation_at_y(
    env: GridWorld,
    values: np.ndarray,
    y_grid: np.ndarray,
    state: int,
    y: float,
    kappa: float | np.ndarray,
    gamma: float = 0.95,
    extrapolation: str = "risk_neutral",
) -> tuple[int, np.ndarray]:
    if env.is_terminal_state(state):
        return 0, np.ones(1, dtype=float)
    best_value = float("inf")
    best_action = 0
    best_allocation: np.ndarray | None = None
    for action in range(env.n_actions):
        local_kappa = (
            float(kappa[state, action])
            if isinstance(kappa, np.ndarray)
            else float(kappa)
        )
        robust_next, allocation = _robust_continuation_pwl(
            values,
            y_grid,
            env.transition_states[state][action],
            env.transition_probs[state][action],
            env.transition_costs[state][action],
            float(y),
            local_kappa,
            gamma,
            extrapolation=extrapolation,
            return_allocation=True,
        )
        candidate = robust_next
        if candidate < best_value:
            best_value = candidate
            best_action = action
            best_allocation = allocation
    if best_allocation is None:
        raise RuntimeError(
            f"No feasible NCVaR allocation at state={state}, y={y:.6g}."
        )
    return best_action, best_allocation


def extract_single_alpha_path(
    env: GridWorld,
    result: SingleAlphaResult,
    max_steps: int = 1_000,
) -> list[tuple[int, int]]:
    state = env.start_state
    path = [env.to_xy(state)]
    seen: dict[tuple[int, int], int] = {}

    for _ in range(max_steps):
        if env.is_terminal_state(state):
            break
        xy = env.to_xy(state)
        seen[xy] = seen.get(xy, 0) + 1
        if seen[xy] > 8:
            break

        action = int(result.policy[state])
        states = env.transition_states[state][action]
        probs = env.transition_probs[state][action]
        next_state = int(states[int(np.argmax(probs))])
        state = next_state
        path.append(env.to_xy(state))

    return path


def save_path(path: list[tuple[int, int]], file_path) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write("step,x,y\n")
        for step, (x, y) in enumerate(path):
            f.write(f"{step},{x},{y}\n")


def _robust_continuation_pwl(
    values: np.ndarray,
    y_grid: np.ndarray,
    states: np.ndarray,
    probs: np.ndarray,
    costs: np.ndarray,
    y: float,
    kappa: float,
    gamma: float,
    extrapolation: str = "risk_neutral",
    return_allocation: bool = False,
) -> float | tuple[float, np.ndarray]:
    active = probs > 0.0
    states = states[active]
    probs = probs[active]
    costs = costs[active]
    if kappa <= 0.0:
        raise ValueError("kappa must be positive.")
    if y <= 1e-12:
        result = float(np.max(costs + gamma * values[states, 0]))
        allocation = np.zeros(len(states), dtype=float)
        return (result, allocation) if return_allocation else result

    budget = y
    objective = 0.0
    pieces: list[tuple[float, float, int]] = []
    upper_u = float(kappa)
    if float(np.sum(probs)) * upper_u + 1e-12 < budget:
        result = float("inf")
        allocation = np.zeros(len(states), dtype=float)
        return (result, allocation) if return_allocation else result
    allocations = np.zeros(len(states), dtype=float)

    for support_index, (state, prob, cost) in enumerate(zip(states, probs, costs)):
        z = y_grid * (float(cost) + gamma * values[int(state)])
        objective += float(prob) * float(z[0])
        for left_index in range(len(y_grid) - 1):
            left = float(y_grid[left_index])
            right = float(y_grid[left_index + 1])
            if right <= 0.0 or left >= upper_u:
                continue
            seg_left = max(left, 0.0)
            seg_right = min(right, upper_u)
            if seg_right <= seg_left:
                continue
            slope = float((z[left_index + 1] - z[left_index]) / (right - left))
            capacity_budget = float(prob) * (seg_right - seg_left)
            pieces.append((slope, capacity_budget, support_index))

        if upper_u > y_grid[-1]:
            if extrapolation == "linear":
                last_span = float(y_grid[-1] - y_grid[-2])
                last_slope = float((z[-1] - z[-2]) / last_span)
            elif extrapolation == "risk_neutral":
                # The paper defines the risk state only on (0, 1]. Extending
                # yV linearly with slope V(x, 1) treats y >= 1 as the
                # risk-neutral endpoint while retaining the perspective form.
                last_slope = float(cost) + gamma * float(values[int(state), -1])
            else:
                raise ValueError(
                    "NCVaR requires y*xi above the published grid domain. "
                    "Use 'risk_neutral', 'linear', or provide an extended grid."
                )
            capacity_budget = float(prob) * (upper_u - float(y_grid[-1]))
            pieces.append((last_slope, capacity_budget, support_index))

    remaining = budget
    for slope, capacity_budget, support_index in sorted(
        pieces, key=lambda item: item[0], reverse=True
    ):
        if remaining <= 1e-12:
            break
        allocation = min(remaining, capacity_budget)
        objective += slope * allocation
        allocations[support_index] += allocation / float(probs[support_index])
        remaining -= allocation

    if remaining > 1e-8:
        raise RuntimeError(
            f"Infeasible NCVaR allocation: remaining budget={remaining:.6g}, "
            f"y={y:.6g}, kappa={kappa:.6g}."
        )

    result = objective / y
    return (result, allocations) if return_allocation else result


def _kl_worst_case_expectation_vectorized(
    scores: np.ndarray,
    probs: np.ndarray,
    radius: float,
) -> np.ndarray:
    if radius <= 1e-12:
        return np.sum(probs * scores, axis=1)

    best = np.argmax(scores, axis=1)
    rows = np.arange(scores.shape[0])
    best_prob = np.maximum(probs[rows, best], 1e-300)
    can_concentrate = -np.log(best_prob) <= radius

    lo = np.full(scores.shape[0], 1e-8, dtype=float)
    hi = np.maximum(np.max(scores, axis=1) - np.min(scores, axis=1), 1.0)

    for _ in range(30):
        kl_hi = _tilted_kl_vectorized(scores, probs, hi)
        grow = kl_hi > radius
        if not np.any(grow):
            break
        hi[grow] *= 2.0

    for _ in range(50):
        mid = 0.5 * (lo + hi)
        kl_mid = _tilted_kl_vectorized(scores, probs, mid)
        high_kl = kl_mid > radius
        lo[high_kl] = mid[high_kl]
        hi[~high_kl] = mid[~high_kl]

    q = _kl_tilted_distribution_vectorized(scores, probs, hi)
    result = np.sum(q * scores, axis=1)
    result[can_concentrate] = scores[rows[can_concentrate], best[can_concentrate]]
    return result


def _kl_tilted_distribution_vectorized(
    scores: np.ndarray,
    probs: np.ndarray,
    eta: np.ndarray,
) -> np.ndarray:
    shifted = (scores - np.max(scores, axis=1, keepdims=True)) / np.maximum(eta[:, None], 1e-12)
    weights = probs * np.exp(shifted)
    totals = np.sum(weights, axis=1, keepdims=True)
    return np.divide(weights, totals, out=np.zeros_like(weights), where=totals > 0.0)


def _tilted_kl_vectorized(scores: np.ndarray, probs: np.ndarray, eta: np.ndarray) -> np.ndarray:
    q = _kl_tilted_distribution_vectorized(scores, probs, eta)
    log_ratio = np.zeros_like(q)
    positive = (q > 0.0) & (probs > 0.0)
    log_ratio[positive] = np.log(q[positive] / probs[positive])
    return np.sum(q * log_ratio, axis=1)
