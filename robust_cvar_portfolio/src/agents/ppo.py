"""PPO and CVaR / robust CVaR PPO agents."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from robust_cvar_portfolio.src.robust_cvar_layer import robust_cvar


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, hidden),
            nn.Tanh(),
        )
        self.actor = nn.Linear(hidden, act_dim)
        self.critic = nn.Linear(hidden, 1)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(obs)
        return self.actor(h), self.critic(h).squeeze(-1)


@dataclass
class TrainConfig:
    lr: float = 3e-4
    gamma: float = 0.99
    clip_eps: float = 0.2
    epochs: int = 8
    rollout_steps: int = 128
    train_iters: int = 80
    alpha: float = 0.05
    objective: str = "return"  # return | cvar | fixed_robust | state_robust | sad_robust
    fixed_kappa: float = 2.0
    kappa_max: float = 1.0
    seed: int = 42


def _collect_rollout(env, model, device, steps: int):
    obs_list, actions, rewards, losses, kappas, values, log_probs = [], [], [], [], [], [], []
    obs = env.reset()
    for _ in range(steps):
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)
        logits, value = model(obs_t)
        scale = torch.ones_like(logits)
        dist = torch.distributions.Normal(logits, scale)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(-1)
        obs2, reward, done, info = env.step(action.squeeze(0).detach().cpu().numpy())
        obs_list.append(obs)
        actions.append(action.squeeze(0))
        rewards.append(reward)
        losses.append(info["loss"])
        kappas.append(info.get("kappa", 1.0))
        values.append(value.squeeze(0))
        log_probs.append(log_prob.squeeze(0))
        obs = obs2 if not done else env.reset()
    return {
        "obs": torch.as_tensor(np.asarray(obs_list), dtype=torch.float32, device=device),
        "actions": torch.stack(actions),
        "rewards": torch.as_tensor(rewards, dtype=torch.float32, device=device),
        "losses": np.asarray(losses, dtype=float),
        "kappas": np.asarray(kappas, dtype=float),
        "values": torch.stack(values).detach(),
        "log_probs": torch.stack(log_probs).detach(),
    }


def _compute_advantages(rewards: torch.Tensor, values: torch.Tensor, gamma: float) -> torch.Tensor:
    adv = torch.zeros_like(rewards)
    gae = 0.0
    next_value = 0.0
    for t in reversed(range(len(rewards))):
        delta = rewards[t] + gamma * next_value - values[t]
        gae = delta + gamma * 0.95 * gae
        adv[t] = gae
        next_value = values[t]
    return adv


def _objective_loss(batch, cfg: TrainConfig) -> torch.Tensor:
    losses = batch["losses"]
    if cfg.objective == "return":
        return -batch["rewards"].mean()
    if cfg.objective == "cvar":
        return torch.as_tensor(robust_cvar(losses, 1.0, cfg.alpha), dtype=torch.float32)
    if cfg.objective == "fixed_robust":
        return torch.as_tensor(robust_cvar(losses, cfg.fixed_kappa, cfg.alpha), dtype=torch.float32)
    if cfg.objective in {"state_robust", "sad_robust"}:
        return torch.as_tensor(robust_cvar(losses, batch["kappas"], cfg.alpha), dtype=torch.float32)
    raise ValueError(cfg.objective)


def train_agent(env, cfg: TrainConfig) -> tuple[ActorCritic, dict]:
    torch.manual_seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device("cpu")
    model = ActorCritic(env.obs_dim, env.act_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr)

    for _ in range(cfg.train_iters):
        batch = _collect_rollout(env, model, device, cfg.rollout_steps)
        advantages = _compute_advantages(batch["rewards"], batch["values"], cfg.gamma)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        old_log_probs = batch["log_probs"]

        for _ in range(cfg.epochs):
            logits, values = model(batch["obs"])
            scale = torch.ones_like(logits)
            dist = torch.distributions.Normal(logits, scale)
            log_probs = dist.log_prob(batch["actions"]).sum(-1)
            ratio = torch.exp(log_probs - old_log_probs)
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1.0 - cfg.clip_eps, 1.0 + cfg.clip_eps) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = ((values - (advantages + batch["values"])) ** 2).mean()
            risk_term = _objective_loss(batch, cfg)
            loss = policy_loss + 0.5 * value_loss + 0.1 * risk_term
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    def policy_fn(obs: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
            logits, _ = model(obs_t)
            return logits.squeeze(0).numpy()

    return model, policy_fn


def evaluate_policy(env, policy_fn) -> dict:
    rollout = env.rollout(policy_fn)
    rollout["turnovers"] = np.zeros_like(rollout["net_returns"])
    return rollout
