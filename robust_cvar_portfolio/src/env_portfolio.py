"""Gym-like portfolio environment for RL training."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import build_market_features, kappa_state_action
from .risk_metrics import net_portfolio_loss, net_portfolio_return, turnover


@dataclass
class PortfolioEnvConfig:
    lookback: int = 20
    cost_rate: float = 0.001
    kappa_max: float = 1.0
    beta_vol: float = 1.0
    beta_dd: float = 1.0
    beta_conc: float = 0.5


class PortfolioEnv:
    def __init__(
        self,
        returns: pd.DataFrame,
        features: pd.DataFrame,
        start: str,
        end: str,
        cfg: PortfolioEnvConfig | None = None,
    ) -> None:
        self.cfg = cfg or PortfolioEnvConfig()
        mask = (returns.index >= pd.Timestamp(start)) & (returns.index <= pd.Timestamp(end))
        self.returns = returns.loc[mask]
        self.features = features.loc[self.returns.index]
        self.dates = list(self.returns.index)
        self.n_assets = self.returns.shape[1]
        self.lookback = self.cfg.lookback
        self.t = self.lookback
        self.w_prev = np.full(self.n_assets, 1.0 / self.n_assets)

    @property
    def obs_dim(self) -> int:
        return self.lookback * self.n_assets + 5 + self.n_assets

    @property
    def act_dim(self) -> int:
        return self.n_assets

    def reset(self) -> np.ndarray:
        self.t = self.lookback
        self.w_prev = np.full(self.n_assets, 1.0 / self.n_assets)
        return self._obs()

    def _obs(self) -> np.ndarray:
        hist = self.returns.iloc[self.t - self.lookback : self.t].values.reshape(-1)
        feat = self.features.iloc[self.t - 1][["Vol_z", "Corr_z", "DD_z", "Tail_z", "Disp_z"]].values
        obs = np.concatenate([hist, feat, self.w_prev])
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def step(self, logits: np.ndarray) -> tuple[np.ndarray, float, bool, dict]:
        w = np.exp(logits - np.max(logits))
        w = w / w.sum()
        date = self.dates[self.t]
        asset_ret = self.returns.iloc[self.t].values
        net_ret = net_portfolio_return(w, asset_ret, self.w_prev, self.cfg.cost_rate)
        loss = net_portfolio_loss(w, asset_ret, self.w_prev, self.cfg.cost_rate)
        kappa = kappa_state_action(
            self.features.iloc[self.t - 1],
            w,
            self.cfg.kappa_max,
            self.cfg.beta_vol,
            self.cfg.beta_dd,
            self.cfg.beta_conc,
        )
        info = {
            "portfolio_return": float(w @ asset_ret),
            "transaction_cost": self.cfg.cost_rate * turnover(w, self.w_prev),
            "net_return": net_ret,
            "loss": loss,
            "turnover": turnover(w, self.w_prev),
            "weights": w.copy(),
            "kappa": kappa,
        }
        self.w_prev = w.copy()
        self.t += 1
        done = self.t >= len(self.dates)
        obs = self._obs() if not done else np.zeros(self.obs_dim)
        return obs, net_ret, done, info

    def rollout(self, policy_fn) -> dict[str, np.ndarray]:
        obs = self.reset()
        rewards, losses, kappas = [], [], []
        while True:
            action = policy_fn(obs)
            obs, reward, done, info = self.step(action)
            rewards.append(reward)
            losses.append(info["loss"])
            kappas.append(info["kappa"])
            if done:
                break
        return {
            "net_returns": np.asarray(rewards, dtype=float),
            "losses": np.asarray(losses, dtype=float),
            "kappas": np.asarray(kappas, dtype=float),
        }
