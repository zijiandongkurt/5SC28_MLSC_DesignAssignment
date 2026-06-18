from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import os
import pathlib
import random

import numpy as np
import torch
from torch import nn

from dqn_common import state_features, swingup_reward
from policy_interface import Policy


ACTION_LIMIT = 3.0


class Actor(nn.Module):
    def __init__(self, state_dim: int = 4, action_limit: float = ACTION_LIMIT):
        super().__init__()
        self.action_limit = action_limit
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Tanh(),
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.action_limit * self.net(state)


class Critic(nn.Module):
    def __init__(self, state_dim: int = 4):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + 1, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
        )

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        if action.ndim == 1:
            action = action[:, None]
        return self.net(torch.cat([state, action / ACTION_LIMIT], dim=1)).squeeze(1)


@dataclass
class Transition:
    state: np.ndarray
    action: float
    reward: float
    next_state: np.ndarray
    done: bool


class ReplayBuffer:
    def __init__(self, capacity: int):
        self.data: deque[Transition] = deque(maxlen=capacity)

    def push(self, transition: Transition) -> None:
        self.data.append(transition)

    def sample(self, batch_size: int):
        batch = random.sample(self.data, batch_size)
        states = torch.tensor(np.stack([b.state for b in batch]), dtype=torch.float32)
        actions = torch.tensor([b.action for b in batch], dtype=torch.float32)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.data)


def soft_update(target: nn.Module, source: nn.Module, tau: float) -> None:
    for target_param, source_param in zip(target.parameters(), source.parameters()):
        target_param.data.mul_(1.0 - tau)
        target_param.data.add_(tau * source_param.data)


class ActorCriticPolicy(Policy):
    def __init__(self, model_path: str, exploration_std: float = 0.0):
        # The checkpoint may have been saved on macOS and can contain PosixPath
        # objects inside the stored argparse namespace. Windows cannot unpickle
        # PosixPath by default, so map it to WindowsPath before torch.load.
        if os.name == "nt":
            pathlib.PosixPath = pathlib.WindowsPath
        checkpoint = torch.load(model_path, map_location="cpu")
        self.actor = Actor()
        self.actor.load_state_dict(checkpoint["actor"])
        self.actor.eval()
        self.exploration_std = exploration_std

    def act(self, obs: np.ndarray) -> float:
        state = torch.tensor(state_features(obs)[None, :], dtype=torch.float32)
        with torch.no_grad():
            action = float(self.actor(state).item())
        if self.exploration_std > 0:
            action += float(np.random.normal(0.0, self.exploration_std))
        return float(np.clip(action, -ACTION_LIMIT, ACTION_LIMIT))
