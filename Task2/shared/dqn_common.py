from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random

import numpy as np
import torch
from torch import nn

from policy_interface import Policy, wrap_to_pi


ACTION_VALUES = np.array([-3.0, -1.5, 0.0, 1.5, 3.0], dtype=np.float32)


def state_features(obs: np.ndarray) -> np.ndarray:
    theta = float(obs[0])
    omega = float(obs[1])
    top_error = wrap_to_pi(theta - np.pi)
    return np.array(
        [
            np.sin(theta),
            np.cos(theta),
            omega / 20.0,
            top_error / np.pi,
        ],
        dtype=np.float32,
    )


def swingup_reward(obs: np.ndarray, action: float) -> float:
    theta = float(obs[0])
    omega = float(obs[1])
    top_error = wrap_to_pi(theta - np.pi)
    upright_score = np.cos(top_error)
    omega_penalty = 0.01 * (omega / 10.0) ** 2
    action_penalty = 0.002 * (float(action) / 3.0) ** 2
    near_top_bonus = 1.0 if abs(top_error) < 0.20 and abs(omega) < 1.0 else 0.0
    return float(upright_score - omega_penalty - action_penalty + near_top_bonus)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int = 4, action_dim: int = len(ACTION_VALUES)):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


@dataclass
class Transition:
    state: np.ndarray
    action_idx: int
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
        actions = torch.tensor([b.action_idx for b in batch], dtype=torch.long)
        rewards = torch.tensor([b.reward for b in batch], dtype=torch.float32)
        next_states = torch.tensor(np.stack([b.next_state for b in batch]), dtype=torch.float32)
        dones = torch.tensor([b.done for b in batch], dtype=torch.float32)
        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.data)


class DqnPolicy(Policy):
    def __init__(self, model_path: str, epsilon: float = 0.0):
        checkpoint = torch.load(model_path, map_location="cpu")
        self.actions = checkpoint.get("actions", ACTION_VALUES)
        self.q_net = QNetwork(action_dim=len(self.actions))
        self.q_net.load_state_dict(checkpoint["q_net"])
        self.q_net.eval()
        self.epsilon = epsilon

    def act(self, obs: np.ndarray) -> float:
        if random.random() < self.epsilon:
            return float(random.choice(self.actions))
        state = torch.tensor(state_features(obs)[None, :], dtype=torch.float32)
        with torch.no_grad():
            action_idx = int(torch.argmax(self.q_net(state), dim=1).item())
        return float(self.actions[action_idx])
