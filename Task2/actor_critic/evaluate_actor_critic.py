from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK2_ROOT = Path(__file__).resolve().parents[1]
GYM_REPO = PROJECT_ROOT / "gym-unbalanced-disk-master"
sys.path.insert(0, str(GYM_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASK2_ROOT / "shared"))

import gymnasium as gym
import gym_unbalanced_disk  # noqa: F401
from gym_unbalanced_disk.envs.UnbalancedDisk import UnbalancedDisk

from actor_critic_common import ACTION_LIMIT, ActorCriticPolicy
from dqn_common import swingup_reward
from policy_interface import wrap_to_pi


def run_episode(model_path: Path, steps: int, render: bool, sleep: float, exploration_std: float):
    env = UnbalancedDisk(dt=0.025, umax=ACTION_LIMIT, render_mode="human") if render else gym.make("unbalanced-disk-v0", dt=0.025, umax=ACTION_LIMIT)
    policy = ActorCriticPolicy(str(model_path), exploration_std=exploration_std)
    obs, info = env.reset()
    logs = {"theta": [], "omega": [], "u": [], "reward": [], "top_error": []}

    try:
        for _ in range(steps):
            action = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(action)
            reward = swingup_reward(obs, action)
            top_error = wrap_to_pi(float(obs[0]) - np.pi)
            logs["theta"].append(float(obs[0]))
            logs["omega"].append(float(obs[1]))
            logs["u"].append(float(action))
            logs["reward"].append(float(reward))
            logs["top_error"].append(top_error)
            if render:
                env.render()
                time.sleep(sleep)
            if terminated or truncated:
                obs, info = env.reset()
    finally:
        env.close()

    return {k: np.asarray(v) for k, v in logs.items()}


def save_outputs(logs: dict[str, np.ndarray], out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "actor_critic_eval_logs.npz", **logs)

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = np.arange(len(logs["theta"])) * 0.025
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, logs["theta"], label="theta")
    ax[0].axhline(np.pi, color="k", linestyle="--", linewidth=0.8, label="upright")
    ax[0].set_ylabel("theta [rad]")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.25)
    ax[1].plot(t, logs["top_error"], color="tab:orange")
    ax[1].set_ylabel("top error [rad]")
    ax[1].grid(True, alpha=0.25)
    ax[2].plot(t, logs["omega"], color="tab:green")
    ax[2].set_ylabel("omega [rad/s]")
    ax[2].grid(True, alpha=0.25)
    ax[3].plot(t, logs["u"], color="tab:red")
    ax[3].axhline(3.0, color="k", linestyle="--", linewidth=0.8)
    ax[3].axhline(-3.0, color="k", linestyle="--", linewidth=0.8)
    ax[3].set_ylabel("u [V]")
    ax[3].set_xlabel("time [s]")
    ax[3].grid(True, alpha=0.25)
    fig.suptitle("Actor-critic swing-up evaluation")
    fig.tight_layout()
    fig.savefig(out_dir / "actor_critic_eval.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained actor-critic policy.")
    parser.add_argument("--model", type=Path, default=Path("project_framework/results/actor_critic/actor_critic_policy.pt"))
    parser.add_argument("--steps", type=int, default=1200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--exploration-std", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("project_framework/results/actor_critic"))
    args = parser.parse_args()

    logs = run_episode(args.model, args.steps, args.render, args.sleep, args.exploration_std)
    save_outputs(logs, args.out_dir)
    last_window = logs["top_error"][-200:]
    print("Actor-critic evaluation finished.")
    print(f"steps: {args.steps}")
    print(f"max |u|: {np.max(np.abs(logs['u'])):.3f} V")
    print(f"mean reward: {np.mean(logs['reward']):.6f}")
    print(f"final theta: {logs['theta'][-1]:.3f} rad")
    print(f"final omega: {logs['omega'][-1]:.3f} rad/s")
    print(f"mean |top error| over last 200 steps: {np.mean(np.abs(last_window)):.3f} rad")
    print(f"saved plot: {args.out_dir / 'actor_critic_eval.png'}")


if __name__ == "__main__":
    main()
