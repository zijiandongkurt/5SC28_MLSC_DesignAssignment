from __future__ import annotations

import argparse
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TASK2_ROOT = Path(__file__).resolve().parents[1]
GYM_REPO = PROJECT_ROOT / "gym-unbalanced-disk-master"
sys.path.insert(0, str(GYM_REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(TASK2_ROOT / "shared"))

import gymnasium as gym
import gym_unbalanced_disk  # noqa: F401

from actor_critic_common import ACTION_LIMIT, Actor, Critic, ReplayBuffer, Transition, soft_update
from dqn_common import state_features, swingup_reward
from policy_interface import wrap_to_pi


def make_env():
    return gym.make("unbalanced-disk-v0", dt=0.025, umax=ACTION_LIMIT)


def optimize(
    actor: Actor,
    critic: Critic,
    target_actor: Actor,
    target_critic: Critic,
    replay: ReplayBuffer,
    actor_optimizer,
    critic_optimizer,
    batch_size: int,
    gamma: float,
    tau: float,
):
    states, actions, rewards, next_states, dones = replay.sample(batch_size)

    with torch.no_grad():
        next_actions = target_actor(next_states).squeeze(1)
        target_q = target_critic(next_states, next_actions)
        y = rewards + gamma * (1.0 - dones) * target_q

    q = critic(states, actions)
    critic_loss = nn.functional.smooth_l1_loss(q, y)
    critic_optimizer.zero_grad(set_to_none=True)
    critic_loss.backward()
    nn.utils.clip_grad_norm_(critic.parameters(), 10.0)
    critic_optimizer.step()

    actor_actions = actor(states).squeeze(1)
    actor_loss = -critic(states, actor_actions).mean()
    actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()
    nn.utils.clip_grad_norm_(actor.parameters(), 10.0)
    actor_optimizer.step()

    soft_update(target_actor, actor, tau)
    soft_update(target_critic, critic, tau)
    return float(actor_loss.item()), float(critic_loss.item())


def exploration_std_by_episode(episode: int, start: float, end: float, decay_episodes: int) -> float:
    fraction = min(1.0, episode / max(1, decay_episodes))
    return start + fraction * (end - start)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DDPG-style actor-critic policy.")
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--max-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--buffer-size", type=int, default=100_000)
    parser.add_argument("--warmup-steps", type=int, default=2_000)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--tau", type=float, default=0.01)
    parser.add_argument("--actor-lr", type=float, default=1e-4)
    parser.add_argument("--critic-lr", type=float, default=3e-4)
    parser.add_argument("--exploration-start", type=float, default=1.5)
    parser.add_argument("--exploration-end", type=float, default=0.15)
    parser.add_argument("--exploration-decay-episodes", type=int, default=650)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--out-dir", type=Path, default=Path("project_framework/results/actor_critic"))
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    env = make_env()
    actor = Actor()
    critic = Critic()
    target_actor = Actor()
    target_critic = Critic()
    target_actor.load_state_dict(actor.state_dict())
    target_critic.load_state_dict(critic.state_dict())
    actor_optimizer = torch.optim.AdamW(actor.parameters(), lr=args.actor_lr)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=args.critic_lr)
    replay = ReplayBuffer(args.buffer_size)

    global_step = 0
    episode_returns = []
    episode_top_errors = []
    actor_losses = []
    critic_losses = []

    for episode in range(args.episodes):
        obs, info = env.reset()
        state = state_features(obs)
        ep_return = 0.0
        top_errors = []
        noise_std = exploration_std_by_episode(
            episode,
            args.exploration_start,
            args.exploration_end,
            args.exploration_decay_episodes,
        )

        for _ in range(args.max_steps):
            if global_step < args.warmup_steps:
                action = float(np.random.uniform(-ACTION_LIMIT, ACTION_LIMIT))
            else:
                with torch.no_grad():
                    action = float(actor(torch.tensor(state[None, :], dtype=torch.float32)).item())
                action += float(np.random.normal(0.0, noise_std))
                action = float(np.clip(action, -ACTION_LIMIT, ACTION_LIMIT))

            next_obs, _, terminated, truncated, info = env.step(action)
            reward = swingup_reward(next_obs, action)
            next_state = state_features(next_obs)
            done = bool(terminated or truncated)
            replay.push(Transition(state, action, reward, next_state, done))

            state = next_state
            ep_return += reward
            top_errors.append(abs(wrap_to_pi(float(next_obs[0]) - np.pi)))
            global_step += 1

            if len(replay) >= args.batch_size and global_step > args.warmup_steps:
                a_loss, c_loss = optimize(
                    actor,
                    critic,
                    target_actor,
                    target_critic,
                    replay,
                    actor_optimizer,
                    critic_optimizer,
                    args.batch_size,
                    args.gamma,
                    args.tau,
                )
                actor_losses.append(a_loss)
                critic_losses.append(c_loss)

            if done:
                break

        episode_returns.append(ep_return)
        episode_top_errors.append(float(np.mean(top_errors[-100:])))

        if (episode + 1) % 25 == 0:
            print(
                f"episode {episode + 1:4d} | "
                f"noise {noise_std:.3f} | "
                f"return {np.mean(episode_returns[-25:]):8.2f} | "
                f"last100 top error {np.mean(episode_top_errors[-25:]):.3f} | "
                f"buffer {len(replay)}"
            )

    env.close()

    torch.save(
        {
            "actor": actor.state_dict(),
            "critic": critic.state_dict(),
            "args": vars(args),
        },
        args.out_dir / "actor_critic_policy.pt",
    )
    np.savez(
        args.out_dir / "actor_critic_training_log.npz",
        episode_returns=np.asarray(episode_returns),
        episode_top_errors=np.asarray(episode_top_errors),
        actor_losses=np.asarray(actor_losses),
        critic_losses=np.asarray(critic_losses),
    )

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    fig, ax = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax[0].plot(episode_returns)
    ax[0].set_ylabel("episode return")
    ax[0].grid(True, alpha=0.25)
    ax[1].plot(episode_top_errors)
    ax[1].set_ylabel("mean |top error|")
    ax[1].set_xlabel("episode")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(args.out_dir / "actor_critic_training_curve.png", dpi=180)
    plt.close(fig)
    print(f"Saved actor-critic policy to {args.out_dir / 'actor_critic_policy.pt'}")


if __name__ == "__main__":
    main()
