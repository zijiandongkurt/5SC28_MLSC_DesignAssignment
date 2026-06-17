from __future__ import annotations

import argparse
from dataclasses import dataclass
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


@dataclass
class SafetyConfig:
    voltage_limit: float = ACTION_LIMIT
    rate_limit: float = 3.0
    omega_limit: float = 38.0
    theta_abs_limit: float = 1.0e6


class SafePolicy:
    def __init__(self, policy: ActorCriticPolicy, safety: SafetyConfig):
        self.policy = policy
        self.safety = safety
        self.prev_u = 0.0

    def reset(self) -> None:
        self.prev_u = 0.0

    def validate_obs(self, obs: np.ndarray) -> tuple[bool, str]:
        if len(obs) < 2:
            return False, "observation has fewer than 2 entries"
        theta = float(obs[0])
        omega = float(obs[1])
        if not np.isfinite(theta) or not np.isfinite(omega):
            return False, "non-finite theta/omega"
        if abs(theta) > self.safety.theta_abs_limit:
            return False, f"|theta|>{self.safety.theta_abs_limit}"
        if abs(omega) > self.safety.omega_limit:
            return False, f"|omega|>{self.safety.omega_limit}"
        return True, "ok"

    def act(self, obs: np.ndarray) -> float:
        raw_u = float(self.policy.act(obs))
        clipped_u = float(np.clip(raw_u, -self.safety.voltage_limit, self.safety.voltage_limit))
        delta_u = float(np.clip(clipped_u - self.prev_u, -self.safety.rate_limit, self.safety.rate_limit))
        safe_u = float(np.clip(self.prev_u + delta_u, -self.safety.voltage_limit, self.safety.voltage_limit))
        self.prev_u = safe_u
        return safe_u


def make_env(env_kind: str, render: bool):
    if env_kind == "sim":
        if render:
            return UnbalancedDisk(dt=0.025, umax=ACTION_LIMIT, render_mode="human")
        return gym.make("unbalanced-disk-v0", dt=0.025, umax=ACTION_LIMIT)
    if env_kind == "real":
        return gym.make("unbalanced-disk-exp-v0", dt=0.025, umax=ACTION_LIMIT)
    raise ValueError(f"Unknown env kind: {env_kind}")


def send_zero_action(env) -> None:
    try:
        env.step(0.0)
    except Exception as exc:
        print(f"Warning: failed to send zero action during shutdown: {exc}")


def save_outputs(logs: dict[str, list[float]], out_dir: Path, prefix: str) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = {k: np.asarray(v) for k, v in logs.items()}
    np.savez(out_dir / f"{prefix}_logs.npz", **arrays)

    if len(arrays["theta"]) == 0:
        return

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = np.arange(len(arrays["theta"])) * 0.025
    fig, ax = plt.subplots(4, 1, figsize=(12, 9), sharex=True)
    ax[0].plot(t, arrays["theta"], label="theta")
    ax[0].axhline(np.pi, color="k", linestyle="--", linewidth=0.8, label="upright")
    ax[0].set_ylabel("theta [rad]")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(t, arrays["top_error"], color="tab:orange")
    ax[1].set_ylabel("top error [rad]")
    ax[1].grid(True, alpha=0.25)

    ax[2].plot(t, arrays["omega"], color="tab:green")
    ax[2].set_ylabel("omega [rad/s]")
    ax[2].grid(True, alpha=0.25)

    ax[3].plot(t, arrays["u"], color="tab:red")
    ax[3].axhline(3.0, color="k", linestyle="--", linewidth=0.8)
    ax[3].axhline(-3.0, color="k", linestyle="--", linewidth=0.8)
    ax[3].set_ylabel("u [V]")
    ax[3].set_xlabel("time [s]")
    ax[3].grid(True, alpha=0.25)

    fig.suptitle(f"Safe actor-critic evaluation ({prefix})")
    fig.tight_layout()
    fig.savefig(out_dir / f"{prefix}_plot.png", dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely evaluate an actor-critic policy before real-system testing.")
    parser.add_argument("--env", choices=["sim", "real"], default="sim")
    parser.add_argument("--model", type=Path, default=Path("Task2/actor_critic/actor_critic_policy.pt"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--out-dir", type=Path, default=Path("Task2/actor_critic/safe_test"))
    parser.add_argument("--rate-limit", type=float, default=3.0)
    parser.add_argument("--omega-limit", type=float, default=38.0)
    parser.add_argument("--theta-abs-limit", type=float, default=1.0e6)
    parser.add_argument("--no-confirm-real", action="store_true")
    args = parser.parse_args()

    if args.env == "real" and not args.no_confirm_real:
        confirmation = input("You are about to run on the REAL setup. Type RUN_REAL to continue: ")
        if confirmation.strip() != "RUN_REAL":
            print("Cancelled real-system run.")
            return

    safety = SafetyConfig(rate_limit=args.rate_limit, omega_limit=args.omega_limit, theta_abs_limit=args.theta_abs_limit)
    policy = SafePolicy(ActorCriticPolicy(str(args.model)), safety)
    env = make_env(args.env, args.render)
    logs: dict[str, list[float]] = {"theta": [], "omega": [], "u": [], "raw_reward": [], "top_error": []}
    stop_reason = "completed"

    try:
        obs, info = env.reset()
        policy.reset()

        for k in range(args.steps):
            ok, reason = policy.validate_obs(obs)
            if not ok:
                stop_reason = f"safety stop at step {k}: {reason}"
                print(stop_reason)
                send_zero_action(env)
                break

            u = policy.act(obs)
            obs, _, terminated, truncated, info = env.step(u)

            theta = float(obs[0])
            omega = float(obs[1])
            top_error = wrap_to_pi(theta - np.pi)
            logs["theta"].append(theta)
            logs["omega"].append(omega)
            logs["u"].append(float(u))
            logs["raw_reward"].append(float(swingup_reward(obs, u)))
            logs["top_error"].append(float(top_error))

            if args.render:
                env.render()
                time.sleep(args.sleep)

            if terminated or truncated:
                stop_reason = f"environment ended at step {k}"
                break

    except KeyboardInterrupt:
        stop_reason = "manual keyboard interrupt"
        print("Interrupted. Sending zero action before shutdown.")
        send_zero_action(env)
    except Exception as exc:
        stop_reason = f"exception: {exc}"
        print(stop_reason)
        send_zero_action(env)
        raise
    finally:
        send_zero_action(env)
        env.close()
        prefix = f"safe_{args.env}"
        save_outputs(logs, args.out_dir, prefix)

    if logs["u"]:
        last_window = np.asarray(logs["top_error"][-min(200, len(logs["top_error"])) :])
        print("Safe actor-critic evaluation finished.")
        print(f"environment: {args.env}")
        print(f"steps requested: {args.steps}")
        print(f"steps completed: {len(logs['u'])}")
        print(f"stop reason: {stop_reason}")
        print(f"max |u|: {np.max(np.abs(logs['u'])):.3f} V")
        print(f"max |du|: {np.max(np.abs(np.diff([0.0] + logs['u']))):.3f} V/step")
        print(f"mean reward: {np.mean(logs['raw_reward']):.6f}")
        print(f"mean |top error| over last window: {np.mean(np.abs(last_window)):.3f} rad")
        print(f"saved outputs in: {args.out_dir}")
    else:
        print(f"No data collected. Stop reason: {stop_reason}")


if __name__ == "__main__":
    main()
