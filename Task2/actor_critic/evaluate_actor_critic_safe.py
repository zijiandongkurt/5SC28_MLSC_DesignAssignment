from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import re
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


@dataclass
class ObservationCalibration:
    theta_offset: float = 0.0
    omega_bias: float = 0.0
    omega_sign: float = 1.0
    action_sign: float = 1.0

    def apply(self, obs: np.ndarray) -> np.ndarray:
        corrected = np.asarray(obs, dtype=np.float32).copy()
        corrected[0] = float(corrected[0]) + self.theta_offset
        corrected[1] = self.omega_sign * (float(corrected[1]) - self.omega_bias)
        return corrected


@dataclass
class ActionDebug:
    raw_u: float
    signed_u: float
    clipped_u: float
    safe_u: float


class SafePolicy:
    def __init__(self, policy: ActorCriticPolicy, safety: SafetyConfig, calibration: ObservationCalibration):
        self.policy = policy
        self.safety = safety
        self.calibration = calibration
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

    def act(self, obs: np.ndarray) -> ActionDebug:
        raw_u = float(self.policy.act(obs))
        signed_u = self.calibration.action_sign * raw_u
        clipped_u = float(np.clip(signed_u, -self.safety.voltage_limit, self.safety.voltage_limit))
        delta_u = float(np.clip(clipped_u - self.prev_u, -self.safety.rate_limit, self.safety.rate_limit))
        safe_u = float(np.clip(self.prev_u + delta_u, -self.safety.voltage_limit, self.safety.voltage_limit))
        self.prev_u = safe_u
        return ActionDebug(raw_u=raw_u, signed_u=signed_u, clipped_u=clipped_u, safe_u=safe_u)


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


def next_output_stem(out_dir: Path, prefix: str) -> str:
    version_pattern = re.compile(rf"^{re.escape(prefix)}_v(\d+)_")
    highest_version = 0
    for path in out_dir.glob(f"{prefix}_v*_*"):
        match = version_pattern.match(path.name)
        if match:
            highest_version = max(highest_version, int(match.group(1)))
    return f"{prefix}_v{highest_version + 1}"


def save_outputs(logs: dict[str, list[float]], out_dir: Path, prefix: str, metadata: dict[str, object]) -> str:
    out_dir.mkdir(parents=True, exist_ok=True)
    output_stem = next_output_stem(out_dir, prefix)
    logs_path = out_dir / f"{output_stem}_logs.npz"
    csv_path = out_dir / f"{output_stem}_logs.csv"
    metadata_path = out_dir / f"{output_stem}_metadata.json"
    plot_path = out_dir / f"{output_stem}_plot.png"

    if logs_path.exists() or csv_path.exists() or metadata_path.exists() or plot_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing output files for {output_stem}")

    arrays = {k: np.asarray(v) for k, v in logs.items()}
    np.savez(logs_path, **arrays)
    if logs:
        fieldnames = list(logs.keys())
        with csv_path.open("x", newline="", encoding="utf-8") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
            writer.writeheader()
            for row_idx in range(len(next(iter(logs.values())))):
                writer.writerow({key: logs[key][row_idx] for key in fieldnames})
    with metadata_path.open("x", encoding="utf-8") as metadata_file:
        json.dump(metadata, metadata_file, indent=2)

    if len(arrays["theta_corrected"]) == 0:
        return output_stem

    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = arrays["time_s"]
    fig, ax = plt.subplots(5, 1, figsize=(12, 11), sharex=True)
    ax[0].plot(t, arrays["theta_raw"], label="theta raw", alpha=0.55)
    ax[0].plot(t, arrays["theta_corrected"], label="theta corrected")
    ax[0].axhline(np.pi, color="k", linestyle="--", linewidth=0.8, label="upright")
    ax[0].set_ylabel("theta [rad]")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.25)

    ax[1].plot(t, arrays["top_error"], color="tab:orange")
    ax[1].set_ylabel("top error [rad]")
    ax[1].grid(True, alpha=0.25)

    ax[2].plot(t, arrays["omega_raw"], color="tab:gray", label="omega raw", alpha=0.55)
    ax[2].plot(t, arrays["omega_corrected"], color="tab:green", label="omega corrected")
    ax[2].set_ylabel("omega [rad/s]")
    ax[2].legend(loc="best")
    ax[2].grid(True, alpha=0.25)

    ax[3].plot(t, arrays["u_policy_raw"], color="tab:purple", label="policy raw")
    ax[3].plot(t, arrays["u_signed"], color="tab:blue", label="after action sign", alpha=0.75)
    ax[3].set_ylabel("policy u [V]")
    ax[3].legend(loc="best")
    ax[3].grid(True, alpha=0.25)

    ax[4].plot(t, arrays["u_safe"], color="tab:red")
    ax[4].plot(t, arrays["u_commanded"], color="tab:pink", linestyle="--", alpha=0.85)
    ax[4].axhline(float(metadata["voltage_limit"]), color="k", linestyle="--", linewidth=0.8)
    ax[4].axhline(-float(metadata["voltage_limit"]), color="k", linestyle="--", linewidth=0.8)
    ax[4].set_ylabel("safe u [V]")
    ax[4].set_xlabel("time [s]")
    ax[4].grid(True, alpha=0.25)

    fig.suptitle(f"Safe actor-critic evaluation ({output_stem})")
    fig.tight_layout()
    fig.savefig(plot_path, dpi=180)
    plt.close(fig)
    return output_stem


def main() -> None:
    parser = argparse.ArgumentParser(description="Safely evaluate an actor-critic policy before real-system testing.")
    parser.add_argument("--env", choices=["sim", "real"], default="sim")
    parser.add_argument("--model", type=Path, default=Path("Task2/actor_critic/actor_critic_policy.pt"))
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--sleep", type=float, default=0.025)
    parser.add_argument("--out-dir", type=Path, default=Path("Task2/actor_critic/safe_test"))
    parser.add_argument("--mode", choices=["policy", "zero", "constant"], default="policy")
    parser.add_argument("--constant-u", type=float, default=0.0)
    parser.add_argument("--voltage-limit", type=float, default=ACTION_LIMIT)
    parser.add_argument("--rate-limit", type=float, default=3.0)
    parser.add_argument("--omega-limit", type=float, default=38.0)
    parser.add_argument("--theta-abs-limit", type=float, default=1.0e6)
    parser.add_argument("--theta-offset", type=float, default=0.0)
    parser.add_argument("--omega-bias", type=float, default=0.0)
    parser.add_argument("--omega-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--action-sign", type=float, choices=[-1.0, 1.0], default=1.0)
    parser.add_argument("--no-confirm-real", action="store_true")
    args = parser.parse_args()

    if args.env == "real" and not args.no_confirm_real:
        confirmation = input("You are about to run on the REAL setup. Press Enter to continue, or type anything to cancel: ")
        if confirmation.strip():
            print("Cancelled real-system run.")
            return

    safety = SafetyConfig(
        voltage_limit=args.voltage_limit,
        rate_limit=args.rate_limit,
        omega_limit=args.omega_limit,
        theta_abs_limit=args.theta_abs_limit,
    )
    calibration = ObservationCalibration(
        theta_offset=args.theta_offset,
        omega_bias=args.omega_bias,
        omega_sign=args.omega_sign,
        action_sign=args.action_sign,
    )
    policy = SafePolicy(ActorCriticPolicy(str(args.model)), safety, calibration)
    env = make_env(args.env, args.render)
    logs: dict[str, list[float]] = {
        "step": [],
        "time_s": [],
        "theta_raw": [],
        "omega_raw": [],
        "theta_corrected": [],
        "omega_corrected": [],
        "top_error": [],
        "u_policy_raw": [],
        "u_signed": [],
        "u_clipped": [],
        "u_safe": [],
        "u_commanded": [],
        "raw_reward": [],
    }
    stop_reason = "completed"

    try:
        obs, info = env.reset()
        policy.reset()

        for k in range(args.steps):
            obs_raw = np.asarray(obs, dtype=np.float32)
            obs_corrected = calibration.apply(obs_raw)
            ok, reason = policy.validate_obs(obs_corrected)
            if not ok:
                stop_reason = f"safety stop at step {k}: {reason}"
                print(stop_reason)
                send_zero_action(env)
                break

            if args.mode == "zero":
                action_debug = ActionDebug(raw_u=0.0, signed_u=0.0, clipped_u=0.0, safe_u=0.0)
            elif args.mode == "constant":
                signed_u = calibration.action_sign * float(args.constant_u)
                clipped_u = float(np.clip(signed_u, -safety.voltage_limit, safety.voltage_limit))
                delta_u = float(np.clip(clipped_u - policy.prev_u, -safety.rate_limit, safety.rate_limit))
                safe_u = float(np.clip(policy.prev_u + delta_u, -safety.voltage_limit, safety.voltage_limit))
                policy.prev_u = safe_u
                action_debug = ActionDebug(raw_u=float(args.constant_u), signed_u=signed_u, clipped_u=clipped_u, safe_u=safe_u)
            else:
                action_debug = policy.act(obs_corrected)

            obs, _, terminated, truncated, info = env.step(action_debug.safe_u)

            theta = float(obs_corrected[0])
            omega = float(obs_corrected[1])
            top_error = wrap_to_pi(theta - np.pi)
            logs["step"].append(float(k))
            logs["time_s"].append(float(k * 0.025))
            logs["theta_raw"].append(float(obs_raw[0]))
            logs["omega_raw"].append(float(obs_raw[1]))
            logs["theta_corrected"].append(theta)
            logs["omega_corrected"].append(omega)
            logs["top_error"].append(float(top_error))
            logs["u_policy_raw"].append(float(action_debug.raw_u))
            logs["u_signed"].append(float(action_debug.signed_u))
            logs["u_clipped"].append(float(action_debug.clipped_u))
            logs["u_safe"].append(float(action_debug.safe_u))
            logs["u_commanded"].append(float(action_debug.safe_u))
            logs["raw_reward"].append(float(swingup_reward(obs_corrected, action_debug.safe_u)))

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
        metadata = {
            "environment": args.env,
            "model": str(args.model),
            "mode": args.mode,
            "constant_u": args.constant_u,
            "steps_requested": args.steps,
            "steps_completed": len(logs["u_safe"]),
            "stop_reason": stop_reason,
            "render": args.render,
            "sleep": args.sleep,
            "voltage_limit": args.voltage_limit,
            "rate_limit": args.rate_limit,
            "omega_limit": args.omega_limit,
            "theta_abs_limit": args.theta_abs_limit,
            "theta_offset": args.theta_offset,
            "omega_bias": args.omega_bias,
            "omega_sign": args.omega_sign,
            "action_sign": args.action_sign,
        }
        output_stem = save_outputs(logs, args.out_dir, prefix, metadata)

    if logs["u_safe"]:
        last_window = np.asarray(logs["top_error"][-min(200, len(logs["top_error"])) :])
        print("Safe actor-critic evaluation finished.")
        print(f"environment: {args.env}")
        print(f"mode: {args.mode}")
        print(f"steps requested: {args.steps}")
        print(f"steps completed: {len(logs['u_safe'])}")
        print(f"stop reason: {stop_reason}")
        print(f"mean raw omega: {np.mean(logs['omega_raw']):.3f} rad/s")
        print(f"mean corrected omega: {np.mean(logs['omega_corrected']):.3f} rad/s")
        print(f"max |u|: {np.max(np.abs(logs['u_safe'])):.3f} V")
        print(f"max |du|: {np.max(np.abs(np.diff([0.0] + logs['u_safe']))):.3f} V/step")
        print(f"mean reward: {np.mean(logs['raw_reward']):.6f}")
        print(f"mean |top error| over last window: {np.mean(np.abs(last_window)):.3f} rad")
        print(f"saved outputs in: {args.out_dir} ({output_stem})")
    else:
        print(f"No data collected. Stop reason: {stop_reason}")
        print(f"saved outputs in: {args.out_dir} ({output_stem})")


if __name__ == "__main__":
    main()
