from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path
import sys
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
gym.logger.min_level = 40


CURRENT_DIR = Path(__file__).resolve().parent
GYM_DIR = CURRENT_DIR.parent.parent / "gym-unbalanced-disk-master"
sys.path.append(str(GYM_DIR))
sys.path.append(str(CURRENT_DIR))

import gym_unbalanced_disk  # noqa: F401
from agent import RBFFeatureExtractor, RBFQLearningAgent
from reward import calculate_custom_reward


ACTIONS = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
SEED = 42
STAB_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]


def expected_model_dir(batch_dir: Path, model_number: int, source_config: dict[str, object], w_stab: float) -> Path:
    return batch_dir / (
        f"model_{model_number:02d}_rank_{source_config['rank']}_"
        f"trial_{source_config['trial_number']:03d}_wstab_{str(w_stab).replace('.', 'p')}"
    )


def batch_is_complete(batch_dir: Path, top_configs: list[dict[str, object]]) -> bool:
    model_number = 0
    for source_config in top_configs:
        for w_stab in STAB_VALUES:
            model_number += 1
            model_dir = expected_model_dir(batch_dir, model_number, source_config, w_stab)
            if not (model_dir / "best_rbf_weights.npy").exists() or not (model_dir / "config.json").exists():
                return False
    return (batch_dir / "batch_summary.json").exists()


def get_resume_or_next_batch_dir(root: Path, top_configs: list[dict[str, object]]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    existing_versions = sorted(
        [int(path.name[1:]) for path in root.glob("v*") if path.is_dir() and path.name[1:].isdigit()]
    )
    if existing_versions:
        latest_batch = root / f"v{existing_versions[-1]}"
        if not batch_is_complete(latest_batch, top_configs):
            return latest_batch

    batch_dir = root / f"v{max(existing_versions, default=0) + 1}"
    batch_dir.mkdir()
    return batch_dir


def load_top5_configs(top5_root: Path) -> list[dict[str, object]]:
    configs = []
    trial_dirs = sorted(
        [path for path in top5_root.iterdir() if path.is_dir() and "_trial_" in path.name],
        key=lambda path: int(path.name.split("_", 1)[0]),
    )
    for trial_dir in trial_dirs[:5]:
        with (trial_dir / "config.json").open("r", encoding="utf-8") as f:
            config = json.load(f)
        configs.append(
            {
                "rank": int(trial_dir.name.split("_", 1)[0]),
                "source_dir": trial_dir.name,
                "trial_number": int(config["trial_number"]),
                "source_eval_score": float(config["eval_score"]),
                "params": config["params"],
            }
        )
    if len(configs) != 5:
        raise RuntimeError(f"Expected 5 top model configs in {top5_root}, found {len(configs)}")
    return configs


def train_model(params: dict[str, float | int], episodes: int, label: str, model_dir: Path) -> tuple[RBFQLearningAgent, list[float]]:
    model_dir.mkdir(exist_ok=True)
    env = gym.make("unbalanced-disk-sincos-v0")
    feature_extractor = RBFFeatureExtractor(params["n_bins"], params["sigma"])
    agent = RBFQLearningAgent(
        feature_extractor.num_features,
        ACTIONS,
        params["alpha"],
        params["gamma"],
        1.0,
        params["epsilon_decay"],
    )
    rewards = []
    start_episode = 0
    checkpoint_path = model_dir / "checkpoint.npz"

    if checkpoint_path.exists():
        checkpoint = np.load(checkpoint_path, allow_pickle=True)
        start_episode = int(checkpoint["episode"]) + 1
        agent.weights = checkpoint["weights"]
        agent.epsilon = float(checkpoint["epsilon"])
        rewards = checkpoint["rewards"].tolist()
        if start_episode < episodes:
            print(f"{label} | resuming from episode {start_episode + 1}/{episodes}")
        else:
            print(f"{label} | training already complete in checkpoint; finalizing outputs")

    try:
        for episode in range(start_episode, episodes):
            state, _ = env.reset(seed=SEED if episode == 0 else None)
            features = feature_extractor.get_features(state)
            done = False
            total_reward = 0.0

            while not done:
                action_idx, action_val = agent.select_action(features, evaluate=False)
                next_state, _, terminated, truncated, _ = env.step(action_val)
                done = terminated or truncated
                reward = calculate_custom_reward(
                    next_state,
                    action_val,
                    params["w_energy"],
                    params["w_position"],
                    params.get("w_balance", params["w_stab"]),
                    params["w_stab"],
                )
                next_features = feature_extractor.get_features(next_state)
                agent.update(features, action_idx, reward, next_features, done)
                features = next_features
                total_reward += reward

            agent.decay_epsilon()
            rewards.append(float(total_reward))
            np.savez(
                checkpoint_path,
                episode=episode,
                weights=agent.weights,
                epsilon=agent.epsilon,
                rewards=np.asarray(rewards),
            )
            if episode % 50 == 0 or episode == episodes - 1:
                print(f"{label} | episode {episode + 1}/{episodes} | reward {total_reward:8.2f}")
    finally:
        env.close()

    return agent, rewards


def evaluate_model(params: dict[str, float | int], agent: RBFQLearningAgent, eval_episodes: int) -> tuple[float, dict[str, list[float]]]:
    env = gym.make("unbalanced-disk-sincos-v0")
    feature_extractor = RBFFeatureExtractor(params["n_bins"], params["sigma"])
    total_score = 0.0
    first_episode = {"theta": [], "omega": [], "u": [], "top_error": []}

    try:
        for episode in range(eval_episodes):
            state, _ = env.reset(seed=SEED + episode)
            features = feature_extractor.get_features(state)
            done = False
            voltages = []
            position_score = 0.0

            while not done:
                _, action_val = agent.select_action(features, evaluate=True)
                next_state, _, terminated, truncated, _ = env.step(action_val)
                done = terminated or truncated
                position_score += -next_state[1]
                voltages.append(action_val)

                if episode == 0:
                    theta = float(np.arctan2(next_state[0], next_state[1]))
                    omega = float(next_state[2])
                    first_episode["theta"].append(theta)
                    first_episode["omega"].append(omega)
                    first_episode["u"].append(float(action_val))
                    first_episode["top_error"].append(float(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))))

                features = feature_extractor.get_features(next_state)
                state = next_state

            total_variation = float(np.sum(np.abs(np.diff(voltages)))) if len(voltages) > 1 else 0.0
            total_score += float(position_score - (0.5 * total_variation))
    finally:
        env.close()

    return total_score / eval_episodes, first_episode


def save_learning_curve(rewards: list[float], model_dir: Path, title: str) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(rewards, label="Episode reward", alpha=0.65)
    if len(rewards) >= 20:
        rolling = np.convolve(rewards, np.ones(20) / 20, mode="valid")
        plt.plot(np.arange(19, len(rewards)), rolling, color="red", label="20-episode average")
    plt.xlabel("Episode")
    plt.ylabel("Cumulative reward")
    plt.title(title)
    plt.grid(alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(model_dir / "learning_curve.png", dpi=160)
    plt.close()


def save_eval_plot(eval_trace: dict[str, list[float]], model_dir: Path, title: str) -> None:
    if not eval_trace["theta"]:
        return
    t = np.arange(len(eval_trace["theta"])) * 0.025
    fig, ax = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    ax[0].plot(t, eval_trace["theta"], label="theta")
    ax[0].axhline(np.pi, color="green", linestyle="--", linewidth=0.8)
    ax[0].axhline(-np.pi, color="green", linestyle="--", linewidth=0.8)
    ax[0].set_ylabel("theta [rad]")
    ax[0].grid(alpha=0.3)
    ax[1].plot(t, eval_trace["omega"], color="tab:orange")
    ax[1].set_ylabel("omega [rad/s]")
    ax[1].grid(alpha=0.3)
    ax[2].step(t, eval_trace["u"], color="tab:red", where="post")
    ax[2].axhline(3.0, color="black", linestyle=":", linewidth=0.8)
    ax[2].axhline(-3.0, color="black", linestyle=":", linewidth=0.8)
    ax[2].set_ylabel("u [V]")
    ax[2].set_xlabel("time [s]")
    ax[2].grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(model_dir / "evaluation_trajectory.png", dpi=160)
    plt.close(fig)


def save_model(
    model_dir: Path,
    source_config: dict[str, object],
    params: dict[str, float | int],
    agent: RBFQLearningAgent,
    rewards: list[float],
    score: float,
    eval_trace: dict[str, list[float]],
    episodes: int,
    eval_episodes: int,
) -> None:
    model_dir.mkdir(exist_ok=True)
    np.save(model_dir / "best_rbf_weights.npy", agent.weights)
    np.save(model_dir / "training_rewards.npy", np.asarray(rewards))
    np.save(model_dir / "last_eval_data.npy", eval_trace)

    title = f"Rank {source_config['rank']} trial {source_config['trial_number']} | w_stab={params['w_stab']}"
    save_learning_curve(rewards, model_dir, title)
    save_eval_plot(eval_trace, model_dir, title)

    config = {
        "source_rank": source_config["rank"],
        "source_trial_number": source_config["trial_number"],
        "source_dir": source_config["source_dir"],
        "source_eval_score": source_config["source_eval_score"],
        "episodes": episodes,
        "eval_episodes": eval_episodes,
        "eval_score": score,
        "params": params,
        "note": "Top-5 model retrained with hardcoded w_stab sweep for later hardware testing.",
    }
    with (model_dir / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config, f, indent=4)
    checkpoint_path = model_dir / "checkpoint.npz"
    if checkpoint_path.exists():
        checkpoint_path.unlink()


def completed_model_summary(model_dir: Path) -> dict[str, object] | None:
    config_path = model_dir / "config.json"
    weights_path = model_dir / "best_rbf_weights.npy"
    if not config_path.exists() or not weights_path.exists():
        return None
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)
    return {
        "model_dir": model_dir.name,
        "source_rank": config["source_rank"],
        "source_trial_number": config["source_trial_number"],
        "w_stab": config["params"]["w_stab"],
        "eval_score": config["eval_score"],
        "params": config["params"],
    }


def write_batch_summary(batch_dir: Path, args: argparse.Namespace, summary: list[dict[str, object]], complete: bool) -> None:
    sorted_summary = sorted(summary, key=lambda item: item["eval_score"], reverse=True)
    batch_summary = {
        "status": "complete" if complete else "in_progress",
        "episodes_per_model": args.episodes,
        "eval_episodes": args.eval_episodes,
        "stab_values": STAB_VALUES,
        "models_completed": len(sorted_summary),
        "models_expected": 25,
        "models": sorted_summary,
    }
    summary_path = batch_dir / ("batch_summary.json" if complete else "batch_progress.json")
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(batch_summary, f, indent=4)


def planned_model_jobs(batch_dir: Path, top_configs: list[dict[str, object]]) -> list[dict[str, object]]:
    jobs = []
    model_number = 0
    total_models = len(top_configs) * len(STAB_VALUES)
    for source_config in top_configs:
        for w_stab in STAB_VALUES:
            model_number += 1
            model_dir = expected_model_dir(batch_dir, model_number, source_config, w_stab)
            jobs.append(
                {
                    "model_number": model_number,
                    "total_models": total_models,
                    "source_config": source_config,
                    "w_stab": w_stab,
                    "model_dir": model_dir,
                }
            )
    return jobs


def count_pending_jobs(jobs: list[dict[str, object]]) -> int:
    return sum(1 for job in jobs if completed_model_summary(job["model_dir"]) is None)


def main() -> None:
    parser = argparse.ArgumentParser(description="Retrain the top 5 RBF models across hardcoded w_stab values.")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--top5-root", type=Path, default=CURRENT_DIR / "top5_results")
    parser.add_argument("--out-root", type=Path, default=CURRENT_DIR / "top5_stab_sweep_results")
    args = parser.parse_args()

    top_configs = load_top5_configs(args.top5_root)
    batch_dir = get_resume_or_next_batch_dir(args.out_root, top_configs)
    jobs = planned_model_jobs(batch_dir, top_configs)
    total_models = len(jobs)
    pending_at_start = count_pending_jobs(jobs)
    session_completed = 0
    start_time = time.time()
    summary = []

    print(f"Saving hardcoded top-5 stab sweep to {batch_dir}")
    print(f"Retraining {total_models} models: 5 source models x w_stab {STAB_VALUES}")
    print(f"Episodes per model: {args.episodes}")
    print(f"Pending this session: {pending_at_start}/{total_models} models")

    for job in jobs:
        source_config = job["source_config"]
        base_params = dict(source_config["params"])
        w_stab = job["w_stab"]
        model_number = job["model_number"]
        model_dir = job["model_dir"]
        params = dict(base_params)
        params["w_stab"] = w_stab
        label = (
            f"[Model {model_number}/{total_models}] "
            f"rank {source_config['rank']} trial {source_config['trial_number']} w_stab={w_stab}"
        )
        existing_summary = completed_model_summary(model_dir)
        if existing_summary is not None:
            summary.append(existing_summary)
            print(f"\n{label} | already complete, skipping {model_dir.name}")
            continue

        print(f"\n{label}")
        agent, rewards = train_model(params, args.episodes, label, model_dir)
        score, eval_trace = evaluate_model(params, agent, args.eval_episodes)
        save_model(model_dir, source_config, params, agent, rewards, score, eval_trace, args.episodes, args.eval_episodes)

        session_completed += 1
        elapsed = time.time() - start_time
        avg = elapsed / session_completed
        remaining_pending = max(0, pending_at_start - session_completed)
        remaining = remaining_pending * avg
        print(
            f"{label} | eval score {score:.2f} | saved to {model_dir.name} | "
            f"session {session_completed}/{pending_at_start} pending done | "
            f"ETA {datetime.timedelta(seconds=int(remaining))}"
        )
        summary.append(
            {
                "model_dir": model_dir.name,
                "source_rank": source_config["rank"],
                "source_trial_number": source_config["trial_number"],
                "w_stab": w_stab,
                "eval_score": score,
                "params": params,
            }
        )
        write_batch_summary(batch_dir, args, summary, complete=False)

    write_batch_summary(batch_dir, args, summary, complete=True)
    summary.sort(key=lambda item: item["eval_score"], reverse=True)

    print("\nTop-5 stab sweep complete.")
    print(f"Best candidate: {summary[0]['model_dir']} | score {summary[0]['eval_score']:.2f}")
    print(f"Saved batch summary to {batch_dir / 'batch_summary.json'}")


if __name__ == "__main__":
    main()
