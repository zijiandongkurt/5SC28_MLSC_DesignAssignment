import os
import sys
import time
import json
import datetime
import multiprocessing
import argparse
from pathlib import Path
from tqdm import tqdm

# Force Matplotlib to run in the background (headless)
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.animation as animation

import numpy as np
import optuna
import gymnasium as gym

# Setup paths based on your file structure
CURRENT_DIR = Path(__file__).resolve().parent
GYM_DIR = CURRENT_DIR.parent.parent / "gym-unbalanced-disk-master"
sys.path.append(str(GYM_DIR))
sys.path.append(str(CURRENT_DIR))

# Import your custom modules
import gym_unbalanced_disk
from agent import RBFFeatureExtractor, RBFQLearningAgent
from wrappers import AdaptiveBiasWrapper, StateRandomizationWrapper
from evaluate_robustness import HardwareImpairmentWrapper
from reward import calculate_custom_reward

# --- PIPELINE CONFIGURATION ---
TOTAL_MODELS = 5
EPISODES_PER_MODEL = 1200
ACTIONS = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
SEED = 42

def train_agent(bp, trial_dir, rank, trial_num, is_transfer=False):
    """Trains the RBF agent with intra-episode checkpointing."""
    env = gym.make('unbalanced-disk-sincos-v0')
    env = StateRandomizationWrapper(env, death_spiral_prob=0.2)
    if is_transfer:
        env = HardwareImpairmentWrapper(env, delay_steps=2, stiction_v=0.1, bias_omega=0.965)
    env = AdaptiveBiasWrapper(env, alpha=0.05, dt=0.025)
    feature_extractor = RBFFeatureExtractor(10, bp['sigma'])
    
    epsilon_start = 0.2 if is_transfer else 1.0
    agent = RBFQLearningAgent(feature_extractor.num_features, ACTIONS, 
                              bp['alpha'], bp['gamma'], epsilon_start, bp['epsilon_decay'])

    # --- TRANSFER LEARNING INITIALIZATION ---
    if is_transfer:
        source_dir = CURRENT_DIR / "top5_results" / "v2" / f"{rank}_trial_{trial_num:03d}"
        source_weights_file = source_dir / 'best_rbf_weights.npy'
        if source_weights_file.exists():
            agent.weights = np.load(source_weights_file)
        else:
            print(f"Warning: Could not find source weights for trial {trial_num} in v2. Starting from scratch.")

    checkpoint_file = trial_dir / 'checkpoint.npz'
    start_ep = 0
    reward_history = []

    # --- RESUME FROM CHECKPOINT LOGIC ---
    if checkpoint_file.exists():
        with np.load(checkpoint_file, allow_pickle=True) as data:
            start_ep = data['ep'].item() + 1
            agent.weights = data['weights']
            agent.epsilon = data['epsilon'].item()
            reward_history = data['reward_history'].tolist()

    pbar = tqdm(total=EPISODES_PER_MODEL, initial=start_ep, position=rank-1, 
                desc=f"Rank {rank} (Trial {trial_num:03d})", leave=True, ncols=110, bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}")

    for ep in range(start_ep, EPISODES_PER_MODEL):
        state, _ = env.reset(seed=SEED if ep == 0 else None)
        
        features = feature_extractor.get_features(state)
        
        done = False
        total_reward = 0

        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=False)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            reward = calculate_custom_reward(
                next_state,
                action_val,
                bp['w_energy'],
                bp['w_position'],
                bp.get("w_balance", bp["w_stab"]),
                bp["w_stab"],
            )
            
            # --- Early Termination "Death Penalty" ---
            if abs(next_state[2]) > 40.0:
                done = True
                reward = -500.0
            
            next_features = feature_extractor.get_features(next_state)
            
            agent.update(features, action_idx, reward, next_features, done)
            features = next_features
            total_reward += reward

        agent.decay_epsilon()
        reward_history.append(total_reward)
        pbar.update(1)
        pbar.set_postfix_str(f"Rwd={total_reward:6.0f}, Eps={agent.epsilon:.2f}")
        
        if ep % 25 == 0 or ep == EPISODES_PER_MODEL - 1:
            # --- SAVE CHECKPOINT ---
            np.savez(checkpoint_file, ep=ep, weights=agent.weights, 
                     epsilon=agent.epsilon, reward_history=np.array(reward_history))

    pbar.close()

    # Clean up checkpoint since we finished successfully
    if checkpoint_file.exists():
        try:
            checkpoint_file.unlink()
        except PermissionError:
            pass

    # Save Final Weights
    np.save(trial_dir / 'best_rbf_weights.npy', agent.weights)
    env.close()

    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Episode Reward', alpha=0.6)
    window = 20
    if len(reward_history) >= window:
        rolling_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(reward_history)), rolling_avg, color='red', label='Moving Average')
    plt.title(f'Training Progress - Rank {rank} (Trial {trial_num})')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig(trial_dir / 'learning_curve.png')
    plt.close()

    return agent


def evaluate_safely(agent, bp, trial_dir):
    """Runs a 7.5s evaluation with hardware-safe kinematic kill-switches."""
    env = gym.make('unbalanced-disk-sincos-v0')
    env = AdaptiveBiasWrapper(env, alpha=0.05, dt=0.025)
    feature_extractor = RBFFeatureExtractor(10, bp['sigma'])
    
    state, _ = env.reset(seed=SEED)
    features = feature_extractor.get_features(state)
    done = False
    
    thetas, omegas, voltages, times = [], [], [], []
    dt = 0.025
    step = 0
    stop_reason = "Completed Successfully"

    while not done:
        theta = np.arctan2(state[0], state[1])
        omega = state[2]
        
        if abs(omega) > 38.0 or abs(theta) > 1.0e6 or not np.isfinite(omega):
            stop_reason = f"SAFETY KILL-SWITCH ENGAGED! Omega: {omega:.2f}, Theta: {theta:.2f}"
            env.step(0.0) 
            break

        _, action_val = agent.select_action(features, evaluate=True)
        next_state, _, terminated, truncated, _ = env.step(action_val)
        done = terminated or truncated
        
        times.append(step * dt)
        thetas.append(theta)
        omegas.append(omega)
        voltages.append(action_val)
        
        features = feature_extractor.get_features(next_state)
        state = next_state
        step += 1

    env.close()

    data = {'thetas': thetas, 'omegas': omegas, 'voltages': voltages, 'stop_reason': stop_reason}
    np.save(trial_dir / 'last_eval_data.npy', data)

    if len(times) > 0:
        import matplotlib.pyplot as plt
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        ax1.plot(times, thetas, label='Angle (rad)', color='blue')
        ax1.plot(times, omegas, label='Angular Velocity (rad/s)', color='orange', alpha=0.7)
        ax1.axhline(np.pi, color='green', linestyle='--', label='Target Upright (+pi)')
        ax1.axhline(-np.pi, color='green', linestyle='--')
        ax1.set_title(f'Safe Swing-up Trajectory\nStatus: {stop_reason}')
        ax1.set_ylabel('State Value')
        ax1.legend(loc='upper right')
        ax1.grid()

        ax2.step(times, voltages, label='Input Voltage (V)', color='red')
        ax2.axhline(3.0, color='black', linestyle=':')
        ax2.axhline(-3.0, color='black', linestyle=':')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_ylim(-3.5, 3.5)
        ax2.legend(loc='upper right')
        ax2.grid()

        plt.tight_layout()
        plt.savefig(trial_dir / 'evaluation_trajectory.png')
        plt.close()


def generate_animation(trial_dir):
    """Generates the GIF using the saved evaluation data."""
    try:
        data = np.load(trial_dir / 'last_eval_data.npy', allow_pickle=True).item()
        thetas = np.array(data['thetas'])
        voltages = np.array(data['voltages'])
    except Exception:
        return

    if len(thetas) == 0:
        return 

    dt = 0.025
    times = np.arange(len(thetas)) * dt
    import matplotlib.pyplot as plt
    fig, (ax_pend, ax_ctrl) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle(f'RBF Policy: {trial_dir.name}', fontsize=16)

    ax_pend.set_xlim(-1.5, 1.5)
    ax_pend.set_ylim(-1.5, 1.5)
    ax_pend.set_aspect('equal')
    ax_pend.grid(True, linestyle='--', alpha=0.5)
    ax_pend.plot([0], [1.0], marker='*', color='green', markersize=15, alpha=0.5)
    ax_pend.plot([0], [0], marker='o', color='black', markersize=8)

    arm_line, = ax_pend.plot([], [], lw=4, color='#8B4513')
    mass_circle, = ax_pend.plot([], [], marker='o', markersize=20, color='blue')
    trail_line, = ax_pend.plot([], [], lw=2, color='blue', alpha=0.3)

    ax_ctrl.set_xlim(times[0], times[-1])
    ax_ctrl.set_ylim(-3.5, 3.5)
    ax_ctrl.set_ylabel('Voltage (V)')
    ax_ctrl.grid(True)
    ax_ctrl.step(times, voltages, color='red', alpha=0.3, where='mid')
    
    voltage_line, = ax_ctrl.step([], [], color='red', lw=2, where='mid')
    playhead = ax_ctrl.axvline(times[0], color='blue', lw=2, linestyle='--')

    def update(frame):
        current_time = times[frame]
        current_theta = thetas[frame]
        x, y = np.sin(current_theta), -np.cos(current_theta)
        
        arm_line.set_data([0, x], [0, y])
        mass_circle.set_data([x], [y])
        
        start_trail = max(0, frame - 15)
        trail_x = np.sin(thetas[start_trail:frame+1])
        trail_y = -np.cos(thetas[start_trail:frame+1])
        trail_line.set_data(trail_x, trail_y)
        
        voltage_line.set_data(times[:frame+1], voltages[:frame+1])
        playhead.set_xdata([current_time])
        
        return arm_line, mass_circle, trail_line, voltage_line, playhead

    anim = animation.FuncAnimation(fig, update, frames=len(times), interval=50, blit=True)
    anim.save(trial_dir / 'swingup_policy.gif', writer='pillow')
    plt.close(fig)


def process_model(args):
    rank, trial_number, trial_value, trial_params, results_root, is_transfer = args
    trial_dir = results_root / f"{rank}_trial_{trial_number:03d}"
    trial_dir.mkdir(exist_ok=True)
    
    if (trial_dir / 'swingup_policy.gif').exists() and (trial_dir / 'best_rbf_weights.npy').exists():
        return # Skip completed
        
    config_data = {"trial_number": trial_number, "eval_score": trial_value, "params": trial_params, "is_transfer": is_transfer}
    with open(trial_dir / 'config.json', 'w') as f:
        json.dump(config_data, f, indent=4)

    agent = train_agent(trial_params, trial_dir, rank, trial_number, is_transfer=is_transfer)
    evaluate_safely(agent, trial_params, trial_dir)
    generate_animation(trial_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Top 5 RBF Models')
    parser.add_argument('--transfer', action='store_true', help='Use transfer learning from v2 models for fine-tuning')
    args = parser.parse_args()

    global EPISODES_PER_MODEL
    EPISODES_PER_MODEL = 600 if args.transfer else 1200

    print(f"Initializing Multiprocessed Resumable Top 5 Pipeline (Transfer Learning: {args.transfer})...\n")
    
    try:
        journal_path = CURRENT_DIR / 'rbf_tuning_journal.log'
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(str(journal_path))
        )
        study = optuna.load_study(
            study_name="rbf_swingup_balance_robust",
            storage=storage,
        )
    except Exception as e:
        print(f"Error: Could not load journal file: {e}")
        sys.exit()

    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    completed_trials.sort(key=lambda t: t.value, reverse=True) 
    top_5_trials = completed_trials[:TOTAL_MODELS]

    base_results_root = CURRENT_DIR / "top5_results"
    base_results_root.mkdir(exist_ok=True)

    top_trial_numbers = {t.number for t in top_5_trials}
    # Find highest version
    existing_versions = [int(d.name[1:]) for d in base_results_root.iterdir() if d.is_dir() and d.name.startswith("v") and d.name[1:].isdigit()]
    
    if not existing_versions:
        version = 1
    else:
        latest_version = max(existing_versions)
        latest_dir = base_results_root / f"v{latest_version}"
        
        # Check if latest dir is fully complete for all top trials
        is_complete = True
        for trial in top_5_trials:
            trial_dirs = list(latest_dir.glob(f"*_trial_{trial.number:03d}"))
            if not trial_dirs:
                is_complete = False
                break
            tdir = trial_dirs[0]
            if not (tdir / 'best_rbf_weights.npy').exists() or not (tdir / 'swingup_policy.gif').exists():
                is_complete = False
                break
                
        if is_complete:
            version = latest_version + 1
        else:
            version = latest_version

    results_root = base_results_root / f"v{version}"

    results_root.mkdir(exist_ok=True)
    print(f"[INFO] Saving models to versioned folder: {results_root.relative_to(CURRENT_DIR)}\n")

    print(f"{'='*50}")
    print(f"TOP {TOTAL_MODELS} TRIALS LOADED:")
    for rank, trial in enumerate(top_5_trials, start=1):
        print(f"Rank {rank} | Trial {trial.number:03d} | Score: {trial.value:.4f}")
    print(f"{'='*50}\n")

    session_start_time = time.time()

    # Prepare arguments for multiprocessing
    args_list = []
    for rank, trial in enumerate(top_5_trials, start=1):
        args_list.append((rank, trial.number, trial.value, trial.params, results_root, args.transfer))

    # Launch 5 workers
    print(f"Launching {TOTAL_MODELS} parallel workers...\n")
    with multiprocessing.Pool(processes=TOTAL_MODELS) as pool:
        pool.map(process_model, args_list)

    total_time = time.time() - session_start_time
    # Add some newlines so we don't print over the tqdm progress bars
    print("\n" * (TOTAL_MODELS + 1))
    print(f"{'='*50}")
    print(f"PIPELINE COMPLETE!")
    print(f"Session Runtime: {datetime.timedelta(seconds=int(total_time))}")
    print(f"{'='*50}")
