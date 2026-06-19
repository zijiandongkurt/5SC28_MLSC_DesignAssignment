import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'gym-unbalanced-disk-master'))
sys.path.append(GYM_DIR)

import gymnasium as gym
import gym_unbalanced_disk
import numpy as np
import optuna
import matplotlib.pyplot as plt

from agent import RBFFeatureExtractor, RBFQLearningAgent

SEED = 42
np.random.seed(SEED)

if __name__ == "__main__":
    try:
        journal_path = os.path.join(BASE_DIR, 'models', 'rbf_tuning_journal.log')
        storage = optuna.storages.JournalStorage(
            optuna.storages.journal.JournalFileBackend(journal_path)
        )
        study = optuna.load_study(
            study_name="rbf_swingup_balance_robust",
            storage=storage,
        )
        bp = study.best_params
    except Exception as e:
        print(f"Exception: {e}")
        print("Could not load database. Run tune.py first!")
        sys.exit()
    
    env = gym.make('unbalanced-disk-sincos-v0', render_mode='human')
    actions = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    
    feature_extractor = RBFFeatureExtractor(bp['n_bins'], bp['sigma'])
    agent = RBFQLearningAgent(feature_extractor.num_features, actions, bp['alpha'], bp['gamma'], 0.0, 1.0)
    
    try:
        agent.weights = np.load(os.path.join(BASE_DIR, 'models', 'best_rbf_weights.npy'))
        print("Loaded trained weights.")
    except Exception as e:
        print("Could not find weights. Run train_final.py first!")
        sys.exit()

    state, _ = env.reset(seed=SEED)
    features = feature_extractor.get_features(state)
    done = False
    
    thetas, omegas, voltages, times = [], [], [], []
    dt = 0.025
    step = 0

    print("Simulating swing-up...")
    while not done:
        action_idx, action_val = agent.select_action(features, evaluate=True)
        next_state, _, terminated, truncated, _ = env.step(action_val)
        done = terminated or truncated
        
        theta = np.arctan2(state[0], state[1])
        omega = state[2]
        
        times.append(step * dt)
        thetas.append(theta)
        omegas.append(omega)
        voltages.append(action_val)
        
        features = feature_extractor.get_features(next_state)
        state = next_state
        step += 1

    data = {
    'thetas': thetas,
    'omegas': omegas,
    'voltages': voltages
    }
    os.makedirs(os.path.join(BASE_DIR, 'data'), exist_ok=True)
    np.save(os.path.join(BASE_DIR, 'data', 'last_eval_data.npy'), data)

    env.close()

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    ax1.plot(times, thetas, label='Angle (rad)', color='blue')
    ax1.plot(times, omegas, label='Angular Velocity (rad/s)', color='orange', alpha=0.7)
    ax1.axhline(np.pi, color='green', linestyle='--', label='Target Upright (+pi)')
    ax1.axhline(-np.pi, color='green', linestyle='--')
    ax1.set_title('Swing-up State Trajectory')
    ax1.set_ylabel('State Value')
    ax1.legend(loc='upper right')
    ax1.grid()

    ax2.step(times, voltages, label='Input Voltage (V)', color='red')
    ax2.axhline(3.0, color='black', linestyle=':', label='Max Saturation')
    ax2.axhline(-3.0, color='black', linestyle=':')
    ax2.set_title('Control Signal Effort')
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Voltage (V)')
    ax2.set_ylim(-3.5, 3.5)
    ax2.legend(loc='upper right')
    ax2.grid()

    plt.tight_layout()
    os.makedirs(os.path.join(BASE_DIR, 'visualizations'), exist_ok=True)
    save_path = os.path.join(BASE_DIR, 'visualizations', 'evaluation_trajectory.png')
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to: file:///{save_path.replace(os.sep, '/')}")
