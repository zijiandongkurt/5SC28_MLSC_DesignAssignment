import sys
import os
# Fix path: Go up one level, then into gym-unbalanced-disk-master
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-unbalanced-disk-master')))

import gymnasium as gym
gym.logger.set_level(40)
import gym_unbalanced_disk
import optuna
import time
from datetime import datetime, timedelta

from agent import RBFFeatureExtractor, RBFQLearningAgent
from reward import calculate_custom_reward

START_TIME = None

def objective(trial):
    # 1. Suggest Hyperparameters
    n_bins = trial.suggest_int('n_bins', 5, 12)
    sigma = trial.suggest_float('sigma', 0.2, 1.5)
    alpha = trial.suggest_float('alpha', 1e-4, 5e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    eps_decay = trial.suggest_float('epsilon_decay', 0.98, 0.999)
    w_energy = trial.suggest_float('w_energy', 0.1, 2.0)
    w_position = trial.suggest_float('w_position', 1.0, 5.0)
    w_stab = trial.suggest_float('w_stab', 0.0, 2.0)

    # 2. Setup Environment and Agent
    env = gym.make('unbalanced-disk-sincos-v0')
    
    # The 15 ultra-fine actions for smooth control at the top
    actions = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    
    feature_extractor = RBFFeatureExtractor(n_bins, sigma)
    agent = RBFQLearningAgent(feature_extractor.num_features, actions, alpha, gamma, 1.0, eps_decay)

    # 3. Training Loop (Extended to 400 episodes)
    for episode in range(400):
        # Seed locked for deterministic noise
        state, _ = env.reset(seed=42)
        features = feature_extractor.get_features(state)
        done = False
        episode_reward = 0

        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=False)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            reward = calculate_custom_reward(next_state, action_val, w_energy, w_position, w_stab)
            next_features = feature_extractor.get_features(next_state)
            
            agent.update(features, action_idx, reward, next_features, done)
            features = next_features
            episode_reward += reward

        agent.decay_epsilon()

        # 4. Pruning Check (Delayed to episode 200)
        if episode == 200:
            trial.report(episode_reward, step=200)
            if trial.should_prune():
                env.close()
                # Clean exit for Windows multiprocessing
                raise optuna.TrialPruned()

    # 5. Evaluation Loop (Pure physics score)
    # 5. Evaluation Loop (Physics + Smoothness Score)
    import numpy as np # Ensure this is at the top of your file
    
    eval_episodes = 5
    total_eval_score = 0.0

    for _ in range(eval_episodes):
        state, _ = env.reset(seed=42)
        features = feature_extractor.get_features(state)
        done = False
        
        episode_voltages = []
        episode_position_score = 0.0

        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=True)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            # Position score: +1.0 for perfectly upright
            episode_position_score += -next_state[1] 
            
            # Track the exact voltage used
            episode_voltages.append(action_val)
            features = feature_extractor.get_features(next_state)

        # Calculate Total Variation (Actuator Chattering)
        voltage_diffs = np.abs(np.diff(episode_voltages))
        total_variation = np.sum(voltage_diffs)
        
        # THE SMOOTHNESS PENALTY
        # We multiply TV by 0.5 so it doesn't completely overwhelm the position score.
        # This means every 1V of unnecessary jitter costs the agent 0.5 points.
        episode_final_score = episode_position_score - (0.5 * total_variation)
        
        total_eval_score += episode_final_score

    env.close()
    
    # 6. Return Final Score
    mean_score = total_eval_score / eval_episodes
    return mean_score


if __name__ == "__main__":
    print("Starting Optimization...")
    START_TIME = time.time()
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="rbf_swingup",
        storage="sqlite:///rbf_tuning.db", 
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    TOTAL_TARGET_TRIALS = 250
    completed_trials = len(study.trials)
    trials_to_run = TOTAL_TARGET_TRIALS - completed_trials
    
    study.optimize(objective, n_trials=trials_to_run, n_jobs=4, show_progress_bar=True)
    print(f"Best Score: {study.best_value:.2f}")