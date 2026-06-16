import sys
import os
# Fix path: Go up one level, then into gym-unbalanced-disk-master
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-unbalanced-disk-master')))

import gymnasium as gym
import gym_unbalanced_disk
import optuna
import time
from datetime import datetime, timedelta

from agent import RBFFeatureExtractor, RBFQLearningAgent
from reward import calculate_custom_reward

START_TIME = None
TRIALS_COMPLETED = 0
TOTAL_TRIALS = 100 

def objective(trial):
    global TRIALS_COMPLETED, START_TIME

    n_bins = trial.suggest_int('n_bins', 5, 12)
    sigma = trial.suggest_float('sigma', 0.2, 1.5)
    alpha = trial.suggest_float('alpha', 1e-4, 5e-2, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    eps_decay = trial.suggest_float('epsilon_decay', 0.98, 0.999)
    w_energy = trial.suggest_float('w_energy', 0.1, 2.0)
    w_position = trial.suggest_float('w_position', 1.0, 5.0)

    env = gym.make('unbalanced-disk-sincos-v0')
    
    # Coarse for swing-up, ultra-fine for balancing
    actions = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    
    feature_extractor = RBFFeatureExtractor(n_bins, sigma)
    agent = RBFQLearningAgent(feature_extractor.num_features, actions, alpha, gamma, 1.0, eps_decay)

    for episode in range(400):
        state, _ = env.reset()
        features = feature_extractor.get_features(state)
        done = False
        episode_reward = 0

        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=False)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            reward = calculate_custom_reward(next_state, action_val, w_energy, w_position)
            next_features = feature_extractor.get_features(next_state)
            
            agent.update(features, action_idx, reward, next_features, done)
            features = next_features
            episode_reward += reward

        agent.decay_epsilon()

        # --- THE PRUNING FIX ---
        if episode == 200:
            trial.report(episode_reward, step=100)
            if trial.should_prune():
                env.close()
                
                # Update the counter and print the ETA before ejecting
                TRIALS_COMPLETED += 1
                elapsed_time = time.time() - START_TIME
                avg_time = elapsed_time / TRIALS_COMPLETED
                eta_sec = avg_time * (TOTAL_TRIALS - TRIALS_COMPLETED)
                finish_time = datetime.now() + timedelta(seconds=eta_sec)
                
                print(f"[Trial {TRIALS_COMPLETED}/{TOTAL_TRIALS}] PRUNED (Poor Performance) | "
                      f"Avg: {avg_time:.1f}s | ETA: {finish_time.strftime('%H:%M:%S')}")
                      
                raise optuna.TrialPruned()

    eval_episodes = 5
    total_eval_reward = 0.0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        features = feature_extractor.get_features(state)
        done = False
        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=True)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            total_eval_reward += -next_state[1] # Reward proximity to top
            features = feature_extractor.get_features(next_state)

    env.close()
    mean_reward = total_eval_reward / eval_episodes
    
    # --- SUCCESSFUL TRIAL COMPLETION ---
    TRIALS_COMPLETED += 1
    elapsed_time = time.time() - START_TIME
    avg_time = elapsed_time / TRIALS_COMPLETED
    eta_sec = avg_time * (TOTAL_TRIALS - TRIALS_COMPLETED)
    finish_time = datetime.now() + timedelta(seconds=eta_sec)
    
    print(f"[Trial {TRIALS_COMPLETED}/{TOTAL_TRIALS}] SUCCESS | Score: {mean_reward:.2f} | "
          f"Avg: {avg_time:.1f}s | ETA: {finish_time.strftime('%H:%M:%S')}")
          
    return mean_reward

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
    
    study.optimize(objective, n_trials=TOTAL_TRIALS)
    print(f"Best Score: {study.best_value:.2f}")