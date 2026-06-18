import sys
import os
# Fix path: Go up one level, then into gym-unbalanced-disk-master
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-unbalanced-disk-master')))

import gymnasium as gym
import gym_unbalanced_disk
import numpy as np
import optuna
import matplotlib.pyplot as plt

from agent import RBFFeatureExtractor, RBFQLearningAgent
from reward import calculate_custom_reward

if __name__ == "__main__":
    try:
        study = optuna.load_study(study_name="rbf_swingup", storage="sqlite:///rbf_tuning.db")
        bp = study.best_params
        print("Loaded best parameters from database:", bp)
    except Exception as e:
        print("Could not load database. Run tune.py first!")
        sys.exit()

    env = gym.make('unbalanced-disk-sincos-v0')
    actions = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    
    feature_extractor = RBFFeatureExtractor(bp['n_bins'], bp['sigma'])
    agent = RBFQLearningAgent(feature_extractor.num_features, actions, bp['alpha'], bp['gamma'], 1.0, bp['epsilon_decay'])

    episodes = 800
    reward_history = []

    print("Training final agent...")
    for ep in range(episodes):
        state, _ = env.reset()
        features = feature_extractor.get_features(state)
        done = False
        total_reward = 0

        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=False)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            reward = calculate_custom_reward(next_state, action_val, bp['w_energy'], bp['w_position'], bp["w_stab"])
            next_features = feature_extractor.get_features(next_state)
            
            agent.update(features, action_idx, reward, next_features, done)
            features = next_features
            total_reward += reward

        agent.decay_epsilon()
        reward_history.append(total_reward)
        if ep % 25 == 0:
            print(f"Episode {ep}/{episodes} | Reward: {total_reward:.2f}")

    np.save('best_rbf_weights.npy', agent.weights)
    print("Saved weights to 'best_rbf_weights.npy'")
    env.close()

    plt.figure(figsize=(10, 5))
    plt.plot(reward_history, label='Episode Reward', alpha=0.6)
    
    window = 20
    if len(reward_history) >= window:
        rolling_avg = np.convolve(reward_history, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(window-1, len(reward_history)), rolling_avg, color='red', label='Moving Average')
        
    plt.title('Final Agent Training Progress')
    plt.xlabel('Episode')
    plt.ylabel('Cumulative Custom Reward')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.savefig('learning_curve.png')
    plt.show()