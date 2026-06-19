import os
import sys
import json
import numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- PATH SETUP ---
CURRENT_DIR = Path(__file__).resolve().parent
GYM_DIR = CURRENT_DIR.parent.parent / "gym-unbalanced-disk-master"
sys.path.append(str(GYM_DIR))
sys.path.append(str(CURRENT_DIR))

import gymnasium as gym
import gym_unbalanced_disk
from agent import RBFFeatureExtractor, RBFQLearningAgent
from wrappers import AdaptiveBiasWrapper

class HardwareImpairmentWrapper(gym.Wrapper):
    """Injects measured hardware impairments into the simulation."""
    def __init__(self, env, delay_steps=1, stiction_v=1.5, theta_offset=0.0003, omega_bias=-0.0001, omega_noise=0.0029):
        super().__init__(env)
        self.delay_steps = delay_steps
        self.stiction_v = stiction_v
        self.theta_offset = theta_offset
        self.omega_bias = omega_bias
        self.omega_noise = omega_noise
        self.action_buffer = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.action_buffer = [0.0] * max(1, self.delay_steps)
        return self._corrupt_obs(obs), info

    def step(self, action):
        # 1. Stiction check (if omega is very small and voltage is below stiction, it doesn't move)
        true_omega = self.env.unwrapped.omega
        if abs(true_omega) < 0.5 and abs(action) < self.stiction_v:
            effective_action = 0.0
        else:
            effective_action = action

        # 2. Delay
        if self.delay_steps > 0:
            self.action_buffer.append(effective_action)
            delayed_action = self.action_buffer.pop(0)
        else:
            delayed_action = effective_action

        obs, reward, terminated, truncated, info = self.env.step(delayed_action)
        return self._corrupt_obs(obs), reward, terminated, truncated, info

    def _corrupt_obs(self, obs):
        true_theta = np.arctan2(obs[0], obs[1])
        true_omega = obs[2]
        
        corrupted_theta = true_theta + self.theta_offset
        noise = np.random.normal(0, self.omega_noise / 3.0) # assuming 3-sigma was given
        corrupted_omega = true_omega + self.omega_bias + noise
        
        obs_new = obs.copy()
        obs_new[0] = np.sin(corrupted_theta)
        obs_new[1] = np.cos(corrupted_theta)
        obs_new[2] = corrupted_omega
        return obs_new

def load_model(trial_dir):
    with open(trial_dir / 'config.json', 'r') as f:
        config = json.load(f)
        
    bp = config['params']
    feature_extractor = RBFFeatureExtractor(bp.get('n_bins', 10), bp['sigma'])
    # Actions used during training
    actions = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    agent = RBFQLearningAgent(feature_extractor.num_features, actions, bp['alpha'], bp['gamma'], 0.0, 1.0)
    
    agent.weights = np.load(trial_dir / 'best_rbf_weights.npy')
    return agent, feature_extractor

def get_latest_models_dir():
    base_root = CURRENT_DIR / "top5_results"
    if base_root.exists():
        versions = sorted([int(p.name[1:]) for p in base_root.glob("v*") if p.is_dir() and p.name[1:].isdigit()])
        if versions: return base_root / f"v{versions[-1]}"
    return None

def main():
    models_dir = get_latest_models_dir()
    if not models_dir:
        print("No trained models found.")
        return

    models = sorted([d for d in models_dir.iterdir() if d.is_dir() and (d / 'config.json').exists()])
    print(f"Evaluating {len(models)} models from {models_dir.name} with hardware impairments injected...\n")

    fig, axes = plt.subplots(len(models), 1, figsize=(10, 3 * len(models)), sharex=True)
    if len(models) == 1: axes = [axes]
    
    results = []

    for idx, model_dir in enumerate(models):
        agent, feature_extractor = load_model(model_dir)
        
        # Build environment
        env = gym.make('unbalanced-disk-sincos-v0')
        
        # Load real hardware impairments
        try:
            sys_id = np.load(CURRENT_DIR / 'data' / 'sys_id_data.npy', allow_pickle=True).item()
            delay_steps = sys_id.get('delay_steps', 2)
            stiction_v = sys_id.get('stiction_voltage', 0.1)
            theta_offset = sys_id.get('theta_offset', 0.0)
            omega_bias = sys_id.get('omega_bias', 0.96)
            omega_noise = np.sqrt(sys_id.get('omega_variance', 0.4)) * 3 # 3-sigma
        except:
            print("Could not load data/sys_id_data.npy. Using defaults.")
            delay_steps, stiction_v, theta_offset, omega_bias, omega_noise = 2, 0.1, 0.0, 0.96, 1.88

        # 1. Inject impairments
        env = HardwareImpairmentWrapper(env, delay_steps=delay_steps, stiction_v=stiction_v, 
                                        theta_offset=theta_offset, omega_bias=omega_bias, omega_noise=omega_noise)
        # 2. Add the agent's filter
        env = AdaptiveBiasWrapper(env, alpha=0.05, dt=0.025)

        state, _ = env.reset(seed=42)
        features = feature_extractor.get_features(state)
        
        thetas, omegas, voltages, times = [], [], [], []
        dt = 0.025
        
        # Run for 300 steps (7.5 seconds)
        for step in range(300):
            theta = np.arctan2(state[0], state[1])
            omega = state[2]
            
            _, action_val = agent.select_action(features, evaluate=True)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            
            times.append(step * dt)
            thetas.append(theta)
            omegas.append(omega)
            voltages.append(action_val)
            
            features = feature_extractor.get_features(next_state)
            state = next_state
            
        env.close()
        
        # Plot trajectory
        ax = axes[idx]
        ax.plot(times, thetas, label='Angle (rad)', color='blue')
        ax.plot(times, voltages, label='Voltage (V)', color='red', alpha=0.3, drawstyle='steps-mid')
        ax.axhline(np.pi, color='green', linestyle='--', alpha=0.5)
        ax.axhline(-np.pi, color='green', linestyle='--', alpha=0.5)
        
        # Check if successful (is balanced at top near the end?)
        final_thetas = np.array(thetas[-40:]) # last 1 second
        # normalize to [-pi, pi]
        final_thetas = (final_thetas + np.pi) % (2 * np.pi) - np.pi
        # check distance to 0 (top is pi, but we normalized, wait, top is pi or -pi. Distance to pi/ -pi is same as abs(abs(final_thetas) - pi))
        dist_to_top = np.abs(np.abs(final_thetas) - np.pi)
        
        is_stable = np.all(dist_to_top < 0.3)
        status = "SUCCESS" if is_stable else "FAILED"
        
        results.append((model_dir.name, status))
        
        ax.set_title(f"Model: {model_dir.name} | Status: {status}")
        ax.set_ylabel("Rad / V")
        ax.grid(True)
        if idx == 0:
            ax.legend(loc='upper left')
            
    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "robustness_evaluation.png")
    
    print("="*50)
    print("ROBUSTNESS EVALUATION RESULTS (w/ Delay, Stiction, Bias)")
    print("="*50)
    for name, status in results:
        print(f"[{status}] {name}")
    print("\nPlot saved to robustness_evaluation.png")

if __name__ == "__main__":
    main()
