import sys
import os
# Fix path: Go up one level, then into gym-unbalanced-disk-master
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'gym-unbalanced-disk-master')))

import gymnasium as gym
import gym_unbalanced_disk
import numpy as np
import optuna
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from agent import RBFFeatureExtractor, RBFQLearningAgent

SEED = 42
np.random.seed(SEED)

# ==========================================
# GIF CONFIGURATION SETTINGS
# ==========================================
START_TIME = 0.0        # Seconds to start the GIF (e.g., 1.0 to skip the beginning)
END_TIME = 7.5          # Seconds to end the GIF (e.g., 3.0 to focus on the catch)
PLAYBACK_SPEED = 0.5    # 1.0 is real-time. 0.5 is half-speed (slow motion). 2.0 is double speed.
TRAIL_LENGTH = 15       # How many frames the motion trail should last
# ==========================================

if __name__ == "__main__":
    print("Loading Agent and simulating physics...")
    
    # 1. Load Parameters and Weights
    try:
        study = optuna.load_study(study_name="rbf_swingup", storage="sqlite:///rbf_tuning.db")
        bp = study.best_params
    except Exception as e:
        print("Could not load database. Run tune.py first!")
        sys.exit()
    
    env = gym.make('unbalanced-disk-sincos-v0')
    actions = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
    
    feature_extractor = RBFFeatureExtractor(bp['n_bins'], bp['sigma'])
    agent = RBFQLearningAgent(feature_extractor.num_features, actions, bp['alpha'], bp['gamma'], 0.0, 1.0)
    agent.weights = np.load('best_rbf_weights.npy')

    # 2. Run the Simulation to gather data
    state, _ = env.reset(seed=SEED)
    features = feature_extractor.get_features(state)
    done = False
    
    all_times, all_thetas, all_voltages = [], [], []
    dt = 0.025
    step = 0

    while not done:
        action_idx, action_val = agent.select_action(features, evaluate=True)
        next_state, _, terminated, truncated, _ = env.step(action_val)
        done = terminated or truncated
        
        all_times.append(step * dt)
        all_thetas.append(np.arctan2(state[0], state[1]))
        all_voltages.append(action_val)
        
        features = feature_extractor.get_features(next_state)
        state = next_state
        step += 1

    env.close()

    # 3. Crop Data based on User Settings
    all_times = np.array(all_times)
    all_thetas = np.array(all_thetas)
    all_voltages = np.array(all_voltages)

    mask = (all_times >= START_TIME) & (all_times <= END_TIME)
    times = all_times[mask]
    thetas = all_thetas[mask]
    voltages = all_voltages[mask]

    if len(times) == 0:
        print("Error: Cropped window contains no data. Adjust START_TIME and END_TIME.")
        sys.exit()

    # 4. Set up the Figure and Layout
    print("Rendering GIF... This may take a minute depending on duration.")
    fig, (ax_pend, ax_ctrl) = plt.subplots(2, 1, figsize=(8, 10), gridspec_kw={'height_ratios': [2, 1]})
    fig.suptitle('RBF Agent Swing-up Control Policy', fontsize=16)

    # --- Setup Top Panel (Physical Pendulum) ---
    ax_pend.set_xlim(-1.5, 1.5)
    ax_pend.set_ylim(-1.5, 1.5)
    ax_pend.set_aspect('equal') # Forces it to be a perfect square so the circle doesn't stretch
    ax_pend.grid(True, linestyle='--', alpha=0.5)
    ax_pend.axhline(0, color='black', linewidth=0.5)
    ax_pend.axvline(0, color='black', linewidth=0.5)
    
    # Draw reference target (top upright)
    ax_pend.plot([0], [1.0], marker='*', color='green', markersize=15, alpha=0.5, label="Target")
    ax_pend.plot([0], [0], marker='o', color='black', markersize=8) # Central pivot

    # Initialize drawing objects
    arm_line, = ax_pend.plot([], [], lw=4, color='#8B4513') # Brown arm
    mass_circle, = ax_pend.plot([], [], marker='o', markersize=20, color='blue') # Blue disc
    trail_line, = ax_pend.plot([], [], lw=2, color='blue', alpha=0.3) # Faded trail

    # --- Setup Bottom Panel (Control Signal) ---
    ax_ctrl.set_xlim(times[0], times[-1])
    ax_ctrl.set_ylim(-3.5, 3.5)
    ax_ctrl.set_ylabel('Voltage (V)')
    ax_ctrl.set_xlabel('Time (s)')
    ax_ctrl.grid(True)
    ax_ctrl.axhline(3.0, color='black', linestyle=':', label='Saturation')
    ax_ctrl.axhline(-3.0, color='black', linestyle=':')
    
    # Pre-draw the entire voltage history faintly in the background
    ax_ctrl.step(times, voltages, color='red', alpha=0.3, where='mid')
    
    # Initialize the dynamic objects
    voltage_line, = ax_ctrl.step([], [], color='red', lw=2, where='mid')
    playhead = ax_ctrl.axvline(times[0], color='blue', lw=2, linestyle='--')

    # 5. The Animation Engine Function
    def update(frame):
        # Current index in the cropped arrays
        idx = frame
        current_time = times[idx]
        current_theta = thetas[idx]
        
        # Calculate Kinematics (0 is down, PI is up)
        # x = sin(theta), y = -cos(theta)
        x = np.sin(current_theta)
        y = -np.cos(current_theta)
        
        # Update Pendulum Arm
        arm_line.set_data([0, x], [0, y])
        mass_circle.set_data([x], [y])
        
        # Update Trail
        start_trail = max(0, idx - TRAIL_LENGTH)
        trail_thetas = thetas[start_trail:idx+1]
        trail_x = np.sin(trail_thetas)
        trail_y = -np.cos(trail_thetas)
        trail_line.set_data(trail_x, trail_y)
        
        # Update Control Plot
        voltage_line.set_data(times[:idx+1], voltages[:idx+1])
        playhead.set_xdata([current_time])
        
        return arm_line, mass_circle, trail_line, voltage_line, playhead

    # 6. Build and Save
    # Calculate interval: dt (0.025s) * 1000 = 25ms per frame. Divide by playback speed.
    frame_delay = (dt * 1000) / PLAYBACK_SPEED 
    
    anim = animation.FuncAnimation(
        fig, update, frames=len(times), interval=frame_delay, blit=True
    )

    gif_filename = 'swingup_policy.gif'
    anim.save(gif_filename, writer='pillow')
    print(f"Success! Saved beautiful animation to: {gif_filename}")