import os
import sys
import time
import glob
import argparse
from pathlib import Path
import numpy as np

# --- PATH SETUP ---
CURRENT_DIR = Path(__file__).resolve().parent
GYM_DIR = CURRENT_DIR.parent.parent / "gym-unbalanced-disk-master"
sys.path.append(str(GYM_DIR))
sys.path.append(str(CURRENT_DIR))

import gymnasium as gym
import gym_unbalanced_disk
from agent import RBFFeatureExtractor, RBFQLearningAgent

import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="gymnasium")

# --- DEPLOYMENT CONSTANTS ---
ACTIONS = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
DT = 0.025
MAX_STEPS = 300 # 7.5 seconds
OMEGA_KILL_LIMIT = 60.0
GRACE_PERIOD_SEC = 2.0

def latest_sweep_batch():
    sweep_root = CURRENT_DIR / "top5_stab_sweep_results"
    if not sweep_root.exists():
        return None
    versions = sorted(
        [int(path.name[1:]) for path in sweep_root.glob("v*") if path.is_dir() and path.name[1:].isdigit()]
    )
    if not versions:
        return None
    return sweep_root / f"v{versions[-1]}"

def default_models_root():
    sweep_batch = latest_sweep_batch()
    if sweep_batch is not None:
        return sweep_batch
    return CURRENT_DIR / "top5_results"

def has_loadable_model(model_dir):
    return (model_dir / "config.json").exists() and (model_dir / "best_rbf_weights.npy").exists()

def get_safe_filename(trial_dir, partial=False):
    """Ensures zero-overwrites by finding the next sequential _vX file."""
    base_name = "hardware_eval_v"
    existing_files = glob.glob(str(trial_dir / f"{base_name}*"))
    
    # Extract version numbers safely
    versions = []
    for f in existing_files:
        try:
            v_str = f.split(base_name)[1].split('_')[0].split('.npy')[0]
            versions.append(int(v_str))
        except:
            pass
            
    next_v = max(versions) + 1 if versions else 1
    suffix = "_partial" if partial else ""
    return trial_dir / f"{base_name}{next_v}{suffix}.npy"

def high_res_countdown(seconds):
    """A live, 4-decimal precision countdown that overwrites the terminal line."""
    start = time.perf_counter()
    while True:
        elapsed = time.perf_counter() - start
        remain = seconds - elapsed
        if remain <= 0:
            sys.stdout.write(f"\r[!] CLEAR HANDS! Motor engages in: 0.0000    \n")
            sys.stdout.flush()
            break
        # \r sends the cursor to the start of the line to overwrite it
        sys.stdout.write(f"\r[!] CLEAR HANDS! Motor engages in: {remain:.4f}    ")
        sys.stdout.flush()
        time.sleep(0.01) # Sleep slightly to prevent maxing out the CPU

def settle_disk(env):
    """Forces 0.0V and waits until the disk is completely resting at bottom dead-center."""
    print("\n[~] Settling disk to Bottom Dead-Center (0.0V)...")
    consecutive_settled_steps = 0
    
    while True:
        obs, _, _, _, _ = env.step(0.0)
        theta, omega = obs[0], obs[1]

        print(f"DEBUG -> Theta: {theta:.3f} | Cos(Theta): {np.cos(theta):.3f} | Omega: {omega:.3f}   ", end="\r")
        
        # Check if resting at bottom: low speed AND cos(theta) is close to 1.0 (bottom)
        if abs(omega) < 6 and np.cos(theta) > 0.95:
            consecutive_settled_steps += 1
        else:
            consecutive_settled_steps = 0
            
        # If it has been resting for 1.5 seconds continuously, it is settled
        if consecutive_settled_steps > (1.5 / DT):
            break

def load_model(trial_dir):
    """Loads the agent and extracts its parameters from its config."""
    import json
    with open(trial_dir / 'config.json', 'r') as f:
        config = json.load(f)
        
    bp = config['params']
    feature_extractor = RBFFeatureExtractor(bp['n_bins'], bp['sigma'])
    agent = RBFQLearningAgent(feature_extractor.num_features, ACTIONS, bp['alpha'], bp['gamma'], 0.0, 1.0)
    
    agent.weights = np.load(trial_dir / 'best_rbf_weights.npy')
    return agent, feature_extractor

def run_episode(env, agent, feature_extractor, trial_dir, ep_num, pacing_mode):
    """Executes a single secure test flight."""
    settle_disk(env)
    
    if pacing_mode == "manual":
        input("\n[?] DISK SETTLED. Press [ENTER] to spin up the motor...")
    else:
        print("-" * 50)
        print("[!] DISK SETTLED AT DEAD-CENTER.")
        print("[!] AUTO-FLOW GRACE PERIOD ACTIVE...")
        high_res_countdown(GRACE_PERIOD_SEC)
        print("-" * 50)
        
    print(f"\n[ >>> DEPLOYING: {trial_dir.name} | EPISODE {ep_num} | Status: LIVE <<< ]")
    
    obs, _ = env.reset() # Clear any internal env buffers
    
    thetas, omegas, voltages, times = [], [], [], []
    partial_flag = False
    stop_reason = "Completed"
    
    try:
        for step in range(MAX_STEPS):
            # 1. KINEMATIC BRIDGE: Convert raw [theta, omega] to [sin, cos, omega]
            theta, omega = obs[0], obs[1]
            features = feature_extractor.get_features(np.array([np.sin(theta), np.cos(theta), omega]))
            
            # 2. HARDWARE KILL-SWITCH
            if abs(omega) > OMEGA_KILL_LIMIT or abs(theta) > 1.0e6 or not np.isfinite(omega):
                stop_reason = f"HARDWARE KILL-SWITCH TRIPPED! Omega: {omega:.2f}"
                partial_flag = True
                break
                
            # 3. ACTION & STEP
            _, action_val = agent.select_action(features, evaluate=True)
            obs, _, terminated, truncated, _ = env.step(action_val)
            
            times.append(step * DT)
            thetas.append(theta)
            omegas.append(omega)
            voltages.append(action_val)
            
            if terminated or truncated:
                break
                
    except KeyboardInterrupt:
        # 4. SOFTWARE KILL-SWITCH (Ctrl + C)
        stop_reason = "ABORTED BY USER (Ctrl + C)"
        partial_flag = True
        print(f"\n[!!!] EMERGENCY ABORT DETECTED. Killing motor power [!!!]")
        
    finally:
       # Guarantee motor power is cut immediately, ignoring Gym's strict API errors
        try:
            env.step(0.0) 
        except:
            pass
        
        # Safe Data Logging
        if len(times) > 0:
            save_path = get_safe_filename(trial_dir, partial=partial_flag)
            data = {'thetas': thetas, 'omegas': omegas, 'voltages': voltages, 'stop_reason': stop_reason}
            np.save(save_path, data)
            print(f"[v] Saved {'partial ' if partial_flag else ''}telemetry to: {save_path.name}")
        
        print(f"[ ] Episode Status: {stop_reason}")
    return partial_flag

def main():
    parser = argparse.ArgumentParser(description="Deploy saved RBF models on the real unbalanced disk setup.")
    parser.add_argument(
        "--models-root",
        type=Path,
        default=default_models_root(),
        help="Directory containing model folders with config.json and best_rbf_weights.npy.",
    )
    args = parser.parse_args()

    results_dir = args.models_root
    if not results_dir.exists():
        print(f"Error: Could not find models directory: {results_dir}")
        return
        
    available_models = sorted([d for d in results_dir.iterdir() if d.is_dir() and has_loadable_model(d)])
    if not available_models:
        print(f"Error: No loadable models found in {results_dir}")
        print("Each model folder needs config.json and best_rbf_weights.npy.")
        return
    
    print("\n" + "="*50)
    print("   PHYSICAL HARDWARE DEPLOYMENT SUITE   ")
    print("="*50)
    print(f"Model root: {results_dir}")
    print("Detected Models:")
    for i, m in enumerate(available_models):
        print(f"  [{i+1}] {m.name}")
        
    print("\n--- PLAYLIST BUILDER ---")
    print("[A] Test ONE specific model for N episodes")
    print("[B] Test ALL models (1 episode each)")
    print("[C] Custom Queue (Specify numbers e.g., '1,1,2,5')")
    
    choice = input("\nSelect an option (A/B/C): ").strip().upper()
    queue = []
    
    if choice == 'A':
        idx = int(input(f"Which model number? (1-{len(available_models)}): ")) - 1
        if idx < 0 or idx >= len(available_models):
            print("Invalid model number.")
            return
        n_eps = int(input("How many episodes?: "))
        queue = [available_models[idx]] * n_eps
    elif choice == 'B':
        queue = available_models
    elif choice == 'C':
        seq = input("Enter comma-separated model sequence (e.g. 1,1,3,5): ")
        indices = [int(x.strip()) - 1 for x in seq.split(',')]
        if any(i < 0 or i >= len(available_models) for i in indices):
            print("Invalid model number in sequence.")
            return
        queue = [available_models[i] for i in indices]
    else:
        print("Invalid choice.")
        return

    print("\n--- PACING ---")
    print("[1] Auto-Flow (2.0s Grace Period)")
    print("[2] Manual (Press Enter between episodes)")
    pacing_choice = input("Select pacing (1/2): ").strip()
    pacing_mode = "auto" if pacing_choice == '1' else "manual"

    confirmation = input("\n[!] PLAYLIST READY. Press Enter to connect to the hardware, or type anything to cancel... ")
    if confirmation.strip():
        print("Cancelled hardware run.")
        return
    
    print("\nConnecting to physical lab setup...")
    env = gym.make('unbalanced-disk-exp-v0', dt=DT, umax=3.0)
    env.reset()
    
    try:
        for ep_count, model_dir in enumerate(queue, 1):
            agent, feature_extractor = load_model(model_dir)
            aborted = run_episode(env, agent, feature_extractor, model_dir, ep_count, pacing_mode)
            
            # If user hits Ctrl+C, pause the queue and ask if they want to continue the playlist or exit completely
            if aborted and input("\nPlaylist paused due to abort. Continue to next item? (y/n): ").strip().lower() != 'y':
                break
                
    finally:
        try:
            env.step(0.0) # Absolute final safety shutdown
        except:
            pass
        env.close()
        print("\nHardware disconnected. Deployment suite closed.")

if __name__ == "__main__":
    main()
