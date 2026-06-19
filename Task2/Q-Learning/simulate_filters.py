import os
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR
OUTPUT_DIR = CURRENT_DIR / "filter_comparisons"
DT = 0.025

def adaptive_bias_filter(raw_omega, derived_omega, alpha=0.05):
    """
    Simple low-pass filter on the bias.
    """
    filtered_omega = np.zeros_like(raw_omega)
    bias_est = 0.0
    for i in range(len(raw_omega)):
        instant_bias = raw_omega[i] - derived_omega[i]
        bias_est = (1 - alpha) * bias_est + alpha * instant_bias
        filtered_omega[i] = raw_omega[i] - bias_est
    return filtered_omega

def kalman_filter(thetas, raw_omegas, dt=0.025):
    """
    1D Kinematic Kalman Filter that estimates theta, omega, and sensor bias.
    """
    x = np.array([thetas[0], 0.0, 0.0]) # [theta, omega, bias]
    P = np.eye(3) * 1.0
    
    F = np.array([
        [1.0, dt,  0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ])
    
    H = np.array([
        [1.0, 0.0, 0.0], # We observe theta
        [0.0, 1.0, 1.0]  # We observe raw_omega = true_omega + bias
    ])
    
    # Tuning matrices
    Q = np.array([
        [1e-4, 0.0, 0.0],
        [0.0, 1e-1, 0.0],  # Omega can change rapidly
        [0.0, 0.0, 1e-5]   # Bias drifts very slowly
    ])
    
    R = np.array([
        [1e-3, 0.0],
        [0.0, 5.0]         # High variance/distrust in raw_omega
    ])
    
    filtered_omega = np.zeros_like(raw_omegas)
    
    for i in range(len(thetas)):
        # Predict
        x = F @ x
        P = F @ P @ F.T + Q
        
        # Update
        z = np.array([thetas[i], raw_omegas[i]])
        y = z - (H @ x)
        
        # Handle theta wraparound in the innovation
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        
        x = x + K @ y
        P = (np.eye(3) - K @ H) @ P
        
        filtered_omega[i] = x[1]
        
    return filtered_omega

def theta_only_kalman_filter(thetas, dt=0.025):
    x = np.array([thetas[0], 0.0])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-4, 0.0], [0.0, 1e-1]])
    R = np.array([[1e-3]])
    
    filtered_omega = np.zeros_like(thetas)
    for i in range(len(thetas)):
        x = F @ x
        P = F @ P @ F.T + Q
        z = np.array([thetas[i]])
        y = z - (H @ x)
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(2) - K @ H) @ P
        filtered_omega[i] = x[1]
    return filtered_omega

def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    eval_files = list(RESULTS_DIR.rglob("hardware_eval_*.npy"))
    
    if not eval_files:
        print("No evaluation files found!")
        sys.exit(0)
        
    print(f"Found {len(eval_files)} files. Simulating filters...")
    
    for idx, file_path in enumerate(eval_files, 1):
        data = np.load(file_path, allow_pickle=True).item()
        
        if 'thetas' not in data or 'omegas' not in data:
            continue
            
        thetas = np.array(data['thetas'])
        raw_omegas = np.array(data['omegas'])
        
        if len(thetas) < 2: continue
        
        # 1. Derived Omega
        dtheta = thetas[1:] - thetas[:-1]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        derived_omegas = np.zeros_like(thetas)
        derived_omegas[1:] = dtheta / DT
        derived_omegas[0] = derived_omegas[1]
        
        # 2. Adaptive Bias Filter
        adaptive_omegas = adaptive_bias_filter(raw_omegas, derived_omegas, alpha=0.05)
        
        # 3. Kalman Filter
        kf_omegas = kalman_filter(thetas, raw_omegas, dt=DT)
        
        # 4. Theta-Only Kalman Filter
        theta_only_omegas = theta_only_kalman_filter(thetas, dt=DT)
        
        # Plotting
        t = np.arange(len(thetas)) * DT
        fig, axes = plt.subplots(4, 1, figsize=(14, 16), sharex=True)
        
        # Plot 1: Position
        axes[0].plot(t, thetas, color='black', linewidth=2)
        axes[0].set_ylabel("Theta (rad)")
        axes[0].set_title(f"Position (Theta) - {file_path.parent.name}")
        axes[0].grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Adaptive Bias Filter
        axes[1].plot(t, raw_omegas, color='red', alpha=0.3, label='Raw Hardware Omega')
        axes[1].plot(t, derived_omegas, color='blue', alpha=0.4, label='Derived Omega (Ground Truth Approx)')
        axes[1].plot(t, adaptive_omegas, color='green', linewidth=2, label='Adaptive Bias Filtered')
        axes[1].set_ylabel("Omega (rad/s)")
        axes[1].set_title("Option 1: Adaptive Bias Filter")
        axes[1].legend()
        axes[1].grid(True, linestyle='--', alpha=0.6)
        
        # Plot 3: Kalman Filter
        axes[2].plot(t, raw_omegas, color='red', alpha=0.3, label='Raw Hardware Omega')
        axes[2].plot(t, derived_omegas, color='blue', alpha=0.4, label='Derived Omega (Ground Truth Approx)')
        axes[2].plot(t, kf_omegas, color='purple', linewidth=2, label='Kalman Filtered')
        axes[2].set_ylabel("Omega (rad/s)")
        axes[2].set_title("Option 2: Regular Kalman Filter")
        axes[2].legend()
        axes[2].grid(True, linestyle='--', alpha=0.6)
        
        # Plot 4: Theta-Only Kalman Filter
        axes[3].plot(t, raw_omegas, color='red', alpha=0.3, label='Raw Hardware Omega')
        axes[3].plot(t, derived_omegas, color='blue', alpha=0.4, label='Derived Omega (Ground Truth Approx)')
        axes[3].plot(t, theta_only_omegas, color='orange', linewidth=2, label='Theta-Only Kalman Filtered')
        axes[3].set_xlabel("Time (s)")
        axes[3].set_ylabel("Omega (rad/s)")
        axes[3].set_title("Option 3: Theta-Only Kalman Filter")
        axes[3].legend()
        axes[3].grid(True, linestyle='--', alpha=0.6)
        
        plt.tight_layout()
        save_path = OUTPUT_DIR / f"{file_path.parent.name}_{file_path.stem}.png"
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        print(f"[{idx}/{len(eval_files)}] Saved filter comparison.")

    print(f"\nSUCCESS! All filter comparisons saved to: {OUTPUT_DIR.relative_to(CURRENT_DIR)}")

if __name__ == "__main__":
    main()
