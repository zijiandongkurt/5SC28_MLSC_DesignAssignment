import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import sys
import numpy as np
from pathlib import Path

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR
DT = 0.025

def adaptive_bias_filter(raw_omega, derived_omega, alpha=0.05):
    filtered_omega = np.zeros_like(raw_omega)
    bias_est = 0.0
    for i in range(len(raw_omega)):
        instant_bias = raw_omega[i] - derived_omega[i]
        bias_est = (1 - alpha) * bias_est + alpha * instant_bias
        filtered_omega[i] = raw_omega[i] - bias_est
    return filtered_omega

def kalman_filter(thetas, raw_omegas, dt=0.025):
    x = np.array([thetas[0], 0.0, 0.0])
    P = np.eye(3) * 1.0
    F = np.array([[1.0, dt, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    H = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0]])
    Q = np.array([[1e-4, 0.0, 0.0], [0.0, 1e-1, 0.0], [0.0, 0.0, 1e-5]])
    R = np.array([[1e-3, 0.0], [0.0, 5.0]])
    
    filtered_omega = np.zeros_like(raw_omegas)
    for i in range(len(thetas)):
        x = F @ x
        P = F @ P @ F.T + Q
        z = np.array([thetas[i], raw_omegas[i]])
        y = z - (H @ x)
        y[0] = (y[0] + np.pi) % (2 * np.pi) - np.pi
        S = H @ P @ H.T + R
        K = P @ H.T @ np.linalg.inv(S)
        x = x + K @ y
        P = (np.eye(3) - K @ H) @ P
        filtered_omega[i] = x[1]
    return filtered_omega

def theta_only_kalman_filter(thetas, dt=0.025):
    # Ignore raw omega completely. Only use theta.
    x = np.array([thetas[0], 0.0])
    P = np.eye(2) * 1.0
    F = np.array([[1.0, dt], [0.0, 1.0]])
    H = np.array([[1.0, 0.0]])
    Q = np.array([[1e-4, 0.0], [0.0, 1e-1]]) # Process noise
    R = np.array([[1e-3]]) # Measurement noise on theta
    
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
    eval_files = list(RESULTS_DIR.rglob("hardware_eval_*.npy"))
    if not eval_files: return
    
    metrics = {'adaptive': {'integral_err': [], 'smoothness': []}, 
               'kalman': {'integral_err': [], 'smoothness': []},
               'theta_only': {'integral_err': [], 'smoothness': []},
               'raw': {'integral_err': [], 'smoothness': []},
               'derived': {'integral_err': [], 'smoothness': []}}
               
    for file_path in eval_files:
        data = np.load(file_path, allow_pickle=True).item()
        if 'thetas' not in data or 'omegas' not in data: continue
        thetas = np.array(data['thetas'])
        raw_omegas = np.array(data['omegas'])
        if len(thetas) < 2: continue
        
        dtheta = thetas[1:] - thetas[:-1]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        derived_omegas = np.zeros_like(thetas)
        derived_omegas[1:] = dtheta / DT
        derived_omegas[0] = derived_omegas[1]
        
        adaptive_omegas = adaptive_bias_filter(raw_omegas, derived_omegas)
        kf_omegas = kalman_filter(thetas, raw_omegas, dt=DT)
        theta_only_omegas = theta_only_kalman_filter(thetas, dt=DT)
        
        # 1. Integral Consistency (How well does velocity integral match position diff?)
        # For each filter, integral is cumsum(omega * dt)
        # We compare to actual theta evolution (unwrapped)
        unwrapped_thetas = np.unwrap(thetas)
        true_diff = unwrapped_thetas - unwrapped_thetas[0]
        
        for name, sig in [('raw', raw_omegas), ('derived', derived_omegas), ('adaptive', adaptive_omegas), ('kalman', kf_omegas), ('theta_only', theta_only_omegas)]:
            integral = np.cumsum(sig * DT)
            # Center it roughly or compare relative changes
            mae = np.mean(np.abs(integral - true_diff))
            metrics[name]['integral_err'].append(mae)
            
            # 2. Smoothness (Variance of velocity derivative)
            # A physical pendulum cannot accelerate infinitely fast.
            # Lower variance of second derivative means smoother.
            accel = np.diff(sig) / DT
            smoothness = np.var(accel)
            metrics[name]['smoothness'].append(smoothness)
            
    print("\n" + "="*50)
    print("QUANTITATIVE FILTER BENCHMARK (Average across 19 runs)")
    print("="*50)
    
    def print_stats(name, key):
        print(f"[{name.upper()}]")
        print(f"  - Position Drift (Integral Error): {np.mean(metrics[name]['integral_err']):.4f} rad")
        print(f"  - Acceleration Noise (Smoothness): {np.mean(metrics[name]['smoothness']):.1f} rad/s^2 variance")
        print()
        
    print_stats('raw', 'raw')
    print_stats('derived', 'derived')
    print_stats('adaptive', 'adaptive')
    print_stats('kalman', 'kalman')
    print_stats('theta_only', 'theta_only')

if __name__ == "__main__":
    main()
