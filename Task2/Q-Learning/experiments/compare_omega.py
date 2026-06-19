import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
import os
import sys
import numpy as np
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CURRENT_DIR
OUTPUT_DIR = CURRENT_DIR / "omega_comparisons"
DT = 0.025

def main():
    if not RESULTS_DIR.exists():
        print(f"Error: {RESULTS_DIR} does not exist.")
        sys.exit(1)
        
    OUTPUT_DIR.mkdir(exist_ok=True)
    
    # Find all hardware_eval files
    eval_files = list(RESULTS_DIR.rglob("hardware_eval_*.npy"))
    
    if not eval_files:
        print(f"No hardware_eval_*.npy files found in {RESULTS_DIR}")
        sys.exit(0)
        
    print(f"Found {len(eval_files)} evaluation files. Processing...")
    
    for idx, file_path in enumerate(eval_files, 1):
        try:
            data = np.load(file_path, allow_pickle=True).item()
        except Exception as e:
            print(f"Skipping {file_path.name} (Load error: {e})")
            continue
            
        if 'thetas' not in data or 'omegas' not in data:
            print(f"Skipping {file_path.name} (Missing keys)")
            continue
            
        thetas = np.array(data['thetas'])
        raw_omegas = np.array(data['omegas'])
        
        if len(thetas) < 2:
            continue
            
        # Derive omega mathematically
        derived_omegas = np.zeros_like(thetas)
        
        # Calculate finite difference
        dtheta = thetas[1:] - thetas[:-1]
        
        # Handle wrap-around (-pi to pi)
        # Normalize to [-pi, pi]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        
        # Calculate velocity
        derived_omegas[1:] = dtheta / DT
        # First point can't be derived backwards, so assume it's the same as the second
        derived_omegas[0] = derived_omegas[1]
        
        # Time axis
        t = np.arange(len(thetas)) * DT
        
        # Generate plot
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
        
        # Plot 1: Theta
        ax1.plot(t, thetas, color='green', linewidth=2)
        ax1.set_ylabel("Theta (rad)")
        ax1.set_title(f"Position (Theta) - {file_path.parent.name} / {file_path.name}")
        ax1.grid(True, linestyle='--', alpha=0.7)
        
        # Plot 2: Omegas
        ax2.plot(t, raw_omegas, color='red', alpha=0.6, linewidth=1.5, label='Raw Hardware Omega')
        ax2.plot(t, derived_omegas, color='blue', alpha=0.8, linewidth=2, label='Derived Omega (dTheta/dt)')
        ax2.set_xlabel("Time (s)")
        ax2.set_ylabel("Omega (rad/s)")
        ax2.set_title("Raw vs Derived Velocity")
        ax2.legend()
        ax2.grid(True, linestyle='--', alpha=0.7)
        
        plt.tight_layout()
        
        # Make a safe filename incorporating parent folder names
        parent_name = file_path.parent.name
        safe_name = f"{parent_name}_{file_path.stem}.png"
        save_path = OUTPUT_DIR / safe_name
        
        plt.savefig(save_path, dpi=200)
        plt.close()
        
        print(f"[{idx}/{len(eval_files)}] Saved {safe_name}")

    print(f"\n{'-'*50}")
    print(f"SUCCESS! All comparisons saved to: {OUTPUT_DIR.relative_to(CURRENT_DIR)}")
    print(f"{'-'*50}")

if __name__ == "__main__":
    main()
