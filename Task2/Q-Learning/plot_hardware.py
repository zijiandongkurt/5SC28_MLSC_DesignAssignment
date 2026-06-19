import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

CURRENT_DIR = Path(__file__).resolve().parent

HW_TESTS_DIR = CURRENT_DIR / "hardware_tests"
HW_NPY_DIR = HW_TESTS_DIR / "npy"
HW_PNG_DIR = HW_TESTS_DIR / "png"

HW_NPY_DIR.mkdir(parents=True, exist_ok=True)
HW_PNG_DIR.mkdir(parents=True, exist_ok=True)

print("Scanning for hardware telemetry...")
found_files = False

for npy_file in sorted(HW_NPY_DIR.glob("*.npy")):
    save_name = npy_file.stem + "_wrapped.png"
    save_path = HW_PNG_DIR / save_name
    
    # Skip if already plotted
    if save_path.exists():
        continue
        
    found_files = True
    print(f"Plotting Wrapped: {npy_file.name}")
        
        data = np.load(npy_file, allow_pickle=True).item()
        
        thetas = np.array(data['thetas'])
        omegas = np.array(data['omegas'])
        voltages = np.array(data['voltages'])
        stop_reason = data.get('stop_reason', 'Unknown')
        
        if len(thetas) == 0:
            continue
            
        dt = 0.025
        times = np.arange(len(thetas)) * dt
        
        # --- THE WRAPPER LOGIC ---
        # Wrap angles strictly to [-pi, pi]
        wrapped_thetas = (thetas + np.pi) % (2 * np.pi) - np.pi
        
        # Prevent Matplotlib from drawing ugly vertical lines when jumping from +pi to -pi
        diffs = np.abs(np.diff(wrapped_thetas))
        wrapped_thetas_plot = wrapped_thetas.copy()
        wrapped_thetas_plot[1:][diffs > np.pi] = np.nan
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 1. Plot Wrapped Theta on the Primary Y-Axis (Left)
        line1 = ax1.plot(times, wrapped_thetas_plot, label='Angle (Wrapped)', color='blue')
        ax1.set_ylabel('Angle (rad)', color='blue')
        ax1.tick_params(axis='y', labelcolor='blue')
        
        # Draw target lines exactly at +pi and -pi
        target_line = ax1.axhline(np.pi, color='green', linestyle='--', alpha=0.5, label='Target Upright')
        ax1.axhline(-np.pi, color='green', linestyle='--', alpha=0.5)
        bottom_line = ax1.axhline(0, color='gray', linestyle=':', alpha=0.3, label='Bottom')
        
        # Set Y-limits slightly outside [-pi, pi] for perfect framing
        ax1.set_ylim(-np.pi - 0.5, np.pi + 0.5)
        ax1.set_yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
        ax1.set_yticklabels(['-π', '-π/2', '0', 'π/2', 'π'])
        
        # 2. Plot Omega on a Secondary Y-Axis (Right)
        ax1_twin = ax1.twinx()
        line2 = ax1_twin.plot(times, omegas, label='Angular Velocity', color='orange', alpha=0.7)
        ax1_twin.set_ylabel('Velocity (rad/s)', color='orange')
        ax1_twin.tick_params(axis='y', labelcolor='orange')
        
        # Combine legends for the top graph
        lines = line1 + line2 + [target_line, bottom_line]
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        
        ax1.set_title(f'Wrapped Physical Trajectory: {npy_file.stem}\nStatus: {stop_reason}')
        ax1.grid(True, alpha=0.3)
        
        # 3. Plot Voltages
        ax2.step(times, voltages, label='Input Voltage (V)', color='red', where='post')
        ax2.axhline(3.0, color='black', linestyle=':')
        ax2.axhline(-3.0, color='black', linestyle=':')
        ax2.set_xlabel('Time (s)')
        ax2.set_ylabel('Voltage (V)')
        ax2.set_ylim(-3.5, 3.5)
        ax2.legend(loc='upper left')
        ax2.grid()
        
        plt.tight_layout()
        
        # Save the plot
        save_name = npy_file.stem + "_wrapped.png"
        plt.savefig(HW_PNG_DIR / save_name)
        plt.close()

if not found_files:
    print("No 'hardware_eval_*.npy' files found! Did you run the hardware test yet?")
else:
    print("\n[v] All wrapped trajectories plotted successfully! Check your model folders.")