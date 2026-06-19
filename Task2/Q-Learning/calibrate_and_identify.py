import os
import sys
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import argparse
from pathlib import Path

# --- PATH SETUP ---
CURRENT_DIR = Path(__file__).resolve().parent
GYM_DIR = CURRENT_DIR.parent.parent / "gym-unbalanced-disk-master"
sys.path.append(str(GYM_DIR))
sys.path.append(str(CURRENT_DIR))

import gymnasium as gym
import gym_unbalanced_disk

DT = 0.025

def settle_disk(env):
    print("\n[~] Settling disk to Bottom Dead-Center (0.0V)...")
    consecutive_settled_steps = 0
    # Add a max wait time so we don't get stuck forever
    max_wait = int(10.0 / DT) 
    
    for i in range(max_wait):
        obs, _, _, _, _ = env.step(0.0)
        theta, omega = obs[0], obs[1]
        
        # Check if resting at bottom: low speed AND cos(theta) is close to 1.0
        if abs(omega) < 6 and np.cos(theta) > 0.95:
            consecutive_settled_steps += 1
        else:
            consecutive_settled_steps = 0
            
        if consecutive_settled_steps > (2.0 / DT):
            print(f"[v] Disk settled after {i * DT:.2f} seconds.")
            return True
            
    print("[!] Warning: Disk did not perfectly settle. Proceeding anyway.")
    return False

def main():
    parser = argparse.ArgumentParser(description="Calibrate and identify system parameters.")
    parser.add_argument("--env", type=str, default="unbalanced-disk-exp-v0", help="Gym environment ID to use.")
    args = parser.parse_args()

    print(f"Connecting to environment '{args.env}' for calibration...")
    env = gym.make(args.env, dt=DT, umax=3.0)
    env.reset()
    
    data_dict = {}
    
    try:
        # --- STAGE 1: SETTLING ---
        settle_disk(env)
        
        # --- STAGE 2: STATIC CALIBRATION ---
        print("\n[2] Static Calibration (Theta Offset & Omega Bias/Variance)")
        static_thetas = []
        static_omegas = []
        for _ in range(int(5.0 / DT)): # 5 seconds
            obs, _, _, _, _ = env.step(0.0)
            static_thetas.append(obs[0])
            static_omegas.append(obs[1])
            
        theta_offset = np.mean(static_thetas)
        omega_bias = np.mean(static_omegas)
        omega_variance = np.var(static_omegas)
        omega_noise_floor = np.std(static_omegas) * 3 # 3 sigma
        
        data_dict['theta_offset'] = theta_offset
        data_dict['omega_bias'] = omega_bias
        data_dict['omega_variance'] = omega_variance
        
        print(f"    Theta Offset: {theta_offset:.4f} rad")
        print(f"    Omega Bias  : {omega_bias:.4f} rad/s")
        print(f"    Omega Var   : {omega_variance:.4f} (rad/s)^2")
        print(f"    Omega Noise : ±{omega_noise_floor:.4f} rad/s (3-sigma)")
        
        # --- STAGE 3: MOTOR DELAY ---
        print("\n[3] Motor Delay Measurement")
        settle_disk(env)
        delay_steps = 0
        step_voltages = []
        step_omegas = []
        step_thetas = []
        
        # Apply 3.0V for 1 second
        for i in range(int(1.0 / DT)):
            obs, _, _, _, _ = env.step(3.0)
            step_voltages.append(3.0)
            step_thetas.append(obs[0])
            step_omegas.append(obs[1])
            
            # Check if omega exceeded noise floor (adjusted by bias)
            if delay_steps == 0 and abs(obs[1] - omega_bias) > max(1.0, omega_noise_floor):
                delay_steps = i + 1 # 1-indexed (first step is 1)
                
        motor_delay_sec = delay_steps * DT
        data_dict['motor_delay_sec'] = motor_delay_sec
        data_dict['delay_steps'] = delay_steps
        print(f"    Motor Delay: {delay_steps} steps ({motor_delay_sec:.3f} s)")
        
        data_dict['step_response'] = {'thetas': step_thetas, 'omegas': step_omegas, 'voltages': step_voltages}
        
        # --- STAGE 4: STATIC FRICTION (STICTION) ---
        print("\n[4] Static Friction (Stiction) Measurement")
        settle_disk(env)
        stiction_voltage = None
        ramp_voltages = []
        ramp_omegas = []
        
        max_ramp_time = 10.0 # 10 seconds max
        steps = int(max_ramp_time / DT)
        for i in range(steps):
            # Ramp from 0.0V to 1.5V over 10 seconds (0.15V per second)
            current_v = (i / steps) * 1.5
            obs, _, _, _, _ = env.step(current_v)
            ramp_voltages.append(current_v)
            ramp_omegas.append(obs[1])
            
            if stiction_voltage is None and abs(obs[1] - omega_bias) > max(1.0, omega_noise_floor):
                stiction_voltage = current_v
                break # Stop ramping once it moves
                
        if stiction_voltage is None:
            stiction_voltage = 1.5 # fallback
            
        data_dict['stiction_voltage'] = stiction_voltage
        print(f"    Stiction Voltage: {stiction_voltage:.3f} V")
        
        # --- STAGE 5: DYNAMIC FRICTION (FREE-FALL) ---
        print("\n[5] Dynamic Friction (Free-fall Decay) Measurement")
        settle_disk(env)
        # Swing up
        print("    Swinging up...")
        for _ in range(int(1.5 / DT)):
            env.step(3.0)
            
        # Free fall
        print("    Free falling...")
        freefall_thetas = []
        freefall_omegas = []
        freefall_times = []
        for i in range(int(8.0 / DT)): # 8 seconds of freefall
            obs, _, _, _, _ = env.step(0.0)
            freefall_thetas.append(obs[0])
            freefall_omegas.append(obs[1])
            freefall_times.append(i * DT)
            
        data_dict['freefall'] = {'thetas': freefall_thetas, 'omegas': freefall_omegas, 'times': freefall_times}
        
        # --- STAGE 6: DYNAMIC VALIDATION ---
        print("\n[6] Dynamic Validation (Omega vs dTheta/dt)")
        thetas_arr = np.array(freefall_thetas)
        dtheta = thetas_arr[1:] - thetas_arr[:-1]
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        derived_omegas = np.zeros_like(thetas_arr)
        derived_omegas[1:] = dtheta / DT
        derived_omegas[0] = derived_omegas[1]
        
        data_dict['freefall']['derived_omegas'] = derived_omegas.tolist()
        
    except KeyboardInterrupt:
        print("\n[!] User aborted. Saving partial data.")
    finally:
        try:
            env.step(0.0)
        except:
            pass
        env.close()
        
    # --- SAVE DATA ---
    print("\nSaving data and plots...")
    np.save(CURRENT_DIR / "sys_id_data.npy", data_dict)
    
    fig, axs = plt.subplots(3, 1, figsize=(10, 12))
    
    # Plot 1: Step Response (Motor Delay)
    if 'step_response' in data_dict:
        t_step = np.arange(len(data_dict['step_response']['omegas'])) * DT
        axs[0].plot(t_step, data_dict['step_response']['omegas'], label='Omega')
        axs[0].plot(t_step, data_dict['step_response']['voltages'], label='Voltage', alpha=0.5)
        
        if 'motor_delay_sec' in data_dict:
            motor_delay_sec = data_dict['motor_delay_sec']
            axs[0].axvline(x=motor_delay_sec, color='r', linestyle='--', label=f'Delay ({motor_delay_sec:.3f}s)')
            
        axs[0].set_title("Motor Delay (Step Response)")
        axs[0].set_ylabel("Omega (rad/s) / V")
        axs[0].legend()
        axs[0].grid(True)
        
    # Plot 2: Freefall Decay
    if 'freefall' in data_dict:
        t_free = data_dict['freefall']['times']
        axs[1].plot(t_free, data_dict['freefall']['thetas'], label='Theta')
        axs[1].set_title("Free-fall Decay (Friction Identification)")
        axs[1].set_ylabel("Theta (rad)")
        axs[1].set_xlabel("Time (s)")
        axs[1].grid(True)
        
    # Plot 3: Dynamic Validation
    if 'freefall' in data_dict:
        axs[2].plot(t_free, data_dict['freefall']['omegas'], label='Raw Hardware Omega', alpha=0.6)
        axs[2].plot(t_free, data_dict['freefall']['derived_omegas'], label='Derived Omega (dTheta/dt)', alpha=0.6)
        axs[2].set_title("Dynamic Validation (Raw vs Derived Omega)")
        axs[2].set_ylabel("Omega (rad/s)")
        axs[2].set_xlabel("Time (s)")
        axs[2].legend()
        axs[2].grid(True)
        
    plt.tight_layout()
    plt.savefig(CURRENT_DIR / "system_identification_results.png")
    plt.close()
    
    print(f"\nSUCCESS! Results saved to {CURRENT_DIR / 'system_identification_results.png'}")

if __name__ == "__main__":
    main()
