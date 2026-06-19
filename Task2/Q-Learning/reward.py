import numpy as np

def calculate_custom_reward(state, action, w_position, w_balance, w_stab=None):
    if w_stab is None:
        w_stab = w_balance

    sin_th, cos_th, omega = state
    
    # --- Logarithmic Position Reward ---
    # Give 0 reward for the entire bottom half (cos_th >= 0)
    # Exponentially reward balancing in the top half (cos_th < 0)
    if cos_th >= 0:
        position_reward = 0.0
    else:
        # distance_to_top is 1.0 at horizontal, and 0.0 at perfectly vertical
        distance_to_top = (1.0 + cos_th) 
        # Peaks at 2.0 when perfectly balanced
        position_reward = -np.log10(distance_to_top + 0.01)
    
    if cos_th < -0.8: 
        balance_bonus = w_position * (1.0 - min(1.0, abs(omega) / 6.0))
        stab_penalty = w_position * (0.15 * (omega ** 2) + 0.5 * (action ** 2))
    else:
        balance_bonus = 0.0
        stab_penalty = 0.0

    return (w_position * position_reward) + (w_balance * balance_bonus) - (w_stab * stab_penalty)
