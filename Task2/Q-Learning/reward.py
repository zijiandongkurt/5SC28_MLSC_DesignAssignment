def calculate_custom_reward(state, action, w_energy, w_position, w_balance, w_stab=None):
    if w_stab is None:
        w_stab = w_balance

    sin_th, cos_th, omega = state
    
    position_reward = -cos_th 
    
    E_kin = 0.5 * (omega ** 2)
    E_pot = position_reward * 100.0 
    E_current = E_kin + E_pot
    
    E_target = 100.0 
    energy_penalty = -abs(E_current - E_target) / 100.0 
    
    # --- Strict Speed Limit for "Unnecessary Kinetic Energy" ---
    # We allow a 10% margin above E_target (110.0) before aggressively penalizing
    E_excess = max(0.0, E_current - 110.0)
    excess_penalty = -(E_excess ** 2) / 1000.0
    energy_penalty += excess_penalty 
    
    if cos_th < -0.8: 
        balance_bonus = w_position * (1.0 - min(1.0, abs(omega) / 6.0))
        stab_penalty = w_position * (0.15 * (omega ** 2) + 0.5 * (action ** 2))
    else:
        balance_bonus = 0.0
        stab_penalty = 0.0

    return (w_position * position_reward) + (w_energy * energy_penalty) + (w_balance * balance_bonus) - (w_stab * stab_penalty)
