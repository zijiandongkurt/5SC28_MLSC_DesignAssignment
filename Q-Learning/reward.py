def calculate_custom_reward(state, action, w_energy, w_position):
    sin_th, cos_th, omega = state
    
    position_reward = -cos_th 
    
    E_kin = 0.5 * (omega ** 2)
    E_pot = position_reward * 100.0 
    E_current = E_kin + E_pot
    
    E_target = 100.0 
    energy_penalty = -abs(E_current - E_target) / 100.0 
    
    if cos_th < -0.8: 
        stab_penalty = w_position * (0.1 * (omega ** 2) + 0.5 * (action ** 2))
    else:
        stab_penalty = 0.0

    return (w_position * position_reward) + (w_energy * energy_penalty) - stab_penalty