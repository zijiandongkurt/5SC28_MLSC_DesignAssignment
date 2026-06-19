import numpy as np
import gymnasium as gym

class AdaptiveBiasWrapper(gym.ObservationWrapper):
    """
    A Gym wrapper that removes wandering sensor bias from the omega (velocity) sensor.
    It works by tracking the long-term difference between the raw omega sensor and
    the mathematically derived velocity (dTheta / dt) and actively subtracting the error.
    """
    def __init__(self, env, alpha=0.05, dt=0.025):
        super().__init__(env)
        self.alpha = alpha
        self.dt = dt
        self.bias_est = 0.0
        self.prev_theta = None

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.bias_est = 0.0
        self.prev_theta = np.arctan2(obs[0], obs[1])
        
        # On first step, assume derived omega is 0.0
        inst_bias = obs[2] - 0.0
        self.bias_est = (1 - self.alpha) * self.bias_est + self.alpha * inst_bias
        
        obs_new = obs.copy()
        obs_new[2] = obs[2] - self.bias_est
        return obs_new, info

    def observation(self, obs):
        theta = np.arctan2(obs[0], obs[1])
        
        if self.prev_theta is None:
            self.prev_theta = theta
            return obs
            
        dtheta = theta - self.prev_theta
        # Handle wrap-around
        dtheta = (dtheta + np.pi) % (2 * np.pi) - np.pi
        derived_omega = dtheta / self.dt
        
        raw_omega = obs[2]
        
        # Calculate instantaneous bias error
        inst_bias = raw_omega - derived_omega
        
        # Smooth the bias estimate
        self.bias_est = (1 - self.alpha) * self.bias_est + self.alpha * inst_bias
        
        self.prev_theta = theta
        
        # Output cleanly filtered omega
        obs_new = obs.copy()
        obs_new[2] = raw_omega - self.bias_est
        return obs_new

class StateRandomizationWrapper(gym.Wrapper):
    """
    A Gym wrapper that intercepts the environment reset and occasionally
    forces the disk into a high-speed spin (death spiral). This ensures
    the agent explores OOD states and learns how to brake effectively.
    """
    def __init__(self, env, death_spiral_prob=0.2):
        super().__init__(env)
        self.death_spiral_prob = death_spiral_prob

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        
        # Determine if we should start this episode in a death spiral
        if self.env.unwrapped.np_random.random() < self.death_spiral_prob:
            # Set a random angle
            self.env.unwrapped.th = self.env.unwrapped.np_random.uniform(-np.pi, np.pi)
            
            # Set a high velocity between 15.0 and 30.0 rad/s
            sign = 1 if self.env.unwrapped.np_random.random() < 0.5 else -1
            self.env.unwrapped.omega = sign * self.env.unwrapped.np_random.uniform(15.0, 30.0)
            
            # Update the observation array with the injected state
            obs = self.env.unwrapped.get_obs()
            
        return obs, info
