import numpy as np

class RBFFeatureExtractor:
    def __init__(self, n_bins_per_dim, sigma):
        self.sigma = sigma
        self.omega_max = 40.0
        
        grid_1d = np.linspace(-1.0, 1.0, n_bins_per_dim)
        s_centers, c_centers, w_centers = np.meshgrid(grid_1d, grid_1d, grid_1d)
        
        self.centers = np.vstack([s_centers.ravel(), c_centers.ravel(), w_centers.ravel()]).T
        self.num_features = len(self.centers)

    def normalize_state(self, state):
        sin_th, cos_th, omega = state
        omega_norm = np.clip(omega / self.omega_max, -1.0, 1.0)
        return np.array([sin_th, cos_th, omega_norm])

    def get_features(self, state):
        norm_state = self.normalize_state(state)
        sq_distances = np.sum((self.centers - norm_state) ** 2, axis=1)
        return np.exp(-sq_distances / (2 * self.sigma ** 2))

class RBFQLearningAgent:
    def __init__(self, num_features, actions, alpha, gamma, epsilon_start, epsilon_decay):
        self.actions = actions
        self.num_actions = len(actions)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = 0.01
        
        self.weights = np.random.uniform(-0.01, 0.01, (self.num_actions, num_features))

    def get_q_values(self, features):
        return np.dot(self.weights, features)

    def select_action(self, features, evaluate=False):
        if not evaluate and np.random.rand() < self.epsilon:
            action_idx = np.random.randint(self.num_actions)
        else:
            action_idx = np.argmax(self.get_q_values(features))
        return action_idx, self.actions[action_idx]

    def update(self, features, action_idx, reward, next_features, done):
        current_q = np.dot(self.weights[action_idx], features)
        
        if done:
            target_q = reward
        else:
            target_q = reward + self.gamma * np.max(self.get_q_values(next_features))
            
        td_error = target_q - current_q
        td_error = np.clip(td_error, -50.0, 50.0)
        self.weights[action_idx] += self.alpha * td_error * features

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)