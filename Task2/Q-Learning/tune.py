import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GYM_DIR = os.path.abspath(os.path.join(BASE_DIR, '..', '..', 'gym-unbalanced-disk-master'))
sys.path.append(GYM_DIR)

# pyrefly: ignore [missing-import]
import gymnasium as gym
# pyrefly: ignore [missing-import]
gym.logger.min_level = 40
# pyrefly: ignore [missing-import]
import gym_unbalanced_disk
# pyrefly: ignore [missing-import]
import optuna
import time
from datetime import datetime, timedelta
# pyrefly: ignore [missing-import]
import numpy as np

from agent import RBFFeatureExtractor, RBFQLearningAgent
from wrappers import AdaptiveBiasWrapper
from reward import calculate_custom_reward

START_TIME = None
ACTIONS = [-3.0, -2.0, -1.0, -0.5, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 3.0]
SEED = 42
TRAIN_EPISODES = 400
TRAIN_DOMAIN_RANDOMIZATION = True
BASE_ENV_PARAMS = {
    "omega0": 11.339846957335382,
    "gamma": 1.3328339309394384,
    "Ku": 28.136158407237073,
    "Fc": 6.062729509386865,
    "delta_th": 0.0,
}
ROBUST_EVAL_VARIANTS = [
    {"name": "nominal", "omega0_scale": 1.00, "gamma_scale": 1.00, "ku_scale": 1.00, "fc_scale": 1.00, "delta_th": 0.000, "action_lag": 0.00},
    {"name": "high_friction", "omega0_scale": 1.00, "gamma_scale": 1.15, "ku_scale": 0.95, "fc_scale": 1.15, "delta_th": 0.000, "action_lag": 0.10},
    {"name": "low_friction", "omega0_scale": 1.00, "gamma_scale": 0.85, "ku_scale": 1.05, "fc_scale": 0.85, "delta_th": 0.000, "action_lag": 0.05},
    {"name": "weak_motor", "omega0_scale": 1.03, "gamma_scale": 1.05, "ku_scale": 0.90, "fc_scale": 1.00, "delta_th": 0.010, "action_lag": 0.10},
    {"name": "strong_motor", "omega0_scale": 0.97, "gamma_scale": 0.95, "ku_scale": 1.10, "fc_scale": 1.00, "delta_th": -0.010, "action_lag": 0.05},
]


def make_env():
    return gym.make('unbalanced-disk-sincos-v0')


def apply_env_variant(env, variant):
    base = env.unwrapped
    base.omega0 = BASE_ENV_PARAMS["omega0"] * variant.get("omega0_scale", 1.0)
    base.gamma = BASE_ENV_PARAMS["gamma"] * variant.get("gamma_scale", 1.0)
    base.Ku = BASE_ENV_PARAMS["Ku"] * variant.get("ku_scale", 1.0)
    base.Fc = BASE_ENV_PARAMS["Fc"] * variant.get("fc_scale", 1.0)
    base.delta_th = BASE_ENV_PARAMS["delta_th"] + variant.get("delta_th", 0.0)


def randomize_env_for_training(env, rng):
    if not TRAIN_DOMAIN_RANDOMIZATION:
        return
    apply_env_variant(
        env,
        {
            "omega0_scale": rng.uniform(0.97, 1.03),
            "gamma_scale": rng.uniform(0.85, 1.15),
            "ku_scale": rng.uniform(0.90, 1.10),
            "fc_scale": rng.uniform(0.85, 1.15),
            "delta_th": rng.uniform(-0.01, 0.01),
        },
    )


def theta_from_sincos(state):
    return float(np.arctan2(state[0], state[1]))


def top_error(theta):
    return float(abs(np.arctan2(np.sin(theta - np.pi), np.cos(theta - np.pi))))


def score_eval_episode(trace, completed):
    errors = np.asarray(trace["top_errors"], dtype=float)
    omegas = np.abs(np.asarray(trace["omegas"], dtype=float))
    voltages = np.asarray(trace["voltages"], dtype=float)
    if len(errors) == 0:
        return -1000.0

    near_top = errors < 0.35
    balancing = near_top & (omegas < 4.0)
    catching = near_top & (omegas < 8.0)
    voltage_tv = float(np.sum(np.abs(np.diff(voltages)))) if len(voltages) > 1 else 0.0
    peak_omega = float(np.max(omegas))
    mean_top_error = float(np.mean(np.sort(errors)[: min(len(errors), 40)]))

    longest_balance_streak = 0
    current_streak = 0
    for is_balancing in balancing:
        if is_balancing:
            current_streak += 1
            longest_balance_streak = max(longest_balance_streak, current_streak)
        else:
            current_streak = 0

    position_score = float(np.sum(np.cos(errors)))
    catch_bonus = 2.0 * float(np.sum(catching))
    balance_bonus = 6.0 * float(np.sum(balancing))
    streak_bonus = 4.0 * float(longest_balance_streak)
    completion_bonus = 25.0 if completed else 0.0
    speed_penalty = 0.015 * float(np.sum(np.square(omegas)))
    unsafe_penalty = 75.0 if peak_omega > 38.0 else 0.0
    chatter_penalty = 0.75 * voltage_tv
    miss_penalty = 20.0 * mean_top_error

    return position_score + catch_bonus + balance_bonus + streak_bonus + completion_bonus - speed_penalty - unsafe_penalty - chatter_penalty - miss_penalty


def evaluate_policy(agent, feature_extractor, eval_episodes):
    scores = []
    for variant_index, variant in enumerate(ROBUST_EVAL_VARIANTS):
        env = make_env()
        env = AdaptiveBiasWrapper(env, alpha=0.05, dt=0.025)
        apply_env_variant(env, variant)
        try:
            for episode in range(eval_episodes):
                state, _ = env.reset(seed=SEED + 1000 * variant_index + episode)
                features = feature_extractor.get_features(state)
                done = False
                lagged_action = 0.0
                trace = {"top_errors": [], "omegas": [], "voltages": []}

                while not done:
                    _, requested_action = agent.select_action(features, evaluate=True)
                    lag = variant.get("action_lag", 0.0)
                    applied_action = (1.0 - lag) * requested_action + lag * lagged_action
                    next_state, _, terminated, truncated, _ = env.step(applied_action)
                    done = terminated or truncated

                    theta = theta_from_sincos(next_state)
                    omega = float(next_state[2])
                    trace["top_errors"].append(top_error(theta))
                    trace["omegas"].append(omega)
                    trace["voltages"].append(float(applied_action))

                    features = feature_extractor.get_features(next_state)
                    state = next_state
                    lagged_action = applied_action

                scores.append(score_eval_episode(trace, completed=True))
        finally:
            env.close()

    scores = np.asarray(scores, dtype=float)
    return float(np.mean(scores) - 0.5 * np.std(scores))

def objective(trial):
    # 1. Suggest Hyperparameters
    n_bins = 10
    sigma = trial.suggest_float('sigma', 0.20, 0.31)
    alpha = trial.suggest_float('alpha', 0.010, 0.040, log=True)
    gamma = trial.suggest_float('gamma', 0.95, 0.999)
    eps_decay = trial.suggest_float('epsilon_decay', 0.98, 0.999)
    w_position = trial.suggest_float('w_position', 1.0, 5.0)
    w_balance = trial.suggest_float('w_balance', 0.0, 4.0)
    w_stab = trial.suggest_float('w_stab', 0.0, 3.0)

    # 2. Setup Environment and Agent
    env = make_env()
    env = AdaptiveBiasWrapper(env, alpha=0.05, dt=0.025)
    rng = np.random.default_rng(SEED + trial.number)
    feature_extractor = RBFFeatureExtractor(n_bins, sigma)
    agent = RBFQLearningAgent(feature_extractor.num_features, ACTIONS, alpha, gamma, 1.0, eps_decay)

    # 3. Training Loop
    for episode in range(TRAIN_EPISODES):
        randomize_env_for_training(env, rng)
        state, _ = env.reset(seed=SEED + episode)
        
        features = feature_extractor.get_features(state)
        
        done = False
        episode_reward = 0

        while not done:
            action_idx, action_val = agent.select_action(features, evaluate=False)
            next_state, _, terminated, truncated, _ = env.step(action_val)
            done = terminated or truncated
            
            reward = calculate_custom_reward(next_state, action_val, w_position, w_balance, w_stab)
            
            next_features = feature_extractor.get_features(next_state)
            
            agent.update(features, action_idx, reward, next_features, done)
            features = next_features
            episode_reward += reward

        agent.decay_epsilon()

        # 4. Pruning Check (Delayed to episode 200)
        if episode == 200:
            trial.report(episode_reward, step=200)
            if trial.should_prune():
                env.close()
                # Clean exit for Windows multiprocessing
                raise optuna.TrialPruned()

    env.close()

    # 5. Robust evaluation: reward catching and balancing upright, not just crossing it.
    return evaluate_policy(agent, feature_extractor, eval_episodes=2)


import multiprocessing

def run_worker(n_trials):
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    journal_path = os.path.join(BASE_DIR, 'rbf_tuning_journal.log')
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path)
    )
    study = optuna.load_study(
        study_name="rbf_log_reward_robust",
        storage=storage,
        pruner=optuna.pruners.MedianPruner()
    )
    study.optimize(objective, n_trials=n_trials, n_jobs=1)

if __name__ == "__main__":
    print("Starting Multi-Process Optimization...")
    START_TIME = time.time()
    
    journal_path = os.path.join(BASE_DIR, 'rbf_tuning_journal.log')
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend(journal_path)
    )
    
    study = optuna.create_study(
        direction="maximize", 
        study_name="rbf_log_reward_robust",
        storage=storage,
        load_if_exists=True,
        pruner=optuna.pruners.MedianPruner()
    )

    finished_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
    if finished_trials:
        finished_trials.sort(key=lambda t: t.value, reverse=True)
        print("\n--- Current Top 5 Trials ---")
        for i, t in enumerate(finished_trials[:5]):
            print(f" {i+1}. Trial {t.number:03d} | score: {t.value:.4f}")
        print("----------------------------\n")

    TOTAL_TARGET_TRIALS = 250
    completed_trials = len(study.trials)
    trials_to_run = TOTAL_TARGET_TRIALS - completed_trials
    
    if trials_to_run > 0:
        import tqdm
        N_WORKERS = 8
        trials_per_worker = (trials_to_run // N_WORKERS) + 1
        processes = []
        for i in range(N_WORKERS):
            p = multiprocessing.Process(target=run_worker, args=(trials_per_worker,))
            p.start()
            processes.append(p)
            
        # Create a unified progress bar in the main process
        with tqdm.tqdm(total=TOTAL_TARGET_TRIALS, initial=completed_trials, desc="Optimizing") as pbar:
            processed_trial_numbers = set(t.number for t in study.trials if t.state.name in ['COMPLETE', 'PRUNED', 'FAIL'])
            
            while any(p.is_alive() for p in processes):
                current_finished_trials = [t for t in study.trials if t.state.name in ['COMPLETE', 'PRUNED', 'FAIL']]
                new_trials = [t for t in current_finished_trials if t.number not in processed_trial_numbers]
                
                if new_trials:
                    new_trials.sort(key=lambda x: x.number)
                    for t in new_trials:
                        try:
                            best_trial = study.best_trial
                            best_str = f"Best: (Trial {best_trial.number:03d} | score: {best_trial.value:.4f})"
                        except ValueError:
                            best_str = "Best: (None)"
                            
                        if t.state.name == 'COMPLETE':
                            score_str = f"{t.value:.4f}" if t.value is not None else "None"
                            msg = f"Trial {t.number:03d}: {score_str} {best_str}"
                        elif t.state.name == 'PRUNED':
                            msg = f"Trial {t.number:03d}: PRUNED {best_str}"
                        else:
                            msg = f"Trial {t.number:03d}: FAIL {best_str}"
                            
                        pbar.write(msg)
                        processed_trial_numbers.add(t.number)
                        
                    pbar.update(len(new_trials))
                
                time.sleep(1.0)
            
            # Final update
            current_finished_trials = [t for t in study.trials if t.state.name in ['COMPLETE', 'PRUNED', 'FAIL']]
            new_trials = [t for t in current_finished_trials if t.number not in processed_trial_numbers]
            if new_trials:
                new_trials.sort(key=lambda x: x.number)
                for t in new_trials:
                    try:
                        best_trial = study.best_trial
                        best_str = f"Best: (Trial {best_trial.number:03d} | score: {best_trial.value:.4f})"
                    except ValueError:
                        best_str = "Best: (None)"
                        
                    if t.state.name == 'COMPLETE':
                        score_str = f"{t.value:.4f}" if t.value is not None else "None"
                        msg = f"Trial {t.number:03d}: {score_str} {best_str}"
                    elif t.state.name == 'PRUNED':
                        msg = f"Trial {t.number:03d}: PRUNED {best_str}"
                    else:
                        msg = f"Trial {t.number:03d}: FAIL {best_str}"
                        
                    pbar.write(msg)
                    processed_trial_numbers.add(t.number)
                pbar.update(len(new_trials))
                
        for p in processes:
            p.join()
            
    print(f"Best Score: {study.best_value:.2f}")
