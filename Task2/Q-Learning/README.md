# Q-Learning with RBF Features (Task 2)

This module implements Q-Learning with Radial Basis Function (RBF) feature extraction for the Unbalanced Disk swing-up and balancing problem.

To fully reproduce the results and run the experiments, simply follow the steps below in order. Make sure to run all scripts from within this `Q-Learning` directory using the `ml4sc` conda environment.

## 1. Hyperparameter Tuning
We use Optuna with multi-processing to find the optimal hyperparameters for the RBF Q-Learning agent (e.g., number of bins, alpha, gamma, and the reward function weights like `w_energy`).

```bash
conda run -n ml4sc python tune.py
```
*Outputs: Tuning trials are saved to `models/rbf_tuning_journal.log`.*

## 2. Analyze Tuning Results
Once tuning is complete (or manually stopped), analyze the top trials, parameter correlations, and importance.

```bash
conda run -n ml4sc python analyse_optuna_trials.py
```

## 3. Train the Final Agent
Extract the single best set of hyperparameters from the tuning journal and train the final production agent.

```bash
conda run -n ml4sc python train_final.py
```
*Outputs: Generates `models/best_rbf_weights.npy` and plots `visualizations/learning_curve.png`.*

## 4. Evaluate the Agent
Test the fully trained agent on the environment to see its performance (trajectory, states, actions). 

```bash
conda run -n ml4sc python evaluate.py
```
*Outputs: Generates evaluation trajectories in `visualizations/evaluation_trajectory.png` and logs raw simulation data to `data/last_eval_data.npy`.*

## 5. Generate Animation
Render a beautiful `.gif` animation of the disk using the evaluation data.

```bash
conda run -n ml4sc python animate.py
```
*Outputs: Saves `swingup_policy.gif` in the root folder.*

---

## Experimental Sweeps & Robustness Testing
If you wish to explore the deeper robustness experiments (e.g., testing different stability weights across the top 5 models, or comparing hardware filters), use the scripts organized in the `experiments/` directory.

**1. Generate the Top 5 Models:**
Extract and train the top 5 parameter configurations from the tuning database.
```bash
conda run -n ml4sc python top5_training.py
```

**2. Hardware Filter Simulations:**
Evaluate how the agents handle simulated hardware impairments (latency, noise, discretization) and run filter comparisons.
```bash
conda run -n ml4sc python experiments/simulate_filters.py
conda run -n ml4sc python experiments/evaluate_filters.py
conda run -n ml4sc python experiments/compare_omega.py
```
