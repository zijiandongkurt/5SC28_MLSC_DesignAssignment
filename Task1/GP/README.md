# Task 1 GP Model

This folder contains the Gaussian Process system-identification method.

## Main Script

```text
gp_core.py -> task1_gp_prediction_model.py -> task1_gp_prediction_metrics.json
gp_core.py -> task1_gp_simulation_model.py -> task1_gp_simulation_metrics.json

gp_core.py -> sweep_prediction.py -> task1_sweep_prediction.json
gp_core.py -> sweep_simulation.py -> task1_sweep_simulation.json
```

It trains a Sparse GP NARX model for one-step prediction.

## Output Files

- `task1_gp_prediction_metrics.json, task1_gp_simulation_metrics.json, task1_sweep_prediction.json, task1_sweep_simulation.json`: internal prediction/simulation performance.
- `hidden-test-prediction-gp-submission-file.npz, hiddent_test_simulation-gp-submission-file.npz`: GP hidden prediction/simulation submission file.

## Run

From the project root:

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 task1_outputs/task1_gp_prediction_model.py --out-dir Task1/GP

python3 task1_outputs/task1_gp_prediction_model.py --out-dir Task1/GP
```

## Report Description

Use this method as the GP-based model required by Task 1.
It learns the one-step angle increment using an 8-lag NARX regressor and an RBF kernel.
