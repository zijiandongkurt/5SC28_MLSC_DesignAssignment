# Task 1 GP Model

This folder contains the Gaussian Process system-identification method.

## Main Script

```text
task1_gp_model.py
```

It trains a Sparse GP NARX model for one-step prediction.

## Output Files

- `task1_gp_metrics.json`: internal prediction performance.
- `hidden-test-prediction-sparse-gp-submission-file.npz`: GP hidden prediction submission file.

## Run

From the project root:

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 task1_outputs/task1_gp_model.py --out-dir Task1/GP
```

## Report Description

Use this method as the GP-based model required by Task 1.
It learns the one-step angle increment using an 8-lag NARX regressor and an RBF kernel.
