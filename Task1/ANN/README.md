# Task 1 ANN Model

This folder contains the neural-network system-identification method.

## Main Script

```text
task1_ann_model.py
```

It trains:

1. a simple ANN NARX model;
2. an advanced ANN NARX model with engineered angle features.

## Output Files

- `task1_ann_metrics.json`: internal prediction and simulation performance.
- `hidden-test-prediction-ann-submission-file.npz`: ANN prediction submission file.
- `hidden-test-simulation-ann-submission-file.npz`: ANN simulation submission file.
- `internal-simulation-ann.png`: free-run simulation plot for the report.
- `internal-simulation-ann.npz`: raw simulation data.

## Run

From the project root:

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 task1_outputs/task1_ann_model.py --out-dir Task1/ANN --save-plots
```

## Report Description

Use this method as the ANN-based model required by Task 1.
It predicts the angle increment and reconstructs the next angle, which improves recursive simulation stability.
