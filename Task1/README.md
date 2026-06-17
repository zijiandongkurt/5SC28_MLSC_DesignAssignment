# Task 1: System Dynamics Modeling

Task 1 is about identifying the unbalanced disk dynamics from data.

It answers:

```text
Given past motor voltages and measured disk angles, can we predict/simulate future disk angle?
```

This task is separate from policy learning. It does not output a controller.

## Folder Structure

```text
Task1/
  shared/
    task1_common.py
  GP/
    task1_gp_model.py
    task1_gp_metrics.json
    hidden-test-prediction-sparse-gp-submission-file.npz
  ANN/
    task1_ann_model.py
    task1_ann_metrics.json
    hidden-test-prediction-ann-submission-file.npz
    hidden-test-simulation-ann-submission-file.npz
    internal-simulation-ann.png
    internal-simulation-ann.npz
```

## Methods

- `GP`: Sparse Gaussian Process NARX prediction model.
- `ANN`: Neural-network NARX prediction and simulation model.

Both methods use only:

- input voltage `u`;
- measured angle `theta`.

Angular velocity is not used in Task 1.

## Recommended Run Commands

The original runnable scripts are still in `task1_outputs/`, which keeps imports simple.

Run GP only:

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 task1_outputs/task1_gp_model.py --out-dir Task1/GP
```

Run ANN only:

```bash
cd /Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment
python3 task1_outputs/task1_ann_model.py --out-dir Task1/ANN --save-plots
```

## Main Results To Use In Report

GP result:

```text
Task1/GP/task1_gp_metrics.json
```

ANN result:

```text
Task1/ANN/task1_ann_metrics.json
Task1/ANN/internal-simulation-ann.png
```
