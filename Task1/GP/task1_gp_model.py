from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler

TASK1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK1_ROOT / "shared"))

from task1_common import DEFAULT_DATA_DIR, make_io, score, save_json


class SparseDeltaGP:
    def __init__(self, na: int, nb: int, subset_size: int, length_scale: float):
        self.na = na
        self.nb = nb
        self.subset_size = subset_size
        self.length_scale = length_scale
        self.x_scaler = StandardScaler()
        self.gp: GaussianProcessRegressor | None = None

    def fit(self, x_raw: np.ndarray, y: np.ndarray) -> None:
        y_delta = y - x_raw[:, self.nb + self.na - 1]
        subset = np.linspace(0, len(x_raw) - 1, self.subset_size, dtype=int)
        x_subset = self.x_scaler.fit_transform(x_raw[subset])
        kernel = ConstantKernel(1.0, constant_value_bounds="fixed") * RBF(
            length_scale=np.full(x_subset.shape[1], self.length_scale),
            length_scale_bounds="fixed",
        ) + WhiteKernel(noise_level=2e-5, noise_level_bounds="fixed")
        self.gp = GaussianProcessRegressor(kernel=kernel, alpha=1e-8, normalize_y=True, optimizer=None)
        self.gp.fit(x_subset, y_delta[subset])

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        if self.gp is None:
            raise RuntimeError("Model is not fitted yet.")
        last_theta = x_raw[:, self.nb + self.na - 1]
        delta = self.gp.predict(self.x_scaler.transform(x_raw))
        return last_theta + delta


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Task 1 Sparse GP NARX model.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("task1_outputs"))
    parser.add_argument("--na", type=int, default=8)
    parser.add_argument("--nb", type=int, default=8)
    parser.add_argument("--subset-size", type=int, default=1800)
    parser.add_argument("--length-scale", type=float, default=12.0)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data_dir / "training-val-test-data.npz")
    u, th = data["u"], data["th"]

    split_train = 25_000
    split_val = 30_000
    x_train, y_train = make_io(u, th, args.na, args.nb, stop=split_train)
    x_test, y_test = make_io(u, th, args.na, args.nb, start=split_val)

    model = SparseDeltaGP(args.na, args.nb, args.subset_size, args.length_scale)
    model.fit(x_train, y_train)
    y_test_hat = model.predict(x_test)

    metrics = {"sparse_gp_prediction_test": score(y_test_hat, y_test).__dict__}

    pred_template = np.load(args.data_dir / "hidden-test-prediction-submission-file.npz")
    upast = pred_template["upast"]
    thpast = pred_template["thpast"]
    x_hidden = np.concatenate([upast[:, -args.nb :], thpast[:, -args.na :]], axis=1)
    hidden_pred = model.predict(x_hidden)
    np.savez(
        args.out_dir / "hidden-test-prediction-sparse-gp-submission-file.npz",
        upast=upast,
        thpast=thpast,
        thnow=hidden_pred,
    )

    save_json(args.out_dir / "task1_gp_metrics.json", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
