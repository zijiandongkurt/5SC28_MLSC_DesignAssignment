from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import torch
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

TASK1_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK1_ROOT / "shared"))

from task1_common import DEFAULT_DATA_DIR, add_ann_features, make_io, save_json, score, simulate_io_model


class TorchDeltaRegressor:
    def __init__(self, in_dim: int, hidden: tuple[int, ...], lr: float, weight_decay: float, seed: int):
        torch.manual_seed(seed)
        layers: list[nn.Module] = []
        last = in_dim
        for width in hidden:
            layers += [nn.Linear(last, width), nn.SiLU(), nn.LayerNorm(width)]
            last = width
        layers.append(nn.Linear(last, 1))
        self.model = nn.Sequential(*layers)
        self.x_scaler = StandardScaler()
        self.y_scaler = StandardScaler()
        self.lr = lr
        self.weight_decay = weight_decay

    def fit(self, x: np.ndarray, y_delta: np.ndarray, x_val: np.ndarray, y_val_delta: np.ndarray, epochs: int):
        x_s = self.x_scaler.fit_transform(x).astype(np.float32)
        y_s = self.y_scaler.fit_transform(y_delta[:, None]).astype(np.float32)
        xv_s = self.x_scaler.transform(x_val).astype(np.float32)
        yv_s = self.y_scaler.transform(y_val_delta[:, None]).astype(np.float32)

        loader = DataLoader(TensorDataset(torch.from_numpy(x_s), torch.from_numpy(y_s)), batch_size=512, shuffle=True)
        opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        loss_fn = nn.MSELoss()
        best_state = None
        best_val = float("inf")
        stale = 0

        for _ in range(epochs):
            self.model.train()
            for xb, yb in loader:
                opt.zero_grad(set_to_none=True)
                loss = loss_fn(self.model(xb), yb)
                loss.backward()
                opt.step()

            self.model.eval()
            with torch.no_grad():
                val = float(loss_fn(self.model(torch.from_numpy(xv_s)), torch.from_numpy(yv_s)))
            if val < best_val:
                best_val = val
                best_state = {k: v.detach().clone() for k, v in self.model.state_dict().items()}
                stale = 0
            else:
                stale += 1
            if stale >= 25:
                break

        if best_state is not None:
            self.model.load_state_dict(best_state)

    def predict_delta(self, x: np.ndarray) -> np.ndarray:
        self.model.eval()
        xs = self.x_scaler.transform(x).astype(np.float32)
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 4096):
                out.append(self.model(torch.from_numpy(xs[i : i + 4096])).numpy())
        return self.y_scaler.inverse_transform(np.vstack(out)).ravel()


class AnnNarxModel:
    def __init__(self, regressor: TorchDeltaRegressor, na: int, nb: int, use_engineered_features: bool):
        self.regressor = regressor
        self.na = na
        self.nb = nb
        self.use_engineered_features = use_engineered_features

    def features(self, x_raw: np.ndarray) -> np.ndarray:
        if self.use_engineered_features:
            return add_ann_features(x_raw, self.nb, self.na)
        return x_raw

    def predict(self, x_raw: np.ndarray) -> np.ndarray:
        last_theta = x_raw[:, self.nb + self.na - 1]
        return last_theta + self.regressor.predict_delta(self.features(x_raw))


def save_simulation_plot(path: Path, th_true: np.ndarray, th_sim: np.ndarray) -> None:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib import pyplot as plt

    t = np.arange(len(th_true))
    fig, ax = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
    ax[0].plot(t, th_true, label="measured theta", linewidth=1.2)
    ax[0].plot(t, th_sim, label="ANN simulation", linewidth=1.0)
    ax[0].axvline(50, color="k", linestyle="--", linewidth=0.8, label="free-run start")
    ax[0].set_ylabel("angle [rad]")
    ax[0].set_title("Internal free-run simulation on held-out data")
    ax[0].legend(loc="best")
    ax[0].grid(True, alpha=0.25)

    residual = th_sim - th_true
    ax[1].plot(t, residual, color="tab:red", linewidth=1.0)
    ax[1].axvline(50, color="k", linestyle="--", linewidth=0.8)
    ax[1].set_xlabel("sample k")
    ax[1].set_ylabel("residual [rad]")
    ax[1].grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)


def train_ann_model(
    u: np.ndarray,
    th: np.ndarray,
    na: int,
    nb: int,
    split_train: int,
    split_val: int,
    hidden: tuple[int, ...],
    use_engineered_features: bool,
    lr: float,
    weight_decay: float,
    seed: int,
    epochs: int,
):
    x_train_raw, y_train = make_io(u, th, na, nb, stop=split_train)
    x_val_raw, y_val = make_io(u, th, na, nb, start=split_train, stop=split_val)
    x_test_raw, y_test = make_io(u, th, na, nb, start=split_val)

    feature = lambda x: add_ann_features(x, nb, na) if use_engineered_features else x
    x_train = feature(x_train_raw)
    x_val = feature(x_val_raw)
    y_train_delta = y_train - x_train_raw[:, nb + na - 1]
    y_val_delta = y_val - x_val_raw[:, nb + na - 1]

    regressor = TorchDeltaRegressor(x_train.shape[1], hidden, lr=lr, weight_decay=weight_decay, seed=seed)
    regressor.fit(x_train, y_train_delta, x_val, y_val_delta, epochs=epochs)
    model = AnnNarxModel(regressor, na, nb, use_engineered_features)
    return model, x_test_raw, y_test


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the Task 1 ANN NARX models.")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--out-dir", type=Path, default=Path("task1_outputs"))
    parser.add_argument("--epochs", type=int, default=220)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--save-plots", action="store_true")
    args = parser.parse_args()

    np.random.seed(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    data = np.load(args.data_dir / "training-val-test-data.npz")
    u, th = data["u"], data["th"]
    split_train = 25_000
    split_val = 30_000
    na = nb = 15

    simple_model, x_test_raw, y_test = train_ann_model(
        u,
        th,
        na,
        nb,
        split_train,
        split_val,
        hidden=(64, 64),
        use_engineered_features=False,
        lr=2e-3,
        weight_decay=1e-4,
        seed=args.seed,
        epochs=args.epochs,
    )

    advanced_model, x_test_raw, y_test = train_ann_model(
        u,
        th,
        na,
        nb,
        split_train,
        split_val,
        hidden=(128, 128, 64),
        use_engineered_features=True,
        lr=1.5e-3,
        weight_decay=2e-4,
        seed=args.seed + 1,
        epochs=args.epochs,
    )

    results: dict[str, dict[str, float]] = {
        "ann_simple_prediction_test": score(simple_model.predict(x_test_raw), y_test).__dict__,
        "ann_advanced_prediction_test": score(advanced_model.predict(x_test_raw), y_test).__dict__,
    }

    sim_start = 30_000
    u_sim, th_sim_true = u[sim_start:], th[sim_start:]
    th_sim = simulate_io_model(advanced_model, u_sim, th_sim_true.copy(), na, nb, skip=50)
    results["ann_advanced_simulation_internal"] = score(th_sim[50:], th_sim_true[50:]).__dict__
    np.savez(args.out_dir / "internal-simulation-ann.npz", u=u_sim, th_true=th_sim_true, th_sim=th_sim)
    if args.save_plots:
        save_simulation_plot(args.out_dir / "internal-simulation-ann.png", th_sim_true, th_sim)

    pred_template = np.load(args.data_dir / "hidden-test-prediction-submission-file.npz")
    upast = pred_template["upast"]
    thpast = pred_template["thpast"]
    x_hidden = np.concatenate([upast[:, -nb:], thpast[:, -na:]], axis=1)
    hidden_pred = advanced_model.predict(x_hidden)
    np.savez(args.out_dir / "hidden-test-prediction-ann-submission-file.npz", upast=upast, thpast=thpast, thnow=hidden_pred)

    sim_template = np.load(args.data_dir / "hidden-test-simulation-submission-file.npz")
    u_hidden = sim_template["u"]
    th_hidden = sim_template["th"].copy()
    hidden_sim = simulate_io_model(advanced_model, u_hidden, th_hidden, na, nb, skip=50)
    np.savez(args.out_dir / "hidden-test-simulation-ann-submission-file.npz", u=u_hidden, th=hidden_sim)

    save_json(args.out_dir / "task1_ann_metrics.json", results)
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
