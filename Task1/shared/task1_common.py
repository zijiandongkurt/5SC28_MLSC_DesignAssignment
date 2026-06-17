from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import mean_squared_error


DEFAULT_DATA_DIR = Path(
    "/Users/zijiandong/Documents/GitHub/5SC28_MLSC_DesignAssignment/"
    "gym-unbalanced-disk-master/disc-benchmark-files"
)


@dataclass
class Metrics:
    rms_rad: float
    rms_deg: float
    nrms_percent: float


def score(y_hat: np.ndarray, y: np.ndarray) -> Metrics:
    rms = float(np.sqrt(mean_squared_error(y, y_hat)))
    return Metrics(rms, rms / (2 * np.pi) * 360.0, rms / float(np.std(y)) * 100.0)


def make_io(u: np.ndarray, th: np.ndarray, na: int, nb: int, start: int = 0, stop: int | None = None):
    stop = len(th) if stop is None else stop
    first = max(start, na, nb)
    x, y = [], []
    for k in range(first, stop):
        x.append(np.concatenate([u[k - nb : k], th[k - na : k]]))
        y.append(th[k])
    return np.asarray(x), np.asarray(y)


def add_ann_features(x_raw: np.ndarray, nb: int, na: int) -> np.ndarray:
    u_lags = x_raw[:, :nb]
    th_lags = x_raw[:, nb : nb + na]
    dth = np.diff(th_lags, axis=1)
    last = th_lags[:, -1:]
    prev = th_lags[:, -2:-1]
    return np.concatenate(
        [
            u_lags,
            th_lags,
            np.sin(th_lags),
            np.cos(th_lags),
            dth,
            last - prev,
            np.sin(last),
            np.cos(last),
        ],
        axis=1,
    )


def simulate_io_model(model, u: np.ndarray, th_seed: np.ndarray, na: int, nb: int, skip: int = 50):
    upast = list(u[skip - nb : skip])
    thpast = list(th_seed[skip - na : skip])
    y = list(th_seed[:skip])
    for uk in u[skip:]:
        raw = np.concatenate([upast, thpast])[None, :]
        yk = float(model.predict(raw)[0])
        y.append(yk)
        upast = upast[1:] + [uk]
        thpast = thpast[1:] + [yk]
    return np.asarray(y)


def save_json(path: Path, data: dict) -> None:
    import json

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
