"""
gp_core.py -- shared GP NARX logic for Task 1.

Single source of truth imported by:
  - gp_prediction.py   (one-step-ahead prediction submission + metrics)
  - gp_simulation.py   (free-run simulation submission + metrics)
  - gp_sweep.py        (hyperparameter tuning over na/nb/inducing)

Because all three import these functions, the model is GUARANTEED identical
across submission and tuning -- there is no duplicated logic to drift.
"""
import numpy as np
import GPy


# ---------------------------------------------------------------- regressor
def encode_theta(th_block, use_trig=True):
    """Encode past angles. (sin, cos) avoids the +-pi wraparound discontinuity."""
    th_block = np.asarray(th_block)
    if not use_trig:
        return th_block
    return np.concatenate([np.sin(th_block), np.cos(th_block)], axis=-1)


def create_IO_data(u, y, na, nb, use_trig=True):
    """Build NARX regressors: x(k) = [u[k-nb..k-1], enc(y[k-na..k-1])]."""
    X, Y = [], []
    for k in range(max(na, nb), len(y)):
        X.append(np.concatenate([u[k-nb:k], encode_theta(y[k-na:k], use_trig)]))
        Y.append(y[k])
    return np.array(X), np.array(Y).reshape(-1, 1)


# ---------------------------------------------------------------- model
class GPNarxModel:
    """A fitted sparse GP NARX model. Holds normalization + lag config so that
    prediction and simulation use exactly the same trained object."""

    def __init__(self, na, nb, use_trig=True, num_inducing=200):
        self.na = na
        self.nb = nb
        self.use_trig = use_trig
        self.num_inducing = num_inducing
        self.m = None
        self.Xmean = None
        self.Xstd = None

    def _norm(self, X):
        return (X - self.Xmean) / self.Xstd

    def fit(self, u_train, th_train, opt_iters=200, restarts=0):
        Xtr, Ytr = create_IO_data(u_train, th_train, self.na, self.nb, self.use_trig)
        self.Xmean, self.Xstd = Xtr.mean(0), Xtr.std(0)
        self.Xstd[self.Xstd == 0] = 1.0

        D = Xtr.shape[1]
        ni = min(self.num_inducing, len(Xtr))
        kernel = GPy.kern.RBF(input_dim=D, ARD=True) + GPy.kern.White(input_dim=D)
        Z = self._norm(Xtr)[np.random.choice(len(Xtr), ni, replace=False)].copy()
        self.m = GPy.models.SparseGPRegression(self._norm(Xtr), Ytr, kernel, Z=Z)
        if restarts > 0:
            self.m.optimize_restarts(num_restarts=restarts, messages=False,
                                     max_iters=opt_iters, verbose=False)
        else:
            self.m.optimize(messages=False, max_iters=opt_iters)
        return self

    def predict_rows(self, X):
        """One-step-ahead prediction for a batch of pre-built regressor rows."""
        mean, _ = self.m.predict(self._norm(X))
        return mean.ravel()

    def predict_one(self, x_raw):
        """Predict y(k) from a single RAW-units state [u_past(nb), th_past(na)]."""
        upast, thpast = x_raw[:self.nb], x_raw[self.nb:]
        feat = np.concatenate([upast, encode_theta(thpast, self.use_trig)])
        mean, _ = self.m.predict(self._norm(feat[None, :]))
        return mean[0, 0]

    def simulate(self, ulist, ylist, skip):
        """Free-run simulation: feed own predictions back. ylist supplies only
        the first `skip` initial outputs; the rest is generated."""
        na, nb = self.na, self.nb
        upast = ulist[skip-nb:skip].tolist()
        ypast = ylist[skip-na:skip].tolist()
        Y = ylist[:skip].tolist()
        for u in ulist[skip:]:
            yp = self.predict_one(np.concatenate([upast, ypast]))
            Y.append(yp)
            upast.append(u);  upast.pop(0)
            ypast.append(yp); ypast.pop(0)
        return np.array(Y)


# ---------------------------------------------------------------- metrics
def rmse(pred, true):
    pred = np.asarray(pred).ravel()
    true = np.asarray(true).ravel()
    r = float(np.mean((pred - true) ** 2) ** 0.5)
    return r, r / (2 * np.pi) * 360  # (rad, deg)


def compute_metrics(pred, true, norm_std):
    r_rad, r_deg = rmse(pred, true)
    return {"rms_rad": r_rad,
            "rms_deg": r_deg,
            "nrms_percent": float(r_rad / norm_std * 100)}


def load_traintest(path, test_frac=0.2):
    """Time-ordered split (no shuffling -> no future leakage)."""
    out = np.load(path)
    th, u = out['th'], out['u']
    n = len(th)
    split = int((1 - test_frac) * n)
    return (u[:split], th[:split]), (u[split:], th[split:])