"""
Task 1 -- GP NARX ONE-STEP-AHEAD PREDICTION.
All model logic lives in gp_core.py (shared with simulation + sweep),
so this script and the sweep are GUARANTEED to use the identical model.
"""
import json
import numpy as np
from gp_core import GPNarxModel, create_IO_data, compute_metrics, load_traintest

# ---------------- config (set these to the sweep's winning values) ----------
DATA_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/training-val-test-data.npz'
HIDDEN_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/hidden-test-prediction-submission-file.npz'
SUBMIT_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/hidden-test-prediction-gp-submission-file.npz'
NA = 5
NB = 5
USE_TRIG = True
NUM_INDUCING = 200
RESTARTS = 0

# ---------------- fit ------------------------------------------------------
(u_train, th_train), (u_test, th_test) = load_traintest(DATA_PATH)
model = GPNarxModel(NA, NB, use_trig=USE_TRIG, num_inducing=NUM_INDUCING)
model.fit(u_train, th_train, restarts=RESTARTS)

# ---------------- evaluate one-step-ahead prediction on held-out test ------
Xte, Yte = create_IO_data(u_test, th_test, NA, NB, USE_TRIG)
pred = model.predict_rows(Xte)
metrics = compute_metrics(pred, Yte, norm_std=th_train.std())
print('sparse_gp_prediction_test:', metrics)
with open('task1_gp_prediction_metrics.json', 'w') as fh:
    json.dump({"sparse_gp_prediction_test": metrics}, fh, indent=2)

# ---------------- hidden-test submission -----------------------------------
data = np.load(HIDDEN_PATH)
upast_test, thpast_test = data['upast'], data['thpast']   # N x 15 each
u_block  = upast_test[:, 15-NB:]
th_block = np.concatenate([np.sin(thpast_test[:, 15-NA:]),
                           np.cos(thpast_test[:, 15-NA:])], axis=1) if USE_TRIG \
           else thpast_test[:, 15-NA:]
Xsub = np.concatenate([u_block, th_block], axis=1)
Ysub = model.predict_rows(Xsub)
assert len(Ysub) == len(upast_test), 'number of samples changed!!'
np.savez(SUBMIT_PATH, upast=upast_test, thpast=thpast_test, thnow=Ysub)
print('saved prediction submission, shape', Ysub.shape)