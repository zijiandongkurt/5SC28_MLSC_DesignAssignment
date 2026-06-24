"""
Task 1 -- GP NARX FREE-RUN SIMULATION.
All model logic lives in gp_core.py (shared with prediction + sweep).
"""
import json
import numpy as np
from gp_core import GPNarxModel, compute_metrics, load_traintest

# ---------------- config (set these to the sweep's winning values) ----------
DATA_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/training-val-test-data.npz'
HIDDEN_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/hidden-test-simulation-submission-file.npz'
SUBMIT_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/hidden-test-simulation-gp-submission-file.npz'
NA = 5
NB = 5
USE_TRIG = True
NUM_INDUCING = 200
RESTARTS = 0

# ---------------- fit ------------------------------------------------------
(u_train, th_train), (u_test, th_test) = load_traintest(DATA_PATH)
model = GPNarxModel(NA, NB, use_trig=USE_TRIG, num_inducing=NUM_INDUCING)
model.fit(u_train, th_train, restarts=RESTARTS)

# ---------------- evaluate free-run simulation on held-out test ------------
skip = max(NA, NB)
th_sim = model.simulate(u_test, th_test, skip=skip)
metrics = compute_metrics(th_sim[skip:], th_test[skip:], norm_std=th_train.std())
print('sparse_gp_simulation_test:', metrics)
with open('task1_gp_simulation_metrics.json', 'w') as fh:
    json.dump({"sparse_gp_simulation_test": metrics}, fh, indent=2)

# ---------------- hidden-test submission -----------------------------------
data = np.load(HIDDEN_PATH)
u_sub, th_sub = data['u'], data['th']    # first 50 th filled, rest zeros
th_sub_sim = model.simulate(u_sub, th_sub, skip=50)
assert len(th_sub_sim) == len(th_sub)
np.savez(SUBMIT_PATH, th=th_sub_sim, u=u_sub)
print('saved simulation submission, shape', th_sub_sim.shape)