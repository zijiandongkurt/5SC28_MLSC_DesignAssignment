"""
Task 1 -- SIMULATION hyperparameter sweep.
Imports the shared model from gp_core (same model gp_simulation.py submits with),
loops over na/nb/num_inducing, and reports FREE-RUN SIMULATION RMSE only.
Produces task1_sweep_simulation.{csv,json}. Pick a good row, then compare with
the prediction sweep to choose a compromise na/nb for both submission scripts.
"""
import json
import itertools
from gp_core import GPNarxModel, rmse, load_traintest

# ---------------- config ----------------------------------------------------
DATA_PATH = '../../gym-unbalanced-disk-master/disc-benchmark-files/training-val-test-data.npz'
USE_TRIG  = True
RESTARTS  = 0

NA_GRID       = [2, 3, 4, 5]
NB_GRID       = [2, 3, 4, 5]
INDUCING_GRID = [200, 500]
COUPLE_NA_NB  = True     # True: only na==nb. False: full grid.

# ---------------- sweep -----------------------------------------------------
def evaluate(na, nb, ni, u_train, th_train, u_test, th_test):
    model = GPNarxModel(na, nb, use_trig=USE_TRIG, num_inducing=ni)
    model.fit(u_train, th_train, restarts=RESTARTS)
    skip = max(na, nb)
    th_sim = model.simulate(u_test, th_test, skip=skip)
    return rmse(th_sim[skip:], th_test[skip:])   # (rad, deg)


def main():
    (u_train, th_train), (u_test, th_test) = load_traintest(DATA_PATH)
    print(f'train points: {len(th_train)}, test points: {len(th_test)}\n')

    combos = ([(a, a, ni) for a in NA_GRID for ni in INDUCING_GRID] if COUPLE_NA_NB
              else list(itertools.product(NA_GRID, NB_GRID, INDUCING_GRID)))

    rows = []
    header = f"{'na':>3} {'nb':>3} {'induc':>6} | {'sim_rad':>9} {'sim_deg':>9}"
    print(header); print('-' * len(header))
    for na, nb, ni in combos:
        try:
            sr, sdg = evaluate(na, nb, ni, u_train, th_train, u_test, th_test)
            print(f"{na:>3} {nb:>3} {ni:>6} | {sr:>9.5f} {sdg:>9.4f}")
            rows.append({'na': na, 'nb': nb, 'num_inducing': ni,
                         'sim_rms_rad': sr, 'sim_rms_deg': sdg})
        except Exception as e:
            print(f"{na:>3} {nb:>3} {ni:>6} | FAILED: {e}")

    if rows:
        best = min(rows, key=lambda r: r['sim_rms_rad'])
        print(f"\nBest simulation: na={best['na']} nb={best['nb']} "
              f"inducing={best['num_inducing']} -> {best['sim_rms_rad']:.5f} rad "
              f"({best['sim_rms_deg']:.3f} deg)")

    with open('task1_sweep_simulation.json', 'w') as fh:
        json.dump(rows, fh, indent=2)
    with open('task1_sweep_simulation.csv', 'w') as fh:
        fh.write('na,nb,num_inducing,sim_rms_rad,sim_rms_deg\n')
        for r in rows:
            fh.write(f"{r['na']},{r['nb']},{r['num_inducing']},"
                     f"{r['sim_rms_rad']:.6f},{r['sim_rms_deg']:.6f}\n")
    print('\nsaved task1_sweep_simulation.json and .csv')


if __name__ == '__main__':
    main()