"""RF and SVM with the "_resid_eq2" construction (reviewers R3-W2/R3-Q3 and the equivariant comparison of all models).

The original RF / SVM (models/random_forest.py, models/svm.py) flatten the 23 x 2 positions of an event into one vector and fit one
forest / one SVR for each of the 44 outputs: every slot has its own weights, as in the other original models, and the data of an
event is one sample. The "_resid_eq2" construction of the other models, applied here:

  association / time gap   the 5 features of preprocess_data/make_assoc_features.py (displacement to the previous event, players
                           associated by position only, the time gap, displacement / gap) as inputs; computed outside the model,
                           as for every other model
  residual                 the target is the velocity minus the position-difference velocity (assoc channels 3, 4 times 10)
  full equivariance        one row per (event, player) and one model for all players: the features of a row are the player's own
                           position, the ball (position, end position, displacement, velocity), the team, the distance to the ball,
                           the player's association features and the event type. A relabelling of the players inside a team only
                           reorders the rows, so the predictions are equivariant by construction (checked below).

RF: RandomForestRegressor (2 outputs), n_estimators chosen from {50, 100} on the validation windows (the grid of the original
model, without cross-validation), seeds 0-4 are the random_state. SVM: SVR(rbf, gamma = "scale") per component on a random subsample
of the training rows (--svm_rows, the full 321k rows are too many for an RBF SVR), C chosen from {0.1, 1, 10} on the validation
windows, seeds 0-4 draw different subsamples. Features of the SVM are standardised. Written like analysis/sklearn_samples.py:
<weights>/<date>/<RF|SVM>_resid_eq2_events_<seed>_rectangular_256/experiments/sample_eval/samples.p

    python analysis/sklearn_resid_eq2.py --date 202610081800 --models RF SVM
"""

import argparse, os, pickle, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import event_config as cfg
from interp_baseline import load_split

SIDE = np.r_[np.zeros(11), np.ones(11)]
N_TYPES = len(cfg.EVENT_TYPE)


def load(split, data_path, dataset):
    d = load_split(dataset, split, data_path)
    z = np.load(os.path.join(data_path, "dataset", dataset, "rectangular", "recurrent", f"{split}_assoc.npz"))["assoc"][:, 0]  # (N, 23, 5)
    return d, z


def features(d, a, mask):
    """(rows, features), base velocity, ground truth; one row per (event in mask, player); rows follow the event, then the slot."""
    n = int(mask.sum())
    pos = d["pos"][mask, -1]  # (n, 22, 2) [m]
    ball = np.repeat(d["ball"][mask][:, None, :], 22, 1)
    bend = np.repeat(d["bend"][mask][:, None, :], 22, 1)
    ap = a[mask]  # (n, 23, 5)
    own = ap[:, 1:, :]  # (n, 22, 5)
    ba = np.repeat(ap[:, 0:1, [0, 1, 3, 4]], 22, 1)  # ball displacement and velocity
    onehot = np.eye(N_TYPES)[np.repeat(d["event_type"][mask][:, None], 22, 1)]
    cols = [pos[..., 0:1], pos[..., 1:2], ball, bend, np.tile(SIDE, (n, 1))[..., None], np.linalg.norm(pos - ball, axis=-1)[..., None],
            own, ba, onehot]
    X = np.concatenate(cols, axis=-1).reshape(n * 22, -1)
    base = (own[..., 3:5] * 10.0).reshape(n * 22, 2)
    return X, base, d["gt"][mask].reshape(n * 22, 2)


def event_rmse(pred, gt, n):
    e = np.sqrt(((pred - gt) ** 2).sum(-1).reshape(n, 22).mean(1))
    return float(e.mean())


def fit_predict_factory(kind, seed, svm_rows, n_estimators=None, C=None):
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    if kind == "RF":
        return lambda X, y: RandomForestRegressor(n_estimators=n_estimators, random_state=seed, n_jobs=-1).fit(X, y).predict
    def make(X, y):
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(X), size=min(svm_rows, len(X)), replace=False)
        sc = StandardScaler().fit(X[idx])
        ms = [SVR(C=C, gamma="scale", kernel="rbf").fit(sc.transform(X[idx]), y[idx, k]) for k in range(2)]
        return lambda Z: np.stack([m.predict(sc.transform(Z)) for m in ms], axis=-1)
    return make


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--date", required=True)
    cli.add_argument("--models", nargs="+", default=["RF", "SVM"], choices=["RF", "SVM"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    cli.add_argument("--svm_rows", type=int, default=20000)
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    opt = cli.parse_args()

    D = {s: load(s, opt.data_path, opt.dataset) for s in ("training", "validation", "test")}
    M = {s: D[s][0]["valid"] for s in D}
    tr = features(D["training"][0], D["training"][1], M["training"])
    va = features(D["validation"][0], D["validation"][1], M["validation"])
    te = features(D["test"][0], D["test"][1], M["test"])
    n_va, n_te = int(M["validation"].sum()), int(M["test"].sum())
    assert n_te == len(D["test"][0]["valid"]), "every test event has a previous event"
    print(f"rows: training {len(tr[0])}, validation {len(va[0])}, test {len(te[0])}, features {tr[0].shape[1]}", flush=True)
    rmse_base = event_rmse(te[1], te[2], n_te)
    print(f"position-difference velocity alone on the test events: RMSE {rmse_base:.4f}", flush=True)

    # test events as in non_recurrent (same order): ball velocity / rule-based velocity / locations for samples.p
    z = np.load(os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "non_recurrent", "test.npz"))
    cies, frames = z["cies"], z["frames"][:, 0].astype(float)
    L, V = len(cfg.LOC_NODES), len(cfg.VEL_NODES)
    binfo = z["vel_stats"][0]
    v_rb = np.concatenate([cies[:, L:L + 1, :], cies[:, L + V:, :]], axis=1).astype(np.float64)
    v_rb[:, 0, 0] = v_rb[:, 0, 0] * binfo[1] + binfo[0]
    v_rb[:, 0, 1] = v_rb[:, 0, 1] * binfo[3] + binfo[2]
    loc = np.concatenate([cies[:, 0:1, :], cies[:, 2:L, :]], axis=1).astype(np.float64)
    loc[:, :, 0] *= cfg.FIELD_LENGTH / 2
    loc[:, :, 1] *= cfg.FIELD_WIDTH / 2
    assert np.allclose(D["test"][0]["gt"].reshape(-1, 2), te[2], atol=1e-4)

    for kind in opt.models:
        t0 = time.time()
        grid = {"RF": [("n_estimators", 50), ("n_estimators", 100)], "SVM": [("C", 0.1), ("C", 1.0), ("C", 10.0)]}[kind]
        scores = {}
        for key, val in grid:  # choice on the validation windows, seed 0
            kw = {key: val}
            predict = fit_predict_factory(kind, 0, opt.svm_rows, **kw)(tr[0], tr[2] - tr[1])
            scores[val] = event_rmse(predict(va[0]) + va[1], va[2], n_va)
            print(f"{kind} {key}={val}: validation RMSE {scores[val]:.4f}  ({time.time() - t0:.0f} s)", flush=True)
        best = min(scores, key=scores.get)
        kw = {grid[0][0]: best}
        print(f"{kind}: {grid[0][0]} = {best}", flush=True)
        for s in opt.seeds:
            predict = fit_predict_factory(kind, s, opt.svm_rows, **kw)(tr[0], tr[2] - tr[1])
            pred = (predict(te[0]) + te[1]).reshape(n_te, 22, 2)
            gt = te[2].reshape(n_te, 22, 2)
            rmse = float(np.sqrt(((pred - gt) ** 2).sum(-1).mean(1)).mean())
            if s == opt.seeds[0]:  # equivariance: relabel the players inside each team, the prediction follows
                rng = np.random.default_rng(1)
                P = np.tile(np.arange(22), (n_te, 1))
                P[:, :11] = np.argsort(rng.random((n_te, 11)), axis=-1)
                P[:, 11:] = 11 + np.argsort(rng.random((n_te, 11)), axis=-1)
                d2 = dict(D["test"][0]); d2["pos"] = np.take_along_axis(d2["pos"], P[:, None, :, None], axis=2)
                a2 = D["test"][1].copy(); a2[:, 1:, :] = np.take_along_axis(a2[:, 1:, :], P[:, :, None], axis=1)
                d2["gt"] = np.take_along_axis(d2["gt"], P[:, :, None], axis=1)
                X2, b2, g2 = features(d2, a2, np.ones(n_te, bool))
                pred2 = (predict(X2) + b2).reshape(n_te, 22, 2)
                back = np.take_along_axis(pred, P[:, :, None], axis=1)
                print(f"{kind}: equivariance check, max |prediction(relabelled input) - relabelled prediction| = {np.abs(pred2 - back).max():.2e}", flush=True)
            run = os.path.join(opt.weights_dir, opt.date, f"{kind}_resid_eq2_events_{s}_rectangular_256")
            out = os.path.join(run, "experiments", "sample_eval", "samples.p")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            others = {"v_pred": pred[None].astype(np.float64), "v_gt": gt[None].astype(np.float64), "v_rb": v_rb[None], "loc_gt": loc[None],
                      "frames": frames[None]}
            pickle.dump([{}, {}, others, [{"L_rec2": rmse}]], open(out, "wb"), protocol=4)
            pickle.dump({"model": f"{kind}_resid_eq2", "seed": s, "selected": kw, "svm_rows": opt.svm_rows}, open(os.path.join(run, "params.p"), "wb"))
            print(f"{kind} seed {s}: test RMSE {rmse:.4f} -> {out}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
