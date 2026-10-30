"""RF and SVM in the form of the neural models: one sample = one event, 23 agents in (ball + 22 players), 22 players out.

Input of an event = 23 agents x 7 channels = 161 features, the same information as MLP_resid_eq2: for every agent the position (x, y [m]) and
the 5 association features of preprocess_data/make_assoc_features.py (displacement, time gap, velocity), the ball first. Target = the
44 residuals (velocity - position-difference velocity) of the 22 players. Number of samples = number of events (14,611 for training).
This is the form of the original RF / SVM (models/random_forest.py, models/svm.py: all slots flattened, one output per slot), with the
association inputs and the residual target added.

  --variant resid      slot order as in the data (the weights of a slot are specific to its index: NOT equivariant)
  --variant resid_eq2  the players of each team are sorted by x (then y) before they enter, the 22 predictions are put back into the
                       slots: a relabelling of the players inside a team does not change the sorted input, the prediction is equivariant
                       by construction (checked below; the ball is slot 0)

RF: n_estimators chosen from {50, 100} on the validation events, seeds 0-4 = random_state. SVM: SVR(rbf, gamma = "scale") for every output,
C chosen from {0.1, 1, 10} on the validation events; deterministic, so the 5 seeds are identical (as for the original SVM). The features of
the SVM are standardised. Written like analysis/sklearn_samples.py to <weights>/<date>/<RF|SVM>_<variant>_events_<seed>_rectangular_256.

    python analysis/sklearn_resid_eq2_event.py --date 202610091000 --models RF SVM --variants resid resid_eq2
"""
import argparse, os, pickle, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import event_config as cfg
from sklearn_resid_eq2 import load


def build(d, a, mask, sort):
    """x (n, 161), base (n, 22, 2), gt (n, 22, 2), order (n, 22): the slot of the i-th input player."""
    n = int(mask.sum())
    pos = d["pos"][mask, -1]  # (n, 22, 2)
    ap = a[mask]  # (n, 23, 5)
    order = np.tile(np.arange(22), (n, 1))
    if sort:
        for lo, hi in ((0, 11), (11, 22)):
            key = pos[:, lo:hi, 0] + 1e-7 * pos[:, lo:hi, 1]
            order[:, lo:hi] = lo + np.argsort(key, axis=1)
    pos_s = np.take_along_axis(pos, order[..., None], axis=1)
    own_s = np.take_along_axis(ap[:, 1:, :], order[..., None], axis=1)
    ball = np.concatenate([d["ball"][mask], ap[:, 0, :]], axis=-1)[:, None, :]  # (n, 1, 7)
    agents = np.concatenate([ball, np.concatenate([pos_s, own_s], axis=-1)], axis=1)  # (n, 23, 7)
    gt_s = np.take_along_axis(d["gt"][mask], order[..., None], axis=1)
    base_s = own_s[..., 3:5] * 10.0
    return agents.reshape(n, -1), base_s, gt_s, order


def to_slots(pred_sorted, order):
    out = np.empty_like(pred_sorted)
    np.put_along_axis(out, order[..., None], pred_sorted, axis=1)
    return out


def event_rmse(pred, gt):
    return float(np.sqrt(((pred - gt) ** 2).sum(-1).mean(1)).mean())


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--date", required=True)
    cli.add_argument("--models", nargs="+", default=["RF", "SVM"], choices=["RF", "SVM"])
    cli.add_argument("--variants", nargs="+", default=["resid", "resid_eq2"], choices=["resid", "resid_eq2"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    opt = cli.parse_args()
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVR

    D = {s: load(s, opt.data_path, opt.dataset) for s in ("training", "validation", "test")}
    M = {s: D[s][0]["valid"] for s in D}
    assert M["test"].all()
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

    for variant in opt.variants:
        sort = variant == "resid_eq2"
        tr = build(D["training"][0], D["training"][1], M["training"], sort)
        va = build(D["validation"][0], D["validation"][1], M["validation"], sort)
        te = build(D["test"][0], D["test"][1], M["test"], sort)
        n_te = len(te[0])
        print(f"[{variant}] events: training {len(tr[0])}, validation {len(va[0])}, test {n_te}; features {tr[0].shape[1]}", flush=True)
        gt_slots = to_slots(te[2], te[3])
        assert np.allclose(gt_slots, D["test"][0]["gt"], atol=1e-4)
        for kind in opt.models:
            t0 = time.time()

            def make(val, seed):
                if kind == "RF":
                    return lambda X, y: RandomForestRegressor(n_estimators=val, random_state=seed, n_jobs=-1).fit(X, y).predict
                def fit(X, y):
                    sc = StandardScaler().fit(X)
                    m = MultiOutputRegressor(SVR(C=val, gamma="scale", kernel="rbf"), n_jobs=-1).fit(sc.transform(X), y)
                    return lambda Z: m.predict(sc.transform(Z))
                return fit
            ytr = (tr[2] - tr[1]).reshape(len(tr[0]), -1)
            grid = [50, 100] if kind == "RF" else [0.1, 1.0, 10.0]
            scores = {}
            for val in grid:
                pr = make(val, 0)(tr[0], ytr)(va[0]).reshape(-1, 22, 2) + va[1]
                scores[val] = event_rmse(to_slots(pr, va[3]), to_slots(va[2], va[3]))
                print(f"{kind}_{variant} {'n_estimators' if kind == 'RF' else 'C'}={val}: validation RMSE {scores[val]:.4f} ({time.time() - t0:.0f} s)", flush=True)
            best = min(scores, key=scores.get)
            print(f"{kind}_{variant}: selected {best}", flush=True)
            cache = None
            for s in opt.seeds:
                if kind == "SVM" and cache is not None:
                    pred_s = cache
                else:
                    predict = make(best, s)(tr[0], ytr)
                    pred_s = predict(te[0]).reshape(-1, 22, 2) + te[1]
                    if kind == "SVM":
                        cache = pred_s
                    if s == opt.seeds[0]:  # equivariance: relabel the players inside each team, the prediction must follow
                        rng = np.random.default_rng(1)
                        P = np.tile(np.arange(22), (n_te, 1))
                        P[:, :11] = np.argsort(rng.random((n_te, 11)), axis=-1)
                        P[:, 11:] = 11 + np.argsort(rng.random((n_te, 11)), axis=-1)
                        d2 = dict(D["test"][0]); d2["pos"] = np.take_along_axis(d2["pos"], P[:, None, :, None], axis=2)
                        a2 = D["test"][1].copy(); a2[:, 1:, :] = np.take_along_axis(a2[:, 1:, :], P[:, :, None], axis=1)
                        d2["gt"] = np.take_along_axis(d2["gt"], P[:, :, None], axis=1)
                        x2, b2, g2, o2 = build(d2, a2, np.ones(n_te, bool), sort)
                        p2 = to_slots(predict(x2).reshape(-1, 22, 2) + b2, o2)
                        back = np.take_along_axis(to_slots(pred_s, te[3]), P[:, :, None], axis=1)
                        print(f"{kind}_{variant}: equivariance check, max |prediction(relabelled input) - relabelled prediction| = {np.abs(p2 - back).max():.2e}", flush=True)
                pred = to_slots(pred_s, te[3])
                rmse = event_rmse(pred, gt_slots)
                run = os.path.join(opt.weights_dir, opt.date, f"{kind}_{variant}_events_{s}_rectangular_256")
                out = os.path.join(run, "experiments", "sample_eval", "samples.p")
                os.makedirs(os.path.dirname(out), exist_ok=True)
                others = {"v_pred": pred[None].astype(np.float64), "v_gt": gt_slots[None].astype(np.float64), "v_rb": v_rb[None], "loc_gt": loc[None], "frames": frames[None]}
                pickle.dump([{}, {}, others, [{"L_rec2": rmse}]], open(out, "wb"), protocol=4)
                pickle.dump({"model": f"{kind}_{variant}", "seed": s, "selected": best}, open(os.path.join(run, "params.p"), "wb"))
                print(f"{kind}_{variant} seed {s}: test RMSE {rmse:.4f}  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
