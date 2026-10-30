"""Does the spread of the predictions across training seeds say anything about the error? (appendix)

A velocity-completion model trained with MSE estimates the conditional mean E[v | X]; the velocity of
a player given only the event-time configuration remains indeterminate, so a large part of the error
is irreducible. As a cheap, training-free look at predictive uncertainty we use the five seeds as an
ensemble (no extra training):

  - ensemble RMSE (mean of the five predictions) vs the mean single-seed RMSE: the gap is the part of
    the error that is model variance (seed-dependent) rather than irreducible
  - per-player spread s = norm of the std (over seeds) of the predicted velocity vector and the error
    e = ||ensemble mean - ground truth||: Spearman correlation and the mean error per spread quintile

    python analysis/seed_uncertainty.py --date 202610021638 --models GRNN_add Transformer RNN
"""

import argparse, json, os, pickle

import numpy as np

cli = argparse.ArgumentParser()
cli.add_argument("--date", required=True)
cli.add_argument("--models", nargs="+", required=True)
cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
cli.add_argument("--sample_dir", default="sample_eval")
cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
cli.add_argument("--out", default=None)
opt = cli.parse_args()
out = opt.out or f"analysis/results/seed_uncertainty_{opt.date}.json"


def ranks(x):
    r = np.empty(len(x))
    r[np.argsort(x, kind="mergesort")] = np.arange(len(x))
    return r


def spearman(a, b):
    ra, rb = ranks(a), ranks(b)
    return float(np.corrcoef(ra, rb)[0, 1])


def event_rmse(pred, gt):
    """Paper definition: per-event sqrt of the mean (over players) squared velocity error, then the mean."""
    return float(np.sqrt(np.sum((pred - gt) ** 2, axis=-1).mean(-1)).mean())


results = {}
print(f"{'model':<18}{'single RMSE':>12}{'ensemble':>10}{'variance part':>14}{'spearman':>10}"
      f"   mean error by spread quintile (low -> high)")
for m in opt.models:
    preds, gt = [], None
    for s in opt.seeds:
        p = os.path.join(opt.weights_dir, opt.date, f"{m}_events_{s}_rectangular_256",
                         "experiments", opt.sample_dir, "samples.p")
        o = pickle.load(open(p, "rb"))[2]
        preds.append(np.mean(o["v_pred"], axis=0))
        g = np.mean(o["v_gt"], axis=0)
        gt = g if gt is None else gt
        assert np.allclose(gt, g, atol=1e-5), f"{m} seed {s}: ground truth differs"
    P = np.stack(preds)  # (S, N, 22, 2)
    single = float(np.mean([event_rmse(p, gt) for p in P]))
    ens = P.mean(0)
    ensemble = event_rmse(ens, gt)
    spread = np.sqrt(P.var(axis=0, ddof=1).sum(-1)).ravel()  # (N*22,)
    err = np.linalg.norm(ens - gt, axis=-1).ravel()
    rho = spearman(spread, err)
    q = np.quantile(spread, [0.2, 0.4, 0.6, 0.8])
    bins = np.digitize(spread, q)
    by_q = [float(err[bins == k].mean()) for k in range(5)]
    # share of the single-seed MSE that is seed variance: mean over players of ||P_s - ens||^2 / ||P_s - gt||^2
    var_share = float(np.mean(np.sum((P - ens) ** 2, -1)) / np.mean(np.sum((P - gt) ** 2, -1)))
    results[m] = {"single_rmse": single, "ensemble_rmse": ensemble, "seed_variance_share_of_mse": var_share,
                  "spearman_spread_error": rho, "error_by_spread_quintile": by_q,
                  "mean_spread": float(spread.mean()), "n_player_observations": int(len(spread))}
    print(f"{m:<18}{single:>12.3f}{ensemble:>10.3f}{var_share:>13.1%}{rho:>10.3f}   "
          + " ".join(f"{x:.2f}" for x in by_q))

os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
json.dump({"date": opt.date, "seeds": opt.seeds, "results": results}, open(out, "w"), indent=1)
print(f"\nSaved: {out}")
