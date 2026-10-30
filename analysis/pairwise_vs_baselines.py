"""Two-level bootstrap of the proposed model against the non-learned baselines (rule-based, position difference).

Same test as analysis/pairwise_significance.py (cluster_bootstrap): the 7 test matches are resampled with replacement
(all events of a drawn match), and the seeds of each model are resampled independently; the statistic is the
difference of the means of the per-event metric. The baselines are deterministic (one "seed").
Metrics: per-event velocity RMSE, PPCF MAE and OBSO MAE (against the maps of the ground-truth velocities).

    python analysis/pairwise_vs_baselines.py --out analysis/results/pairwise_vs_baselines.md
"""

import argparse, os, pickle

import numpy as np

cli = argparse.ArgumentParser()
cli.add_argument("--model", default="Transformer_add_resid_eq2")
cli.add_argument("--date", default="202610091700")
cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
cli.add_argument("--model_per_event", default="analysis/results/per_event_best_aligned_202610091700.npz")
cli.add_argument("--baseline_per_event", default="analysis/results/interp_baseline_best_aligned_per_event.npz")
cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
cli.add_argument("--sample_dir", default="sample_eval")
cli.add_argument("--dataset", default="anonymized")
cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
cli.add_argument("--bootstrap", type=int, default=10000)
cli.add_argument("--out", default=None)
opt = cli.parse_args()


def match_index():
    """Match of every test window (as in pairwise_significance.py): test.pkl is stored match by match."""
    d = os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent")
    tp = pickle.load(open(os.path.join(d, "test.pkl"), "rb"))
    z = np.load(os.path.join(d, "test.npz"))
    fr = np.concatenate([np.asarray(v["frames"]).reshape(len(v["frames"]), -1)[:, -1] for v in tp.values()])
    assert np.array_equal(fr.astype(float), z["frames"][:, -1, 0].astype(float)), "test windows are not in match order"
    return np.concatenate([np.full(len(v["frames"]), i) for i, v in enumerate(tp.values())])


GAME = match_index()
N_GAMES = GAME.max() + 1
COUNT = np.bincount(GAME, minlength=N_GAMES).astype(float)
RNG = np.random.default_rng(0)


def cluster_bootstrap(Ea, Eb):
    """Two-level bootstrap (matches, then seeds per model) of mean(Ea) - mean(Eb); Ea, Eb: (seeds, events)."""
    sa = np.stack([np.bincount(GAME, weights=e, minlength=N_GAMES) for e in Ea])
    sb = np.stack([np.bincount(GAME, weights=e, minlength=N_GAMES) for e in Eb])
    B = opt.bootstrap
    w = RNG.multinomial(N_GAMES, np.full(N_GAMES, 1 / N_GAMES), size=B).astype(float)
    ia = RNG.integers(0, len(Ea), size=(B, len(Ea)))
    ib = RNG.integers(0, len(Eb), size=(B, len(Eb)))
    den = w @ COUNT
    ma = np.einsum("bg,bsg->bs", w, sa[ia]).mean(1) / den
    mb = np.einsum("bg,bsg->bs", w, sb[ib]).mean(1) / den
    d = ma - mb
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(lo), float(hi), float(p)


pm = np.load(opt.model_per_event)
pb = np.load(opt.baseline_per_event)
key = {"rmse": "vel_rmse", "ppcf": "ppcf_mae", "obso": "obso_mae"}
model = {m: np.stack([pm[f"{opt.model}_s{s}_{k}"] for s in opt.seeds]) for m, k in key.items()}
base = {"rule-based": {m: pm[f"rule-based_s0_{k}"][None] for m, k in key.items()},
        "position difference": {m: pb[f"hungarian_{m}"][None] for m in key}}

# self check: the per-event RMSE of the model file is in the order of samples.p (= match order of test.pkl)
for s in opt.seeds:
    p = os.path.join(opt.weights_dir, opt.date, f"{opt.model}_events_{s}_rectangular_256", "experiments", opt.sample_dir, "samples.p")
    o = pickle.load(open(p, "rb"))[2]
    pred, gt = np.mean(o["v_pred"], 0), np.mean(o["v_gt"], 0)
    e = np.sqrt(((pred - gt) ** 2).sum(-1).mean(-1))
    assert np.allclose(e, model["rmse"][opt.seeds.index(s)], atol=1e-5), f"seed {s}: event order differs"

pfmt = lambda p: f"<{1 / opt.bootstrap:.0e}" if p == 0 else f"{p:.3g}"
out = [f"# Proposed model vs non-learned baselines ({opt.model}@{opt.date}, seeds {opt.seeds}, {GAME.size} test events, {N_GAMES} matches)\n",
       f"Two-level bootstrap over matches and seeds ({opt.bootstrap} resamples), as in pairwise_significance.py. "
       "Difference = baseline - proposed (positive = the proposed model is better); relative = difference / baseline.\n",
       "| baseline | metric | baseline | proposed | difference | relative | 95 % interval | p (bootstrap) |",
       "|---|---|---|---|---|---|---|---|"]
for b, eb in base.items():
    for m in key:
        lo, hi, p = cluster_bootstrap(eb[m], model[m])
        mb, mp = eb[m].mean(), model[m].mean()
        fmt = ".4f" if m == "rmse" else (".5f" if m == "ppcf" else ".6f")
        out.append(f"| {b} | {m} | {mb:{fmt}} | {mp:{fmt}} | {mb - mp:+{fmt}} | {(mb - mp) / mb:.1%} | "
                   f"[{lo:+{fmt}}, {hi:+{fmt}}] | {pfmt(p)} |")
text = "\n".join(out) + "\n"
print(text)
if opt.out:
    open(opt.out, "w").write(text)
    print(f"Saved: {opt.out}")
