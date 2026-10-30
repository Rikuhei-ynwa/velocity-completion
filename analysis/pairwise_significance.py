"""Paired significance tests between models (reviewer R3-Q3: "statistical significance across multiple seeds for all main models").

Main test (cluster bootstrap): the test events come from 7 matches and are not independent; the models come from 5 training seeds.
A two-level bootstrap resamples the matches (with replacement, all events of a drawn match) and, for each model separately, the
seeds (with replacement); the statistic is the difference of the mean per-event RMSE. Reported: the 95 % interval of the
difference and the two-sided bootstrap p (twice the share of resamples on the other side of 0). This is the test of a
difference between the models across seeds and matches.

Also reported, for reference only: Unit of the test: the 2958 test events. For every model the per-event RMSE is averaged over the seeds, and two models are
compared with the paired Wilcoxon signed-rank test on these per-event values (as in the A2 table). The p-values of all
pairs are corrected with the Holm method. A second, coarser view uses the seeds: the mean over events of each seed
(n = number of seeds) is compared with a two-sided Welch t-test (small n, reported for completeness).

    python analysis/pairwise_significance.py --date 202610021638 \
        --models Transformer_add Transformer GRNN_add GATRNN GRNN RNN TGAT MLP \
        --extra Transformer_add:202610030117 --out analysis/results/pairwise_significance_202610021638.md
    --reference RNN lists every other model against RNN only (e.g. the GRNN vs RNN question of R3-W2)
"""

import argparse, itertools, os, pickle

import numpy as np
from scipy.stats import wilcoxon, ttest_ind

cli = argparse.ArgumentParser()
cli.add_argument("--date", required=True)
cli.add_argument("--models", nargs="+", required=True)
cli.add_argument("--extra", nargs="*", default=[], help="MODEL:DATE for runs trained on another date")
cli.add_argument("--reference", default=None)
cli.add_argument("--pairs", nargs="*", default=[], help="A,B pairs of the --models entries (instead of all pairs / --reference)")
cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
cli.add_argument("--sample_dir", default="sample_eval")
cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
cli.add_argument("--out", default=None)
cli.add_argument("--bootstrap", type=int, default=10000)
cli.add_argument("--dataset", default="anonymized")
cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
opt = cli.parse_args()
dates = dict(e.split(":") for e in opt.extra)


def per_event(spec):  # (seeds, events) per-event RMSE; spec = "model" or "model@date" (date of this run)
    model, _, own_date = spec.partition("@")
    rows = []
    for s in opt.seeds:
        p = os.path.join(opt.weights_dir, own_date or dates.get(model, opt.date), f"{model}_events_{s}_rectangular_256",
                         "experiments", opt.sample_dir, "samples.p")
        o = pickle.load(open(p, "rb"))[2]
        pred, gt = np.mean(o["v_pred"], 0), np.mean(o["v_gt"], 0)
        rows.append(np.sqrt(((pred - gt) ** 2).sum(-1).mean(-1)))
    return np.array(rows)


def match_index():
    """Match of every test window: the recurrent test set is stored match by match (the order of test.pkl), checked with the frames."""
    import pickle as pk
    d = pk.load(open(os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent", "test.pkl"), "rb"))
    z = np.load(os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent", "test.npz"))
    fr = np.concatenate([np.asarray(v["frames"]).reshape(len(v["frames"]), -1)[:, -1] for v in d.values()])
    assert np.array_equal(fr.astype(float), z["frames"][:, -1, 0].astype(float)), "test windows are not in match order"
    return np.concatenate([np.full(len(v["frames"]), i) for i, v in enumerate(d.values())])


GAME = match_index()
N_GAMES = GAME.max() + 1
COUNT = np.bincount(GAME, minlength=N_GAMES).astype(float)
RNG = np.random.default_rng(0)


def cluster_bootstrap(Ea, Eb):
    """Two-level bootstrap (matches, then seeds per model) of mean(Ea) - mean(Eb); Ea, Eb: (seeds, events)."""
    sa = np.stack([np.bincount(GAME, weights=e, minlength=N_GAMES) for e in Ea])  # (seeds, games) sums
    sb = np.stack([np.bincount(GAME, weights=e, minlength=N_GAMES) for e in Eb])
    B = opt.bootstrap
    w = RNG.multinomial(N_GAMES, np.full(N_GAMES, 1 / N_GAMES), size=B).astype(float)  # (B, games) match weights
    ia = RNG.integers(0, len(Ea), size=(B, len(Ea)))
    ib = RNG.integers(0, len(Eb), size=(B, len(Eb)))
    den = w @ COUNT
    ma = np.einsum("bg,bsg->bs", w, sa[ia]).mean(1) / den
    mb = np.einsum("bg,bsg->bs", w, sb[ib]).mean(1) / den
    d = ma - mb
    lo, hi = np.percentile(d, [2.5, 97.5])
    p = min(1.0, 2 * min((d <= 0).mean(), (d >= 0).mean()))
    return float(lo), float(hi), float(p)


def pfmt(p):
    return f"<{1 / opt.bootstrap:.0e}" if p == 0 else f"{p:.3g}"


def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, (len(p) - rank) * p[i])
        adj[i] = min(1.0, run)
    return adj


E_ = {m: per_event(m) for m in opt.models}
if opt.pairs:
    pairs = [tuple(x.split(",")) for x in opt.pairs]
elif opt.reference:
    pairs = [(m, opt.reference) for m in opt.models if m != opt.reference]
else:
    pairs = list(itertools.combinations(opt.models, 2))
rows = []
for a, b in pairs:
    ea, eb = E_[a].mean(0), E_[b].mean(0)
    d = ea - eb
    p = wilcoxon(ea, eb)[1] if np.any(d != 0) else 1.0
    pt = ttest_ind(E_[a].mean(1), E_[b].mean(1), equal_var=False)[1] if len(opt.seeds) > 1 else float("nan")
    lo, hi, pb = cluster_bootstrap(E_[a], E_[b])
    rows.append((a, b, E_[a].mean(), E_[b].mean(), d.mean(), np.median(d), p, pt, lo, hi, pb))
padj = holm([r[6] for r in rows])
out = [f"# Pairwise significance (anonymized test data, {E_[opt.models[0]].shape[1]} events, seeds {opt.seeds}, `{opt.sample_dir}`)\n",
       "RMSE = mean over events of the per-event RMSE, then mean over seeds. Wilcoxon: paired, on the seed-averaged per-event RMSE; "
       "p (Holm) corrects over all listed pairs. Welch: over the per-seed RMSE (small n). Negative difference = the first model is better.\n",
       f"Main test: two-level bootstrap over the {N_GAMES} test matches and the seeds ({opt.bootstrap} resamples). The Wilcoxon / Welch columns "
       "are for reference only (Wilcoxon treats the 2958 events of fixed predictions as independent, it is not a test across seeds).\n",
       "| A | B | RMSE A | RMSE B | mean diff | 95 % interval (matches x seeds) | p (bootstrap) | median diff | p (Wilcoxon, events) | p (Holm) | p (Welch, seeds) |",
       "|---|---|---|---|---|---|---|---|---|---|---|"]
for r, pa in zip(rows, padj):
    out.append(f"| {r[0]} | {r[1]} | {r[2]:.3f} | {r[3]:.3f} | {r[4]:+.4f} | [{r[8]:+.4f}, {r[9]:+.4f}] | {pfmt(r[10])} | {r[5]:+.4f} | {r[6]:.1e} | {pa:.1e} | {r[7]:.2g} |")
text = "\n".join(out) + "\n"
print(text)
if opt.out:
    open(opt.out, "w").write(text)
    print(f"Saved: {opt.out}")
