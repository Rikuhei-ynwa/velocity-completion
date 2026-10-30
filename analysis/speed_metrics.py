"""Speed-stratified evaluation of completed velocities from samples.p (reviewer requests R2-4, R3, A1).

For each run (model x seed) it reports, from the test-set predictions in samples.p:
  - overall RMSE as in the paper (mean over events of the per-event RMSE over players)
  - RMSE, mean/median angle error and mean speed bias (predicted - true speed) per
    ground-truth speed band, pooled over player observations
  - the rule-based baseline (v_rb) with the same metrics
Across seeds it reports mean +- std, and between models a paired Wilcoxon signed-rank test on
per-event RMSE (seed-averaged per event), which answers "is the difference significant".

Usage (from the repo root, where <results_root> is visible):
    python analysis/speed_metrics.py --date 202610011200 --models GRNN_add GRNN_add_hs5 Transformer --seeds 0 1 2 3 4
    python analysis/speed_metrics.py --samples path/to/samples.p [...]   # explicit files, one run each
"""

import argparse, itertools, json, os, pickle, sys

import numpy as np
from scipy.stats import wilcoxon

BANDS = {  # ground-truth speed bands (m/s); 5.5 and 7 follow the paper's high-speed / sprint definitions
    "all": (0.0, np.inf), "0-1": (0.0, 1.0), "1-3": (1.0, 3.0), "3-5.5": (3.0, 5.5),
    "<=5.5": (0.0, 5.5), ">5.5": (5.5, np.inf), ">7": (7.0, np.inf),
}
EPS = 1e-8  # as models/utils.angular_error


def band_mask(speed, lo, hi):
    return (speed > lo) & (speed <= hi) if lo > 0 else speed <= hi


def metrics(pred, gt):
    """pred, gt: (N, 22, 2) in m/s -> dict of metrics and per-event RMSE (N,)."""
    se = np.sum((pred - gt) ** 2, axis=-1)  # (N, 22)
    per_event_rmse = np.sqrt(se.mean(axis=1))
    s_true = np.linalg.norm(gt, axis=-1)
    s_pred = np.linalg.norm(pred, axis=-1)
    cos = np.sum(pred * gt, axis=-1) / (np.sqrt(s_pred ** 2 + EPS) * np.sqrt(s_true ** 2 + EPS))
    ang = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
    out = {"rmse_event_mean": float(per_event_rmse.mean())}
    for name, (lo, hi) in BANDS.items():
        # The stored ground truth differs by ~1e-7 between single-timestep and sequence models, which moves
        # observations at exactly 7.0 m/s in or out of the ">7" band (522 vs 524); round to make the bands identical
        m = band_mask(np.round(s_true, 4), lo, hi)
        n = int(m.sum())
        out[name] = {
            "n": n,
            "rmse": float(np.sqrt(se[m].mean())) if n else float("nan"),
            "angle_mean": float(ang[m].mean()) if n else float("nan"),
            "angle_median": float(np.median(ang[m])) if n else float("nan"),
            "speed_bias": float((s_pred[m] - s_true[m]).mean()) if n else float("nan"),
        }
    return out, per_event_rmse


def load(path):
    others = pickle.load(open(path, "rb"))[2]
    mean = lambda k: np.mean(others[k], axis=0)  # average over test runs
    # VRNN does not store frames in samples.p; the ground truth is still compared across runs
    frames = mean("frames") if "frames" in others else None
    return mean("v_pred"), mean("v_gt"), mean("v_rb")[:, 1:, :], frames


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--date", help="date_experiment of the runs")
    cli.add_argument("--models", nargs="+", default=[], help="directory model names, e.g. GRNN_add_hs5")
    cli.add_argument("--seeds", nargs="+", type=int, default=[0])
    cli.add_argument("--samples", nargs="+", default=[], help="explicit samples.p files (label = parent dirs)")
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--data", default="events")
    cli.add_argument("--batchsize", type=int, default=256)
    cli.add_argument("--out", default="analysis/results/speed_metrics.json")
    cli.add_argument("--sample_dir", default="sample", help="experiments/<sample_dir>/samples.p: sample (train-mode test, as in the paper) or sample_eval (model.eval())")
    opt = cli.parse_args()

    runs = {}  # label -> {seed: path}
    for m in opt.models:
        for s in opt.seeds:
            p = os.path.join(opt.weights_dir, str(opt.date), f"{m}_{opt.data}_{s}_rectangular_{opt.batchsize}",
                             "experiments", opt.sample_dir, "samples.p")
            runs.setdefault(m, {})[s] = p
    for p in opt.samples:
        runs.setdefault(os.path.basename(os.path.dirname(os.path.dirname(os.path.dirname(p)))), {})[0] = p
    missing = [p for r in runs.values() for p in r.values() if not os.path.isfile(p)]
    if missing:
        sys.exit("missing samples.p:\n  " + "\n  ".join(missing))

    results, per_event, gt_ref, frames_ref = {}, {}, None, None
    for label, by_seed in runs.items():
        seed_metrics, seed_events = [], []
        for s, p in sorted(by_seed.items()):
            pred, gt, rb, frames = load(p)
            if gt_ref is None:
                gt_ref, frames_ref = gt, frames
                results["rule-based"], rb_events = metrics(rb, gt)
                per_event["rule-based"] = rb_events
            elif not (np.allclose(gt, gt_ref, atol=1e-5)
                      and (frames is None or frames_ref is None or np.array_equal(frames, frames_ref))):
                sys.exit(f"{label} seed {s}: test set differs from the first run (different dataset?)")
            if frames_ref is None and frames is not None:
                frames_ref = frames
            m, ev = metrics(pred, gt)
            seed_metrics.append(m)
            seed_events.append(ev)
        results[label] = {"seeds": sorted(by_seed), "per_seed": seed_metrics}
        per_event[label] = np.mean(seed_events, axis=0)

    # ---- report ----
    def get(r, band, key):
        if "per_seed" not in r:
            return np.array([r["rmse_event_mean"] if band is None else r[band][key]])
        return np.array([m["rmse_event_mean"] if band is None else m[band][key] for m in r["per_seed"]])

    def fmt(v):
        return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f}±{v.std(ddof=1):.3f}"

    labels = list(results)
    print(f"test events: {len(frames_ref)}   (mean±std over seeds where >1 seed)")
    print(f"\n{'':<14}" + "".join(f"{l:>18}" for l in labels))
    print(f"{'RMSE (paper)':<14}" + "".join(f"{fmt(get(results[l], None, None)):>18}" for l in labels))
    for key, title in (("rmse", "RMSE"), ("angle_mean", "angle mean"), ("angle_median", "angle median"),
                       ("speed_bias", "speed bias")):
        print(f"\n{title} by true speed [m/s]   (n player-observations: "
              + ", ".join(f"{b}={results['rule-based'][b]['n']}" for b in BANDS) + ")")
        for b in BANDS:
            print(f"  {b:<12}" + "".join(f"{fmt(get(results[l], b, key)):>18}" for l in labels))

    tests = []
    print("\nPaired Wilcoxon signed-rank test on per-event RMSE (seed-averaged):")
    for a, b in itertools.combinations(labels, 2):
        stat, pval = wilcoxon(per_event[a], per_event[b])
        diff = float(np.median(per_event[a] - per_event[b]))
        tests.append({"a": a, "b": b, "median_diff": diff, "statistic": float(stat), "p_value": float(pval)})
        ptxt = f"p = {pval:.3g}" if pval > 0 else "p < 1e-300"  # underflows to 0 for huge effects
        print(f"  {a} vs {b}: median diff {diff:+.4f} m/s, {ptxt}")

    os.makedirs(os.path.dirname(opt.out) or ".", exist_ok=True)
    json.dump({"bands": {k: list(v) for k, v in BANDS.items()}, "runs": {l: {str(s): p for s, p in r.items()}
               for l, r in runs.items()}, "results": results, "wilcoxon": tests},
              open(opt.out, "w"), indent=1, default=float)
    print(f"\nSaved: {opt.out}")


if __name__ == "__main__":
    main()
