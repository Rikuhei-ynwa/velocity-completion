"""Errors by tactical situation (reviewer R2-5; R3 question 2 on counterattacks/pressing/sprints).

The data have no counterattack or pressing labels, so situations are defined from the 10-event
window of each test sample (target = last event), with explicit proxy definitions:
  phase      set_piece  : a restart (throw-in, corner, free kick, goal kick) by the team in
                          possession within the last 3 events, with no possession change since
             transition : possession changed within the last 3 events (counterattack proxy)
             open_play  : neither
  event      type of the target event: pass, dribble, cross, shot, restart, defensive
  intensity  number of players faster than 5.5 m/s at the target event (ground truth): 0, 1-2, >=3
For each group it reports the velocity RMSE (paper definition) and the PPCF/OBSO mean absolute
error against PPCF/OBSO computed with ground-truth velocities (obso_settings.py definitions), for
the rule-based baseline and any number of model runs (samples.p, mean +- std over seeds).
PPCF/OBSO per run are cached in --cache_dir, so adding runs does not recompute the others.

    python analysis/scenario_metrics.py                                   # rule-based only
    python analysis/scenario_metrics.py --date 202609301700 --models GRNN_add Transformer --seeds 0 1 2 3 4
"""

import argparse, hashlib, json, multiprocessing, os, pickle, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_based_check import velocity_part, worker_state, _init_worker, _ppcf_obso

RESTARTS = {"throw_in", "corner_crossed", "corner_short", "freekick_crossed", "freekick_short",
            "goalkick", "shot_freekick", "shot_penalty"}
DEFENSIVE = {"tackle", "interception", "clearance", "foul", "keeper_save"}
RECENT = 3  # events


def possession_team(onballs, direcs):
    """(N, T) attack direction (+-1) of the team in possession, 0 if nobody is on the ball."""
    has = onballs.sum(-1) > 0
    who = np.argmax(onballs, axis=-1)
    team = np.take_along_axis(direcs, who[..., None], axis=-1)[..., 0]
    return np.where(has, team, 0)


def situations(types, onballs, direcs, gt_speed):
    """Per-event group labels. types: (N, T) str; onballs/direcs: (N, T, 22); gt_speed: (N, 22)."""
    N, T = types.shape
    team = possession_team(onballs, direcs)
    phase = np.empty(N, dtype=object)
    for i in range(N):
        since_change, restart = None, False
        for k in range(1, T + 1):  # walk back from the target event
            t = T - k
            if k <= RECENT and types[i, t] in RESTARTS and team[i, t] in (0, team[i, -1]):
                restart = True
            if t > 0 and team[i, t] != 0 and team[i, t - 1] != 0 and team[i, t] != team[i, t - 1]:
                since_change = k - 1  # events since the possession change
                break
        if since_change is not None and since_change < RECENT:
            phase[i] = "transition"
        elif restart:
            phase[i] = "set_piece"
        else:
            phase[i] = "open_play"
    last = types[:, -1]
    event = np.array(["restart" if t in RESTARTS else "defensive" if t in DEFENSIVE else t
                      if t in ("pass", "dribble", "cross", "shot") else "other" for t in last], dtype=object)
    n_fast = (gt_speed > 5.5).sum(axis=1)
    intensity = np.where(n_fast == 0, "0 fast", np.where(n_fast <= 2, "1-2 fast", ">=3 fast")).astype(object)
    return {"phase": phase, "event": event, "intensity": intensity}


def per_event_rmse(pred, gt):
    return np.sqrt(np.sum((pred - gt) ** 2, axis=-1).mean(axis=1))


def ppcf_obso_errors(label, vels, state, gt_po, opt):
    """Per-event PPCF/OBSO MAE vs ground truth for one velocity set (cached)."""
    key = hashlib.sha1(np.ascontiguousarray(vels).tobytes()).hexdigest()[:16]
    path = os.path.join(opt.cache_dir, f"{label}_{key}.npz")
    if os.path.isfile(path):
        z = np.load(path)
        return z["ppcf"], z["obso"]
    ctx = multiprocessing.get_context(opt.start_method)
    t0 = time.time()
    with ctx.Pool(opt.n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        res = pool.map(_ppcf_obso, [(i, vels[i]) for i in range(len(vels))], chunksize=16)
    ppcf = np.array([np.mean(np.abs(P - gt_po[i][0])) for i, (P, _) in enumerate(res)])
    obso = np.array([np.mean(np.abs(O - gt_po[i][1])) for i, (_, O) in enumerate(res)])
    np.savez(path, ppcf=ppcf, obso=obso)
    print(f"  PPCF/OBSO for {label}: {time.time() - t0:.0f}s", flush=True)
    return ppcf, obso


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--date")
    cli.add_argument("--models", nargs="+", default=[])
    cli.add_argument("--seeds", nargs="+", type=int, default=[0])
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--n_jobs", type=int, default=8)
    cli.add_argument("--sample_dir", default="sample", help="experiments/<sample_dir>/samples.p: sample (train-mode test, as in the paper) or sample_eval (model.eval())")
    cli.add_argument("--start_method", default=None, choices=[None, "fork", "spawn", "forkserver"])
    cli.add_argument("--cache_dir", default="analysis/results/scenario_cache")
    cli.add_argument("--out", default="analysis/results/scenario_metrics.json")
    cli.add_argument("--per_event_out", default=None,
                     help="also save per-event velocity RMSE / PPCF MAE / OBSO MAE of every run to this .npz "
                          "(keys <label>_s<seed>_<metric>; the rule-based baseline is rule-based_s0_<metric>)")
    opt = cli.parse_args()
    os.makedirs(opt.cache_dir, exist_ok=True)

    arr, _ = velocity_part(argparse.Namespace(data_path=opt.data_path, n_sample=10), opt.dataset)
    vel_ball = arr["v_rb"][:, 0, :]
    gt_pl = arr["v_gt"]
    vels_gt = np.concatenate([vel_ball[:, None, :], gt_pl], axis=1)
    state = worker_state(opt.data_path, opt.dataset, arr)

    test = pickle.load(open(state["args"].data_test_path, "rb"))
    cat = lambda k: np.concatenate([test[g][k] for g in test])
    types = cat("types")[:, :, 0]
    groups = situations(types, cat("onballs")[..., 0], cat("direcs")[..., 0], np.linalg.norm(gt_pl, axis=-1))

    # ground-truth PPCF/OBSO once (cached as full surfaces would be large, so keep in memory)
    ctx = multiprocessing.get_context(opt.start_method)
    t0 = time.time()
    with ctx.Pool(opt.n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        gt_po = pool.map(_ppcf_obso, [(i, vels_gt[i]) for i in range(len(vels_gt))], chunksize=16)
    print(f"ground-truth PPCF/OBSO: {time.time() - t0:.0f}s", flush=True)

    runs = {"rule-based": {None: arr["v_rb"][:, 1:, :]}}
    for m in opt.models:
        for s in opt.seeds:
            p = os.path.join(opt.weights_dir, str(opt.date), f"{m}_events_{s}_rectangular_256",
                             "experiments", opt.sample_dir, "samples.p")
            others = pickle.load(open(p, "rb"))[2]
            if not np.allclose(np.mean(others["v_gt"], axis=0), gt_pl, atol=1e-5):
                sys.exit(f"{p}: ground truth differs from {opt.dataset} (wrong dataset?)")
            runs.setdefault(m, {})[s] = np.mean(others["v_pred"], axis=0)

    per_event = {}  # label -> list over seeds of dict(metric -> (N,))
    for label, by_seed in runs.items():
        per_event[label] = []
        for s, pred in by_seed.items():
            vels = np.concatenate([vel_ball[:, None, :], pred], axis=1)
            ppcf, obso = ppcf_obso_errors(f"{label}_s{s}", vels, state, gt_po, opt)
            per_event[label].append({"vel_rmse": per_event_rmse(pred, gt_pl), "ppcf_mae": ppcf, "obso_mae": obso})

    if opt.per_event_out:
        flat = {}
        for label, seeds_ in per_event.items():
            for k, r in zip(runs[label], seeds_):
                for metric, v in r.items():
                    flat[f"{label}_s{0 if k is None else k}_{metric}"] = v
        np.savez(opt.per_event_out, **flat)
        print(f"Saved per-event errors: {opt.per_event_out}")

    results = {}
    for gname, lab in groups.items():
        results[gname] = {}
        for g in sorted(set(lab), key=lambda x: -np.sum(lab == x)):
            m = lab == g
            results[gname][g] = {"n": int(m.sum())}
            for label, seeds in per_event.items():
                results[gname][g][label] = {k: [float(r[k][m].mean()) for r in seeds] for k in seeds[0]}

    labels = list(per_event)
    fmt = lambda v: f"{np.mean(v):.4f}" if len(v) == 1 else f"{np.mean(v):.4f}±{np.std(v, ddof=1):.4f}"
    for metric, title in (("vel_rmse", "velocity RMSE (m/s)"), ("ppcf_mae", "PPCF MAE"), ("obso_mae", "OBSO MAE")):
        print(f"\n{title}")
        print(f"  {'group':<22}{'n':>6}" + "".join(f"{l:>18}" for l in labels))
        for gname in groups:
            for g, r in results[gname].items():
                digits = 6 if metric == "obso_mae" else 4
                cells = "".join(f"{(fmt(r[l][metric]) if digits == 4 else f'{np.mean(r[l][metric]):.6f}'):>18}"
                                for l in labels)
                print(f"  {gname + ': ' + g:<22}{r['n']:>6}" + cells)

    os.makedirs(os.path.dirname(opt.out) or ".", exist_ok=True)
    json.dump({"dataset": opt.dataset, "definitions": {"recent_events": RECENT, "restarts": sorted(RESTARTS),
               "defensive": sorted(DEFENSIVE), "fast_mps": 5.5},
               "group_sizes": {g: {k: int(np.sum(v == k)) for k in set(v)} for g, v in groups.items()},
               "results": results}, open(opt.out, "w"), indent=1)
    print(f"\nSaved: {opt.out}")


if __name__ == "__main__":
    main()
