"""Position-difference ("position first") baselines for velocity completion (reviewer R2-1).

The reviewer asks for a quantitative comparison with computing the velocity from positions. At test time
only event data are used: the positions of the 22 players at the target event and at the previous event of
the window, and the frame times of the two events (25 Hz). No tracking data.

    v_j ~ (x_j(t) - x_j(t-1)) / dt          (backward difference over the previous inter-event interval)

The methods differ in how the player of the target event is associated with the player of the previous event:

  identity       the true player (restored through <split>_permutations.npy; the original slot order for an
                 ID-ful dataset). An upper bound, it needs the player identity
  hungarian      one-to-one assignment inside each team that minimises the total distance (positions only)
  nearest        nearest player of the previous event (many-to-one allowed)
  team_centroid  the displacement of the team centroid is used for every player of the team
  rank_x         the i-th player along the pitch length is paired with the i-th of the previous event
  same_slot      slot k with slot k (what a model without association would do; slots are re-randomised in
                 the anonymized data, so this is meaningless there)
  zero           zero velocity (reference)
  rule_based     the rule-based velocities of the paper (the baseline of the submitted paper), stored in the dataset

Reported (test set, paper definitions): velocity RMSE (mean over events of the per-event RMSE), mean angle error,
speed bias, RMSE by true-speed band (speed_metrics.BANDS), by the time gap to the previous event, by tactical
situation (scenario_metrics.situations), the share of players associated correctly, and with --ppcf the
PPCF/OBSO error against PPCF/OBSO computed with the ground-truth velocities. Trained models can be added with
--models LABEL:DATE (mean over seeds) to put them in the same gap / band / situation tables.

Self check: the ground truth rebuilt here must equal v_gt of a model's samples.p, and the event RMSE computed here
from that model's v_pred must equal the L_rec2 it logged.

    python analysis/interp_baseline.py --models Transformer_add:202610030117 GRNN_add:202610021638 --ppcf
"""

import argparse, json, multiprocessing, os, pickle, sys, time

import numpy as np
from scipy.optimize import linear_sum_assignment

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
import event_config as cfg
from speed_metrics import BANDS, EPS, metrics

FPS = 25.0  # tracking frame rate; "frames" of the dataset are frame numbers
TEAMS = (np.arange(0, 11), np.arange(11, 22))  # slots of the home and away players
GAP_BINS = [(0, 1), (1, 2), (2, 3), (3, 5), (5, np.inf)]  # seconds to the previous event
METHODS = ["rule_based", "identity", "hungarian", "nearest", "team_centroid", "rank_x", "same_slot", "zero"]


# ---------------------------------------------------------------- data
def load_split(dataset, split, data_path):
    """Positions [m] of the 22 players at every event of the window, the ground-truth velocity [m/s] of the
    last (target) event, the time gap to the previous event and the identity of every slot."""
    d = os.path.join(data_path, "dataset", dataset, "rectangular", "recurrent")
    z = np.load(os.path.join(d, split + ".npz"), allow_pickle=True)
    cies = z["cies"].astype(np.float64)
    frames = z["frames"][:, :, 0].astype(np.float64)
    vs = z["vel_stats"]  # row 1 = players: mean_x, std_x, mean_y, std_y
    L, V = len(cfg.LOC_NODES), len(cfg.VEL_NODES)
    pos = cies[:, :, 2:24, :].copy()
    pos[..., 0] *= cfg.FIELD_LENGTH / 2
    pos[..., 1] *= cfg.FIELD_WIDTH / 2
    rb = cies[:, -1, L + V:, :].copy()  # rule-based player velocities [m/s] of the target event (not normalised)
    ball = cies[:, -1, 0, :].copy()
    ball[:, 0] *= cfg.FIELD_LENGTH / 2
    ball[:, 1] *= cfg.FIELD_WIDTH / 2
    ball_prev = cies[:, -2, 0, :].copy()  # ball at the start of the previous event
    ball_prev[:, 0] *= cfg.FIELD_LENGTH / 2
    ball_prev[:, 1] *= cfg.FIELD_WIDTH / 2
    bend = cies[:, -1, 1, :].copy()
    bend[:, 0] *= cfg.FIELD_LENGTH / 2
    bend[:, 1] *= cfg.FIELD_WIDTH / 2
    gt = cies[:, -1, L + 1:L + V, :].copy()
    gt[..., 0] = gt[..., 0] * vs[1, 1] + vs[1, 0]
    gt[..., 1] = gt[..., 1] * vs[1, 3] + vs[1, 2]
    dt = (frames[:, -1] - frames[:, -2]) / FPS
    valid = (frames[:, -2] > 0) & (dt > 0)  # frame 0 = padding at the start of a match: no previous event
    pp = os.path.join(d, split + "_permutations.npy")
    if os.path.isfile(pp):  # anonymized: new slot k holds the original player P[k] (ball = slot 0)
        ident = np.load(pp)[:, :, 1:].astype(np.int64) - 1
    else:  # ID-ful: the slot is the player
        ident = np.broadcast_to(np.arange(22), (len(cies), cies.shape[1], 22)).copy()
    return dict(pos=pos, ball=ball, ball_prev=ball_prev, bend=bend, gt=gt, rb=rb, frames=frames, dt=dt, valid=valid, ident=ident,
                event_type=z["eventtypes"][:, -1].argmax(-1), vel_stats=vs)


# ---------------------------------------------------------------- baselines
def _scale(dt):
    return np.maximum(dt, 1e-6)[:, None, None]


def displacement(d, method):
    """(N, 22, 2) displacement [m] from the previous event, and the per-event share of correctly associated
    players (NaN where the method does not associate players)."""
    pos, valid, ident = d["pos"], d["valid"], d["ident"]
    N = len(pos)
    disp, acc = np.zeros((N, 22, 2)), np.full(N, np.nan)
    if method == "same_slot":
        disp = pos[:, -1] - pos[:, -2]
    elif method == "identity":
        last, prev = np.zeros((N, 22, 2)), np.zeros((N, 22, 2))
        for i in range(N):
            last[i, ident[i, -1]] = pos[i, -1]
            prev[i, ident[i, -2]] = pos[i, -2]
        # displacement of every original player, read back in the slot order of the target event
        disp = np.take_along_axis(last - prev, ident[:, -1][..., None].repeat(2, -1), axis=1)
    elif method == "team_centroid":
        for t in TEAMS:
            disp[:, t] = (pos[:, -1, t].mean(1) - pos[:, -2, t].mean(1))[:, None, :]
    elif method == "rank_x":
        for i in range(N):
            for t in TEAMS:
                o1, o0 = np.argsort(pos[i, -1, t][:, 0]), np.argsort(pos[i, -2, t][:, 0])
                disp[i, t[o1]] = pos[i, -1, t][o1] - pos[i, -2, t][o0]
    elif method in ("hungarian", "nearest"):
        for i in np.where(valid)[0]:
            ok = 0
            for t in TEAMS:
                P0, P1 = pos[i, -2, t], pos[i, -1, t]
                C = np.linalg.norm(P1[:, None, :] - P0[None, :, :], axis=-1)  # target player x previous player
                if method == "hungarian":
                    r, c = linear_sum_assignment(C)
                else:
                    r, c = np.arange(len(t)), C.argmin(1)
                disp[i, t[r]] = P1[r] - P0[c]
                ok += int((ident[i, -1][t][r] == ident[i, -2][t][c]).sum())
            acc[i] = ok / 22
    elif method != "zero":
        raise ValueError(method)
    if method == "zero":
        disp = np.zeros((N, 22, 2))
    return disp, acc


def baseline_velocity(d, method):
    if method == "rule_based":
        return d["rb"].copy(), np.full(len(d["rb"]), np.nan)
    disp, acc = displacement(d, method)
    v = np.zeros_like(disp) if method == "zero" else disp / _scale(d["dt"])
    v[~d["valid"]] = 0.0
    return v, acc


# ---------------------------------------------------------------- metrics
def angle_deg(p, g):
    """Per-player angle [deg] between predicted and true velocity, as models/utils.angular_error."""
    dot = (p * g).sum(-1)
    pm, tm = np.sqrt((p ** 2).sum(-1) + EPS), np.sqrt((g ** 2).sum(-1) + EPS)
    return np.degrees(np.arccos(np.clip(dot / (pm * tm + EPS), -1.0, 1.0)))


def event_rmse(p, g):
    return np.sqrt(((p - g) ** 2).sum(-1).mean(-1))


def per_event(v, gt):
    return event_rmse(v, gt), angle_deg(v, gt).mean(-1)


def summarize(v, gt, mask):
    out, _ = metrics(v[mask], gt[mask])
    ev, an = per_event(v[mask], gt[mask])
    return {"rmse": float(ev.mean()), "angle": float(an.mean()), "bands": out}


# ---------------------------------------------------------------- trained models
def load_model_runs(label, weights_dir, seeds):
    model, date = label.split(":")
    runs = []
    for s in seeds:
        p = os.path.join(weights_dir, date, f"{model}_events_{s}_rectangular_256", "experiments", "sample_eval", "samples.p")
        d = pickle.load(open(p, "rb"))
        runs.append({"v_pred": np.mean(d[2]["v_pred"], 0), "v_gt": np.mean(d[2]["v_gt"], 0), "L_rec2": float(d[3][0]["L_rec2"])})
    return model, runs


# ---------------------------------------------------------------- PPCF / OBSO
def ppcf_obso(vels, dataset, data_path, n_jobs, cache_dir, start_method, gt_check):
    """Per-event PPCF / OBSO mean absolute error against PPCF/OBSO of the ground-truth velocities
    (the definitions of obso_settings.py, as analysis/scenario_metrics.py)."""
    from rule_based_check import velocity_part, worker_state, _init_worker, _ppcf_obso
    from scenario_metrics import ppcf_obso_errors

    arr, _ = velocity_part(argparse.Namespace(data_path=data_path, n_sample=10), dataset)
    assert np.allclose(arr["v_gt"], gt_check, atol=1e-4), "ground truth of the PPCF/OBSO pipeline differs"
    vel_ball = arr["v_rb"][:, 0, :]
    state = worker_state(data_path, dataset, arr)
    full = lambda v: np.concatenate([vel_ball[:, None, :], v], axis=1)
    os.makedirs(cache_dir, exist_ok=True)
    opt = argparse.Namespace(cache_dir=cache_dir, start_method=start_method, n_jobs=n_jobs)
    ctx = multiprocessing.get_context(start_method)
    with ctx.Pool(n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        gt_po = pool.map(_ppcf_obso, [(i, full(gt_check)[i]) for i in range(len(gt_check))], chunksize=16)
    out = {}
    for name, v in vels.items():
        p, o = ppcf_obso_errors(f"interp_{name}", full(v), state, gt_po, opt)
        out[name] = (p, o)
    return out


def situation_labels(dataset, data_path, gt):
    from scenario_metrics import situations

    test = pickle.load(open(os.path.join(data_path, "dataset", dataset, "rectangular", "recurrent", "test.pkl"), "rb"))
    cat = lambda k: np.concatenate([test[g][k] for g in test])
    return situations(cat("types")[:, :, 0], cat("onballs")[..., 0], cat("direcs")[..., 0], np.linalg.norm(gt, axis=-1))


# ---------------------------------------------------------------- main
def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--methods", nargs="+", default=METHODS, choices=METHODS)
    cli.add_argument("--models", nargs="*", default=[], help="MODEL:DATE of trained runs (mean over --seeds) to compare with")
    cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--ppcf", action="store_true", help="also PPCF/OBSO errors of the baselines (about 45 s per method on 16 cores)")
    cli.add_argument("--model_per_event", nargs="*", default=[],
                     help=".npz files written by analysis/scenario_metrics.py --per_event_out; the PPCF/OBSO of the --models "
                          "(mean over seeds, keys <MODEL>_s<seed>_ppcf_mae / _obso_mae) are read from them")
    cli.add_argument("--n_jobs", type=int, default=16)
    cli.add_argument("--start_method", default=None, choices=[None, "fork", "spawn", "forkserver"])
    cli.add_argument("--cache_dir", default="analysis/results/scenario_cache")
    cli.add_argument("--out", default=None)
    opt = cli.parse_args()
    tag = opt.dataset
    out_json = opt.out or f"analysis/results/interp_baseline_{tag}.json"

    d = load_split(opt.dataset, "test", opt.data_path)
    gt, valid, dt = d["gt"], d["valid"], d["dt"]
    N = len(gt)
    print(f"dataset {opt.dataset}: {N} test events, {int(valid.sum())} with a previous event ({valid.mean():.1%})")

    # ---- self check against a trained model
    runs = {}
    for lab in opt.models:
        runs[lab] = load_model_runs(lab, opt.weights_dir, opt.seeds)[1]
    if runs:
        r0 = next(iter(runs.values()))[0]
        diff = float(np.abs(r0["v_gt"] - gt).max())
        ev = float(event_rmse(r0["v_pred"], r0["v_gt"]).mean())
        print(f"self check: max |v_gt(samples.p) - ground truth rebuilt here| = {diff:.2e}; "
              f"event RMSE recomputed {ev:.5f} vs logged L_rec2 {r0['L_rec2']:.5f}")
        assert diff < 1e-4 and abs(ev - r0["L_rec2"]) < 1e-4, "self check failed"

    # ---- baselines
    vel, acc = {}, {}
    t0 = time.time()
    for m in opt.methods:
        vel[m], acc[m] = baseline_velocity(d, m)
    print(f"baselines computed in {time.time() - t0:.0f}s")
    res = {m: summarize(vel[m], gt, valid) for m in vel}
    ev = {m: per_event(vel[m], gt) for m in vel}  # (rmse, angle) per event (invalid events carry zero velocity)

    # ---- trained models: mean over seeds of the same quantities
    model_ev, model_res = {}, {}
    for lab, rs in runs.items():
        model_ev[lab] = tuple(np.mean([per_event(r["v_pred"], gt)[k] for r in rs], axis=0) for k in (0, 1))
        sums = [summarize(r["v_pred"], gt, valid) for r in rs]
        model_res[lab] = {"rmse": float(np.mean([s["rmse"] for s in sums])), "angle": float(np.mean([s["angle"] for s in sums])),
                          "bands": {b: {k: float(np.mean([s["bands"][b][k] for s in sums])) for k in sums[0]["bands"][b] if k != "n"}
                                    | {"n": sums[0]["bands"][b]["n"]} for b in BANDS}}

    # ---- PPCF / OBSO
    po = {}
    if opt.ppcf:
        po = ppcf_obso({m: vel[m] for m in opt.methods if m != "same_slot"}, opt.dataset, opt.data_path, opt.n_jobs,
                       opt.cache_dir, opt.start_method, gt)

    for lab in runs:  # PPCF/OBSO of the trained models: already computed by scenario_metrics, read from its per-event files
        model = lab.split(":")[0]
        for path in opt.model_per_event:
            z = np.load(path)
            keys = [f"{model}_s{s}_{k}" for s in opt.seeds for k in ("ppcf_mae", "obso_mae")]
            if all(k in z.files for k in keys):
                po[lab] = tuple(np.mean([z[f"{model}_s{s}_{k}"] for s in opt.seeds], axis=0) for k in ("ppcf_mae", "obso_mae"))

    # ---- print
    allev = {**{m: ev[m] for m in vel}, **model_ev}
    print(f"\n{'method':<28}{'RMSE':>8}{'angle':>8}{'RMSE>5.5':>10}{'RMSE>7':>8}{'bias>5.5':>9}{'assoc.':>8}"
          + (f"{'PPCF':>9}{'OBSO':>11}" if po else ""))
    for m in vel:
        s = res[m]
        a = np.nanmean(acc[m][valid]) if not np.all(np.isnan(acc[m])) else float("nan")
        line = (f"{m:<28}{s['rmse']:>8.3f}{s['angle']:>7.1f}°{s['bands']['>5.5']['rmse']:>10.3f}{s['bands']['>7']['rmse']:>8.3f}"
                f"{s['bands']['>5.5']['speed_bias']:>9.2f}{'' if np.isnan(a) else format(a, '.1%'):>8}")
        if po:
            line += f"{po[m][0].mean():>9.4f}{po[m][1].mean():>11.6f}" if m in po else f"{'':>20}"
        print(line)
    for lab, s in model_res.items():
        line = (f"{lab:<28}{s['rmse']:>8.3f}{s['angle']:>7.1f}°{s['bands']['>5.5']['rmse']:>10.3f}{s['bands']['>7']['rmse']:>8.3f}"
                f"{s['bands']['>5.5']['speed_bias']:>9.2f}{'':>8}")
        if po:
            line += f"{po[lab][0].mean():>9.4f}{po[lab][1].mean():>11.6f}" if lab in po else f"{'':>20}"
        print(line)

    name_of = lambda lo, hi: f"{lo}-{hi}s" if np.isfinite(hi) else f">{lo}s"
    gap_label = np.array([next((name_of(lo, hi) for lo, hi in GAP_BINS if lo < g <= hi), "n/a") if v else "n/a"
                          for g, v in zip(dt, valid)])
    gap_order = [name_of(lo, hi) for lo, hi in GAP_BINS]
    cols = [m for m in ("rule_based", "identity", "hungarian", "nearest", "team_centroid", "rank_x", "zero") if m in vel] + list(runs)
    print("\nby time gap to the previous event: mean event RMSE / mean angle")
    print(f"{'gap':<9}{'n':>6}{'assoc.':>8}" + "".join(f"{c[:20]:>26}" for c in cols))
    gaps = {}
    for g in gap_order:
        m = gap_label == g
        a = np.nanmean(acc["hungarian"][m]) if "hungarian" in acc and m.any() else float("nan")
        gaps[g] = {"n": int(m.sum()), "hungarian_assoc": float(a)} | {c: [float(allev[c][0][m].mean()), float(allev[c][1][m].mean())] for c in cols}
        print(f"{g:<9}{m.sum():>6}{a:>8.1%}" + "".join(f"{allev[c][0][m].mean():>14.3f}{allev[c][1][m].mean():>11.1f}°" for c in cols))

    groups_out = {}
    try:
        grp = situation_labels(opt.dataset, opt.data_path, gt)
        print("\nby situation: mean event RMSE" + (" / PPCF MAE" if po else ""))
        for gname, lab in grp.items():
            groups_out[gname] = {}
            for g in sorted(set(lab), key=lambda x: -np.sum(lab == x)):
                m = (lab == g) & valid
                row = {c: float(allev[c][0][m].mean()) for c in cols}
                if po:
                    row |= {f"{c}_ppcf": float(po[c][0][m].mean()) for c in cols if c in po}
                groups_out[gname][g] = {"n": int(m.sum())} | row
                print(f"  {gname + ': ' + g:<24}{m.sum():>5}" + "".join(f"{row[c]:>10.3f}" for c in cols)
                      + ("".join(f"{row[c + '_ppcf']:>9.4f}" for c in cols if c + "_ppcf" in row) if po else ""))
    except Exception as e:  # the situation labels need the test .pkl and the scenario code
        print(f"\n(situation table skipped: {e})")

    bands_out = {b: {m: res[m]["bands"][b] for m in vel} | {lab: model_res[lab]["bands"][b] for lab in model_res} for b in BANDS}
    json.dump({"dataset": opt.dataset, "n_events": N, "n_with_previous_event": int(valid.sum()),
               "overall": {**{m: {k: v for k, v in res[m].items() if k != "bands"} | {"assoc": None if np.all(np.isnan(acc[m])) else float(np.nanmean(acc[m][valid]))} for m in vel},
                           **{lab: {"rmse": s["rmse"], "angle": s["angle"]} for lab, s in model_res.items()}},
               "bands": bands_out, "gaps": gaps, "situations": groups_out,
               "ppcf_obso": {m: {"ppcf": float(p.mean()), "obso": float(o.mean())} for m, (p, o) in po.items()}},
              open(out_json, "w"), indent=1, default=float)
    flat = {f"{m}_{k}": v for m in vel for k, v in zip(("rmse", "angle"), ev[m])}
    flat |= {f"{m}_{k}": po[m][j] for m in po for j, k in enumerate(("ppcf", "obso"))}
    flat["dt"], flat["valid"] = dt, valid
    np.savez(out_json.replace(".json", "_per_event.npz"), **flat)
    print(f"\nSaved: {out_json}")


if __name__ == "__main__":
    main()
