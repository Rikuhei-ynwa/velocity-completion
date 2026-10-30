"""Check that the rule-based baseline results are reproducible, on the original and anonymized datasets.

Recomputes, without any trained model, the rule-based rows of the paper:
  Table 2 : velocity RMSE                          (main_completion.py L_efobso)
  Table 9 : angle error, x/y axis RMSE             (L_angular_rb, L_axis_x_rb, L_axis_y_rb)
  Table 5 : Er_ppcf / Er_obso mean, min, max       (rule-based vs ground-truth velocities)
The ground-truth and rule-based arrays are rebuilt from test.npz exactly as models/grnn.py
sample(TEST=True) and main_completion.py write them to samples.p. The rule-based baseline has
no learned parameters, so relabeling players (anonymized dataset) must not change any number.

Every metric is checked against the original runs' values and, across datasets, against the
first dataset; the script exits with status 1 if any check fails.

Run from the repo root:
    python analysis/rule_based_check.py --datasets source anonymized
"""

import argparse, json, multiprocessing, os, pickle, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
import event_config as cfg
from datasets import EventDataset
from models.utils import loss_reconst, angular_error, axis_error
import DRSO.obso_player as obs
import metrica.Metrica_PitchControl as mpc
import obso_settings

LOC_NODES, VEL_NODES = cfg.LOC_NODES, cfg.VEL_NODES
N_PLAYERS = 22

# values printed by the original runs (logs/test_*.log, slurm-6787096.out) that the paper reports
REFERENCE = {
    "vel_rmse": 4.507522132126778, "angle_deg": 67.76, "axis_x": 3.709160714926392, "axis_y": 2.5010176184529924,
    "ppcf_mean": 0.057033352998192705, "ppcf_min": 0.019613834215182452, "ppcf_max": 0.15646158724003206,
    "obso_mean": 0.0008167412553948271, "obso_min": 8.615289947319811e-05, "obso_max": 0.0059447475170701855,
}
# (kind, tolerance) against REFERENCE. Velocity metrics come from float32 torch ops whose
# summation order differs between the original GPU run and CPU (observed ~1e-8 relative);
# the reference angle was printed with 2 decimals; PPCF/OBSO are float64 and match exactly.
TOL_REFERENCE = {
    "vel_rmse": ("rel", 1e-6), "axis_x": ("rel", 1e-6), "axis_y": ("rel", 1e-6), "angle_deg": ("abs", 0.005),
    "ppcf_mean": ("rel", 1e-9), "ppcf_min": ("rel", 1e-9), "ppcf_max": ("rel", 1e-9),
    "obso_mean": ("rel", 1e-9), "obso_min": ("rel", 1e-9), "obso_max": ("rel", 1e-9),
}
TOL_ACROSS = 1e-6  # relative, between datasets (only float32 summation order changes)


def velocity_part(opt, date):
    """Rebuild samples.p arrays and the rule-based velocity metrics for one dataset."""
    dataset_args = argparse.Namespace(data_path=opt.data_path, batchsize=256, model="GRNN_add",
                                 n_all_agents=23, date_preprocessed_data=date)
    dataset = EventDataset(dataset_args, train=-1)
    loader = DataLoader(dataset, batch_size=256, shuffle=False, drop_last=False)
    players_info = dataset.players_vel_info
    ball_info = dataset.ball_vel_info.numpy()
    N = len(dataset)
    others = {k: np.zeros((opt.n_sample, N, 23 if k in ("loc_gt", "v_rb") else 22, 2))
              for k in ("loc_gt", "v_rb", "v_gt")}
    per_event = {k: np.zeros((opt.n_sample, N)) for k in ("L_efobso", "L_angular_rb", "L_axis_x_rb", "L_axis_y_rb")}
    for r in range(opt.n_sample):
        start = 0
        for items in loader:
            cies = items["cies"].permute(1, 0, 2, 3).clone()
            B, t = cies.size(1), cies.size(0) - 1
            x_t = torch.cat([cies[t, :, 0:1, :], cies[t, :, 2:len(LOC_NODES), :]], dim=1)
            v_t = cies[t, :, len(LOC_NODES):len(LOC_NODES) + len(VEL_NODES), :]
            gt = v_t[:, 1:, :]  # self.restore, in place
            gt[:, :, 0] = gt[:, :, 0] * players_info[1] + players_info[0]
            gt[:, :, 1] = gt[:, :, 1] * players_info[3] + players_info[2]
            rb = cies[t, :, len(LOC_NODES + VEL_NODES):, :]
            sl = slice(start, start + B)
            per_event["L_efobso"][r, sl] = loss_reconst(rb, gt, N_PLAYERS, Sum=False).numpy()
            per_event["L_angular_rb"][r, sl] = angular_error(rb, gt, N_PLAYERS, Sum=False).numpy()
            per_event["L_axis_x_rb"][r, sl] = axis_error(rb, gt, N_PLAYERS, axis="x", Sum=False).numpy()
            per_event["L_axis_y_rb"][r, sl] = axis_error(rb, gt, N_PLAYERS, axis="y", Sum=False).numpy()
            pos = x_t.numpy()
            pos[:, :, 0] *= cfg.FIELD_LENGTH / 2
            pos[:, :, 1] *= cfg.FIELD_WIDTH / 2
            v_rb = np.concatenate((np.expand_dims(v_t[:, 0, :].numpy(), 1), rb.numpy()), axis=1).astype(np.float64)
            v_rb[:, 0, 0] = v_rb[:, 0, 0] * ball_info[1] + ball_info[0]
            v_rb[:, 0, 1] = v_rb[:, 0, 1] * ball_info[3] + ball_info[2]
            others["v_rb"][r, sl] += v_rb
            others["v_gt"][r, sl] += gt.numpy()
            others["loc_gt"][r, sl] += pos
            start += B
    # main_completion.py: per-run mean over events (sqrt for RMSE), then mean over runs
    run = lambda k, f: np.mean([np.mean(f(per_event[k][r])) for r in range(opt.n_sample)])
    metrics = {"vel_rmse": run("L_efobso", np.sqrt), "angle_deg": run("L_angular_rb", lambda x: x),
               "axis_x": run("L_axis_x_rb", np.sqrt), "axis_y": run("L_axis_y_rb", np.sqrt)}
    return {k: np.mean(v, axis=0) for k, v in others.items()}, metrics


# ---- PPCF/OBSO workers: module-level so they work with both fork and spawn ----
_W = {}


def _init_worker(state):
    _W.update(state)


def _ppcf_obso(task):
    idx, vel = task
    args, locs = _W["args"], _W["locs"]
    direcs, onballs, gks = _W["direcs"], _W["onballs"], _W["gks"]
    team1, team2 = np.where(direcs[idx] == 1)[0], np.where(direcs[idx] == -1)[0]
    actor = np.where(onballs[idx] == 1)[0][0]
    if actor in team1:
        actor_team, opponent_team, direction = team1, team2, 1
    elif actor in team2:
        actor_team, opponent_team, direction = team2, team1, -1
    gk_att = actor_team[gks[idx][actor_team] == 1][0]
    gk_def = opponent_team[gks[idx][opponent_team] == 1][0]
    P = mpc.generate_pitch_control_for_event(args, locs[idx], vel, _W["eventtypes"][idx][0], actor_team,
                                             opponent_team, [gk_att, gk_def], direction)
    O, _ = obs.calc_obso(P, args.Trans, args.EPV, locs[idx, 0, :], args.n_grid_cells_x,
                         args.n_grid_cells_y, attack_direction=direction)
    return P, O


def worker_state(data_path, date, arr):
    """PPCF/OBSO settings (obso_settings.py) and per-event metadata needed by _ppcf_obso."""
    args = obso_settings.settings(data_path, date)
    data_test = pickle.load(open(args.data_test_path, "rb"))
    L = args.seq_len - 1
    cat = lambda key, sl: np.array([x for g in data_test for x in data_test[g][key][sl]])
    return {
        "args": args, "locs": arr["loc_gt"],
        "eventtypes": cat("types", (slice(None), L, 0)),
        "onballs": cat("onballs", (slice(None), L, slice(None), 0)),
        "gks": cat("gks", (slice(None), L, slice(None), 0)),
        "direcs": cat("direcs", (slice(None), L, slice(None), 0)),
    }


def obso_part(opt, date, arr):
    state = worker_state(opt.data_path, date, arr)
    vel_ball = arr["v_rb"][:, 0, :]
    vels_gt = np.concatenate([vel_ball[:, np.newaxis, :], arr["v_gt"]], axis=1)
    vels_rb = arr["v_rb"]
    N = len(vels_gt)
    ctx = multiprocessing.get_context(opt.start_method)
    with ctx.Pool(opt.n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        gt = pool.map(_ppcf_obso, [(i, vels_gt[i]) for i in range(N)], chunksize=16)
        rb = pool.map(_ppcf_obso, [(i, vels_rb[i]) for i in range(N)], chunksize=16)
    out = {}
    for j, key in enumerate(("ppcf", "obso")):
        err = np.abs(np.stack([x[j] for x in rb]) - np.stack([x[j] for x in gt]))
        grid = np.sum(err, axis=(1, 2)) / (err.shape[1] * err.shape[2])
        out[f"{key}_mean"] = float(np.sum(err) / err.size)
        out[f"{key}_min"] = float(np.min(grid))
        out[f"{key}_max"] = float(np.max(grid))
    return out


def within(value, target, kind, tol):
    diff = abs(value - target)
    return diff <= tol if kind == "abs" else diff <= tol * abs(target)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--datasets", nargs="+", default=["source", "anonymized"])
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--n_sample", type=int, default=10)  # main_completion.py default
    cli.add_argument("--n_jobs", type=int, default=8)
    cli.add_argument("--start_method", default=None, choices=[None, "fork", "spawn", "forkserver"],
                     help="multiprocessing start method (default: platform default)")
    cli.add_argument("--skip_obso", action="store_true")
    cli.add_argument("--no_reference", action="store_true",
                     help="only compare the data sets with each other, not with the values of the paper (other data)")
    cli.add_argument("--out_dir", default="analysis/results")
    opt = cli.parse_args()

    os.makedirs(opt.out_dir, exist_ok=True)
    results = {}
    for date in opt.datasets:
        t0 = time.time()
        arr, m = velocity_part(opt, date)
        if not opt.skip_obso:
            m.update(obso_part(opt, date, arr))
        results[date] = {k: float(v) for k, v in m.items()}
        print(f"[{date}] done in {time.time() - t0:.0f}s", flush=True)

    names = list(results)
    print(f"\n{'metric':<10} | {'reference (paper logs)':>24} | " + " | ".join(f"{n:>24}" for n in names))
    for k, ref in REFERENCE.items():
        if k in results[names[0]]:
            print(f"{k:<10} | {ref:>24.17g} | " + " | ".join(f"{results[n][k]:>24.17g}" for n in names))

    failures, checks = [], []
    for n in names:
        for k, v in (() if opt.no_reference else results[n].items()):
            kind, tol = TOL_REFERENCE[k]
            ok = within(v, REFERENCE[k], kind, tol)
            checks.append({"dataset": n, "metric": k, "against": "reference", "tol": f"{kind} {tol:g}", "ok": ok})
            if not ok:
                failures.append(f"{n} {k}: {v!r} vs reference {REFERENCE[k]!r} ({kind} tol {tol:g})")
        if n != names[0]:
            for k, v in results[n].items():
                ok = within(v, results[names[0]][k], "rel", TOL_ACROSS)
                checks.append({"dataset": n, "metric": k, "against": names[0], "tol": f"rel {TOL_ACROSS:g}", "ok": ok})
                if not ok:
                    failures.append(f"{n} {k}: {v!r} vs {names[0]} {results[names[0]][k]!r} (rel tol {TOL_ACROSS:g})")

    json.dump({"reference": REFERENCE, "results": results, "checks": checks, "passed": not failures},
              open(os.path.join(opt.out_dir, "rule_based_check.json"), "w"), indent=1)
    print(f"Saved: {opt.out_dir}/rule_based_check.json")
    if failures:
        print(f"\nFAILED ({len(failures)} of {len(checks)} checks):")
        for f in failures:
            print("  " + f)
        sys.exit(1)
    print(f"\nPASSED: all {len(checks)} checks within tolerance")


if __name__ == "__main__":
    main()
