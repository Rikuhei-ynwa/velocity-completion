"""Monte Carlo sensitivity of PPCF/OBSO to velocity noise (appendix of the paper); no trained model is needed.

The MC analysis only uses ground-truth locations/velocities, which the original pipeline read from
samples.p (written by main_completion.py --TEST). Here the same arrays are rebuilt directly from
test.npz, replicating models/grnn.py sample(TEST=True) and main_completion.py's aggregation.

Run from the velocity-completion repo root:
    .venv/bin/python <this file> [--n_jobs 8] [--out DIR]
"""

import argparse, json, os, pickle, random, sys, time
import multiprocessing

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
import event_config as cfg
from datasets import EventDataset
import DRSO.obso_player as obs
import metrica.Metrica_PitchControl as mpc
import obso_settings

LOC_NODES = cfg.LOC_NODES
VEL_NODES = cfg.VEL_NODES

# ---- PPCF/OBSO workers: module-level so they work with both fork and spawn ----
_W = {}


def _init_worker(state):
    _W.update(state)


def _ppcf_obso(task):
    idx, vel = task
    args, locs = _W["args"], _W["locs"]
    direcs, onballs, gks = _W["direcs"], _W["onballs"], _W["gks"]
    team1 = np.where(direcs[idx] == 1)[0]
    team2 = np.where(direcs[idx] == -1)[0]
    actor = np.where(onballs[idx] == 1)[0][0]
    event = _W["eventtypes"][idx][0]
    if actor in team1:
        actor_team, opponent_team, direction = team1, team2, 1
    elif actor in team2:
        actor_team, opponent_team, direction = team2, team1, -1
    gk_att = actor_team[gks[idx][actor_team] == 1][0]
    gk_def = opponent_team[gks[idx][opponent_team] == 1][0]
    PPCF = mpc.generate_pitch_control_for_event(
        args, locs[idx], vel, event, actor_team, opponent_team, [gk_att, gk_def], direction)
    OBSO, _ = obs.calc_obso(PPCF, args.Trans, args.EPV, locs[idx, 0, :],
                            args.n_grid_cells_x, args.n_grid_cells_y, attack_direction=direction)
    return PPCF, OBSO


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--n_jobs", type=int, default=8)
    cli.add_argument("--n_sample", type=int, default=10)  # main_completion.py default
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--dataset", default="source",
                     help="the paper used the data set with the original slot order; PPCF/OBSO do not depend on the slot "
                          "order, but the order in which the noise is drawn does")
    cli.add_argument("--out", type=str, default="repro_out")
    cli.add_argument("--limit_levels", type=int, default=None)  # for quick timing tests
    cli.add_argument("--start_method", default=None, choices=[None, "fork", "spawn", "forkserver"],
                     help="multiprocessing start method (default: platform default)")
    opt = cli.parse_args()

    # ---- PPCF/OBSO settings (obso_settings.py) ----
    args = obso_settings.settings(opt.data_path, opt.dataset)
    args.date_preprocessed_data = opt.dataset
    args.seed = 0

    # ---- 1. rebuild samples.p ground-truth arrays from test.npz ----
    dataset_args = argparse.Namespace(
        data_path=opt.data_path,
        batchsize=256, model="GRNN_add", n_all_agents=23,
        date_preprocessed_data=args.date_preprocessed_data,
    )
    dataset_test = EventDataset(dataset_args, train=-1)
    loader = DataLoader(dataset_test, batch_size=256, shuffle=False, drop_last=False)
    players_info = dataset_test.players_vel_info
    ball_info = dataset_test.ball_vel_info.detach().cpu().numpy()
    N = len(dataset_test)

    others = {k: np.zeros((opt.n_sample, N, 23 if k in ("loc_gt", "v_rb") else 22, 2))
              for k in ("loc_gt", "v_rb", "v_gt")}
    others["frames"] = np.zeros((opt.n_sample, N))

    for r in range(opt.n_sample):
        start = 0
        for items in loader:
            for key in items:
                if key != "frames":
                    items[key] = items[key].permute(1, 0, 2, 3)
            cies = items["cies"].clone().clone()
            B, t = cies.size(1), cies.size(0) - 1
            x_t = torch.cat([cies[t, :, 0:1, :], cies[t, :, 2:len(LOC_NODES), :]], dim=1)
            v_t = cies[t, :, len(LOC_NODES):len(LOC_NODES) + len(VEL_NODES), :]
            # self.restore (in place, float32)
            player_vel_GT = v_t[:, 1:, :]
            player_vel_GT[:, :, 0] = player_vel_GT[:, :, 0] * players_info[1] + players_info[0]
            player_vel_GT[:, :, 1] = player_vel_GT[:, :, 1] * players_info[3] + players_info[2]
            pos_GT = x_t.numpy()
            pos_GT[:, :, 0] = pos_GT[:, :, 0] * (cfg.FIELD_LENGTH / 2)
            pos_GT[:, :, 1] = pos_GT[:, :, 1] * (cfg.FIELD_WIDTH / 2)
            ball_vel_rb = np.expand_dims(v_t[:, 0, :].numpy(), axis=1)
            player_vel_rb = cies[t, :, len(LOC_NODES + VEL_NODES):, :]

            d = {k: np.zeros((1, B) + s) for k, s in
                 (("v_gt", (22, 2)), ("v_rb", (23, 2)), ("loc_gt", (23, 2)), ("frames", ()))}
            d["v_gt"][0] = player_vel_GT.numpy()
            d["v_rb"][0] = np.concatenate((ball_vel_rb, player_vel_rb.numpy()), axis=1)
            d["loc_gt"][0] = pos_GT
            d["frames"][0] += items["frames"][:, t, :].numpy().T[0]

            sl = slice(start, start + B)
            v_rb = np.mean(d["v_rb"], axis=0)
            v_rb[:, 0, 0] = v_rb[:, 0, 0] * ball_info[1] + ball_info[0]
            v_rb[:, 0, 1] = v_rb[:, 0, 1] * ball_info[3] + ball_info[2]
            others["v_rb"][r, sl] += v_rb
            for k in ("v_gt", "loc_gt", "frames"):
                others[k][r, sl] += np.mean(d[k], axis=0)
            start += B

    # ---- 2. test metadata (test.pkl) and seed ----
    random.seed(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    np.random.seed(args.seed)

    data_test = pickle.load(open(args.data_test_path, "rb"))
    L = args.seq_len - 1
    cat = lambda key, sl: np.array([x for g in data_test for x in data_test[g][key][sl]])
    frames = cat("frames", (slice(None), L, 0))
    eventtypes = cat("types", (slice(None), L, 0))
    onballs = cat("onballs", (slice(None), L, slice(None), 0))
    gks = cat("gks", (slice(None), L, slice(None), 0))
    direcs = cat("direcs", (slice(None), L, slice(None), 0))
    assert np.all(frames == np.mean(others["frames"], axis=0)), "frame order mismatch with test.pkl"

    locs = np.mean(others["loc_gt"], axis=0)
    vel_ball = np.mean(others["v_rb"], axis=0)[:, 0, :]
    vels_gt = np.concatenate([vel_ball[:, np.newaxis, :], np.mean(others["v_gt"], axis=0)], axis=1)




    # ---- 3. Monte Carlo trials, same RNG call order as the original analysis ----
    noise_levels = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    n_monte_carlo = 5
    n_samples_for_mc = min(500, vels_gt.shape[0])
    mc_indices = np.random.choice(vels_gt.shape[0], n_samples_for_mc, replace=False)

    tasks = []  # (level, mc_iter, idx, vel_noisy), drawn sequentially like the original loop
    for noise_level in noise_levels:
        for mc_iter in range(n_monte_carlo):
            for idx in mc_indices:
                vel_noisy = vels_gt[idx].copy()
                if noise_level > 0:
                    noise = np.random.normal(0, noise_level, vels_gt[idx, 1:, :].shape)
                    vel_noisy[1:, :] += noise
                tasks.append((noise_level, mc_iter, idx, vel_noisy))

    levels_run = noise_levels[: opt.limit_levels] if opt.limit_levels else noise_levels
    t0 = time.time()
    state = {"args": args, "locs": locs, "eventtypes": eventtypes,
             "onballs": onballs, "gks": gks, "direcs": direcs}
    ctx = multiprocessing.get_context(opt.start_method)
    with ctx.Pool(opt.n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        gt = dict(zip(mc_indices, pool.map(_ppcf_obso, [(i, vels_gt[i]) for i in mc_indices], chunksize=8)))
        run = [t for t in tasks if t[0] in levels_run]
        res = pool.map(_ppcf_obso, [(t[2], t[3]) for t in run], chunksize=8)
    print(f"PPCF/OBSO computed for {len(run) + len(mc_indices)} samples in {time.time() - t0:.0f}s")

    ppcf_sens = {l: [[] for _ in range(n_monte_carlo)] for l in levels_run}
    obso_sens = {l: [[] for _ in range(n_monte_carlo)] for l in levels_run}
    for (lvl, it, idx, _), (P, O) in zip(run, res):
        ppcf_sens[lvl][it].append(np.mean(np.abs(P - gt[idx][0])))
        obso_sens[lvl][it].append(np.mean(np.abs(O - gt[idx][1])))
    ppcf_trials = {l: [np.mean(e) for e in ppcf_sens[l]] for l in levels_run}
    obso_trials = {l: [np.mean(e) for e in obso_sens[l]] for l in levels_run}

    print("\nMonte Carlo Sensitivity Results:")
    print("Noise (m/s) | PPCF MAE (mean±std) | OBSO MAE (mean±std)")
    for l in levels_run:
        print(f"    {l:.1f}      |   {np.mean(ppcf_trials[l]):.4f}±{np.std(ppcf_trials[l]):.4f}   |"
              f"   {np.mean(obso_trials[l]):.6f}±{np.std(obso_trials[l]):.6f}")

    os.makedirs(opt.out, exist_ok=True)
    json.dump({"noise_levels": levels_run, "mc_indices": mc_indices.tolist(),
               "ppcf_trials": {str(k): v for k, v in ppcf_trials.items()},
               "obso_trials": {str(k): v for k, v in obso_trials.items()}},
              open(os.path.join(opt.out, "mc_results.json"), "w"), indent=1, default=float)

    if levels_run == noise_levels:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        pm = [np.mean(ppcf_trials[l]) for l in noise_levels]; ps = [np.std(ppcf_trials[l]) for l in noise_levels]
        om = [np.mean(obso_trials[l]) for l in noise_levels]; os_ = [np.std(obso_trials[l]) for l in noise_levels]
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].errorbar(noise_levels, pm, yerr=ps, fmt="o-", capsize=5)
        axes[0].set_xlabel("Velocity Noise Level (m/s)", fontsize=14); axes[0].set_ylabel("PPCF MAE", fontsize=14)
        axes[0].set_title("PPCF Sensitivity to Velocity Noise", fontsize=14); axes[0].grid(True, alpha=0.3)
        axes[1].errorbar(noise_levels, om, yerr=os_, fmt="o-", capsize=5, color="red")
        axes[1].set_xlabel("Velocity Noise Level (m/s)", fontsize=14); axes[1].set_ylabel("OBSO MAE", fontsize=14)
        axes[1].set_title("OBSO Sensitivity to Velocity Noise", fontsize=14); axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(opt.out, "monte_carlo_sensitivity.png"), dpi=150)
        print(f"Saved: {opt.out}/monte_carlo_sensitivity.png")


if __name__ == "__main__":
    main()
