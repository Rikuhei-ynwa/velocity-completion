"""Sensitivity of PPCF/OBSO to direction-only velocity errors (reviewer R4-3).

The Monte Carlo analysis of the paper (analysis/mc_sensitivity.py) adds isotropic Gaussian noise,
which changes speed and direction at once. Here each player's ground-truth velocity is rotated by
a fixed angle theta with a random sign per player while its speed is kept, and PPCF/OBSO are
recomputed and compared with the ground-truth PPCF/OBSO (MAE), on the same 500 test events as
the Monte Carlo analysis (np.random.seed(0) then np.random.choice, as in analysis/mc_sensitivity.py).

For comparison with the Gaussian analysis, the velocity RMSE induced by each perturbation is
recorded (a rotation by theta moves a velocity of speed s by 2 s sin(theta/2)), and the figure
plots both perturbation types against that velocity RMSE (Gaussian: RMSE = sigma * sqrt(2),
results from repro_mc/mc_results.json).

Run from the repo root:
    python analysis/angular_sensitivity.py [--n_jobs 8] [--out_dir analysis/results]
"""

import argparse, json, multiprocessing, os, sys, time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rule_based_check import velocity_part, worker_state, _init_worker, _ppcf_obso

ANGLES = [0, 10, 20, 30, 40, 50, 60, 70, 90]  # degrees; covers GRNN_dec_add (48.1) and rule-based (67.8)
N_TRIALS = 5
N_EVENTS = 500


def rotate(vel, theta_deg, rng):
    """Rotate each player's velocity (ball = row 0 untouched) by +-theta, keeping its speed."""
    out = vel.copy()
    sign = rng.choice([-1.0, 1.0], size=vel.shape[0] - 1)
    t = np.radians(theta_deg) * sign
    c, s = np.cos(t), np.sin(t)
    vx, vy = vel[1:, 0], vel[1:, 1]
    out[1:, 0] = c * vx - s * vy
    out[1:, 1] = s * vx + c * vy
    return out


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", default="source")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--n_jobs", type=int, default=8)
    cli.add_argument("--start_method", default=None, choices=[None, "fork", "spawn", "forkserver"])
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--mc_results", default="repro_mc/mc_results.json")
    cli.add_argument("--out_dir", default="analysis/results")
    opt = cli.parse_args()

    arr, _ = velocity_part(argparse.Namespace(data_path=opt.data_path, n_sample=10), opt.dataset)
    vel_ball = arr["v_rb"][:, 0, :]
    vels_gt = np.concatenate([vel_ball[:, np.newaxis, :], arr["v_gt"]], axis=1)

    np.random.seed(opt.seed)  # same event sample as the Monte Carlo analysis
    idx = np.random.choice(vels_gt.shape[0], min(N_EVENTS, vels_gt.shape[0]), replace=False)
    rng = np.random.default_rng(opt.seed)

    tasks, meta = [], []
    for theta in ANGLES:
        for trial in range(N_TRIALS):
            for i in idx:
                v = rotate(vels_gt[i], theta, rng)
                tasks.append((i, v))
                meta.append((theta, trial, i, float(np.mean(np.sum((v[1:] - vels_gt[i][1:]) ** 2, axis=-1)))))

    state = worker_state(opt.data_path, opt.dataset, arr)
    ctx = multiprocessing.get_context(opt.start_method)
    t0 = time.time()
    with ctx.Pool(opt.n_jobs, initializer=_init_worker, initargs=(state,)) as pool:
        gt = dict(zip(idx, pool.map(_ppcf_obso, [(i, vels_gt[i]) for i in idx], chunksize=8)))
        res = pool.map(_ppcf_obso, tasks, chunksize=8)
    print(f"PPCF/OBSO computed for {len(tasks) + len(idx)} samples in {time.time() - t0:.0f}s")

    acc = {th: {"ppcf": [[] for _ in range(N_TRIALS)], "obso": [[] for _ in range(N_TRIALS)],
                "mse": [[] for _ in range(N_TRIALS)]} for th in ANGLES}
    for (th, trial, i, mse), (P, O) in zip(meta, res):
        acc[th]["ppcf"][trial].append(np.mean(np.abs(P - gt[i][0])))
        acc[th]["obso"][trial].append(np.mean(np.abs(O - gt[i][1])))
        acc[th]["mse"][trial].append(mse)
    summary = {}
    for th in ANGLES:
        p = [np.mean(x) for x in acc[th]["ppcf"]]
        o = [np.mean(x) for x in acc[th]["obso"]]
        r = [np.sqrt(np.mean(x)) for x in acc[th]["mse"]]
        summary[th] = {"ppcf_trials": p, "obso_trials": o, "vel_rmse_trials": r}

    print("\nDirection-only perturbation (speed kept, +-theta per player):")
    print("theta (deg) | vel RMSE (m/s) | PPCF MAE (mean±std) | OBSO MAE (mean±std)")
    for th in ANGLES:
        s = summary[th]
        print(f"   {th:>4}     |     {np.mean(s['vel_rmse_trials']):.3f}      |   "
              f"{np.mean(s['ppcf_trials']):.4f}±{np.std(s['ppcf_trials']):.4f}   |   "
              f"{np.mean(s['obso_trials']):.6f}±{np.std(s['obso_trials']):.6f}")

    os.makedirs(opt.out_dir, exist_ok=True)
    json.dump({"dataset": opt.dataset, "angles_deg": ANGLES, "n_trials": N_TRIALS, "event_indices": idx.tolist(),
               "results": {str(k): v for k, v in summary.items()}},
              open(os.path.join(opt.out_dir, "angular_sensitivity.json"), "w"), indent=1, default=float)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for col, key, label, color in ((0, "ppcf", "PPCF MAE", "tab:blue"), (1, "obso", "OBSO MAE", "red")):
        m = [np.mean(summary[t][f"{key}_trials"]) for t in ANGLES]
        sd = [np.std(summary[t][f"{key}_trials"]) for t in ANGLES]
        axes[0, col].errorbar(ANGLES, m, yerr=sd, fmt="o-", capsize=5, color=color)
        for ref, name in ((48.1, "GRNN$_{dec\\_add}$"), (67.8, "rule-based")):
            axes[0, col].axvline(ref, color="gray", ls="--", lw=1)
            axes[0, col].text(ref, max(m) * 0.05, f" {name}\n mean angle", fontsize=9, color="gray")
        axes[0, col].set_xlabel("Rotation angle θ (deg)", fontsize=13)
        axes[0, col].set_ylabel(label, fontsize=13)
        axes[0, col].set_title(f"{label.split()[0]} sensitivity to direction-only error", fontsize=13)
        axes[0, col].grid(True, alpha=0.3)
        # both perturbation types against the induced velocity RMSE
        rm = [np.mean(summary[t]["vel_rmse_trials"]) for t in ANGLES]
        axes[1, col].plot(rm, m, "o-", color=color, label="direction only (rotation)")
        if os.path.isfile(opt.mc_results):
            mc = json.load(open(opt.mc_results))
            sig = mc["noise_levels"]
            gm = [np.mean(mc[f"{key}_trials"][str(s)]) for s in sig]
            axes[1, col].plot([s * np.sqrt(2) for s in sig], gm, "s--", color="black",
                              label="isotropic Gaussian (paper MC)")
        axes[1, col].set_xlabel("Induced player-velocity RMSE (m/s)", fontsize=13)
        axes[1, col].set_ylabel(label, fontsize=13)
        axes[1, col].grid(True, alpha=0.3)
        axes[1, col].legend()
    plt.tight_layout()
    out_png = os.path.join(opt.out_dir, "angular_sensitivity.png")
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


if __name__ == "__main__":
    main()
