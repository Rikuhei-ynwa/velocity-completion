"""Create an anonymized copy of a preprocessed dataset (protocol A of the fair-comparison design).

In the original dataset, player slots are sorted by jersey number within each team and keep the
same player across consecutive events, so every model can exploit a jersey-number role prior
and cross-event identity continuity. This script relabels the players of every event with an
independent random permutation within each team (ball stays in slot 0, Home stays in slots
1-11, Away in slots 12-22), which matches the task premise that player identities are not
available.

The permutation of an event depends only on (seed, game, frame), so the same event gets the
same labeling wherever it appears (recurrent windows, non-recurrent samples, npz and pkl),
while different events in a window are relabeled independently.

Relabeled consistently:
    npz: cies (locations, velocities, rule-based velocities), node_type, edge_type
    pkl: locs, vels, vel_fakes, direcs, onballs, gks
Unchanged: frames, eventtypes/types, vel_stats (pooled over players), idx_* arrays.

Usage (from the repo root):
    python preprocess_data/anonymize_players.py --src source
    -> <data_root>/dataset/anonymized/
Then train/test with --date_preprocessed_data anonymized.
"""

import argparse, json, os, pickle, subprocess, sys

import numpy as np

N_AGENTS = 23  # ball + 22 players
HOME = np.arange(1, 12)
AWAY = np.arange(12, 23)
N_PLAYERS = 22
# cies node layout (preprocess_data/make_dataset.py): 24 locations, 23 velocities, 22 rule-based velocities
LOC0, VEL0, FAKE0 = 0, 24, 47

cli = argparse.ArgumentParser()
cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
cli.add_argument("--src", default="source", help="source dataset date")
cli.add_argument("--dst", default=None, help="output dataset name (default: <src>_anonymized)")
cli.add_argument("--seed", type=int, default=0)
cli.add_argument("--splits", nargs="+", default=["test", "validation", "training"])
cli.add_argument("--datatypes", nargs="+", default=["recurrent", "non_recurrent"])
opt = cli.parse_args()

src_dir = os.path.join(opt.data_path, "dataset", opt.src, "rectangular")
dst_name = opt.dst or f"{opt.src}_anonymized"
dst_root = os.path.join(opt.data_path, "dataset", dst_name)
if os.path.exists(dst_root):
    sys.exit(f"{dst_root} already exists; refusing to overwrite")

recv, send = np.where(np.ones((N_AGENTS, N_AGENTS)) - np.eye(N_AGENTS))  # GNN edge order
pair2edge = -np.ones((N_AGENTS, N_AGENTS), dtype=np.int64)
pair2edge[recv, send] = np.arange(len(recv))


def event_perm(game, frame):
    """Agent permutation for one event: new slot k takes old agent P[k]."""
    rng = np.random.default_rng([opt.seed, int(game), int(frame)])
    P = np.arange(N_AGENTS)
    P[HOME] = rng.permutation(HOME)
    P[AWAY] = rng.permutation(AWAY)
    return P


def take(a, idx, axis):
    """a[..., idx[...], ...] along `axis`, with idx shaped like a's leading dims + (k,)."""
    idx = idx.reshape(idx.shape + (1,) * (a.ndim - idx.ndim))
    return np.take_along_axis(a, idx, axis=axis)


def cies_index(P):
    pl = P[..., 1:]
    lead = P.shape[:-1]
    return np.concatenate([
        np.broadcast_to([LOC0, LOC0 + 1], lead + (2,)), LOC0 + 1 + pl,
        np.broadcast_to([VEL0], lead + (1,)), VEL0 + pl,
        FAKE0 - 1 + pl,
    ], axis=-1)


def relabel_npz(z, P):
    """P: (N, 23) or (N, T, 23) matching the sample/time layout of the arrays."""
    out = dict(z)
    ax = P.ndim - 1  # agent/node axis
    out["cies"] = take(z["cies"], cies_index(P), ax)
    out["node_type"] = take(z["node_type"], P, ax)
    out["edge_type"] = take(z["edge_type"], pair2edge[P[..., recv], P[..., send]], ax)
    return out


def relabel_pkl(d, P):
    """P: (n, 23) or (n, 10, 23) for one game's arrays."""
    out = dict(d)
    ax = P.ndim - 1
    pl = P[..., 1:] - 1  # player index 0..21
    out["locs"] = take(d["locs"], cies_index(P)[..., :24], ax)
    out["vels"] = take(d["vels"], P, ax)
    for k in ("vel_fakes", "direcs", "onballs", "gks"):
        out[k] = take(d[k], pl, ax)
    return out


def edge_type_from(onballs):
    """Rebuild edge_type from onballs (..., 22) with the layout of docs/DATA_SCHEMA.md."""
    teammates = np.zeros((N_PLAYERS + 1, N_PLAYERS))
    teammates[1:12, 1:11] = 1
    teammates[12:23, 12:22] = 1
    lead = onballs.shape[:-1]
    onb = np.zeros(lead + (N_PLAYERS + 1, N_PLAYERS))
    onb[..., 0, :] = onballs
    who = np.argmax(onballs, axis=-1)
    valid = onballs.sum(axis=-1) > 0
    sel = np.nonzero(valid)
    onb[sel + (who[valid] + 1, np.zeros(valid.sum(), dtype=int))] = 1
    et = np.zeros(lead + (N_AGENTS * N_PLAYERS, 2))
    et[..., 0] = np.broadcast_to(teammates.ravel(), lead + (N_AGENTS * N_PLAYERS,))
    et[..., 1] = onb.reshape(lead + (-1,))
    return et


def check(name, cond):
    if not cond:
        raise AssertionError(f"check failed: {name}")
    print(f"    ok  {name}")


meta = {"source": opt.src, "name": dst_name, "seed": opt.seed,
        "method": "independent within-team player permutation per event, seeded by (seed, game, frame); "
                  "ball slot 0 and Home/Away slot ranges kept",
        "git_commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
        "files": {}}

for dt in opt.datatypes:
    os.makedirs(os.path.join(dst_root, "rectangular", dt))
    for split in opt.splits:
        print(f"[{dt}/{split}]")
        pkl = pickle.load(open(os.path.join(src_dir, dt, f"{split}.pkl"), "rb"))
        z = dict(np.load(os.path.join(src_dir, dt, f"{split}.npz")))
        games = list(pkl.keys())
        sizes = [pkl[g]["frames"].shape[0] for g in games]
        game_of = np.repeat(np.array([int(g) for g in games]), sizes)
        frames = np.concatenate([pkl[g]["frames"] for g in games])[..., 0]  # (N,) or (N, T)
        check("npz and pkl sample order match", np.array_equal(frames, z["frames"][..., 0]))

        # one permutation per distinct (game, frame) event
        gf = np.broadcast_to(game_of.reshape((-1,) + (1,) * (frames.ndim - 1)), frames.shape)
        keys, inv = np.unique(np.stack([gf.ravel(), frames.ravel()], 1), axis=0, return_inverse=True)
        table = np.stack([event_perm(g, f) for g, f in keys])
        P = table[inv.ravel()].reshape(frames.shape + (N_AGENTS,))

        zn = relabel_npz(z, P)
        offs = np.cumsum([0] + sizes)
        pn = {g: relabel_pkl(pkl[g], P[offs[i]:offs[i + 1]]) for i, g in enumerate(games)}

        # ---- checks ----
        Pinv = np.argsort(P, axis=-1)
        back = relabel_npz(zn, Pinv)
        check("inverse relabeling restores npz", all(np.array_equal(back[k], z[k]) for k in z))
        g0 = games[0]
        back_p = relabel_pkl(pn[g0], Pinv[:sizes[0]])
        check("inverse relabeling restores pkl", all(np.array_equal(back_p[k], pkl[g0][k]) for k in pkl[g0]))
        onb_new = np.concatenate([pn[g]["onballs"] for g in games])[..., 0]
        check("edge_type == rebuilt from relabeled onballs", np.array_equal(zn["edge_type"], edge_type_from(onb_new)))
        check("original edge_type == rebuilt from original onballs",
              np.array_equal(z["edge_type"], edge_type_from(np.concatenate([pkl[g]["onballs"] for g in games])[..., 0])))
        nt_pl = zn["node_type"][..., 1:, :]
        for col, key in ((3, "onballs"), (4, "gks")):
            src_match = np.array_equal(z["node_type"][..., 1:, col],
                                       np.concatenate([pkl[g][key] for g in games])[..., 0])
            if src_match:
                check(f"node_type[:, {col}] == relabeled pkl {key}",
                      np.array_equal(nt_pl[..., col], np.concatenate([pn[g][key] for g in games])[..., 0]))
        check("vel_stats unchanged", np.array_equal(zn["vel_stats"], z["vel_stats"]))
        pv = lambda a: np.sort(a[..., VEL0 + 1:VEL0 + 23, :].reshape(-1, 22, 2)[..., 0], axis=-1)
        check("per-event multiset of player velocities unchanged", np.array_equal(pv(zn["cies"]), pv(z["cies"])))
        n_moved = (P[..., 1:] != np.arange(1, N_AGENTS)).mean()
        print(f"    share of player slots relabeled: {n_moved:.3f}")

        np.savez(os.path.join(dst_root, "rectangular", dt, f"{split}.npz"), **zn)
        pickle.dump(pn, open(os.path.join(dst_root, "rectangular", dt, f"{split}.pkl"), "wb"), protocol=4)
        np.save(os.path.join(dst_root, "rectangular", dt, f"{split}_permutations.npy"), P)
        meta["files"][f"{dt}/{split}"] = {"n_samples": int(frames.shape[0]), "n_events": int(len(keys))}
        del z, zn, pkl, pn, back

json.dump(meta, open(os.path.join(dst_root, "anonymization.json"), "w"), indent=1)
print(f"Saved: {dst_root}")
