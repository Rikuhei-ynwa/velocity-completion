"""Align the players of every window by position, so that a slot means the same (estimated) player at every event of the window.

In the anonymized data (preprocess_data/anonymize_players.py) the players of every event are relabelled independently: slot k at an
earlier event of a window is, in general, another player than slot k at the target (last) event. Sequence models that connect a slot
through time (the GRU state of a slot, the temporal attention of a slot) are then equivariant only to a relabelling that is the same
for the whole window, not to the symmetry of the data (an independent relabelling of every event).

This script re-orders the players of the earlier events of every window: going back from the target event, the players of event t are
associated, inside each team, with the (already aligned) players of event t+1 by position only (one-to-one, minimum total distance:
Hungarian algorithm; no identity, no tracking data), and put into the slots of their partners. The target event keeps its order, so
the targets, the association features (<split>_assoc.npz) and every evaluation stay unchanged. Because the association depends only on
the positions, the aligned window depends on the slot order of the target event only: a model that is equivariant to a relabelling
of the whole window becomes, on aligned windows, equivariant to an independent relabelling of every event (checked below).

Writes <dataset>_aligned/rectangular/recurrent/<split>.npz (cies, node_type, edge_type relabelled; the rest copied),
<split>_assoc.npz (copied) and <split>_permutations.npy (the original identity of every aligned slot, for the diagnostics).

    python preprocess_data/align_windows.py [--dataset anonymized]
"""
import argparse, os, shutil, sys

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import event_config as cfg

N_AGENTS = 23
TEAMS = (np.arange(1, 12), np.arange(12, 23))
LOC0, VEL0, FAKE0 = 0, 24, 47
recv, send = np.where(np.ones((N_AGENTS, N_AGENTS)) - np.eye(N_AGENTS))
pair2edge = -np.ones((N_AGENTS, N_AGENTS), dtype=np.int64)
pair2edge[recv, send] = np.arange(len(recv))


def node_index(P):  # rows of cies for the relabelling P (new slot k holds old agent P[k]); same as analysis/equivariance_check.py
    pl = P[..., 1:]
    lead = P.shape[:-1]
    return np.concatenate([np.broadcast_to([LOC0, LOC0 + 1], lead + (2,)), LOC0 + 1 + pl, np.broadcast_to([VEL0], lead + (1,)), VEL0 + pl,
                           FAKE0 - 1 + pl], axis=-1)


def positions(cies):
    pos = cies[..., 2:24, :].astype(np.float64).copy()  # (N, T, 22, 2) players, agent k = index k - 1
    pos[..., 0] *= cfg.FIELD_LENGTH / 2
    pos[..., 1] *= cfg.FIELD_WIDTH / 2
    return pos


def alignment(cies):
    """P (N, T, 23): for every event, the old agent in each aligned slot; the target event keeps its order."""
    N, T = cies.shape[:2]
    pos = positions(cies)
    P = np.tile(np.arange(N_AGENTS), (N, T, 1))
    for t in range(T - 2, -1, -1):
        for n in range(N):
            for S in TEAMS:
                ref = pos[n, t + 1, P[n, t + 1, S] - 1]  # aligned positions at event t+1 (slots of the target event)
                cand = pos[n, t, S - 1]  # raw positions at event t
                cost = np.linalg.norm(ref[:, None, :] - cand[None, :, :], axis=-1)
                r, c = linear_sum_assignment(cost)
                P[n, t, S[r]] = S[c]
    return P


def apply(z, P):
    cies = np.take_along_axis(z["cies"], node_index(P)[..., None], axis=2)
    out = {"cies": cies}
    if "node_type" in z:
        out["node_type"] = np.take_along_axis(z["node_type"], P[..., None], axis=2)
    if "edge_type" in z:
        out["edge_type"] = np.take_along_axis(z["edge_type"], pair2edge[P[..., recv], P[..., send]][..., None], axis=2)
    for k in z:
        if k not in out:
            out[k] = z[k]
    return out


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--check_events", type=int, default=300)
    opt = cli.parse_args()
    src = os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent")
    dst = os.path.join(opt.data_path, "dataset", opt.dataset + "_aligned", "rectangular", "recurrent")
    os.makedirs(dst, exist_ok=True)
    rng = np.random.default_rng(0)
    for split in ("test", "validation", "training"):
        out_npz = os.path.join(dst, f"{split}.npz")
        if os.path.exists(out_npz):
            print(f"{out_npz} exists; skipped", flush=True)
            continue
        z = dict(np.load(os.path.join(src, f"{split}.npz")))
        P = alignment(z["cies"])
        a = apply(z, P)
        ident = np.load(os.path.join(src, f"{split}_permutations.npy"))
        ident_al = np.take_along_axis(ident, P, axis=2)
        # diagnostics: share of aligned slots that hold the same true player as at the target event, per step back
        valid = z["frames"][:, :, 0] > 0
        same = ident_al[:, :, 1:] == ident_al[:, -1:, 1:]
        acc = [float(same[:, t][valid[:, t]].mean()) for t in range(same.shape[1] - 1)]
        raw = [float((ident[:, t, 1:] == ident[:, -1, 1:])[valid[:, t]].mean()) for t in range(ident.shape[1] - 1)]
        T = len(acc)
        print(f"{split}: {len(P)} windows; same true player as at the target event, steps back 1..{T}: aligned "
              + " ".join(f"{acc[T - k]:.2f}" for k in range(1, T + 1)) + " | without alignment " + " ".join(f"{raw[T - k]:.2f}" for k in range(1, T + 1)), flush=True)
        if split == "test":  # equivariance of the alignment: relabel every event independently, align, compare
            n = min(opt.check_events, len(P))
            Q = np.tile(np.arange(N_AGENTS), (n, z["cies"].shape[1], 1))
            for S in TEAMS:
                Q[:, :, S] = S[np.argsort(rng.random((n, Q.shape[1], len(S))), axis=-1)]
            zq = apply({k: v[:n] for k, v in z.items()}, Q)
            aq = apply(zq, alignment(zq["cies"]))
            expect = apply({k: v[:n] for k, v in a.items()}, np.repeat(Q[:, -1:, :], Q.shape[1], axis=1))
            d = max(float(np.abs(aq[k] - expect[k]).max()) for k in ("cies", "node_type", "edge_type"))
            print(f"test: equivariance of the alignment (independent relabelling of every event, {n} windows): max |difference| = {d:.2e}", flush=True)
        np.savez(out_npz, **a)
        np.save(os.path.join(dst, f"{split}_permutations.npy"), ident_al)
        shutil.copy(os.path.join(src, f"{split}_assoc.npz"), os.path.join(dst, f"{split}_assoc.npz"))
        print(f"-> {out_npz}", flush=True)


if __name__ == "__main__":
    main()
