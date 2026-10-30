"""Synthetic data sets in the format of docs/DATA_SCHEMA.md, for running the pipeline without the licensed data.

The matches are simulated, not real: 22 players move with smoothed random accelerations around a formation that
follows the play towards the goal attacked by the team in possession, the ball moves from player to player at
irregular event times, and the instantaneous player velocities at the event frames serve as the ground truth. The
rule-based velocities are computed with preprocess_data/rule_based_velocity.py. Numbers obtained on these data check
that the code runs; they say nothing about the accuracy of any method on real matches.

    python examples/make_synthetic_dataset.py [--data_root data] [--dataset source] [--n_matches 4 2 2]
Writes <data_root>/dataset/<dataset>/rectangular/{recurrent,non_recurrent}/<split>.{pkl,npz}, with the player slots in
a fixed order (as the source data of the paper); preprocess_data/anonymize_players.py then relabels them.
"""

import argparse, os, pickle, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import event_config as cfg
from preprocess_data.rule_based_velocity import rule_based_velocities

FPS = cfg.FPS
L2, W2 = cfg.FIELD_LENGTH / 2, cfg.FIELD_WIDTH / 2
T_WIN = 10  # events per window
N_AGENTS = 23
# 4-4-2 formation (x towards the attacked goal, y), goalkeeper first
FORMATION = np.array([[-48, 0], [-30, -24], [-32, -8], [-32, 8], [-30, 24], [-10, -22], [-12, -7], [-12, 7], [-10, 22],
                      [8, -8], [8, 8]], float)
EVENT_P = {"pass": 0.42, "dribble": 0.25, "cross": 0.05, "shot": 0.04, "clearance": 0.04, "tackle": 0.04,
           "interception": 0.04, "throw_in": 0.03, "foul": 0.02, "freekick_short": 0.02, "corner_crossed": 0.01,
           "goalkick": 0.01, "keeper_save": 0.01, "freekick_crossed": 0.01, "corner_short": 0.005,
           "shot_freekick": 0.0025, "shot_penalty": 0.0025}
TURNOVER = {"tackle", "interception", "clearance", "keeper_save"}


# ------------------------------------------------------------------ simulation
def simulate_half(rng, n_frames, attack_dir_home, home_ball):
    """Positions and velocities (n_frames, 22, 2) of one half. The formation follows a play line that moves towards
    the goal attacked by the team in possession (home_ball: (n_frames,) bool)."""
    target = np.where(home_ball, attack_dir_home, -attack_dir_home) * 32.0
    play_x = np.zeros(n_frames)
    for t in range(1, n_frames):
        play_x[t] = play_x[t - 1] + (target[t] - play_x[t - 1]) / (6.0 * FPS) + rng.normal(0, 0.05)
    base = np.concatenate([FORMATION * [attack_dir_home, 1], FORMATION * [-attack_dir_home, 1]])  # (22, 2)
    pos = np.zeros((n_frames, 22, 2))
    vel = np.zeros((n_frames, 22, 2))
    p = base.copy()
    v = np.zeros((22, 2))
    for t in range(n_frames):
        anchor = base.copy()
        anchor[:, 0] = anchor[:, 0] * 0.6 + play_x[t]
        anchor[0, 0], anchor[11, 0] = base[0, 0], base[11, 0]  # goalkeepers stay near their goal
        acc = 0.4 * (anchor - p) - 0.6 * v + rng.normal(0, 4.0, (22, 2))
        v = v + acc / FPS
        sp = np.linalg.norm(v, axis=1, keepdims=True)
        v = v * np.minimum(1.0, 9.5 / np.maximum(sp, 1e-9))
        p = np.clip(p + v / FPS, [-L2, -W2], [L2, W2])
        pos[t], vel[t] = p, v
    return pos, vel


def simulate_match(rng, half_seconds, mean_gap):
    """List of events; every event is a dict of the per-event arrays of docs/DATA_SCHEMA.md plus 'half' and 'target'."""
    events = []
    frame0 = 1  # frame 0 marks padding
    types, p_type = list(EVENT_P), np.array(list(EVENT_P.values())) / sum(EVENT_P.values())
    for half, attack_dir_home in ((1, 1), (2, -1)):
        n = int(half_seconds * FPS)
        ev = np.cumsum(np.maximum(1, (rng.exponential(mean_gap, int(3 * half_seconds / mean_gap)) * FPS).astype(int)))
        ev = ev[ev < n - 1]
        etypes = [str(x) for x in rng.choice(types, size=len(ev), p=p_type)]
        home = np.zeros(len(ev), bool)  # team in possession at every event
        home[0] = half == 1
        for k in range(1, len(ev)):
            home[k] = home[k - 1] != (etypes[k - 1] in TURNOVER or rng.random() < 0.12)
        last_event = np.maximum(np.searchsorted(ev, np.arange(n), side="right") - 1, 0)
        pos, vel = simulate_half(rng, n, attack_dir_home, home[last_event])
        actors = []  # an outfield player of the team in possession, often the most advanced one
        for k, t in enumerate(ev):
            team = np.arange(1, 11) if home[k] else np.arange(12, 22)
            adir = attack_dir_home if home[k] else -attack_dir_home
            actors.append(int(team[np.argmax(adir * pos[t, team, 0])]) if rng.random() < 0.5 else int(rng.choice(team)))
        direc = np.r_[np.full(11, attack_dir_home), np.full(11, -attack_dir_home)]
        for k, t in enumerate(ev):
            etype, actor = etypes[k], actors[k]
            adir = attack_dir_home if home[k] else -attack_dir_home
            start = pos[t, actor]
            t_next = ev[k + 1] if k + 1 < len(ev) else min(t + int(1.5 * FPS), n - 1)
            if etype == "dribble":
                end = np.clip(start + vel[t, actor] * (t_next - t) / FPS, [-L2, -W2], [L2, W2])
                ball_v = vel[t, actor]
            elif "shot" in etype:
                end = np.array([adir * L2, rng.uniform(-4, 4)])
                ball_v = (end - start) / 0.8
            else:
                end = pos[t_next, actors[k + 1] if k + 1 < len(ev) else actor]
                ball_v = (end - start) / max((t_next - t) / FPS, 0.2)
            rb = rule_based_velocities(etype, pos[t], np.ones(22, bool), actor, 0, 11, end, ball_v, attack_dir_home)
            onball = np.zeros(22, int)
            onball[actor] = 1
            gk = np.zeros(22, int)
            gk[[0, 11]] = 1
            events.append({
                "frames": np.array([frame0 + int(t)]), "types": np.array([etype], dtype=object),
                "locs": np.vstack([start, end, pos[t]]), "vels": np.vstack([ball_v, vel[t]]), "vel_fakes": rb,
                "direcs": direc[:, None], "onballs": onball[:, None], "gks": gk[:, None], "half": half,
                "target": adir * start[0] >= cfg.ATTCKING_THIRD and adir * end[0] >= cfg.ATTCKING_THIRD,
            })
        frame0 += n
    return events


# ------------------------------------------------------------------ data set files
KEYS = ("frames", "types", "locs", "vels", "vel_fakes", "direcs", "onballs", "gks")


def padding_event():
    return {"frames": np.array([0]), "types": np.array([cfg.EVENT_TYPE[15]], dtype=object),  # non_action
            "locs": np.tile([cfg.MISSING_CIE_X, cfg.MISSING_CIE_Y], (24, 1)), "vels": np.zeros((23, 2)),
            "vel_fakes": np.zeros((22, 2)), "direcs": np.zeros((22, 1), int), "onballs": np.zeros((22, 1), int),
            "gks": np.zeros((22, 1), int)}


def match_samples(events):
    """recurrent: windows of T_WIN events ending at a target event (events of the other half are padding);
    non_recurrent: the target events."""
    idx = [i for i, e in enumerate(events) if e["target"]]
    windows = []
    for i in idx:
        win = [events[j] if j >= 0 and events[j]["half"] == events[i]["half"] else padding_event()
               for j in range(i - T_WIN + 1, i + 1)]
        windows.append(win)
    rec = {k: np.stack([np.stack([e[k] for e in w]) for w in windows]) for k in KEYS}
    rec["idx_series_filtered"] = np.array([np.arange(i - T_WIN + 1, i + 1) for i in idx])
    rec["idx_filtered"] = np.array(idx)
    nrec = {k: np.stack([events[i][k] for i in idx]) for k in KEYS}
    nrec["idx_filtered"] = np.array(idx)
    return rec, nrec


def edge_types(onballs):
    """(..., 22) on-the-ball flags -> (..., 506, 2): same team, ball-actor relation (edge layout of docs/DATA_SCHEMA.md)."""
    teammates = np.zeros((N_AGENTS, 22))
    teammates[1:12, 1:11] = 1
    teammates[12:23, 12:22] = 1
    lead = onballs.shape[:-1]
    onb = np.zeros(lead + (N_AGENTS, 22))
    onb[..., 0, :] = onballs
    who = np.argmax(onballs, axis=-1)
    valid = onballs.sum(axis=-1) > 0
    sel = np.nonzero(valid)
    onb[sel + (who[valid] + 1, np.zeros(valid.sum(), dtype=int))] = 1
    et = np.zeros(lead + (N_AGENTS * 22, 2))
    et[..., 0] = np.broadcast_to(teammates.ravel(), lead + (N_AGENTS * 22,))
    et[..., 1] = onb.reshape(lead + (-1,))
    return et


def model_inputs(data):
    """{match: samples} -> the arrays of <split>.npz."""
    cat = lambda k: np.concatenate([data[g][k] for g in data])
    locs, vels, fakes = cat("locs").astype(float), cat("vels").astype(float), cat("vel_fakes").astype(float)
    locs[..., 0] /= L2
    locs[..., 1] /= W2
    stats = []
    for sl in (np.s_[..., 0:1, :], np.s_[..., 1:, :]):  # ball, players
        v = vels[sl]
        m = [v[..., 0].mean(), v[..., 0].std(), v[..., 1].mean(), v[..., 1].std()]
        v[..., 0] = (v[..., 0] - m[0]) / m[1]
        v[..., 1] = (v[..., 1] - m[2]) / m[3]
        vels[sl] = v
        stats.append(m)
    direcs, onballs, gks = cat("direcs")[..., 0], cat("onballs")[..., 0], cat("gks")[..., 0]
    lead = direcs.shape[:-1]
    node = np.zeros(lead + (N_AGENTS, 5))
    node[..., 0, 0] = 1
    node[..., 1:, 1] = 1
    node[..., 1:, 2], node[..., 1:, 3], node[..., 1:, 4] = direcs, onballs, gks
    types = cat("types")[..., 0]
    onehot = (types[..., None] == np.array(cfg.EVENT_TYPE, dtype=object)).astype(float)
    return {"cies": np.concatenate([locs, vels, fakes], axis=-2), "vel_stats": np.array(stats), "node_type": node,
            "edge_type": edge_types(onballs), "eventtypes": onehot, "frames": cat("frames")}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data_root", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--dataset", default="source")
    cli.add_argument("--n_matches", type=int, nargs=3, default=[4, 2, 2], metavar=("TRAIN", "VAL", "TEST"))
    cli.add_argument("--half_seconds", type=float, default=300.0)
    cli.add_argument("--mean_gap", type=float, default=1.8, help="mean time between events [s]")
    cli.add_argument("--seed", type=int, default=0)
    opt = cli.parse_args()
    root = os.path.join(opt.data_root, "dataset", opt.dataset, "rectangular")
    if os.path.exists(root):
        sys.exit(f"{root} already exists; refusing to overwrite")
    rng = np.random.default_rng(opt.seed)
    bounds = np.cumsum([0] + opt.n_matches)
    for s, split in enumerate(("training", "validation", "test")):
        rec, nrec = {}, {}
        for m in range(bounds[s], bounds[s + 1]):
            rec[f"{m:03d}"], nrec[f"{m:03d}"] = match_samples(simulate_match(rng, opt.half_seconds, opt.mean_gap))
        for datatype, data in (("recurrent", rec), ("non_recurrent", nrec)):
            d = os.path.join(root, datatype)
            os.makedirs(d, exist_ok=True)
            pickle.dump(data, open(os.path.join(d, f"{split}.pkl"), "wb"), protocol=4)
            np.savez(os.path.join(d, f"{split}.npz"), **model_inputs(data))
        print(f"{split}: {bounds[s + 1] - bounds[s]} synthetic matches, {len(model_inputs(nrec)['frames'])} target events")
    print(f"-> {root}")


if __name__ == "__main__":
    main()
