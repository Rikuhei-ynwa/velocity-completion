"""Association features for the "position difference" inputs of the models (R2-1, advisor comment).

For every window (10 events, target = last) this stores what can be computed from the event data alone:
  - the time gap dt to the previous event (frame numbers / 25 Hz)
  - the displacement of every player since the previous event, with the player of the previous event
    associated by position only (one-to-one assignment inside each team that minimises the total distance:
    Hungarian algorithm; no player identity, no tracking data), and the velocity displacement / dt
  - the same for the ball (no association needed)
The result is analysis/interp_baseline.baseline_velocity(..., "hungarian") written per split to
<dataset>/rectangular/recurrent/<split>_assoc.npz with the key
  assoc  (N, 1, 23, 5) float32   agent 0 = ball, 1..22 = players in the slot order of the target event
         channel 0,1  displacement [m] / 10 (clipped to +-5)
         channel 2    min(dt, 10 s) / 10
         channel 3,4  displacement / dt [m/s] / 10; the speed is capped at 12 m/s for players, 40 m/s for the ball
Windows without a previous event (frame 0 padding) are all zero. The physical velocity of the players
is channels 3,4 * 10 (used by the residual model: prediction = network output + this velocity).

    python preprocess_data/make_assoc_features.py [--dataset anonymized]
"""

import argparse, os, sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "analysis"))
from interp_baseline import load_split, displacement

PLAYER_CAP, BALL_CAP = 12.0, 40.0  # m/s


def cap(v, vmax):
    sp = np.linalg.norm(v, axis=-1, keepdims=True)
    return v * np.minimum(1.0, vmax / np.maximum(sp, 1e-9))


def build(dataset, split, data_path, method="hungarian"):
    d = load_split(dataset, split, data_path)
    valid, dt = d["valid"], d["dt"]
    disp, _ = displacement(d, method)
    g = np.maximum(dt, 1e-6)
    v = cap(disp / g[:, None, None], PLAYER_CAP)
    bdisp = d["ball"] - d["ball_prev"]
    bv = cap(bdisp / g[:, None], BALL_CAP)
    N = len(dt)
    A = np.zeros((N, 1, 23, 5), np.float32)
    A[:, 0, 1:, 0:2] = np.clip(disp / 10.0, -5, 5)
    A[:, 0, 0, 0:2] = np.clip(bdisp / 10.0, -5, 5)
    A[:, 0, :, 2] = (np.minimum(dt, 10.0) / 10.0)[:, None]
    A[:, 0, 1:, 3:5] = v / 10.0
    A[:, 0, 0, 3:5] = bv / 10.0
    A[~valid] = 0.0
    return A, valid, d


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--method", default="hungarian", choices=["hungarian", "identity"],
                     help="identity: the players are known (the ID-ful data set source, the slot is the player)")
    cli.add_argument("--splits", nargs="+", default=["test", "validation", "training"])
    opt = cli.parse_args()
    for split in opt.splits:
        A, valid, d = build(opt.dataset, split, opt.data_path, opt.method)
        out = os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent", f"{split}_assoc.npz")
        if os.path.exists(out):
            sys.exit(f"{out} already exists; refusing to overwrite")
        np.savez(out, assoc=A)
        # sanity: the stored velocity must reproduce the Hungarian baseline of analysis/interp_baseline.py
        v = A[:, 0, 1:, 3:5] * 10.0
        rm = np.sqrt(((v - d["gt"]) ** 2).sum(-1).mean(-1))[valid].mean()
        print(f"{split}: {len(A)} windows, {int(valid.sum())} with a previous event; "
              f"RMSE of the stored difference velocity {rm:.3f} m/s -> {out}")


if __name__ == "__main__":
    main()
