"""Association features for the single-event (non_recurrent) data sets of the models without a window.

The features of preprocess_data/make_assoc_features.py are computed per window of the recurrent data set (target =
last event). A non_recurrent sample is the same event with the same slot order, so every window is matched to its
event by the locations of the 24 location rows and the frame number. Events without a window (the first events of a
half: they have no previous event) get zeros, as the windows without a previous event do.

    python preprocess_data/make_assoc_nonrecurrent.py [--dataset anonymized]
Writes <dataset>/rectangular/non_recurrent/<split>_assoc.npz with assoc (N, 1, 23, 5), N = number of non_recurrent events.
"""

import argparse, os, sys

import numpy as np


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    opt = cli.parse_args()
    D = os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular")
    for split in ("test", "validation", "training"):
        out = os.path.join(D, "non_recurrent", f"{split}_assoc.npz")
        if os.path.exists(out):
            print(f"{out} already exists; skipped", flush=True)
            continue
        a = np.load(os.path.join(D, "non_recurrent", f"{split}.npz"))
        b = np.load(os.path.join(D, "recurrent", f"{split}.npz"))
        assoc_rec = np.load(os.path.join(D, "recurrent", f"{split}_assoc.npz"))["assoc"]
        ca, fa = a["cies"], a["frames"]
        cb, fb = b["cies"], b["frames"]  # load once: every access of an npz key reads the whole array
        key_a = {}
        for i in range(len(ca)):
            key_a.setdefault(ca[i, :24].tobytes() + fa[i].tobytes(), []).append(i)
        A = np.zeros((len(ca), 1, 23, 5), np.float32)
        hit = dup = 0
        for j in range(len(cb)):
            idx = key_a.get(cb[j, -1, :24].tobytes() + fb[j, -1].tobytes())
            if idx is None:
                continue
            dup += len(idx) > 1
            A[idx[0]] = assoc_rec[j]
            hit += 1
        # the first-event windows have all-zero features; compare with the physical velocity of the recurrent set
        print(f"{split}: non_recurrent events {len(ca)}, windows {len(cb)}, matched {hit}, ambiguous {dup} -> {out}", flush=True)
        np.savez(out, assoc=A)


if __name__ == "__main__":
    main()
