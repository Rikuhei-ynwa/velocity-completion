"""Accuracy of the window alignment (preprocess_data/align_windows.py): share of the aligned slots of an earlier event that hold the
same true player as the slot at the target event, per step back (1 = the previous event). Same diagnostic as align_windows.py
prints, recomputed from the saved files: <dataset>_aligned/.../<split>_permutations.npy (the true identity of every aligned slot) and,
for the baseline without alignment, <dataset>/.../<split>_permutations.npy. Events of a window that are padding (frame <= 0) are left out.

    python analysis/alignment_accuracy.py [--dataset anonymized]
"""
import argparse, json, os

import numpy as np


def share_same(ident, valid):  # per step back 1..T-1: share of the 22 player slots holding the same true player as at the target event
    same = ident[:, :, 1:] == ident[:, -1:, 1:]
    T = ident.shape[1]
    return [float(same[:, T - 1 - k][valid[:, T - 1 - k]].mean()) for k in range(1, T)]


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--out", default="analysis/results/alignment_accuracy.json")
    opt = cli.parse_args()
    src = os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent")
    dst = os.path.join(opt.data_path, "dataset", opt.dataset + "_aligned", "rectangular", "recurrent")
    r = {"dataset": opt.dataset, "steps_back": None, "splits": {}}
    for split in ("test", "validation", "training"):
        valid = np.load(os.path.join(src, f"{split}.npz"))["frames"][:, :, 0] > 0
        aligned = share_same(np.load(os.path.join(dst, f"{split}_permutations.npy")), valid)
        raw = share_same(np.load(os.path.join(src, f"{split}_permutations.npy")), valid)
        r["steps_back"] = list(range(1, len(aligned) + 1))
        r["splits"][split] = {"windows": int(len(valid)), "aligned": aligned, "without_alignment": raw}
        print(f"{split}: {len(valid)} windows; steps back 1..{len(aligned)}: aligned " + " ".join(f"{a:.3f}" for a in aligned)
              + " | without alignment " + " ".join(f"{a:.3f}" for a in raw), flush=True)
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    json.dump(r, open(opt.out, "w"), indent=1)
    print(f"-> {opt.out}")


if __name__ == "__main__":
    main()
