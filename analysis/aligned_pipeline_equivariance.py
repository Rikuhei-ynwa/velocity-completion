"""Equivariance of the whole pipeline "relabel every event independently -> align the window by position (preprocess_data/
align_windows.py) -> trained model" (the symmetry of the anonymized data). The model is the trained best model on aligned windows.

    python analysis/aligned_pipeline_equivariance.py --run <results_root>/weights/vel_1_acc_0_norm/202610091700/Transformer_add_resid_eq2_events_0_rectangular_256
"""
import argparse, json, os, pickle, sys

import numpy as np
import torch

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "preprocess_data"))
from align_windows import alignment, apply, TEAMS, N_AGENTS
from models import load_model


def items_of(z, assoc):
    t = lambda a: torch.tensor(np.asarray(a, dtype=np.float32))
    return {"cies": t(z["cies"]).permute(1, 0, 2, 3), "eventtypes": t(z["eventtypes"])[..., None].permute(1, 0, 2, 3),
            "assoc": t(assoc).permute(1, 0, 2, 3), "frames": t(z["frames"])}


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--run", required=True)
    cli.add_argument("--n", type=int, default=128)
    cli.add_argument("--data", default=os.path.join(os.environ.get("VC_DATA_ROOT", "data"), "dataset", "anonymized", "rectangular", "recurrent"))
    cli.add_argument("--out", default="analysis/results/aligned_pipeline_equivariance.json")
    cli.add_argument("--tol", type=float, default=1e-4)
    opt = cli.parse_args()
    z = {k: v[:opt.n] for k, v in np.load(os.path.join(opt.data, "test.npz")).items()}
    assoc = np.load(os.path.join(opt.data, "test_assoc.npz"))["assoc"][:opt.n]  # (n, 1, 23, 5), target event
    vs = np.load(os.path.join(opt.data, "test.npz"))["vel_stats"]
    info = torch.tensor(vs[1].astype(np.float32))
    params = pickle.load(open(os.path.join(opt.run, "params.p"), "rb"))
    model = load_model(params["model"], params)
    model.load_state_dict(torch.load(os.path.join(opt.run, "state_dict_best.pth"), map_location="cpu", weights_only=True))
    model.eval()
    rng = np.random.default_rng(0)
    T = z["cies"].shape[1]
    Q = np.tile(np.arange(N_AGENTS), (opt.n, T, 1))
    for S in TEAMS:
        Q[:, :, S] = S[np.argsort(rng.random((opt.n, T, len(S))), axis=-1)]

    def predict(zz, aa):
        al = apply(zz, alignment(zz["cies"]))
        with torch.no_grad():
            _, _, data = model.sample(items_of(al, aa), info, n_sample=1, TEST=True)
        return np.asarray(data["v_pred"][0])  # (n, 22, 2)

    p0 = predict(z, assoc)
    zq = apply(z, Q)
    aq = np.take_along_axis(assoc, Q[:, -1:, :, None], axis=2)  # the association features belong to the target event
    p1 = predict(zq, aq)
    expect = np.take_along_axis(p0, (Q[:, -1, 1:] - 1)[..., None], axis=1)
    d = np.abs(p1 - expect)
    # the same check without the alignment (raw windows), for comparison
    with torch.no_grad():
        _, _, r0 = model.sample(items_of(z, assoc), info, n_sample=1, TEST=True)
        _, _, r1 = model.sample(items_of(zq, aq), info, n_sample=1, TEST=True)
    dr = np.abs(np.asarray(r1["v_pred"][0]) - np.take_along_axis(np.asarray(r0["v_pred"][0]), (Q[:, -1, 1:] - 1)[..., None], axis=1))
    res = {"n_windows": opt.n, "with_alignment_max": float(d.max()), "with_alignment_mean": float(d.mean()),
           "without_alignment_max": float(dr.max()), "without_alignment_mean": float(dr.mean()), "mean_speed": float(np.linalg.norm(p0, axis=-1).mean())}
    print(json.dumps(res, indent=1))
    json.dump(res, open(opt.out, "w"), indent=1)
    assert res["with_alignment_max"] < opt.tol, f"the aligned pipeline is not equivariant: {res['with_alignment_max']:.2e} >= {opt.tol}"


if __name__ == "__main__":
    main()
