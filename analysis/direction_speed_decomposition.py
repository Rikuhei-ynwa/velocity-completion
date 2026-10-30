"""Direction vs speed errors of a trained model and their effect on PPCF/OBSO (reviewer R4-3; replaces the ratio of
analysis/angular_vs_models.py, which interpolates a fixed-angle rotation curve at the mean angle error and cannot separate the
contributions).

For every test event and seed, three velocity sets are evaluated against the PPCF/OBSO of the ground-truth velocities:
  prediction        the model's velocities
  direction only    the true speed with the predicted direction (only the direction is wrong)
  speed only        the predicted speed with the true direction (only the speed is wrong)
The direction of a velocity below --v_min m/s is not defined: where the true or the predicted speed is below the threshold the
direction-only set uses the true velocity and the speed-only set keeps the predicted speed along the true direction (if the
true speed is below the threshold, the predicted velocity is kept).

    python analysis/direction_speed_decomposition.py --date 202610091700 --model Transformer_add_resid_eq2
"""
import argparse, json, os, pickle, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.getcwd())
from interp_baseline import load_split, ppcf_obso


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--date", required=True)
    cli.add_argument("--model", required=True)
    cli.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    cli.add_argument("--v_min", type=float, default=0.5)
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--n_jobs", type=int, default=16)
    cli.add_argument("--cache_dir", default="analysis/results/scenario_cache")
    cli.add_argument("--out", default=None)
    opt = cli.parse_args()
    gt = load_split(opt.dataset, "test", opt.data_path)["gt"]  # (N, 22, 2)
    sg = np.linalg.norm(gt, axis=-1, keepdims=True)
    ug = np.where(sg >= opt.v_min, gt / np.maximum(sg, 1e-9), 0.0)
    vels = {}
    for s in opt.seeds:
        o = pickle.load(open(os.path.join(opt.weights_dir, opt.date, f"{opt.model}_events_{s}_rectangular_256", "experiments", "sample_eval", "samples.p"), "rb"))[2]
        pred = np.mean(o["v_pred"], 0)
        assert np.allclose(np.mean(o["v_gt"], 0), gt, atol=1e-4)
        sp = np.linalg.norm(pred, axis=-1, keepdims=True)
        up = pred / np.maximum(sp, 1e-9)
        defined = (sg >= opt.v_min) & (sp >= opt.v_min)
        vels[f"s{s}_prediction"] = pred
        vels[f"s{s}_direction_only"] = np.where(defined, sg * up, gt)
        vels[f"s{s}_speed_only"] = np.where(sg >= opt.v_min, sp * ug, pred)
    po = ppcf_obso(vels, opt.dataset, opt.data_path, opt.n_jobs, opt.cache_dir, None, gt)
    res = {}
    for kind in ("prediction", "direction_only", "speed_only"):
        p = [float(po[f"s{s}_{kind}"][0].mean()) for s in opt.seeds]
        o = [float(po[f"s{s}_{kind}"][1].mean()) for s in opt.seeds]
        e = [float(np.sqrt(((vels[f"s{s}_{kind}"] - gt) ** 2).sum(-1).mean(-1)).mean()) for s in opt.seeds]
        res[kind] = {"ppcf_mae": float(np.mean(p)), "ppcf_sd": float(np.std(p, ddof=1)), "obso_mae": float(np.mean(o)), "obso_sd": float(np.std(o, ddof=1)),
                     "velocity_rmse": float(np.mean(e))}
        print(f"{kind:15s} velocity RMSE {res[kind]['velocity_rmse']:.3f}   PPCF MAE {res[kind]['ppcf_mae']:.5f} ± {res[kind]['ppcf_sd']:.5f}   OBSO MAE {res[kind]['obso_mae']:.6f} ± {res[kind]['obso_sd']:.6f}", flush=True)
    out = opt.out or f"analysis/results/direction_speed_decomposition_{opt.model}_{opt.date}.json"
    json.dump({"model": opt.model, "date": opt.date, "v_min": opt.v_min, "seeds": opt.seeds, "results": res}, open(out, "w"), indent=1)
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()
