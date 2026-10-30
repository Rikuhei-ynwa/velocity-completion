"""Write samples.p for fitted SVM/RF runs, so their predictions can be evaluated like the other models.

main_completion.py --TEST stores only RMSE/R^2 for SVM and RF (samples_<model>.p) and discards
the predictions. This reloads <run>/state_dict_fitted.pkl, predicts the test set exactly as
models/{svm,random_forest}.py test() does, and writes <run>/experiments/sample/samples.p in
the format of the torch models ([losses, losses2, others, loss_i] with others = v_pred, v_gt,
v_rb, loc_gt, frames of shape (1, N, ...)), which analysis/speed_metrics.py and
analysis/scenario_metrics.py read. The recomputed RMSE is checked against samples_<model>.p.

    python analysis/sklearn_samples.py --date 202609301700 --models SVM RF --seeds 0 1 2 3 4
"""

import argparse, os, pickle, sys

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.getcwd())
import event_config as cfg
from datasets import EventDataset

N_PLAYERS = 22


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--date", required=True)
    cli.add_argument("--models", nargs="+", default=["SVM", "RF"])
    cli.add_argument("--seeds", nargs="+", type=int, default=[0])
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--date_preprocessed_data", default="anonymized")
    cli.add_argument("--overwrite", action="store_true")
    opt = cli.parse_args()

    L, V = len(cfg.LOC_NODES), len(cfg.VEL_NODES)
    for name in opt.models:
        ds = EventDataset(argparse.Namespace(data_path=opt.data_path, batchsize=256, model=name, n_all_agents=23,
                                               date_preprocessed_data=opt.date_preprocessed_data), train=-1)
        cies = torch.cat([b["cies"] for b in DataLoader(ds, batch_size=256, shuffle=False)]).numpy()
        frames = ds.frame.numpy()[:, 0].astype(float)
        pinfo = ds.players_vel_info.numpy()
        binfo = ds.ball_vel_info.numpy()
        X = np.concatenate([cies[:, 0:1, :], cies[:, 2:L, :]], axis=1).reshape(len(cies), -1)
        gt = cies[:, L + 1:L + V, :].copy()
        gt[:, :, 0] = gt[:, :, 0] * pinfo[1] + pinfo[0]
        gt[:, :, 1] = gt[:, :, 1] * pinfo[3] + pinfo[2]
        v_rb = np.concatenate([cies[:, L:L + 1, :], cies[:, L + V:, :]], axis=1).astype(np.float64)
        v_rb[:, 0, 0] = v_rb[:, 0, 0] * binfo[1] + binfo[0]
        v_rb[:, 0, 1] = v_rb[:, 0, 1] * binfo[3] + binfo[2]
        loc = np.concatenate([cies[:, 0:1, :], cies[:, 2:L, :]], axis=1).astype(np.float64)
        loc[:, :, 0] *= cfg.FIELD_LENGTH / 2
        loc[:, :, 1] *= cfg.FIELD_WIDTH / 2

        for s in opt.seeds:
            run = os.path.join(opt.weights_dir, opt.date, f"{name}_events_{s}_rectangular_256")
            # experiments/sample (the original runs) or experiments/sample_eval (runs with --bn_eval; RF and SVM have no
            # BatchNorm, so the predictions are the same)
            sub = "sample_eval" if os.path.isdir(os.path.join(run, "experiments", "sample_eval")) else "sample"
            out = os.path.join(run, "experiments", sub, "samples.p")
            if os.path.exists(out) and not opt.overwrite:
                print(f"{out} exists; skipping (use --overwrite)")
                continue
            model = pickle.load(open(os.path.join(run, "state_dict_fitted.pkl"), "rb"))
            pred = model.model.predict(X).reshape(-1, N_PLAYERS, 2)
            rmse = float(np.mean(np.sqrt(np.sum((pred - gt) ** 2, axis=-1).mean(axis=1))))
            logged = pickle.load(open(os.path.join(run, "experiments", sub, f"samples_{name}.p"), "rb"))
            logged_rmse = float(np.mean([li["L_rec2"] for li in logged[1]]))
            if not np.isclose(rmse, logged_rmse, rtol=1e-6):
                sys.exit(f"{run}: recomputed RMSE {rmse} != logged {logged_rmse}")
            others = {"v_pred": pred[None].astype(np.float64), "v_gt": gt[None].astype(np.float64),
                      "v_rb": v_rb[None], "loc_gt": loc[None], "frames": frames[None]}
            pickle.dump([{}, {}, others, [{"L_rec2": rmse}]], open(out, "wb"), protocol=4)
            print(f"{name} seed {s}: RMSE {rmse:.4f} (logged {logged_rmse:.4f}) -> {out}")


if __name__ == "__main__":
    main()
