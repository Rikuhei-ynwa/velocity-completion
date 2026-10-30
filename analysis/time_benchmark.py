"""Training time per epoch and inference latency on a GPU that nothing else uses (reviewers R1-6, R2-7).

The times of the experiments (logs/slurm, analysis/training_time.py) were measured on GPUs shared with other jobs. This script measures
the same quantities again, one model after the other, and refuses to start a measurement while another process holds the GPU:

  training epoch   as in main_completion.py: Adam (lr of the run), batch size 256, model.train() for the training pass and
                   model.eval() for the validation pass (BN_EVAL), the same forward/loss/backward/step, the same validation pass
                   (model.sample, n_sample = 1, no torch.no_grad(), as in main_completion.py). One warm-up epoch, then --epochs timed
                   epochs; the median over the epochs is reported (train pass, validation pass, and their sum = the epoch of the logs).
  inference        trained weights (state_dict_best.pth of seed 0), model.eval(), batch size 1 (median over --n_single test events after
                   warm-up) and the throughput with batch size 256 (as analysis/model_cost.py)
  parameters       trainable parameters

Every measurement records the foreign processes on the GPU before and after (flag "shared" if there are any).

    CUDA_VISIBLE_DEVICES=3 python analysis/time_benchmark.py --preset transformer     # the Transformer family incl. the controls
    CUDA_VISIBLE_DEVICES=3 python analysis/time_benchmark.py --preset resid_eq2       # the 16 models with the _resid_eq2 construction
    CUDA_VISIBLE_DEVICES=3 python analysis/time_benchmark.py --preset eq16 --no_train # the 16 models: original, _eq, _eqw (inference only)
    CUDA_VISIBLE_DEVICES=3 python analysis/time_benchmark.py --preset aligned         # the 9 models trained on aligned windows (--data ..._aligned)
    --specs NAME:DATE ...  for any other trained runs; --no_train / --no_infer to skip a part
"""

import argparse, json, os, pickle, subprocess, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())
from datasets import EventDataset, MODELS_SEQ
from models import load_model

W = os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm")
PRESETS = {
    "transformer": {"Transformer": "202610021638", "Transformer_add": "202610030117",
                    "Transformer_add_assoc": "202610031836", "Transformer_add_resid": "202610031836",
                    "Transformer_add_assoc_eq": "202610031948", "Transformer_add_eq": "202610031948", "Transformer_add_resid_eq": "202610031948",
                    "Transformer_add_assoc_eq2": "202610041317", "Transformer_add_eq2": "202610041317", "Transformer_add_resid_eq2": "202610041317",
                    "Transformer_add_flat": "202610041931", "Transformer_add_eq2_wide": "202610041931"},
    "resid_eq2": {**{m + "_resid_eq2": "202610061720" for m in ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "GAT", "CrossModalGating", "TGAT", "GATRNN"]},
                  **{m + "_resid_eq2": "202610071600" for m in ["RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add"]}},
}
# the 16 models of the revision with their original structure (official A2), the equivariant version (_eq) and the parameter-matched one (_eqw)
_B16 = ["MLP", "GNN", "GNN_add", "VAE", "VGNN", "VGNN_add", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add",
        "GAT", "CrossModalGating", "GATRNN", "TGAT"]
# sequence models and Transformer_add_resid_eq2 trained on aligned windows (preprocess_data/align_windows.py): the data set differs
_ALIGNED = ["Transformer_add", "TGAT", "GATRNN", "RNN", "GRNN", "GRNN_add", "VRNN", "GVRNN", "GVRNN_add"]
PRESETS["aligned"] = {m + "_resid_eq2": "202610091700" for m in _ALIGNED}
PRESET_DATA = {"aligned": "anonymized_aligned"}
DATA = "anonymized"  # date_preprocessed_data of the measured runs (set in main)
PRESETS["eq16"] = {**{m: "202610021638" for m in _B16}, **{m + "_eq": "202610041900" for m in _B16},
                   **{m + "_eqw": "202610051115" for m in _B16}}


def gpu_index():
    cv = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    return cv.split(",")[0] if cv else "0"


def foreign_processes():
    """Compute processes on the GPU that are not this process (nvidia-smi sees the processes of every user)."""
    out = subprocess.run(["nvidia-smi", "-i", gpu_index(), "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                         capture_output=True, text=True).stdout.strip().splitlines()
    return [l for l in out if l.strip() and l.split(",")[0].strip() != str(os.getpid())]


def wait_for_idle(limit_s):
    t0 = time.time()
    while foreign_processes():
        if time.time() - t0 > limit_s:
            sys.exit(f"GPU {gpu_index()} is not idle after {limit_s} s: {foreign_processes()}")
        time.sleep(20)


def sync():
    torch.cuda.synchronize()


def to_cuda(items, is_seq):
    if is_seq:  # (batch, T, agents, feat) -> (T, batch, agents, feat), as main_completion.py (all keys)
        items = {k: v.permute(1, 0, 2, 3) for k, v in items.items()}
    return {k: v.cuda() for k, v in items.items()}


def dataset_args(model_name, bs):
    return argparse.Namespace(data_path=os.environ.get("VC_DATA_ROOT", "data"), batchsize=bs, model=model_name, n_all_agents=23,
                              date_preprocessed_data=DATA)


def train_time(model_name, params, n_epochs, bs):
    is_seq = model_name in MODELS_SEQ
    torch.manual_seed(0)
    model = load_model(model_name, params).cuda()
    d_tr, d_va = EventDataset(dataset_args(model_name, bs), train=1), EventDataset(dataset_args(model_name, bs), train=0)
    kw = {"num_workers": 4, "pin_memory": True}
    l_tr = DataLoader(d_tr, batch_size=bs, shuffle=True, **kw)
    l_va = DataLoader(d_va, batch_size=bs, shuffle=False, **kw)
    opt = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params["lr"])
    info_tr, info_va = d_tr.players_vel_info, d_va.players_vel_info
    tr, va = [], []
    for ep in range(n_epochs + 1):
        sync(); t0 = time.perf_counter()
        model.train()
        for items in l_tr:
            items = to_cuda(items, is_seq)
            opt.zero_grad()
            bl, _ = model(items, info_tr)
            (params["Beta"] * bl["L_kl"] + bl["L_rec"]).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0e10)
            opt.step()
        sync(); t1 = time.perf_counter()
        model.eval()
        for items in l_va:
            items = to_cuda(items, is_seq)
            model.sample(items, info_va, n_sample=1)
        sync(); t2 = time.perf_counter()
        if ep > 0:  # the first epoch is the warm-up
            tr.append(t1 - t0); va.append(t2 - t1)
    return float(np.median(tr)), float(np.median(va)), len(d_tr)


def infer_time(model_name, params, run_dir, n_single, n_warmup, bs):
    is_seq = model_name in MODELS_SEQ
    model = load_model(model_name, params)
    model.load_state_dict(torch.load(os.path.join(run_dir, "state_dict_best.pth"), map_location="cpu", weights_only=True))
    model.cuda().eval()
    ds = EventDataset(dataset_args(model_name, bs), train=-1)
    info = ds.players_vel_info

    def batches(d, b):
        for items in DataLoader(d, batch_size=b, shuffle=False):
            items = {k: (v if k == "frames" else (v.permute(1, 0, 2, 3) if is_seq else v)) for k, v in items.items()}
            yield {k: v.cuda() for k, v in items.items()}
    times = []
    with torch.no_grad():
        for k, items in enumerate(batches(Subset(ds, range(min(n_warmup + n_single, len(ds)))), 1)):
            sync(); t0 = time.perf_counter()
            model.sample(items, info, n_sample=1, TEST=True)
            sync()
            if k >= n_warmup:
                times.append(time.perf_counter() - t0)
        loader = list(batches(ds, bs))
        model.sample(loader[0], info, n_sample=1, TEST=True); sync()
        t0 = time.perf_counter()
        for items in loader:
            model.sample(items, info, n_sample=1, TEST=True)
        sync()
        total = time.perf_counter() - t0
    return float(np.median(times) * 1e3), float(np.percentile(times, 90) * 1e3), float(len(ds) / total)


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--preset", choices=list(PRESETS))
    cli.add_argument("--specs", nargs="*", default=[], help="NAME:DATE of a trained run (seed 0)")
    cli.add_argument("--epochs", type=int, default=3)
    cli.add_argument("--n_single", type=int, default=300)
    cli.add_argument("--n_warmup", type=int, default=20)
    cli.add_argument("--batchsize", type=int, default=256)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--data", default=None, help="date_preprocessed_data of the runs (default: that of the preset, else the anonymized data)")
    cli.add_argument("--no_train", action="store_true")
    cli.add_argument("--no_infer", action="store_true")
    cli.add_argument("--wait_idle_s", type=int, default=3600, help="wait this long for a GPU without other processes")
    cli.add_argument("--allow_shared", action="store_true", help="smoke test only: measure even if other processes use the GPU")
    cli.add_argument("--out", default=None)
    opt = cli.parse_args()
    global DATA
    DATA = opt.data or PRESET_DATA.get(opt.preset, DATA)
    specs = dict(PRESETS[opt.preset]) if opt.preset else {}
    specs.update(dict(s.split(":") for s in opt.specs))
    out_path = opt.out or f"analysis/results/time_benchmark_{opt.preset or 'specs'}.json"
    hw = torch.cuda.get_device_name(0)
    print(f"device: {hw} (physical GPU {gpu_index()}), models: {len(specs)}, data: {DATA}", flush=True)
    results = json.load(open(out_path))["results"] if os.path.isfile(out_path) else {}
    for name, date in specs.items():
        run_dir = f"{W}/{date}/{name}_events_{opt.seed}_rectangular_{opt.batchsize}"
        if name in results:
            continue
        if not os.path.isfile(os.path.join(run_dir, "state_dict_best.pth")) or not os.path.isfile(os.path.join(run_dir, "params.p")):
            print(f"{name:<28} no trained run in {run_dir}: skipped", flush=True)
            continue
        params = pickle.load(open(os.path.join(run_dir, "params.p"), "rb"))
        if not opt.allow_shared:
            wait_for_idle(opt.wait_idle_s)
        before = foreign_processes()
        r = {"date": date, "trainable_params": int(sum(p.numel() for p in load_model(params["model"], params).parameters() if p.requires_grad))}
        if not opt.no_train:
            r["train_s"], r["val_s"], r["n_train"] = train_time(params["model"], params, opt.epochs, opt.batchsize)
            r["epoch_s"] = r["train_s"] + r["val_s"]
        if not opt.no_infer:
            r["latency_ms_batch1_median"], r["latency_ms_batch1_p90"], r["throughput_events_per_s"] = infer_time(
                params["model"], params, run_dir, opt.n_single, opt.n_warmup, opt.batchsize)
        after = foreign_processes()
        r["shared"] = bool(before or after)
        r["foreign_processes"] = before + after
        results[name] = r
        print(f"{name:<28} params {r['trainable_params']:>9,}  epoch " + (f"{r['epoch_s']:6.1f} s (train {r['train_s']:.1f} + val {r['val_s']:.1f})" if not opt.no_train else "-")
              + "  latency " + (f"{r['latency_ms_batch1_median']:6.2f} ms" if not opt.no_infer else "-") + ("   SHARED" if r["shared"] else ""), flush=True)
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        json.dump({"device": hw, "gpu": gpu_index(), "data": DATA, "epochs": opt.epochs, "results": results}, open(out_path, "w"), indent=1)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
