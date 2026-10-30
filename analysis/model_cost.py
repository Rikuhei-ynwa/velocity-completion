"""Trainable parameters and inference latency of trained models (reviewers R1-6, R2-7).

For each model it loads params.p and state_dict_best.pth of a trained run and measures, on the
test set, with the same forward pass used for evaluation (model.sample(..., TEST=True)):
  - trainable parameters
  - latency per event with batch size 1 (median over --n_single events, after warm-up)
  - throughput with the evaluation batch size (events/s over the whole test set)
CUDA work is synchronized before each timer read. Models run in eval mode (BatchNorm running
statistics), as needed for single-event inference. SVM/RF are skipped (not torch models).

    python analysis/model_cost.py --date 202609301600 --models MLP GNN_add RNN GRNN GRNN_add Transformer TGAT
"""

import argparse, json, os, pickle, sys, time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

sys.path.insert(0, os.getcwd())
from datasets import EventDataset, MODELS_SEQ
from models import load_model


def batches(dataset, batch_size, is_seq, device):
    for items in DataLoader(dataset, batch_size=batch_size, shuffle=False, drop_last=False):
        if is_seq:
            items = {k: (v if k == "frames" else v.permute(1, 0, 2, 3)) for k, v in items.items()}
        yield {k: v.to(device) for k, v in items.items()}


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--date", required=True)
    cli.add_argument("--models", nargs="+", required=True)
    cli.add_argument("--seed", type=int, default=0)
    cli.add_argument("--weights_dir", default=os.path.join(os.environ.get("VC_RESULTS_ROOT", "results"), "weights", "vel_1_acc_0_norm"))
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--date_preprocessed_data", default="anonymized")
    cli.add_argument("--batchsize", type=int, default=256)
    cli.add_argument("--n_single", type=int, default=300, help="events timed one at a time")
    cli.add_argument("--n_warmup", type=int, default=20)
    cli.add_argument("--out", default="analysis/results/model_cost.json")
    opt = cli.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    hw = torch.cuda.get_device_name(0) if device == "cuda" else f"CPU ({os.cpu_count()} threads)"
    print(f"device: {hw}")
    results = {}
    for name in opt.models:
        run_dir = os.path.join(opt.weights_dir, opt.date, f"{name}_events_{opt.seed}_rectangular_{opt.batchsize}")
        params = pickle.load(open(os.path.join(run_dir, "params.p"), "rb"))
        model_name = params["model"]
        if model_name in ("SVM", "RF"):
            continue
        model = load_model(model_name, params)
        model.load_state_dict(torch.load(os.path.join(run_dir, "state_dict_best.pth"),
                                         map_location="cpu", weights_only=True))
        model.to(device)
        # Deployment mode: BatchNorm uses its running statistics. main_completion.py evaluates in
        # train mode (batch statistics), which cannot predict a single event at all.
        model.eval()
        n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        dataset_args = argparse.Namespace(data_path=opt.data_path, batchsize=opt.batchsize, model=model_name,
                                     n_all_agents=23, date_preprocessed_data=opt.date_preprocessed_data)
        dataset = EventDataset(dataset_args, train=-1)
        info = dataset.players_vel_info
        is_seq = model_name in MODELS_SEQ
        with torch.no_grad():
            # batch size 1: warm-up, then time each event separately
            single = Subset(dataset, range(min(opt.n_warmup + opt.n_single, len(dataset))))
            times = []
            for k, items in enumerate(batches(single, 1, is_seq, device)):
                sync(device)
                t0 = time.perf_counter()
                model.sample(items, info, n_sample=1, TEST=True)
                sync(device)
                if k >= opt.n_warmup:
                    times.append(time.perf_counter() - t0)
            # evaluation batch size over the whole test set (after one warm-up batch)
            loader = list(batches(dataset, opt.batchsize, is_seq, device))
            model.sample(loader[0], info, n_sample=1, TEST=True)
            sync(device)
            t0 = time.perf_counter()
            for items in loader:
                model.sample(items, info, n_sample=1, TEST=True)
            sync(device)
            total = time.perf_counter() - t0
        results[name] = {"model": model_name, "trainable_params": int(n_params),
                         "latency_ms_batch1_median": float(np.median(times) * 1e3),
                         "latency_ms_batch1_p90": float(np.percentile(times, 90) * 1e3),
                         "throughput_events_per_s": float(len(dataset) / total)}
        r = results[name]
        print(f"{name:<18} params {r['trainable_params']:>10,}   batch 1: {r['latency_ms_batch1_median']:7.2f} ms "
              f"(p90 {r['latency_ms_batch1_p90']:.2f})   batch {opt.batchsize}: {r['throughput_events_per_s']:>9.0f} events/s")
        del model
        if device == "cuda":
            torch.cuda.empty_cache()

    os.makedirs(os.path.dirname(opt.out) or ".", exist_ok=True)
    json.dump({"device": hw, "date": opt.date, "seed": opt.seed, "results": results}, open(opt.out, "w"), indent=1)
    print(f"Saved: {opt.out}")


if __name__ == "__main__":
    main()
