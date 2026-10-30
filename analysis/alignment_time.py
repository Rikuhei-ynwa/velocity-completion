"""Cost of the window alignment (preprocess_data/align_windows.py), the preprocessing the sequence models need at inference time.

Times alignment() (Hungarian matching of the players of every earlier event with those of the following event, per team) on the test
windows, on the CPU, one thread, after one warm-up call. Reports the time per window (10 events) and the share of the cost of the
models' inference (analysis/results/time_benchmark_aligned.json, batch size 1).

    python analysis/alignment_time.py [--n 2958]
"""
import argparse, json, os, sys, time

import numpy as np

sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.join(os.getcwd(), "preprocess_data"))
from align_windows import alignment


def main():
    cli = argparse.ArgumentParser()
    cli.add_argument("--data_path", default=os.environ.get("VC_DATA_ROOT", "data"))
    cli.add_argument("--dataset", default="anonymized")
    cli.add_argument("--n", type=int, default=0, help="number of test windows (0 = all)")
    cli.add_argument("--repeats", type=int, default=3)
    cli.add_argument("--out", default="analysis/results/alignment_time.json")
    opt = cli.parse_args()
    z = np.load(os.path.join(opt.data_path, "dataset", opt.dataset, "rectangular", "recurrent", "test.npz"))
    cies = z["cies"][: opt.n or None]
    alignment(cies[:20])  # warm-up
    per_window = []
    for _ in range(opt.repeats):
        t0 = time.perf_counter(); alignment(cies); per_window.append((time.perf_counter() - t0) / len(cies) * 1e3)
    one = []  # one window at a time, as in single-event inference
    for i in range(min(300, len(cies))):
        t0 = time.perf_counter(); alignment(cies[i:i + 1]); one.append((time.perf_counter() - t0) * 1e3)
    r = {"windows": int(len(cies)), "window_events": int(cies.shape[1]), "ms_per_window_batched_median": float(np.median(per_window)),
         "ms_per_window_single_median": float(np.median(one)), "ms_per_window_single_p90": float(np.percentile(one, 90)), "threads": 1}
    os.makedirs(os.path.dirname(opt.out), exist_ok=True)
    json.dump(r, open(opt.out, "w"), indent=1)
    print(json.dumps(r, indent=1))


if __name__ == "__main__":
    main()
