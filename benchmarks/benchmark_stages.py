"""Stage-level benchmark: CPU vs GPU per pipeline stage on one real batch of
training images. Writes results/timing.csv (assignments A2 and A5).

Every stage processes the same N images, so per-stage times are comparable.
The CPU rows are provided. The GPU rows are yours.

Methodology (brief, section 7): warm-up runs, CUDA synchronized around every
timed region (GpuTimer), several repeats, mean +/- std.
"""
import os, sys, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import pandas as pd
import yaml
import torch

from src.utils import GpuTimer, set_seed
from src.data_loader import find_images, split_by_patient, DRDataset
from src import preprocess_cpu, filters_cpu
from src.model import build_model


def bench(fn, device, repeats, warmup=2):
    """Run fn `warmup` times (discarded), then `repeats` timed runs. Returns (mean, std) in s."""
    for _ in range(warmup):
        fn()
    ts = []
    for _ in range(repeats):
        with GpuTimer(device) as t:
            fn()
        ts.append(t.elapsed)
    return float(np.mean(ts)), float(np.std(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--n", type=int, default=64, help="images per stage")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--out", default="results/timing.csv")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    set_seed(cfg["seed"])
    has_cuda = torch.cuda.is_available()
    cpu = torch.device("cpu")
    N, R = args.n, args.repeats
    rows = []

    def add(stage, dev, fn, device, repeats=R, warmup=2, n=N):
        m, s = bench(fn, device, repeats, warmup)
        rows.append({"stage": stage, "device": dev, "mean_s": m, "std_s": s,
                     "n_images": n, "imgs_per_s": n / m})
        print(f"{stage:20s} {dev:4s} {m * 1e3:10.2f} ms +/- {s * 1e3:.2f}", flush=True)

    items = find_images(cfg["data"]["root"])
    train = split_by_patient(items, cfg["data"]["train_frac"], cfg["data"]["val_frac"],
                             cfg["data"].get("split_seed", cfg["seed"]))["train"][:N]
    ds = DRDataset(train, cfg["data"]["image_size"])

    # ---- CPU rows (given) ----
    add("loading", "cpu", lambda: [ds.load(i) for i in range(N)], cpu, repeats=3, warmup=1)
    u8 = np.stack([ds.load(i) for i in range(N)])            # N,H,W,3 uint8
    f32 = u8.astype(np.float32) / 255.0
    add("preprocessing", "cpu", lambda: preprocess_cpu.batch_preprocess_cpu(f32, cfg), cpu)
    add("filter_gaussian", "cpu", lambda: [filters_cpu.gaussian_blur(i) for i in f32], cpu)
    add("filter_median", "cpu", lambda: [filters_cpu.median_filter(i) for i in f32], cpu)
    add("filter_green", "cpu",
        lambda: [np.ascontiguousarray(filters_cpu.green_channel(i)) for i in f32], cpu)
    add("filter_contrast", "cpu", lambda: [filters_cpu.contrast_stretch(i) for i in f32], cpu)
    rng = np.random.default_rng(0)
    add("augmentation", "cpu", lambda: [preprocess_cpu.augment_image(i, rng) for i in f32], cpu)

    x = torch.from_numpy(preprocess_cpu.batch_preprocess_cpu(f32, cfg))
    y = torch.tensor([l for _, l in train])
    model = build_model(cfg["train"]["model"], cfg["data"]["num_classes"])
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
    crit = torch.nn.CrossEntropyLoss()

    def train_step():
        model.train()
        opt.zero_grad(set_to_none=True)
        crit(model(x), y).backward()
        opt.step()

    def infer():
        model.eval()
        with torch.no_grad():
            model(x)

    add("train_step", "cpu", train_step, cpu, repeats=3, warmup=1)
    add("inference", "cpu", infer, cpu, repeats=3, warmup=1)

    # ---- GPU rows (yours) ----
    if has_cuda:
        # TODO(A5.6): add one GPU row for every CPU row above, same N images, same `add` helper:
        #   loading         - pick one: GPU decode with torchvision.io.decode_jpeg(device="cuda")
        #                     (JPEG inputs only) or the uint8 cache from A3. State which one.
        #   h2d_transfer    - a pinned uint8 batch copied to the device (the price every GPU stage pays)
        #   preprocessing   - preprocess_gpu.preprocess_batch_gpu, INCLUDING the host-to-device copy
        #   filter_*        - the four filters_gpu functions on a tensor that is already on the GPU
        #   augmentation    - preprocess_gpu.augment_gpu
        #   train_step / inference - the SAME model as the CPU row on the GPU (so you isolate the
        #                     effect of the device), then once more with AMP. Label the variants.
        # Rule: warm up before timing (the first CUDA call pays for context and kernel loading)
        # and let GpuTimer synchronize.
        pass

    df = pd.DataFrame(rows)
    # TODO(A5.7): add a speedup_vs_cpu column = cpu mean_s / gpu mean_s for every stage that has both.
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
