"""Measure direct loading, cache preparation and reuse on the same images."""
import argparse
import copy
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import yaml
from src.data_loader import DRDataset, make_loaders


def measure_loading(dataset, count, repeats=3):
    timings = []
    for _ in range(repeats):
        start = time.perf_counter()
        for index in range(count):
            # Copy forces reading the pixels, including memory-mapped pages.
            dataset.load(index).copy()
        timings.append(time.perf_counter() - start)
    return {"mean_s": float(np.mean(timings)), "std_s": float(np.std(timings))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/cpu-cache.yaml")
    parser.add_argument("--n", type=int, default=64)
    parser.add_argument("--out", default="results/cache_loading.json")
    args = parser.parse_args()
    if args.n < 1:
        parser.error("--n must be positive")
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    direct_cfg = copy.deepcopy(cfg)
    direct_cfg["data"].pop("cache_dir")
    direct_loaders, splits = make_loaders(direct_cfg)
    count = min(args.n, len(splits["train"]))
    direct = DRDataset(splits["train"], cfg["data"]["image_size"])
    uncached = measure_loading(direct, count)
    cache_dir = Path(cfg["data"]["cache_dir"])
    existing = list(cache_dir.glob("*.npy"))
    start = time.perf_counter()
    loaders, cached_splits = make_loaders(cfg)
    preparation_s = time.perf_counter() - start
    assert splits == cached_splits
    start = time.perf_counter()
    reused, _ = make_loaders(cfg)
    reuse_s = time.perf_counter() - start
    cached = loaders["train"].dataset
    first_read = measure_loading(cached, count, repeats=1)
    for index in range(count):
        np.testing.assert_array_equal(direct.load(index), cached.load(index))
    warm = measure_loading(reused["train"].dataset, count)
    # Exercise the normal Windows DataLoader worker path.
    direct_batch = next(iter(direct_loaders["val"]))
    cached_batch = next(iter(loaders["val"]))
    for expected, actual in zip(direct_batch, cached_batch):
        np.testing.assert_array_equal(expected.numpy(), actual.numpy())
    files = {Path(loader.dataset.cache.filename) for loader in loaders.values()}
    result = {
        "config": cfg, "images_measured": count, "repeats": 3,
        "splits": {key: len(value) for key, value in splits.items()},
        "cache_files_present_before": len(existing),
        "cache_preparation_s": preparation_s, "cache_reopen_s": reuse_s,
        "cache_bytes": sum(path.stat().st_size for path in files),
        "direct_loading": uncached, "first_cached_read": first_read,
        "warm_cached_read": warm,
        "loading_speedup": uncached["mean_s"] / warm["mean_s"],
        "pixel_equality_checked": count, "validation_batch_equal": True,
        "scope": "Loading only; OS page cache is not cleared. End-to-end training measured separately."
    }
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
