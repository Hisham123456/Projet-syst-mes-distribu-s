"""Shared utilities: seeding, timing, JSON helpers."""
import os, random, time, json
import numpy as np
import torch


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # TODO(A1.5): torch.backends.cudnn.benchmark = True picks the fastest conv
    #             algorithm but makes runs less reproducible. Decide what you want
    #             for the baseline and for the benchmarks, and justify it in the report.


class GpuTimer:
    """Wall-clock timer. Use as: `with GpuTimer(device) as t: ...` then `t.elapsed`.

    TODO(A2.1): CUDA kernels are asynchronous, so perf_counter alone only
    measures the time to launch them. When the device is CUDA, call
    torch.cuda.synchronize(self.device) before reading the clock in __enter__
    and again in __exit__.
    """

    def __init__(self, device):
        self.device = torch.device(device)
        self.elapsed = 0.0

    def __enter__(self):
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self._t0
        return False


def save_json(obj, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)


def load_json(path):
    with open(path) as f:
        return json.load(f)
