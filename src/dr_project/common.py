from __future__ import annotations

import contextlib
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
import time

import numpy as np
import torch
import yaml


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")


def digest(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def config(path):
    value = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if "extends" in value:
        parent = config(Path(path).parent / value.pop("extends"))
        parent.update(value)
        value = parent
    required = {"name", "model", "device", "processing", "image_size", "batch_size", "epochs", "seed"}
    if required - value.keys():
        raise ValueError(f"Missing configuration keys: {required - value.keys()}")
    if value["processing"] not in {"cpu", "batch_cpu", "gpu"}:
        raise ValueError("processing must be cpu, batch_cpu or gpu")
    if value["device"] == "cpu" and (value.get("amp") or value["processing"] == "gpu"):
        raise ValueError("CPU runs cannot use CUDA processing or AMP")
    for key in ("image_size", "batch_size", "epochs"):
        if value[key] <= 0:
            raise ValueError(f"{key} must be positive")
    if value.get("workers", 0) < 0:
        raise ValueError("workers must be non-negative")
    return value


def device_for(cfg):
    device = torch.device(cfg["device"])
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable. Select a GPU runtime in Colab; no silent CPU fallback.")
    return device


def seed_all(seed, threads=2):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(_):
    seed = torch.initial_seed() % 2**32
    random.seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(1)
    import cv2
    cv2.setNumThreads(1)


def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


class Timings:
    """Diagnostic timings synchronize stages; normal throughput runs do not."""
    def __init__(self, device, enabled=False):
        self.device = device
        self.enabled = enabled
        self.values = {}

    @contextlib.contextmanager
    def stage(self, name):
        with torch.profiler.record_function(name):
            if not self.enabled:
                yield
                return
            sync(self.device)
            start = time.perf_counter()
            yield
            sync(self.device)
            self.values[name] = self.values.get(name, 0.0) + time.perf_counter() - start


def environment():
    import psutil
    versions = {}
    for name in ["torch", "torchvision", "timm", "numpy", "Pillow", "opencv-python-headless",
                 "scikit-learn", "openpyxl", "PyYAML", "psutil"]:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    try:
        freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        freeze = "pip unavailable"
    try:
        gpu = subprocess.check_output([
            "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"
        ], text=True, stderr=subprocess.DEVNULL).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        gpu = None
    return {
        "python": sys.version, "os": platform.platform(), "cpu": platform.processor(),
        "logical_cpus": os.cpu_count(), "ram_bytes": psutil.virtual_memory().total,
        "cuda_available": torch.cuda.is_available(), "cuda_runtime": torch.version.cuda,
        "gpu": gpu, "packages": versions, "pip_freeze": freeze,
    }
