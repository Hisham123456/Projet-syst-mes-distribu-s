"""Dataset discovery, patient-safe splitting, and DataLoader construction.

The CPU reference below works as is. Assignments A3 and A5 extend it.
"""
import os, random
import hashlib
import json
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def find_images(root):
    """Walk root/<class_id>/*.png and return [(path, label), ...]."""
    items = []
    for dirpath, _, fnames in os.walk(root):
        try:
            label = int(os.path.basename(dirpath))
        except ValueError:
            continue
        if not 0 <= label <= 4:
            continue
        for f in sorted(fnames):
            if f.lower().endswith(IMG_EXTS):
                items.append((os.path.join(dirpath, f), label))
    return items


def patient_id(path):
    """Patient id = filename prefix before the LAST underscore (<patient>_<image>.png)."""
    return os.path.splitext(os.path.basename(path))[0].rsplit("_", 1)[0]


def split_by_patient(items, train_frac, val_frac, seed):
    by_pat = {}
    for p, l in items:
        by_pat.setdefault(patient_id(p), []).append((p, l))
    pats = sorted(by_pat)
    random.Random(seed).shuffle(pats)
    n = len(pats)
    n_train, n_val = int(n * train_frac), int(n * val_frac)
    train_p = set(pats[:n_train])
    val_p = set(pats[n_train:n_train + n_val])
    splits = {"train": [], "val": [], "test": []}
    for p in pats:
        key = "train" if p in train_p else "val" if p in val_p else "test"
        splits[key].extend(by_pat[p])
    return splits


class DRDataset(Dataset):
    """Loads, decodes and resizes an image; returns (C,H,W) tensor + label.

    CPU preprocessing is injected through `transform`.
    With `raw_uint8=True` the image is returned untouched (uint8, CHW) so a
    GPU can preprocess the whole batch after a 4x smaller host-to-device copy.
    """

    def __init__(self, items, image_size, transform=None, raw_uint8=False, cache=None):
        self.items = items
        self.image_size = tuple(image_size)
        self.transform = transform
        self.raw_uint8 = raw_uint8
        self.cache = cache

    def __len__(self):
        return len(self.items)

    def __getstate__(self):
        state = self.__dict__.copy()
        if isinstance(self.cache, np.memmap):
            # Windows workers reopen the file instead of receiving all its pixels.
            state["_cache_spec"] = (str(self.cache.filename), self.cache.dtype,
                                    self.cache.offset, self.cache.shape)
            state["cache"] = None
        return state

    def __setstate__(self, state):
        spec = state.pop("_cache_spec", None)
        self.__dict__.update(state)
        if spec is not None:
            filename, dtype, offset, shape = spec
            self.cache = np.memmap(filename, dtype=dtype, mode="r",
                                   offset=offset, shape=shape)

    def load(self, idx):
        """Decode + resize only (the 'loading' stage). Returns HWC uint8."""
        if self.cache is not None:
            return np.asarray(self.cache[idx])
        path, _ = self.items[idx]
        with Image.open(path) as im:
            im = im.convert("RGB").resize(self.image_size, Image.Resampling.BILINEAR)
            return np.asarray(im, dtype=np.uint8)

    def __getitem__(self, idx):
        u8 = self.load(idx)
        label = int(self.items[idx][1])
        if self.raw_uint8:
            return torch.from_numpy(np.ascontiguousarray(u8.transpose(2, 0, 1))), label
        x = u8.astype(np.float32) / 255.0
        if self.transform is not None:
            x = self.transform(x)
        else:
            x = x.transpose(2, 0, 1)
        return torch.from_numpy(np.ascontiguousarray(x)), label


class CpuTransform:
    """CPU preprocessing for one image: HWC float [0,1] -> CHW float.

    A class (not a lambda) so it pickles into spawned DataLoader workers.
    """

    def __init__(self, cfg, augment=False):
        self.cfg = cfg
        self.augment = augment
        self._rng = None

    def __call__(self, img):
        from src import preprocess_cpu
        if self.augment:
            if self._rng is None:      # created inside the worker, seeded per worker
                self._rng = np.random.default_rng(torch.initial_seed() % 2**32)
            img = preprocess_cpu.augment_image(img, self._rng)
        return preprocess_cpu.preprocess_image(img, self.cfg)


def get_preprocess_transform(cfg, augment=False):
    return CpuTransform(cfg, augment)


def build_cache(items, image_size, cache_dir, workers=8):
    """Decode and resize once into an atomically published, read-only NPY cache.

    Requirements:
      - return a read-only np.memmap of shape (len(items), H, W, 3), dtype uint8,
        where cache[i] equals DRDataset(items, image_size).load(i)
      - image_size is (W, H) as in the config; the array is (N, H, W, 3)
      - write to a temporary file and rename it, so an interrupted run never
        leaves a half-written cache behind (on Windows, delete the memmap object
        before os.replace: a mapped file cannot be renamed)
      - the file name must change when the image list or the size changes
        (hint: hashlib.md5 over the paths), so a stale cache is never reused
      - fill the array with a thread pool (PIL releases the GIL while decoding)
    Test: tests/test_data_pipeline.py::test_cache_matches_direct_decode
    """
    if not items or len(image_size) != 2 or min(image_size) <= 0 or workers < 1:
        raise ValueError("Cache requires images, positive width/height and workers >= 1")
    width, height = image_size
    shape = (len(items), height, width, 3)
    sources = []
    for filename, _ in items:
        path = Path(filename).resolve()
        stat = path.stat()
        sources.append((str(path), stat.st_size, stat.st_mtime_ns))
    key = hashlib.sha256(json.dumps(
        ["RGB-BILINEAR-v1", list(image_size), sources]
    ).encode("utf-8")).hexdigest()
    directory = Path(cache_dir)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{key}.npy"
    if not target.exists():
        fd, temporary = tempfile.mkstemp(prefix=key + "-", suffix=".tmp", dir=directory)
        os.close(fd)
        array = None
        try:
            array = np.lib.format.open_memmap(temporary, mode="w+", dtype=np.uint8, shape=shape)
            dataset = DRDataset(items, image_size)

            def fill(index):
                array[index] = dataset.load(index)

            with ThreadPoolExecutor(max_workers=workers) as pool:
                for _ in pool.map(fill, range(len(items))):
                    pass
            array.flush()
            array._mmap.close()
            array = None
            try:
                os.replace(temporary, target)
            except PermissionError:
                # Another builder may have published and opened the same cache on Windows.
                if not target.exists():
                    raise
        finally:
            if array is not None:
                array._mmap.close()
            Path(temporary).unlink(missing_ok=True)
    cache = np.load(target, mmap_mode="r", allow_pickle=False)
    if cache.shape != shape or cache.dtype != np.uint8:
        cache._mmap.close()
        raise ValueError(f"Invalid cache shape or dtype: {target}")
    return cache


def _worker_init(_):
    # one OpenCV thread per worker: N workers x many cv2 threads oversubscribes the CPU
    import cv2
    cv2.setNumThreads(0)


def make_loaders(cfg, device="cpu", gpu_preprocess=False, which=("train", "val", "test")):
    """Build DataLoaders for the requested splits.

    gpu_preprocess=True: workers only decode/resize and return uint8; the
    training loop preprocesses the batch on the GPU (assignment A5).
    """
    items = find_images(cfg["data"]["root"])
    if not items:
        raise FileNotFoundError(
            f"No images found under {cfg['data']['root']}/<class>/...  "
            "Run scripts/prepare_fdr26.py or point data.root at the dataset.")
    splits = split_by_patient(
        items, cfg["data"]["train_frac"], cfg["data"]["val_frac"],
        cfg["data"].get("split_seed", cfg["seed"]))
    nw = cfg["data"]["num_workers"]
    augment = bool(cfg["preprocess"].get("augment", False))
    loaders = {}
    for k in which:
        # augmentation is for the training split only
        tf = None if gpu_preprocess else get_preprocess_transform(cfg, augment and k == "train")
        cache = None
        if cfg["data"].get("cache_dir"):
            cache = build_cache(splits[k], cfg["data"]["image_size"],
                                cfg["data"]["cache_dir"])
        ds = DRDataset(splits[k], cfg["data"]["image_size"], transform=tf,
                       raw_uint8=gpu_preprocess, cache=cache)
        # TODO(A5.1): add the DataLoader options that feed a GPU efficiently:
        #   pin_memory (only when device == "cuda"), persistent_workers (when nw > 0),
        #   prefetch_factor (cfg["data"].get("prefetch_factor", 2), only valid when nw > 0),
        #   worker_init_fn=_worker_init (when nw > 0).
        loaders[k] = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=(k == "train"),
            num_workers=nw,
            drop_last=(k == "train"),
        )
    return loaders, splits
