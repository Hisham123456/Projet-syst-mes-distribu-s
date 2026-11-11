"""Dataset discovery, patient-safe splitting, and DataLoader construction.

The CPU reference below works as is. Assignments A3 and A5 extend it.
"""
import os, random
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

    def load(self, idx):
        """Decode + resize only (the 'loading' stage). Returns HWC uint8."""
        # TODO(A3.2): if self.cache is not None, return the cached uint8 image
        #             (np.asarray(self.cache[idx])) instead of decoding from disk.
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
    """TODO(A3.1): decode + resize every image ONCE into a uint8 array on disk.

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
    raise NotImplementedError("TODO(A3.1): build_cache")


def _worker_init(_):
    # one OpenCV thread per worker: N workers x many cv2 threads oversubscribes the CPU
    import cv2
    cv2.setNumThreads(0)


def load_frozen_splits(data):
    """Keep a shared, conflict-free validation and test set for both runs."""
    from src.data_splits import read_json, read_split, validate_disjoint

    directory = Path(data["frozen_splits"])
    root = Path(data["root"])
    audit = read_json(directory / "audit.json")
    conflicting_ids = {
        image["image_id"]
        for group in audit["byte_identical_groups"] if group["conflicting_labels"]
        for image in group["images"]
    }
    splits = {}
    for name in ("train", "val", "test"):
        rows = read_split(directory, name)
        if name != "train" or data.get("exclude_conflicts_from_train", False):
            rows = [row for row in rows if row["image_id"] not in conflicting_ids]
        splits[name] = rows
    validate_disjoint(splits)

    items = {}
    for name, rows in splits.items():
        items[name] = []
        for row in rows:
            path = root / row["image_path"]
            if not path.is_file():
                raise FileNotFoundError(f"Missing {name} image from frozen manifest: {path}")
            items[name].append((str(path), row["label"]))
    return items


def make_loaders(cfg, device="cpu", gpu_preprocess=False, which=("train", "val", "test")):
    """Build DataLoaders for the requested splits.

    gpu_preprocess=True: workers only decode/resize and return uint8; the
    training loop preprocesses the batch on the GPU (assignment A5).
    """
    data = cfg["data"]
    frozen_splits = data.get("frozen_splits")
    if frozen_splits:
        splits = load_frozen_splits(data)
    else:
        items = find_images(data["root"])
        if not items:
            raise FileNotFoundError(
                f"No images found under {data['root']}/<class>/...  "
                "Run scripts/prepare_fdr26.py or point data.root at the dataset.")
        splits = split_by_patient(items, data["train_frac"], data["val_frac"],
                                  data.get("split_seed", cfg["seed"]))
    nw = data["num_workers"]
    augment = bool(cfg["preprocess"].get("augment", False))
    loaders = {}
    for k in which:
        # augmentation is for the training split only
        tf = None if gpu_preprocess else get_preprocess_transform(cfg, augment and k == "train")
        # TODO(A3.3): if cfg["data"].get("cache_dir") is set, call build_cache for
        #             this split and pass the result to DRDataset(cache=...).
        ds = DRDataset(splits[k], cfg["data"]["image_size"], transform=tf,
                       raw_uint8=gpu_preprocess)
        # TODO(A5.1): add the DataLoader options that feed a GPU efficiently:
        #   pin_memory (only when device == "cuda"), persistent_workers (when nw > 0),
        #   prefetch_factor (cfg["data"].get("prefetch_factor", 2), only valid when nw > 0),
        #   worker_init_fn=_worker_init (when nw > 0).
        loaders[k] = DataLoader(
            ds,
            batch_size=cfg["data"]["batch_size"],
            shuffle=(k == "train"),
            num_workers=nw,
            drop_last=(k == "train" and not frozen_splits),
        )
    return loaders, splits
