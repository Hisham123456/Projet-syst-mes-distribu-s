"""Load the frozen FDR26 partitions for CPU training and evaluation."""

from pathlib import Path

from PIL import Image
import torch
from torch.utils.data import DataLoader, Dataset

from .common import seed_all
from .preprocess_cpu import IMAGE_SIZE, get_transforms
from .splits import read_split

SPLITS = ("train", "val", "test")
DEFAULT_IMAGES = "data/FDR26_classification/images"
DEFAULT_SPLITS = "data/splits"


class FDR26Dataset(Dataset):
    """Each sample is (CPU image tensor, integer label, image identifier)."""

    def __init__(self, split, images_dir=DEFAULT_IMAGES, splits_dir=DEFAULT_SPLITS):
        if split not in SPLITS:
            raise ValueError(f"Unknown split: {split!r}; expected one of {SPLITS}")
        self.rows = read_split(splits_dir, split)
        self.images_dir = Path(images_dir)
        train, evaluation = get_transforms()
        self.transform = train if split == "train" else evaluation
        for row in self.rows:
            if row["label"] not in range(5):
                raise ValueError(f"Invalid label for {row['image_id']}: {row['label']}")

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        path = self.images_dir / row["image_path"]
        try:
            with Image.open(path) as image:
                image = image.convert("RGB")
                image.load()
        except OSError as error:
            raise OSError(f"Cannot read image {row['image_id']} at {path}: {error}") from error
        return self.transform(image), row["label"], row["image_id"]


def build_data_loader(split, batch_size=8, seed=42,
                      images_dir=DEFAULT_IMAGES, splits_dir=DEFAULT_SPLITS):
    """Seed CPU augmentation RNGs and shuffle only training data.

    With num_workers=0, torchvision augmentations use the main process RNG.
    Recreate the loader with the same seed to replay a run; do not reseed epochs.
    """
    seed_all(seed)
    dataset = FDR26Dataset(split, images_dir=images_dir, splits_dir=splits_dir)
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=split == "train",
                      num_workers=0, generator=generator, drop_last=False)


def build_data_loaders(batch_size=8, seed=42,
                       images_dir=DEFAULT_IMAGES, splits_dir=DEFAULT_SPLITS):
    """Return train/val/test loaders using the same frozen manifests."""
    return {split: build_data_loader(split, batch_size, seed, images_dir, splits_dir)
            for split in SPLITS}


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Verify the first FDR26 CPU batch")
    parser.add_argument("--split", choices=SPLITS, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--images-dir", default=DEFAULT_IMAGES)
    parser.add_argument("--splits-dir", default=DEFAULT_SPLITS)
    args = parser.parse_args()
    loader = build_data_loader(args.split, args.batch_size, args.seed,
                               args.images_dir, args.splits_dir)
    images, labels, image_ids = next(iter(loader))
    count = min(args.batch_size, len(loader.dataset))
    assert images.shape == (count, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert images.dtype == torch.float32 and images.device.type == "cpu"
    assert torch.isfinite(images).all()
    assert labels.shape == (count,) and labels.dtype == torch.int64
    assert ((labels >= 0) & (labels <= 4)).all()
    assert len(image_ids) == count
    rows_by_id = {row["image_id"]: row for row in loader.dataset.rows}
    assert all(rows_by_id[image_id]["label"] == label
               for image_id, label in zip(image_ids, labels.tolist()))
    print(f"Split : {args.split} ({len(loader.dataset)} images), seed={args.seed}, num_workers=0")
    print(f"Images : forme {tuple(images.shape)}, type {images.dtype}, sur {images.device}")
    print(f"Labels : forme {tuple(labels.shape)}, type {labels.dtype}, valeurs {labels.tolist()}")
    print(f"Identifiants : {len(image_ids)} identifiants {list(image_ids)}")
    print("Verification du premier lot : OK")


if __name__ == "__main__":
    main()
