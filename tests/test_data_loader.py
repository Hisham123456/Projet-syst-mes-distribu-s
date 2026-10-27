import csv
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import RandomSampler, SequentialSampler

from dr_project.common import digest
from dr_project.data_loader import FDR26Dataset, build_data_loader, build_data_loaders
from dr_project.preprocess_cpu import get_transforms


class DataLoaderTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.splits = self.root / "splits"
        self.images.mkdir()
        self.splits.mkdir()
        self.rows = []
        rng = np.random.default_rng(123)
        for index in range(10):
            image_id = f"sample_{index}"
            Image.fromarray(rng.integers(0, 256, (29, 43, 3), dtype=np.uint8)).save(
                self.images / f"{image_id}.png")
            self.rows.append(dict(image_path=f"{image_id}.png", image_id=image_id,
                                  label=(index * 3) % 5))
        audit = {"files": {}}
        for split in ("train", "val", "test"):
            target = self.splits / f"{split}.csv"
            with target.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(stream, fieldnames=self.rows[0].keys())
                writer.writeheader()
                writer.writerows(self.rows)
            audit["files"][split] = digest(target)
        (self.splits / "audit.json").write_text(json.dumps(audit), encoding="utf-8")

    def loader(self, split):
        return build_data_loader(split, images_dir=self.images, splits_dir=self.splits)

    def test_batches_preserve_image_label_identifier_and_eval_order(self):
        for split in ("val", "test"):
            loader = self.loader(split)
            self.assertEqual(loader.num_workers, 0)
            self.assertIsInstance(loader.sampler, SequentialSampler)
            seen = []
            for images, labels, ids in loader:
                self.assertEqual(images.shape[1:], (3, 384, 384))
                self.assertEqual(images.dtype, torch.float32)
                self.assertEqual(images.device.type, "cpu")
                self.assertEqual(labels.dtype, torch.int64)
                for tensor, label, image_id in zip(images, labels, ids):
                    row = self.rows[len(seen)]
                    self.assertEqual(image_id, row["image_id"])
                    self.assertEqual(label.item(), row["label"])
                    with Image.open(self.images / row["image_path"]) as image:
                        expected = loader.dataset.transform(image.convert("RGB"))
                    self.assertTrue(torch.equal(tensor, expected))
                    seen.append(image_id)
            self.assertEqual(seen, [row["image_id"] for row in self.rows])
            self.assertEqual(images.shape[0], 2)  # final partial batch is kept

    def test_validation_is_deterministic_and_matches_lab_reference(self):
        dataset = self.loader("val").dataset
        first = dataset[0][0]
        torch.manual_seed(999)
        torch.rand(100)
        self.assertTrue(torch.equal(first, dataset[0][0]))
        with Image.open(self.images / self.rows[0]["image_path"]) as image:
            resized = np.array(image.resize((384, 384), Image.Resampling.BILINEAR))
        lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        rgb = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
        expected = torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255
        mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]
        torch.testing.assert_close(first, (expected - mean) / std, rtol=0, atol=0)

    def test_training_is_shuffled_reproducible_and_keeps_labels(self):
        loader = self.loader("train")
        self.assertIsInstance(loader.sampler, RandomSampler)
        first_images, first_labels, first_ids = next(iter(loader))
        images, labels, ids = next(iter(self.loader("train")))
        self.assertEqual(first_images.shape, (8, 3, 384, 384))
        self.assertTrue(torch.equal(first_images, images))
        self.assertTrue(torch.equal(first_labels, labels))
        self.assertEqual(first_ids, ids)
        self.assertNotEqual(list(ids), [row["image_id"] for row in self.rows[:8]])
        by_id = {row["image_id"]: row["label"] for row in self.rows}
        self.assertEqual(labels.tolist(), [by_id[image_id] for image_id in ids])
        train, _ = get_transforms()
        with Image.open(self.images / self.rows[0]["image_path"]) as image:
            self.assertFalse(torch.equal(train(image), train(image)))

    def test_missing_and_corrupt_images_raise_without_substitution(self):
        dataset = self.loader("val").dataset
        target = self.images / self.rows[0]["image_path"]
        target.unlink()
        with self.assertRaisesRegex(OSError, "sample_0"):
            dataset[0]
        target.write_bytes(b"invalid image")
        with self.assertRaisesRegex(OSError, "sample_0"):
            next(iter(self.loader("val")))

    def test_all_loaders_and_manifest_integrity(self):
        loaders = build_data_loaders(images_dir=self.images, splits_dir=self.splits)
        self.assertEqual(set(loaders), {"train", "val", "test"})
        self.assertTrue(all(loader.batch_size == 8 for loader in loaders.values()))
        with (self.splits / "val.csv").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            FDR26Dataset("val", self.images, self.splits)


if __name__ == "__main__":
    unittest.main()
