import tempfile
import unittest
from pathlib import Path

import openpyxl
from PIL import Image

from src.data_splits import create_splits, digest, metadata_rows, read_split, validate_disjoint


class SplitTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.images = self.root / "images"
        self.images.mkdir()
        self.metadata = self.root / "classification DR.xlsx"
        book = openpyxl.Workbook()
        sheet = book.active
        sheet.append(["image_id", "classe"])
        sheet.append(["2T2SALA702", 0])
        for patient in range(30):
            for label in range(5):
                image_id = f"{patient:08d}{label:02d}"
                sheet.append([image_id, label])
                Image.new("RGB", (4, 4), (patient, label, 0)).save(self.images / f"{image_id}.png")
        book.save(self.metadata)
        book.close()

    def freeze(self, name="splits"):
        return create_splits(self.metadata, self.images, self.root / name)

    def test_freeze_is_complete_disjoint_reproducible_and_immutable(self):
        before = digest(self.metadata)
        audit = self.freeze()
        self.assertEqual(before, digest(self.metadata))
        self.assertEqual(audit["total_images"], 150)
        self.assertEqual(audit["metadata_entries"], 151)
        self.assertEqual(audit["exclusions"][0]["image_id"], "2T2SALA702")
        self.assertTrue(audit["exclusions"][0]["reason"])
        splits = {name: read_split(self.root / "splits", name) for name in ("train", "val", "test")}
        validate_disjoint(splits)
        self.assertEqual(sum(map(len, splits.values())), 150)
        self.assertEqual(len({r["image_id"] for subset in splits.values() for r in subset}), 150)
        again = self.freeze("again")
        self.assertEqual(audit["files"], again["files"])
        with self.assertRaises(FileExistsError):
            self.freeze()
        target = self.root / "splits" / "train.csv"
        target.write_text(target.read_text() + "\n")
        with self.assertRaisesRegex(ValueError, "changed"):
            read_split(self.root / "splits", "train")

    def test_unexpected_missing_image_fails_without_manifests(self):
        (self.images / "0000000000.png").unlink()
        with self.assertRaises(FileNotFoundError):
            self.freeze()
        self.assertFalse((self.root / "splits").exists())

    def test_corrupt_image_fails_without_manifests(self):
        (self.images / "0000000000.png").write_bytes(b"not an image")
        with self.assertRaises(OSError):
            self.freeze()
        self.assertFalse((self.root / "splits").exists())

    def test_unlabelled_image_and_changed_exclusion_are_rejected(self):
        extra = self.images / "9999999900.png"
        Image.new("RGB", (4, 4)).save(extra)
        with self.assertRaisesRegex(ValueError, "without metadata"):
            self.freeze()
        extra.rename(self.images / "2T2SALA702.png")
        with self.assertRaisesRegex(ValueError, "now present"):
            self.freeze()

    def test_fractional_label_is_not_silently_truncated(self):
        book = openpyxl.load_workbook(self.metadata)
        book.active["B3"] = 1.5
        book.save(self.metadata)
        book.close()
        with self.assertRaisesRegex(ValueError, "Invalid class"):
            metadata_rows(self.metadata)

    def test_identical_images_with_conflicting_labels_stay_together_and_are_audited(self):
        source = self.images / "0000000000.png"
        (self.images / "0000000101.png").write_bytes(source.read_bytes())
        audit = self.freeze()
        group = audit["byte_identical_groups"][0]
        self.assertTrue(group["conflicting_labels"])
        self.assertEqual({r["label"] for r in group["images"]}, {0, 1})
        memberships = {}
        for name in ("train", "val", "test"):
            for row in read_split(self.root / "splits", name):
                memberships[row["image_id"]] = name
        self.assertEqual(memberships["0000000000"], memberships["0000000101"])


if __name__ == "__main__":
    unittest.main()
