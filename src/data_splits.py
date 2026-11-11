from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import logging
from pathlib import Path
import re

import numpy as np
import openpyxl
from sklearn.model_selection import GroupShuffleSplit

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

FIELDS = ["image_path", "image_id", "label", "patient_id", "sha256", "split_group"]
EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
EXCLUSIONS = {
    "2T2SALA702": "Image absente de la livraison locale (3 000 entrées Excel, 2 999 images). "
                   "Exclusion explicite avant entraînement ; Excel original conservé.",
}


def patient_id(image_id, rule, mapping=None):
    if mapping is not None:
        if image_id not in mapping or not mapping[image_id]:
            raise ValueError(f"Missing patient mapping: {image_id}")
        return mapping[image_id]
    if rule != "fdr26-prefix" or not re.fullmatch(r"[A-Z0-9]{8}\d{2}", image_id):
        raise ValueError(f"Unrecognized patient identifier: {image_id}; provide an explicit mapping CSV")
    return image_id[:8]


def metadata_rows(path):
    book = openpyxl.load_workbook(path, read_only=True, data_only=True)
    try:
        # Actual release has a single 'Toutes' sheet, unlike the upstream loader.
        for sheet in book:
            values = iter(sheet.values)
            header = next(values, ())
            if "image_id" in header and "classe" in header:
                image_col, label_col = header.index("image_id"), header.index("classe")
                rows = []
                for row in values:
                    if row[image_col] is None:
                        continue
                    label = row[label_col]
                    if isinstance(label, bool) or label not in range(5):
                        raise ValueError(f"Invalid class for {row[image_col]}: {label!r}")
                    rows.append((str(row[image_col]).strip(), int(label)))
                return rows
        raise ValueError("Expected metadata columns image_id and classe")
    finally:
        book.close()


def index_images(root):
    index = {}
    for path in sorted(Path(root).rglob("*")):
        if path.suffix.lower() not in EXTENSIONS or not path.is_file():
            continue
        if path.stem in index:
            raise ValueError(f"Ambiguous image ID {path.stem}: {index[path.stem]} and {path}")
        index[path.stem] = path
    return index


def best_group_split(rows, fraction, seed):
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["split_group"] for r in rows])
    best = None
    splitter = GroupShuffleSplit(n_splits=256, test_size=fraction, random_state=seed)
    overall = np.bincount(labels, minlength=5)
    for keep, hold in splitter.split(labels, labels, groups):
        held = np.bincount(labels[hold], minlength=5)
        kept = np.bincount(labels[keep], minlength=5)
        if np.any(held == 0) or np.any(kept == 0):
            continue
        score = float(np.mean(np.abs(held / overall - fraction)))
        if best is None or score < best[0]:
            best = (score, keep, hold)
    if best is None:
        raise ValueError("Cannot produce patient-disjoint splits containing all five classes")
    return [rows[i] for i in best[1]], [rows[i] for i in best[2]]


def validate_disjoint(splits):
    for name, rows in splits.items():
        if not rows or {r["label"] for r in rows} != set(range(5)):
            raise ValueError(f"{name} must contain all five classes")
        if len({r["image_id"] for r in rows}) != len(rows):
            raise ValueError(f"Duplicate image IDs in {name}")
    names = list(splits)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            for key in ["image_path", "image_id", "patient_id", "sha256", "split_group"]:
                overlap = {r[key] for r in splits[left]} & {r[key] for r in splits[right]}
                if overlap:
                    raise ValueError(f"Leakage across {left}/{right}: {key} ({len(overlap)} overlaps)")


def create_splits(metadata, images, output, rule="fdr26-prefix", mapping_path=None, seed=42):
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite frozen splits: {output}")
    mapping = None
    if mapping_path:
        with open(mapping_path, encoding="utf-8-sig", newline="") as stream:
            entries = list(csv.DictReader(stream))
        mapping = {r["image_id"]: r["patient_id"] for r in entries}
        if len(mapping) != len(entries):
            raise ValueError("Duplicate image IDs in patient mapping")
    root = Path(images).resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"Image directory missing: {root}")
    metadata_hash = digest(metadata)
    metadata_entries = metadata_rows(metadata)
    index = index_images(root)
    rows = []
    excluded = []
    seen = set()
    for image_id, label in metadata_entries:
        if image_id in seen or label not in range(5):
            raise ValueError(f"Invalid or duplicate metadata row: {image_id}, {label}")
        seen.add(image_id)
        if image_id in EXCLUSIONS:
            if image_id in index:
                raise ValueError(f"Excluded image {image_id} is now present; review the exclusion before freezing")
            excluded.append({"image_id": image_id, "label": label,
                             "reason": EXCLUSIONS[image_id], "image_present": False})
            continue
        if image_id not in index:
            raise FileNotFoundError(f"Image missing: {image_id} below {root}")
        path = index[image_id]
        from PIL import Image
        with Image.open(path) as image:
            image.verify()  # Do not silently replace a corrupt image with its neighbor.
        # verify() alone does not decode JPEG pixels, including truncated streams.
        with Image.open(path) as image:
            image.load()
            image.convert("RGB").load()
        patient = patient_id(image_id, rule, mapping)
        rows.append(dict(image_path=path.relative_to(root).as_posix(), image_id=image_id,
                         label=label, patient_id=patient, sha256=digest(path), split_group=patient))
        if len(rows) % 250 == 0 or len(rows) == len(index):
            logging.getLogger(__name__).info("Validated images: %s/%s", len(rows), len(index))
    unlabelled = sorted(set(index) - seen)
    if unlabelled:
        raise ValueError(f"Images without metadata: {unlabelled}")
    if len(rows) != len(index):
        raise ValueError("Not all available images have been included")
    # Merge patient groups linked by byte-identical images, preventing duplicate leakage.
    parent = {r["patient_id"]: r["patient_id"] for r in rows}
    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    hashes = {}
    duplicate_groups = {}
    for row in rows:
        prior = hashes.get(row["sha256"])
        if prior:
            duplicate_groups.setdefault(row["sha256"], [prior]).append(row)
            parent[find(row["patient_id"])] = find(prior["patient_id"])
        else:
            hashes[row["sha256"]] = row
    for row in rows:
        row["split_group"] = find(row["patient_id"])
    train_val, test = best_group_split(rows, 0.15, seed)
    train, val = best_group_split(train_val, 0.15 / 0.85, seed + 1)
    splits = {"train": train, "val": val, "test": test}
    validate_disjoint(splits)
    if sum(map(len, splits.values())) != len(rows):
        raise ValueError("Split coverage does not match available images")
    if digest(metadata) != metadata_hash:
        raise ValueError("Metadata changed during split creation")
    output.mkdir(parents=True, exist_ok=True)
    audit = {"seed": seed, "target_fractions": [0.70, 0.15, 0.15],
             "schema_version": 1,
             "metadata_sha256": metadata_hash, "metadata_file": Path(metadata).name,
             "metadata_entries": len(metadata_entries), "available_images": len(index),
             "exclusions": excluded, "excluded_images": len(excluded),
             "patient_rule": "mapping" if mapping is not None else rule,
             "patient_identification_assumption": (
                 "Hypothèse provisoire : les huit premiers caractères du nom de fichier "
                 "identifient un patient. Cette règle est inférée des noms, sans validation clinique. "
                 "Toutes les images de ce préfixe restent dans le même ensemble."
                 if mapping is None else "Identifiants fournis par le CSV de correspondance."),
             "patient_rule_inferred": mapping is None, "patients": len(parent),
             "total_images": len(rows), "class_counts": dict(Counter(r["label"] for r in rows)),
             "image_validation": {"method": "Pillow verify, full decode and RGB conversion",
                                  "readable_images": len(rows), "unreadable_images": 0},
             "validation": {"complete_coverage": True, "patient_overlap_count": 0,
                            "image_id_overlap_count": 0, "sha256_overlap_count": 0,
                            "unlabelled_images": unlabelled},
             "split_method": "GroupShuffleSplit: best of 256 class-balanced candidates per stage",
             "byte_identical_groups": [
                 {"sha256": sha, "images": [
                     {"image_id": r["image_id"], "label": r["label"], "patient_id": r["patient_id"]}
                     for r in group],
                  "conflicting_labels": len({r["label"] for r in group}) > 1}
                 for sha, group in sorted(duplicate_groups.items())],
             "duplicate_policy": "Conserver toutes les images et les labels Excel ; "
                                 "regrouper les fichiers identiques dans le même ensemble. "
                                 "Les conflits de labels sont signalés pour revue.",
             "reuse_policy": "Réutiliser ces mêmes CSV pour toutes les expériences CPU/GPU ; "
                             "ne pas recalculer les partitions. SHA-256 contrôlé par read_split.",
             "splits": {}, "files": {}}
    if mapping_path:
        audit["patient_mapping_sha256"] = digest(mapping_path)
    for name, subset in splits.items():
        target = output / f"{name}.csv"
        with target.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDS)
            writer.writeheader()
            writer.writerows(subset)
        audit["files"][name] = digest(target)
        audit["splits"][name] = {"images": len(subset), "fraction": len(subset) / len(rows),
                                 "patients": len({r['patient_id'] for r in subset}),
                                 "class_counts": dict(Counter(r["label"] for r in subset))}
    write_json(output / "audit.json", audit)
    return audit


def read_split(directory, name):
    directory = Path(directory)
    audit = read_json(directory / "audit.json")
    target = directory / f"{name}.csv"
    if digest(target) != audit["files"][name]:
        raise ValueError(f"Frozen manifest was changed: {target}")
    with target.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        row["label"] = int(row["label"])
    return rows


def main():
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Audit and freeze patient-disjoint CPU/GPU splits")
    parser.add_argument("--metadata", default="data/FDR26_classification/classification DR.xlsx")
    parser.add_argument("--images", default="data/FDR26_classification/images")
    parser.add_argument("--output", default="data/splits")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mapping", default=None, help="Optional CSV with image_id,patient_id")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    audit = create_splits(args.metadata, args.images, args.output,
                          mapping_path=args.mapping, seed=args.seed)
    print(json.dumps({"total_images": audit["total_images"], "splits": audit["splits"],
                      "exclusions": audit["exclusions"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
