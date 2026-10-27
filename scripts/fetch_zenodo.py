"""Download the public, versioned FDR26 release; verify every MD5 checksum."""
import argparse
import hashlib
import json
from pathlib import Path
import urllib.request
import zipfile

RECORD = "https://zenodo.org/api/records/20605487"
SMALL = {"README.md", "requirements.txt", "FDR26_benchmark.ipynb", "metadata_classification_DR.xlsx"}


def fetch(entry, directory):
    name = entry["key"]
    if Path(name).name != name:
        raise ValueError(f"Unexpected remote filename: {name}")
    target = directory / name
    algorithm, expected = entry["checksum"].split(":", 1)
    def checksum(path):
        h = hashlib.new(algorithm)
        with path.open("rb") as stream:
            for part in iter(lambda: stream.read(1024 * 1024), b""):
                h.update(part)
        return h.hexdigest()
    if target.exists() and checksum(target) == expected:
        print(f"Verified: {name}")
        return target
    part = target.with_suffix(target.suffix + ".part")
    h = hashlib.new(algorithm)
    total = 0
    print(f"Downloading {name}: {entry['size'] / 1e6:.1f} MB", flush=True)
    with urllib.request.urlopen(entry["links"]["self"], timeout=120) as response, part.open("wb") as out:
        while chunk := response.read(1024 * 1024):
            out.write(chunk)
            h.update(chunk)
            total += len(chunk)
    if h.hexdigest() != expected or total != entry["size"]:
        raise ValueError(f"Checksum or size mismatch: {part}")
    part.replace(target)
    return target


def safe_extract(archive, destination):
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as z:
        for info in z.infolist():
            candidate = (destination / info.filename).resolve()
            if not candidate.is_relative_to(destination) or "\\" in info.filename:
                raise ValueError(f"Unsafe ZIP member: {info.filename}")
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError(f"Symlink ZIP member: {info.filename}")
        z.extractall(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=Path("reference/zenodo"))
    parser.add_argument("--dataset", action="store_true", help="Also download the 1.68 GB image archive")
    parser.add_argument("--data-dir", type=Path, default=Path("data/fdr26"))
    args = parser.parse_args()
    args.reference.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(RECORD, timeout=60) as response:
        record = json.load(response)
    (args.reference / "record.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    for entry in record["files"]:
        if entry["key"] in SMALL:
            fetch(entry, args.reference)
        if args.dataset and entry["key"] == "FDR26_classification.zip":
            args.data_dir.mkdir(parents=True, exist_ok=True)
            archive = fetch(entry, args.data_dir)
            safe_extract(archive, args.data_dir / "images")
    print("Source: https://doi.org/10.5281/zenodo.20605487")


if __name__ == "__main__":
    main()
