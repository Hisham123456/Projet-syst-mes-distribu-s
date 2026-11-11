"""Convert the FDR26 classification release into the layout the pipeline expects.

FDR26 (Zenodo 10.5281/zenodo.20605487, CC-BY-4.0): 3,000 fundus images from
1,918 patients, ALFEDIAM grades 0-4. metadata_classification_DR.xlsx has
image_id (10 chars = 8-char patient code + 2-digit image index) and classe.

Output: <out>/<classe>/<patientCode>_<imageIndex>.<ext>
so src.data_loader.patient_id() (prefix before the last "_") returns the
patient code and split_by_patient() keeps every patient in one split,
including the 407 patients whose two eyes have different grades.

    python scripts/prepare_fdr26.py --download     # fetch both files from Zenodo first
    python scripts/prepare_fdr26.py --zip data/_download/FDR26_classification.zip \
        --meta data/_download/metadata_classification_DR.xlsx --out data/dr_dataset
"""
import os, sys, argparse, shutil, zipfile, json, urllib.request
import pandas as pd

ZENODO = "https://zenodo.org/api/records/20605487/files/{}/content"

IMG_EXTS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def download(name, dest):
    if os.path.exists(dest):
        print(f"{dest} already present, skipping download")
        return
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"downloading {name} -> {dest}")
    urllib.request.urlretrieve(ZENODO.format(name), dest)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--download", action="store_true",
                    help="download the archive and metadata from Zenodo if missing")
    ap.add_argument("--zip", default="data/_download/FDR26_classification.zip")
    ap.add_argument("--meta", default="data/_download/metadata_classification_DR.xlsx")
    ap.add_argument("--out", default="data/dr_dataset")
    ap.add_argument("--tmp", default="data/_download/extracted")
    args = ap.parse_args()

    if args.download:
        download("FDR26_classification.zip", args.zip)
        download("metadata_classification_DR.xlsx", args.meta)

    meta = pd.read_excel(args.meta)
    meta["image_id"] = meta.image_id.astype(str).str.strip()
    label = dict(zip(meta.image_id.str.upper(), meta.classe.astype(int)))

    if not os.path.isdir(args.tmp):
        with zipfile.ZipFile(args.zip) as z:
            z.extractall(args.tmp)

    found, unmatched = {}, []
    for dirpath, _, files in os.walk(args.tmp):
        for f in files:
            stem, ext = os.path.splitext(f)
            if ext.lower() not in IMG_EXTS:
                continue
            key = stem.strip().upper()
            if key in label:
                if key in found:
                    sys.exit(f"duplicate image id {key}: {found[key]} and {dirpath}/{f}")
                found[key] = os.path.join(dirpath, f)
            else:
                unmatched.append(os.path.join(dirpath, f))
    missing = sorted(set(label) - set(found))
    print(f"matched {len(found)}/{len(label)} metadata rows; "
          f"{len(unmatched)} image files not in metadata; {len(missing)} rows without image")
    if missing:
        print("first missing:", missing[:10])
    if unmatched:
        print("first unmatched:", unmatched[:10])
    if len(found) != len(label):
        sys.exit("metadata/image mismatch - inspect before continuing")

    # cross-check: if the release stores images in grade folders, they must agree
    disagree = [k for k, p in found.items()
                if os.path.basename(os.path.dirname(p)).isdigit()
                and int(os.path.basename(os.path.dirname(p))) != label[k]]
    print(f"folder-vs-metadata grade disagreements: {len(disagree)}")

    if os.path.isdir(args.out):
        shutil.rmtree(args.out)
    for k, src in found.items():
        d = os.path.join(args.out, str(label[k]))
        os.makedirs(d, exist_ok=True)
        ext = os.path.splitext(src)[1].lower()
        shutil.move(src, os.path.join(d, f"{k[:-2]}_{k[-2:]}{ext}"))
    shutil.rmtree(args.tmp)

    counts = {c: len(os.listdir(os.path.join(args.out, str(c)))) for c in range(5)}
    info = {"source": "FDR26, Zenodo 10.5281/zenodo.20605487 (CC-BY-4.0)",
            "images": len(found), "patients": meta.image_id.str[:-2].nunique(),
            "class_counts": counts, "folder_grade_disagreements": len(disagree)}
    with open(os.path.join(args.out, "DATASET_INFO.json"), "w") as f:
        json.dump(info, f, indent=2)
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
