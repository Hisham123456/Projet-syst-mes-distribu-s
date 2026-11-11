# GPU Acceleration of a Diabetic Retinopathy Classification Pipeline

Engineering project, 2nd year: GPU programming and performance optimization.
Brief: [`Project_GPU_Acceleration_2026_V1.pdf`](Project_GPU_Acceleration_2026_V1.pdf).

**Start here: [`ASSIGNMENTS.md`](ASSIGNMENTS.md).** It lists every step in order, with checks and
questions. You have two weeks, end-to-end run and final report included.

**Team members:** TODO(A8.7) replace with your names

## Référence utilisée dans ce dépôt

La chaîne CPU fournie par le professeur dans `gpua-student` est la référence
du projet. Depuis la racine, lancez `.venv/Scripts/python.exe src/train_cpu.py`
sous Windows. La configuration fournie utilise `ReferenceCNN`, des images
de 224 × 224 et des lots de 32.

Pour ce jeu de données local, les 2 999 images disponibles ont été copiées
dans `data/dr_dataset/<grade>/<patient>_<index>.jpg` à partir des chemins et
des grades vérifiés dans les CSV. L'image absente `2T2SALA702` est exclue.
Le chargeur du professeur recalcule ses partitions par patient. Les CSV
précédents servent ici à préparer les fichiers et leurs grades. Le découpage
fourni répartit 2 116 images en entraînement, 425 en validation et 458 en test.
Il place 32 paires de fichiers identiques dans des ensembles différents, ce
qui peut biaiser les scores. Le module `src/data_splits.py` permet de recréer
les partitions figées utilisées pour les expériences sur les doublons.

## What you will do

The repository holds a working CPU pipeline that grades diabetic retinopathy (five classes) on
3,000 fundus images. It is correct and slow. Your job is to find out why, then make every stage
faster: loading, preprocessing, filtering, augmentation, training and evaluation.

The loop you repeat for each change: **profile, hypothesize, optimize, validate, re-profile.**
You write a prediction in `docs/lab_notebook.md` before each experiment, run it, and explain the
result. Your report only states numbers that come from files in `results/`.

| Days | You | You produce |
|---|---|---|
| 1-3 | run and profile the CPU baseline | a measured bottleneck breakdown |
| 3-6 | redesign loading, write GPU preprocessing and filters | decoded-image cache, batched GPU kernels |
| 7-10 | build the end-to-end input pipeline and accelerate training | prefetching, mixed precision, a justified model choice |
| 11-14 | benchmark, profile the GPU, write up | five figures, the tables, the report, the defense |

Code you have to write is marked `TODO(A<n>.<m>)`. `python scripts/progress.py` lists what is
left, and `python -m pytest tests -v` checks what you have done. Tests that need a GPU are skipped
on machines without CUDA, so you need a CUDA machine to validate the GPU parts.

## Your first hour

1. Read the brief, then `ASSIGNMENTS.md` up to A2.
2. Set up the environment and download the dataset (below).
3. Run `python scripts/progress.py` and `python -m pytest tests -v` to see the starting state.
4. Start A1 (listed in `docs/checklist.md`): environment, data, hardware table, baseline run.

## Layout

```
ASSIGNMENTS.md   what to do, step by step
configs/         baseline.yaml (given), optimized.yaml (yours)
src/             CPU pipeline (given) and GPU modules (yours)
benchmarks/      profiling and benchmark scripts (CPU parts given, GPU parts yours)
tests/           automatic checks for your implementations
scripts/         prepare_fdr26 (dataset), record_env, progress, make_figures (yours)
docs/            checklist, lab notebook, design proposal, pipeline diagram
report/          report outline
environment/     environment.yml, hardware_software.md (Table 1)
results/         everything the scripts produce (commit the small json/csv/md/png files)
```

## Environment setup

TODO(A8.8): keep this section correct for your machine and add the compatibility issues you hit.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1                     # Linux/macOS: source .venv/bin/activate
# NVIDIA GPU: install the CUDA build of PyTorch first (plain pip is CPU-only on Windows)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
python -c "import torch; print(torch.cuda.is_available())"
```

Conda users (Linux): `conda env create -f environment/environment.yml`, activate it, then run the
`pip install torch torchvision --index-url ...` line above.

## Dataset

FDR26: 3,000 images, 1,918 patients, grades 0-4
(Zenodo [10.5281/zenodo.20605487](https://doi.org/10.5281/zenodo.20605487), CC-BY-4.0).

```powershell
python scripts/prepare_fdr26.py --download
```

This creates `data/dr_dataset/<grade 0-4>/<patient>_<index>.<ext>`. The text before the last `_`
is the patient id; `split_by_patient` keeps each patient in one split. To use another location,
change `data.root` in the configs.

## Commands

| Task | Command |
|---|---|
| Progress check | `python scripts/progress.py` |
| Tests | `python -m pytest tests -v` |
| CPU baseline | `python src/train_cpu.py` |
| Evaluate a run | `python src/evaluate.py --config configs/baseline.yaml --run baseline --device cpu` |
| Any config/device | `python src/train.py --config configs/NAME.yaml --device cuda --run NAME [--monitor]` |
| Profile the baseline | `python benchmarks/profile_baseline.py` |
| Stage timings | `python benchmarks/benchmark_stages.py` |
| End-to-end timings | `python benchmarks/benchmark_end_to_end.py --epochs 2 --repeats 3` |
| GPU profile | `python benchmarks/profile_gpu.py` |
| Environment record | `python scripts/record_env.py` |
| Figures | `python scripts/make_figures.py` |

Outputs: `results/<run>/history.json`, `results/<run>/metrics.json`,
`results/<run>/resources.csv` (with `--monitor`), `checkpoints/<run>/best.pt`.

## Notes

- Windows DataLoader workers use `spawn`: transforms must be picklable (no lambdas).
- Without CUDA the code runs on the CPU and the GPU tests are skipped. A4 to A7 need an NVIDIA
  GPU (a lab machine or a cloud GPU is fine).
- This is an educational benchmark, not a diagnostic tool.
