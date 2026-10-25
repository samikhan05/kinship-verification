# Tri-Subject Kinship Verification from Face Images

Given face images of a claimed **father**, **mother** and **child**, predict the
probability that the child is their biological child, and show **which parent the
child resembles more** (via a learned attention gate).

> Status: **M1, M2 done** (baselines run on Kaggle); **M3 code ready** (E2 attention head, E2b on
> FaceNet), not run on Kaggle yet. The full README comes in M5.

## Pipeline (order of scripts)

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/inspect_data.py` | detects the dataset format, prints/saves `outputs/data_summary.md` |
| 1 | `scripts/extract_embeddings.py` *(M2)* | detect + align faces, ArcFace embeddings, gender/age estimates (cached) |
| 2 | `scripts/prepare.py` | family-disjoint splits, triplets, negatives, pairs, `triplet_check.jpg` |
| 3 | `scripts/evaluate.py` *(M2)* | E0 / E1 baselines and all later experiments |
| 4 | `scripts/train_head.py` | E2 / E2-hardneg / E2b: attention-gated head on cached embeddings |
| 4b | `scripts/finetune_backbone.py` *(M4)* | E3: fine-tune FaceNet end to end |
| 5 | `scripts/make_report.py` *(M5)* | `outputs/results.md` + plots |

`extract_embeddings` runs **before** `prepare` because, for the Kaggle dataset
(Option B), mining pseudo-triplets needs each person's estimated gender and age,
and InsightFace produces those in the same pass as the embeddings.

## Datasets

* **Option A (preferred):** full FIW. Upload as a *private* Kaggle dataset (license).
* **Option B (fallback):** Kaggle "Recognizing Faces in the Wild"; triplets are
  *pseudo-triplets* mined from related pairs + estimated gender/age (noisy labels).

The format is detected automatically from `paths.data_root`.

## Same-photo bias and the "different photos" protocol

FIW face crops are named `<photo>_face<k>.jpg`, and relatives are often cut from the
**same photo**. In our test split, 94% of kin triplets share a photo between the
child and at least one parent; non-kin triplets never do. Faces from one photo look
alike for reasons unrelated to kinship (lighting, camera, colour), so a model can
score well by recognising the photo. All experiments are therefore also scored with
the **different-photo protocol**: a parent and the child are compared only through
images from different photos. That is the headline number in `results.md`.
Measured effect (Kaggle run 2026-10-25, test set, frozen ArcFace E0-mean): accuracy
0.862 / ROC-AUC 0.926 with all photos vs 0.809 / 0.901 with different photos only.
Classical and learned models are trained on different-photo data for the same reason.

## Local setup and tests (CPU)

```bash
pip install numpy pandas PyYAML Pillow scikit-learn matplotlib tqdm pytest
python -m pytest -q
# smoke run on synthetic data
python scripts/make_synthetic.py --out data/synth_kaggle --format kaggle
python scripts/inspect_data.py --data_root data/synth_kaggle --work_dir work --output_dir outputs
python scripts/prepare.py --data_root data/synth_kaggle --work_dir work --output_dir outputs \
    --attributes data/synth_kaggle/attributes_gt.csv
```

All settings are in `configs/default.yaml`; override any value with
`--set key.sub=value`, or use the `--data_root/--work_dir/--output_dir` shortcuts.

To run the real face models locally (CPU), use a separate virtual environment.
InsightFace pulls in `onnx`, whose protobuf requirement can clash with other
packages in a global Python install:

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt      # Windows (Linux: .venv/bin/python)
.venv/Scripts/python scripts/extract_embeddings.py --data_root data/fiw --work_dir work/fiw --output_dir outputs --limit 200
```

## Running on Kaggle

**Option 1 - from GitHub (recommended).** Push this repo to GitHub, open
`notebooks/kaggle_run.ipynb` on Kaggle (File → Import Notebook), check `REPO_URL`,
attach the dataset, turn on GPU + Internet, then *Save Version → Save & Run All*.
A **public** repo is simplest. A private repo needs a GitHub token stored as a
Kaggle secret. The code contains no FIW images, so a public repo is fine.

**Option 2 - repo as a Kaggle dataset.** Zip the repo (without `data/`, `work/`,
`outputs/`, `.venv/`), upload it as a private Kaggle dataset (e.g. `kinship-code`),
attach it, and set `CODE_DATASET = "kinship-code"` in the notebook instead of `REPO_URL`.

**Package fixes on Kaggle** (all in `scripts/kaggle_install.sh`, which the notebook runs):
`onnxruntime-gpu==1.24.1` (CUDA 12), `onnx==1.21.0` (has wheels for new Pythons),
`insightface==2.1 --no-deps` (avoids the CPU-only onnxruntime), `facenet-pytorch==2.6.0 --no-deps`
(avoids a torch downgrade).

Every stage is cached in `work_dir` and skipped when it is already done. Embedding
extraction saves every 2,000 images, so a stopped session resumes where it left off.
