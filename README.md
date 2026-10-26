# Tri-Subject Kinship Verification from Face Images

Given face photos of a claimed **father**, **mother** and **child**, the system
estimates the probability that the child is the biological child of this couple,
and shows which parent the child resembles more.

Deep Learning course project (BSCS). The project uses:
- **Siamese / comparator networks** with an **attention gate**,
- **transfer learning** from pretrained face-recognition networks (ArcFace, FaceNet),
- **fine-tuning** of FaceNet end to end,
- classical ML (logistic regression, SVM) as a supporting baseline.

> **Statistical estimate from face images. Not proof of parentage. Not a DNA test.**

- Full results: [`docs/results/results.md`](docs/results/results.md), generated from the run outputs, never typed by hand.
- Plain-English walkthrough and viva notes: [`docs/HOW_IT_WORKS.md`](docs/HOW_IT_WORKS.md).

## Results

Test set of 580 triplets (50% kin / 50% non-kin) from 147 families never seen in
training. People are represented by templates (the average of their photos). A
parent and the child are compared **only through images from different photos**
(see [Same-photo bias](#same-photo-bias)). The decision threshold is chosen on
the validation set.

| Exp | Model | Accuracy | ROC-AUC | Hard-set AUC* |
|---|---|---|---|---|
| E0 | frozen ArcFace, mean cosine similarity, no training | 0.797 | 0.878 | 0.711 |
| E1 | logistic regression on 8 hand-made features (frozen ArcFace) | 0.793 | 0.879 | 0.705 |
| E2 | attention-gated tri-subject head on frozen ArcFace | 0.807 | 0.871 | 0.704 |
| **E2-hardneg** | E2 trained with 50% "one wrong parent" negatives (**used in the demo**) | **0.807** | **0.881** | **0.714** |
| E2-pairs | E2-hardneg with the comparator pretrained on pairs (overfit) | 0.716 | 0.796 | 0.663 |
| E2b | E2 head on frozen FaceNet | 0.736 | 0.830 | 0.709 |
| **E3** | **FaceNet fine-tuned end to end** with the head (from E2b) | **0.747** | **0.842** | 0.709 |

\*Hard set: the real child with one parent swapped for an unrelated adult.

![ROC curves](docs/results/figures/roc_test.png)

**Main findings**
- **Same-photo bias.** FIW relatives are often cut from the same photo, and that
  inflates every model. Frozen ArcFace with cosine scores 0.862 accuracy with all
  photos and 0.797 with different photos only.
- **On ArcFace, the learned head ties the cosine baseline.** E2-hardneg has the
  best AUC, but the gap (0.881 vs 0.878) is within the uncertainty: about ±3
  points of accuracy with 580 test triplets. With about 1,400 training families,
  extra layers on very strong embeddings mostly overfit.
- **On weaker features, learning helps.** FaceNet + head: AUC 0.796 → 0.830.
- **Fine-tuning helps, modestly but in every comparison.** E3 beats E2b (0.842 vs
  0.830 AUC). The fine-tuned embeddings alone also improve: cosine 0.796 → 0.809,
  parent-child pairs 0.767 → 0.780.
- **"One wrong parent" is much harder.** Every model drops to about 0.71 AUC.
- **Attention.** The weights follow the more similar parent for the FaceNet models
  (about 70% of the time), not for the ArcFace models. So the demo shows
  resemblance from face similarity and labels attention only as "model reliance".
- **Context.** Published RFIW tri-subject leaderboard accuracies are roughly
  0.68-0.79. Our split and protocol differ (stricter: different photos only), so
  they are a reference point, not a direct comparison.

## Pipeline

| Step | Script | What it does |
|---|---|---|
| 0 | `scripts/inspect_data.py` | Detects the dataset format and saves `outputs/data_summary.md` |
| 1 | `scripts/extract_embeddings.py` | Detects and aligns faces (InsightFace / MTCNN); computes ArcFace or FaceNet embeddings and gender/age; cached and resumable |
| 2 | `scripts/prepare.py` | Family-disjoint 70/15/15 splits, triplets, main and hard negatives, pairs |
| 3 | `scripts/evaluate.py` | E0 (cosine) and E1 (logistic regression / SVM) |
| 4 | `scripts/train_head.py` | E2 / E2-hardneg / E2-pairs / E2b: attention-gated head on cached embeddings, small validation-only sweep |
| 5 | `scripts/finetune_backbone.py` | E3: fine-tunes FaceNet's last block together with the head (AMP fp16) |
| 6 | `scripts/make_report.py` | `results.md`, ROC curves, histograms, confusion matrix, attention plots, error contact sheets |
| - | `app.py` | Gradio demo (CPU) |

Code: `src/kinship/`
- `data.py`: loading, splits
- `triplets.py`: triplets and negatives
- `align.py`: the one shared preprocessing path
- `embed.py`: embeddings and caching
- `models.py`, `train.py`: the model and its training
- `evaluate.py`: metrics and protocols
- `plots.py`: report figures
- `predict.py`: demo logic

All settings are in `configs/default.yaml`. Any value can be overridden with
`--set key.sub=value`.

## Data

**Families in the Wild (FIW)** has 980 families, 5,030 people and 26,517 face
images. It is available on request from the
[FIW website](https://fulab.sites.northeastern.edu/fiw-download), and its license
forbids redistribution, so **no FIW images are in this repository**.
- Triplets come from each family's relationship matrix: 2,029 father-mother-child
  triplets, a superset of the 350 official RFIW Track-II triplets.
- The Kaggle "Recognizing Faces in the Wild" data is also supported (as noisy
  *pseudo-triplets*, mined from related pairs plus estimated gender and age). The
  format is detected automatically.

### Same-photo bias

FIW face crops are named `<photo>_face<k>.jpg`. In our test split, **94% of kin
triplets share a photo** between the child and a parent; non-kin triplets never do.
Faces cut from one photo share lighting, camera and colour. That similarity has
nothing to do with kinship, and a model can exploit it.

All headline numbers therefore use the **different-photo protocol**: the child's
images come from photos that neither parent's images come from. 97-98% of
triplets remain scorable this way. Models are trained under the same rule.
`results.md` also lists the "all photos" scores for comparison.

## Running it

### Tests (CPU, synthetic data)
```bash
pip install numpy pandas PyYAML Pillow scikit-learn scipy matplotlib tqdm pytest torch
python -m pytest -q
```

### Full pipeline on Kaggle (GPU)
1. Upload FIW as a **private** Kaggle dataset.
2. Import `notebooks/kaggle_run.ipynb`, attach the dataset, and turn on GPU and
   Internet.
3. Click *Save Version → Save & Run All*.

The notebook clones this repo, installs packages (`scripts/kaggle_install.sh`) and
runs every step. Each stage is cached, and attaching an earlier run's output as
an input reuses its embeddings. Package fixes for Kaggle are documented in
`scripts/kaggle_install.sh` and `requirements.txt`:
- `onnxruntime-gpu==1.24.1` for CUDA 12,
- `onnx==1.21.0`,
- `insightface==2.1 --no-deps`, which avoids pulling in the CPU-only onnxruntime,
- `facenet-pytorch==2.6.0 --no-deps`, which avoids a torch downgrade.

Alternative to GitHub: upload the repo as a Kaggle dataset zip and set
`CODE_DATASET` in the notebook.

### Report from a downloaded Kaggle run
```bash
python scripts/make_report.py --output_dir kaggle_outputs/outputs --work_dir work/fiw \
    --data_root data/fiw --copy_to docs/results
```

### Demo (CPU)
Download `outputs.zip` from the Kaggle run and unzip it to `kaggle_outputs/`. Then:
```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt          # Windows (Linux/macOS: .venv/bin/python)
.venv/Scripts/python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/Scripts/python app.py                                       # open http://127.0.0.1:7860
```
Upload three photos. The demo shows:
- the aligned faces,
- a **calibrated** probability (Platt scaling fitted on validation scores),
- a verdict at the validation threshold,
- father / mother resemblance,
- the disclaimer.

A photo without a detectable face gives a clear error. With several faces, the
largest one is used.

Use a separate virtual environment for the face models. InsightFace pulls in
`onnx`, whose protobuf requirement can clash with other packages in a global
Python.

## Limitations and ethics

- **About 75-80% accuracy at best**: roughly one answer in five is wrong. This must
  **never** be used for real parentage, custody, immigration or any other decision
  about real people. Only DNA testing can establish parentage.
- **Dataset bias.** FIW is mostly Western public figures in professional photos.
  Performance on other populations, children's photos and everyday images is
  unknown and probably worse.
- **Age gaps.** Parents and children are often photographed decades apart, which
  confuses face similarity.
- **Privacy and consent.** Face images of real families are sensitive personal
  data. Only use photos with the consent of everyone shown. Never upload other
  people's family photos without permission.
- **Pseudo-triplets (Kaggle data option).** Triplets mined from related pairs with
  estimated gender and age contain wrong labels. Results on that option are less
  reliable.
- **Same-photo bias** (above) inflates any FIW result that does not control for it.

## Acknowledgements

- FIW: J. P. Robinson et al., *Families in the Wild (FIW): Large-Scale Kinship
  Image Database and Benchmarks* (ACM MM 2016), and the RFIW challenges.
- [InsightFace](https://github.com/deepinsight/insightface) (ArcFace, SCRFD).
- [facenet-pytorch](https://github.com/timesler/facenet-pytorch) (MTCNN,
  InceptionResnetV1).
