# How it works (plain English, for the viva)

This file walks through the whole project in the order the data flows. Every
number here comes from `docs/results/results.md`. That file is generated from the
Kaggle run's output files. No number was typed in by hand.

---

## 1. The task

We get three face photos: a claimed **father (F)**, a claimed **mother (M)** and a
**child (C)**. We output the probability that C is the biological child of this
couple, plus which parent the child resembles more.

Why use both parents instead of one? A child can look much more like one parent
than the other. With both parents the model has more evidence, and it can check
the stronger claim "these two people are **both** the parents".

---

## 2. The data

**Dataset: Families in the Wild (FIW).** It has 980 families, 5,030 people and
26,517 face images, mostly public figures. Every family folder has a `mid.csv` file
with a *relationship matrix* (cell (i, j) = "member i is the parent / child /
sibling / spouse of member j") and each member's gender.

**Building triplets.** For every child with a known father and a known mother we
make one triplet (F, M, C). This gives 2,029 triplets from 807 families. That set
contains all 350 official RFIW Track-II triplets, with the same ground-truth labels.

**Splits: family-disjoint, 70 / 15 / 15.** A whole family goes into exactly one of
train / val / test (686 / 147 / 147 families). If a family were in both train and
test, the model could recognise *people* it had already seen instead of learning
*kinship*. A unit test checks that no family is in two splits.

**Negatives (non-kin examples):**
- *Main negative* (F, M, C'): the real couple with a child from another family,
  same gender as the real child.
- *Hard negative* (F', M, C) or (F, M', C): the real child with one parent swapped
  for an unrelated adult of the same gender. This tests "these are **both** her
  parents", which is much harder.

Every validation / test set is exactly 50% kin and 50% non-kin, with a fixed seed.

**The same-photo problem.** FIW face crops are named `<photo>_face<k>.jpg`, and
relatives are often cut from the **same photo**. In our test set, 94% of kin
triplets share a photo between the child and at least one parent. Non-kin
triplets never do. Faces from one photo look alike for reasons that have nothing
to do with family: same lighting, camera and colours. A model can score well just
by recognising "same photo".

We measured this. Frozen ArcFace with cosine (E0-mean) reaches 0.862 accuracy if
same-photo comparisons are allowed. With the shortcut removed it reaches 0.797.
So **all headline numbers use the "different photos" protocol:** the child's
images come only from photos that neither parent's images come from. 97-98% of
triplets can still be scored this way. Models are also *trained* this way, so they
cannot learn the shortcut.

---

## 3. Preprocessing: one function for everything

`src/kinship/align.py`. Training, validation, test and the demo all use the same
code. If they didn't, the model would see differently prepared faces in the demo
than in training.

1. **Detect** the face and 5 landmarks (eyes, nose, mouth corners) with
   InsightFace's SCRFD detector (`buffalo_l` pack).
2. **Align**: rotate, scale and crop so the 5 landmarks land on fixed positions in
   a 112 x 112 image (`norm_crop`). That is exactly the input ArcFace was trained on.
3. **Padding trick.** FIW images are already tight face crops (about 108 x 124 px),
   and detectors miss faces that fill the whole frame. We add a black border before
   detection. On 200 real images: no border found 12/200 faces, a 0.3 border found
   200/200. Over the whole dataset, faces were found in 26,507 of 26,517 images.
4. If no face is found, we fall back to a centre crop and count it (10 images).

The FaceNet path (for fine-tuning) uses MTCNN instead: 160 x 160 crops with a
14-pixel margin.

---

## 4. Embeddings and transfer learning

A face-recognition network turns a face into an **embedding**: 512 numbers,
scaled to length 1. Photos of the same person give nearby embeddings. The
**cosine similarity** (dot product of two unit vectors) measures how alike two
faces are, from -1 to 1.

**Transfer learning:** we never train a face network from scratch. We reuse
networks pretrained on millions of faces:
- **ArcFace** (InsightFace `buffalo_l`, ONNX model). It is very strong, but an ONNX
  file cannot be trained, so it stays frozen.
- **FaceNet** (`facenet_pytorch` InceptionResnetV1, trained on VGGFace2). It is
  a PyTorch model, so it *can* be fine-tuned.

These networks were trained to tell *different people* apart, not to find
relatives. That relatives look more alike than strangers is a useful side
effect, and that side effect is what we use.

Embeddings are computed once and cached (`extract_embeddings.py`, saved in chunks
of 2,000 images), so every later step takes seconds.

---

## 5. The experiments

| Exp | What it is | Why it is there |
|---|---|---|
| **E0** | No training. Score = mean (or max) of cos(F,C) and cos(M,C) on frozen ArcFace | The "before" number: how far pure face similarity gets |
| **E1** | Logistic regression / linear SVM on 8 hand-made features (the two cosines + statistics of the difference vectors) | Classical machine learning as a supporting baseline |
| **E2** | The main deep model: attention-gated tri-subject head on frozen ArcFace (section 6) | Learns how to combine father and mother evidence |
| E2-hardneg | E2, but half of the training negatives swap one parent | Targets the hard "one wrong parent" case |
| E2-pairs | E2-hardneg, but the comparator is first trained on ~11,800 parent-child / relative pairs | An attempt to fight overfitting with more data |
| **E2b** | The E2 head on frozen FaceNet | The fair "before" for E3 (same backbone) |
| **E3** | **Fine-tuning:** FaceNet's last block + the head trained end to end | The course's fine-tuning requirement |

---

## 6. The main model (E2), step by step

`src/kinship/models.py`

1. **Shared comparator (Siamese).** The same small network compares father-child
   and mother-child. Both embeddings first go through a shared projection
   (512 -> 128 numbers). Then it builds `[|p-c|, p*c, (p-c)^2, cos(p,c)]` and an
   MLP turns that into a vector *h* (h_F for the father, h_M for the mother).
   Sharing the weights means both parents are judged in exactly the same way.
2. **Attention gate.** A linear layer looks at [h_F; h_M] and outputs two numbers.
   A softmax turns them into weights alpha_F + alpha_M = 1: how much the decision
   relies on each parent.
3. **Fusion.** z = alpha_F * h_F + alpha_M * h_M.
4. **Extra inputs.** cos((e_F + e_M)/2, e_C) (the child against a "mixed parent"),
   plus cos(F,C) and cos(M,C) directly (skip connections, see section 7).
5. **Classifier.** A small MLP turns all of this into one score (logit).

**Loss:** binary cross-entropy on the triplet label, + 0.3 x the losses of two
small "is this parent related to this child?" outputs (one for F-C, one for M-C).
These auxiliary losses force h_F and h_M to carry real parent-child evidence.

---

## 7. Training

`src/kinship/train.py`

- **Fresh data every epoch.** Every kin triplet gets a newly drawn negative each
  epoch. Each person is represented by the mean of a *random subset* of their
  photos, so the model sees new inputs every time. The photo rules from section 2
  always apply.
- **Optimiser:** AdamW, cosine learning-rate schedule, **early stopping** on
  validation ROC-AUC, fixed seeds.
- **What went wrong first, and the fix (good viva story).** The first E2 overfit:
  its best validation AUC came at epoch 1. Training loss then kept falling while
  validation AUC fell too, because the model memorised the 1,434 training families.
  The fix:
  - a shared 512 -> 128 projection (about 6x fewer weights),
  - the plain cosine similarities fed straight to the classifier ("skip
    connections"), so the model starts from what E0 already knows,
  - a small hyper-parameter sweep (8 settings: learning rate x dropout x weight
    decay), picked **only by validation AUC**.
- **E3 fine-tuning:**
  - Starts from the E2b head.
  - Only FaceNet's last block (`block8`, `last_linear`, `last_bn`, 2.5M of 23.5M
    weights) is unfrozen. The early layers (edges, textures, face parts) stay as
    they are.
  - Two learning rates: 1e-5 for the backbone (gentle changes), 1e-3 for the head.
  - Mixed precision (fp16) on the GPU.
  - Augmentation is only horizontal flips and +-10% brightness / contrast. No
    rotations or crops, because they would undo the alignment.

---

## 8. Evaluation

`src/kinship/evaluate.py`

- **The threshold is chosen on validation only** (the cut-off that gives the best
  validation accuracy), then applied unchanged to the test set. The test set is
  never used for any choice.
- **Accuracy:** fraction of correct kin / non-kin decisions at that threshold.
- **ROC-AUC:** the probability that a random kin triplet scores higher than a random
  non-kin one. It does not depend on the threshold. 0.5 = guessing, 1.0 = perfect.
  It is the fairest single number for comparing models.
- **F1:** the balance between precision and recall for the "kin" class.
- **TAR @ FAR = 0.1:** if we only allow 10% of non-kin to be accepted, what share
  of kin do we accept?
- **Breakdowns:** FMS vs FMD, main vs hard negatives, per-person *template* (mean of
  all photos) vs *per-image* (one random photo), pairwise FS / FD / MS / MD.
- **Uncertainty:** the test set has 580 triplets, so an accuracy carries roughly
  +-3 percentage points of uncertainty (95%). Differences smaller than that are
  ties.

---

## 9. Results and what they mean

Test set, templates, different photos, 580 triplets (50% kin):

| Exp | Accuracy | ROC-AUC | Hard-set AUC |
|---|---|---|---|
| E0-mean (cosine, ArcFace) | 0.797 | 0.878 | 0.711 |
| E1-LR | 0.793 | 0.879 | 0.705 |
| E2 | 0.807 | 0.871 | 0.704 |
| **E2-hardneg (best)** | **0.807** | **0.881** | **0.714** |
| E2-pairs | 0.716 | 0.796 | 0.663 |
| E0-mean (cosine, FaceNet) | 0.721 | 0.796 | 0.673 |
| E2b (head on frozen FaceNet) | 0.736 | 0.830 | 0.709 |
| **E3 (fine-tuned FaceNet)** | **0.747** | **0.842** | 0.709 |

Key findings:
1. **The same-photo shortcut is real.** Allowing it raised E0-mean from 0.797 to
   0.862 accuracy. We detected it, measured it and removed it.
2. **On ArcFace, the learned head ties the simple cosine baseline.** E2-hardneg has
   the best AUC (0.881 vs 0.878), but the gap is far inside the uncertainty. With
   about 1,400 training families, extra layers on top of very strong embeddings
   mostly overfit.
3. **On weaker features, learning clearly helps.** On FaceNet the head adds +0.034
   AUC (0.796 -> 0.830).
4. **Fine-tuning helps, modestly but consistently.** E3 beats E2b (0.842 vs 0.830
   AUC). The fine-tuned embeddings are also better on their own: plain cosine goes
   0.796 -> 0.809, and pairwise parent-child AUC goes 0.767 -> 0.780. FaceNet
   still stays below frozen ArcFace.
5. **A negative result: E2-pairs.** Pre-training the comparator on pairs overfit
   (training loss 0.12 while validation pair AUC stalled around 0.76, below the
   0.82 of plain cosine) and made E2 worse. We report it instead of hiding it.
6. **Hard negatives are much harder.** With one wrong parent, AUC drops to about
   0.71 for every model.
7. **Pairs:** mother-daughter is easiest (AUC 0.865), father-daughter hardest
   (0.773). Daughter triplets (FMD) score a little below son triplets (FMS).
8. **Attention:**
   - **FaceNet heads (E2b, E3):** alpha does follow the more similar parent
     (Spearman rho about 0.5, the right parent about 70% of the time).
   - **ArcFace heads:** alpha is almost constant. The skip-connected cosines carry
     the decision.
   - So the demo shows resemblance from the similarities and labels alpha only as
     "model reliance".
9. **Context:** published RFIW tri-subject accuracies are roughly 0.68-0.79. Our
   protocol differs (stricter: different photos only), so this is a reference
   point, not a direct comparison.

---

## 10. The demo (`app.py`)

1. You upload three photos. The **same** alignment as in training runs.
2. If a photo has no face, a clear error is shown. If it has several faces, the
   largest one is used and the demo says so.
3. The best model (E2-hardneg) gives a raw score. Raw scores are not
   probabilities (they sit around 0.47-0.55). They are **calibrated** with
   one-variable logistic regression fitted on *validation* scores (Platt
   scaling), so "70%" means "about 7 in 10 such validation cases were kin".
4. The verdict uses the threshold chosen on validation.
5. Resemblance (father X% / mother Y%) comes from the child-father and
   child-mother face similarities.
6. The disclaimer is always shown: *Statistical estimate from face images. Not
   proof of parentage. Not a DNA test.*

---

## 11. Limitations and ethics

- About **80% accuracy at best**: one in five answers is wrong. Never use this for
  real parentage decisions. Only DNA tests can establish parentage.
- **Dataset bias:** FIW is mostly Western public figures, photographed
  professionally. Accuracy on other groups and on everyday photos is unknown and
  probably lower.
- **Age gaps:** parents and children are often photographed decades apart.
- **Privacy and consent:** face images of real families are personal data. Only use
  photos with the consent of the people in them. The FIW license forbids
  redistributing its images, so no FIW image is in the public repository.

---

## 12. Likely viva questions

**Why tri-subject instead of one parent at a time?**
The child may resemble one parent more. Two parents give more evidence, and the
claim "both are the parents" is stronger. That's why the hard-negative test exists.

**What is transfer learning here, and what is fine-tuning?**
Transfer learning: reusing face networks pretrained on millions of faces
(ArcFace, FaceNet) as feature extractors. Fine-tuning: E3 unfreezes FaceNet's last
block and trains it with our head, using a small learning rate.

**Why can't you fine-tune ArcFace?**
The InsightFace model is an ONNX file, a frozen export without training support.
So we fine-tune FaceNet, a PyTorch model, and compare against E2b on the same
backbone.

**How do you prevent data leakage?**
- Family-disjoint splits, checked by a unit test.
- Thresholds, scalers, hyper-parameters and early stopping use validation data
  only. The test set is used only for the final numbers.
- The different-photo protocol removes the same-photo leak.

**Why ROC-AUC and not just accuracy?**
Accuracy depends on one threshold. AUC measures the ranking over all thresholds,
so it is a fairer way to compare models.

**Your deep model doesn't beat the baseline on ArcFace. Isn't that a failure?**
No. It's an honest finding. ArcFace embeddings are already excellent, and 1,434
training families are few. The learning curves show overfitting, and the fixes
narrowed the gap but couldn't create information the data doesn't have. On weaker
features (FaceNet), learning and fine-tuning clearly helped.

**What does alpha mean?**
It is how much the decision relies on each parent. Our analysis shows it matches
"who the child looks like" only for the FaceNet models. So the demo computes
resemblance from the similarities and labels alpha as reliance only.

**Why calibrate the probability in the demo?**
The raw sigmoid output was not calibrated (scores clustered near 0.5). Platt
scaling on validation data turns the score into a probability that matches
validation frequencies, without touching the test set.

**What would you do with more time?**
- More families (more data matters most).
- Calibration checks on a separate set.
- Testing on non-Western and everyday photos.
- Fine-tuning a stronger PyTorch backbone (e.g. an ArcFace-style ResNet).
