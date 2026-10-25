"""Training loops: the tri-subject head on cached embeddings (E2 / E2b).
Backbone fine-tuning (E3) is added in M4.

Each epoch is built fresh:
  * every kin triplet of the training split, plus ONE newly drawn negative for it:
      main negative  (F, M, C')  C' = a child of another family, same gender as C
      hard negative  (F', M, C) or (F, M', C), with probability `hard_negative_fraction`
  * every person is represented by the mean of a RANDOM SUBSET of their images
    (sometimes one image, sometimes all), so the model sees new inputs every
    epoch and gets used to both single images and templates;
  * the child's images always come from photos that neither parent's chosen
    images come from, so the model cannot learn the same-photo shortcut.
Early stopping on validation ROC-AUC (headline protocol: templates, different photos).
"""
from __future__ import annotations

import csv
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score

from .align import l2_normalise
from .embed import EmbeddingStore
from .evaluate import choose_threshold, keep_complete_pairs, triplet_embeddings
from .models import TriSubjectHead, tri_subject_loss


class EpochSampler:
    """Draws a fresh training epoch of (eF, eM, eC, y, y_fc, y_mc) arrays."""

    def __init__(self, store: EmbeddingStore, positives: pd.DataFrame, persons: pd.DataFrame,
                 hard_fraction: float = 0.0):
        self.store, self.hard_fraction = store, hard_fraction
        self.pos = positives[["F", "M", "C"]].itertuples(index=False, name=None)
        self.pos = [t for t in self.pos if all(p in store for p in t)]
        info = persons.set_index("pid")
        self.fid = info["fid"].to_dict()
        self.gender = info["gender"].to_dict()

        def pool(pids):
            """{gender: array of pids} for candidate replacements."""
            pids = sorted({p for p in pids if p in store and isinstance(self.gender.get(p), str)})
            return {g: np.array([p for p in pids if self.gender[p] == g], dtype=object) for g in ("m", "f")}

        self.children = pool(t[2] for t in self.pos)              # stand-ins for C
        self.adults = pool(x for t in self.pos for x in t[:2])    # stand-ins for F / M
        self._rows: dict[tuple, tuple | None] = {}

    def _stranger(self, pool: dict, like: str, avoid_fids: set, rng) -> str | None:
        cands = pool.get(self.gender.get(like), np.array([], dtype=object))
        for _ in range(20):                                       # rejection sampling
            if len(cands) == 0:
                return None
            p = cands[rng.integers(len(cands))]
            if self.fid[p] not in avoid_fids:
                return p
        return None

    def _subset_mean(self, rows: np.ndarray, rng) -> np.ndarray:
        k = int(rng.integers(1, len(rows) + 1))
        return l2_normalise(self.store.emb[rng.choice(rows, size=k, replace=False)].mean(axis=0))

    def epoch(self, rng: np.random.Generator):
        items = []
        for f, m, c in self.pos:
            items.append((f, m, c, 1.0, 1.0, 1.0))
            fams = {self.fid[f], self.fid[m], self.fid[c]}
            if rng.random() < self.hard_fraction:                 # hard: swap one parent
                if rng.random() < 0.5:
                    f2 = self._stranger(self.adults, f, fams, rng)
                    if f2 is not None:
                        items.append((f2, m, c, 0.0, 0.0, 1.0))
                else:
                    m2 = self._stranger(self.adults, m, fams, rng)
                    if m2 is not None:
                        items.append((f, m2, c, 0.0, 1.0, 0.0))
            else:                                                 # main: swap the child
                c2 = self._stranger(self.children, c, fams, rng)
                if c2 is not None:
                    items.append((f, m, c2, 0.0, 0.0, 0.0))
        eF, eM, eC, ys = [], [], [], []
        for f, m, c, y, yf, ym in items:
            key = (f, m, c)
            if key not in self._rows:
                self._rows[key] = self.store.triplet_rows(f, m, c)
            rows = self._rows[key]
            if rows is None:
                continue
            eF.append(self._subset_mean(rows[0], rng))
            eM.append(self._subset_mean(rows[1], rng))
            eC.append(self._subset_mean(rows[2], rng))
            ys.append((y, yf, ym))
        ys = np.array(ys, np.float32)
        return np.stack(eF), np.stack(eM), np.stack(eC), ys[:, 0], ys[:, 1], ys[:, 2]


class HeadScorer:
    """Wraps a trained TriSubjectHead so evaluate_triplet_scorer can use it."""

    def __init__(self, model: TriSubjectHead, device: str = "cpu"):
        self.model, self.device = model.eval(), device

    @torch.no_grad()
    def score_details(self, eF, eM, eC):
        outs = []
        for i in range(0, len(eF), 1024):
            t = [torch.as_tensor(x[i:i + 1024], dtype=torch.float32, device=self.device) for x in (eF, eM, eC)]
            o = self.model(*t)
            outs.append({k: v.float().cpu().numpy() for k, v in o.items()})
        cat = {k: np.concatenate([o[k] for o in outs]) for k in outs[0]}
        sig = lambda x: 1 / (1 + np.exp(-x))
        return sig(cat["logit"]), {"alpha_F": cat["alpha_f"], "alpha_M": cat["alpha_m"],
                                   "p_FC": sig(cat["logit_fc"]), "p_MC": sig(cat["logit_mc"])}

    def score(self, eF, eM, eC):
        return self.score_details(eF, eM, eC)[0]


def _val_arrays(store: EmbeddingStore, df: pd.DataFrame):
    """Fixed validation inputs: templates from different photos (headline protocol)."""
    *emb, ok = triplet_embeddings(store, df, "template", cross_photo=True)
    keep = keep_complete_pairs(df, ok)
    return [e[keep] for e in emb], df["label"].to_numpy()[keep]


def train_head(store: EmbeddingStore, train_df: pd.DataFrame, val_df: pd.DataFrame,
               persons: pd.DataFrame, hp: dict, seed: int, log, log_csv: str | Path,
               ckpt_path: str | Path, val_hard_df: pd.DataFrame | None = None,
               device: str | None = None) -> tuple[TriSubjectHead, dict]:
    """Train E2 with AdamW + cosine LR, early stopping on validation ROC-AUC.
    Saves the best checkpoint and a per-epoch CSV log. Returns (best model, summary)."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    sampler = EpochSampler(store, train_df[train_df["label"] == 1], persons, float(hp["hard_negative_fraction"]))
    (vF, vM, vC), v_y = _val_arrays(store, val_df)
    hard = _val_arrays(store, val_hard_df) if val_hard_df is not None else None

    model_args = {"emb_dim": int(store.emb.shape[1]), "proj_dim": int(hp["proj_dim"]),
                  "hidden": int(hp["hidden"]), "pair_dim": int(hp["pair_dim"]),
                  "dropout": float(hp["dropout"])}
    model = TriSubjectHead(**model_args).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(hp["lr"]), weight_decay=float(hp["weight_decay"]))
    epochs, bs = int(hp["epochs"]), int(hp["batch_size"])
    steps = epochs * math.ceil(2 * len(sampler.pos) / bs)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(steps, 1))
    scorer = HeadScorer(model, device)

    best = {"val_auc": -1.0, "epoch": 0}
    wait, history = 0, []
    Path(log_csv).parent.mkdir(parents=True, exist_ok=True)
    Path(ckpt_path).parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        t0 = time.time()
        model.train()
        eF, eM, eC, y, yf, ym = sampler.epoch(rng)
        perm = rng.permutation(len(y))
        total = 0.0
        for i in range(0, len(perm), bs):
            idx = perm[i:i + bs]
            t = [torch.as_tensor(a[idx], device=device) for a in (eF, eM, eC, y, yf, ym)]
            loss = tri_subject_loss(model(t[0], t[1], t[2]), t[3], t[4], t[5], float(hp["aux_weight"]))
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += loss.item() * len(idx)

        model.eval()
        v_s = scorer.score(vF, vM, vC)
        row = {"epoch": epoch, "train_loss": total / len(perm), "train_items": len(perm),
               "val_auc": roc_auc_score(v_y, v_s),
               "val_acc": float(((v_s >= choose_threshold(v_y, v_s)) == v_y).mean()),
               "val_hard_auc": roc_auc_score(hard[1], scorer.score(*hard[0])) if hard else float("nan"),
               "lr": opt.param_groups[0]["lr"], "seconds": time.time() - t0}
        history.append(row)
        improved = row["val_auc"] > best["val_auc"] + 1e-4
        if improved:
            best = {"val_auc": row["val_auc"], "epoch": epoch}
            torch.save({"state_dict": model.state_dict(), "model_args": model_args, "epoch": epoch,
                        "val_auc": row["val_auc"], "hyperparams": hp}, ckpt_path)
            wait = 0
        else:
            wait += 1
        log.info(f"epoch {epoch:2d}  loss {row['train_loss']:.4f}  val AUC {row['val_auc']:.4f}  "
                 f"val acc {row['val_acc']:.3f}  val hard AUC {row['val_hard_auc']:.4f}"
                 + ("  *best*" if improved else ""))
        if wait >= int(hp["patience"]):
            log.info(f"Early stopping: no val AUC improvement for {wait} epochs")
            break

    with open(log_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(history[0]))
        w.writeheader()
        w.writerows(history)
    model = load_head(ckpt_path, device)
    return model, {"best_epoch": best["epoch"], "best_val_auc": best["val_auc"],
                   "epochs_run": len(history), "model_args": model_args, "hyperparams": hp}


def load_head(path: str | Path, device: str = "cpu") -> TriSubjectHead:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = TriSubjectHead(**ckpt["model_args"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    return model.eval()
