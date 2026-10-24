"""Embedding extraction with caching, person attributes, and an embedding store.

Extraction is split into chunks of `embed.chunk_size` images. Each finished chunk
is saved immediately (chunk_00000.csv + chunk_00000.npy), so if a Kaggle session
dies, the next run skips the finished chunks and continues where it stopped.
At the end all chunks are merged into `embeddings.npy` + `index.csv`
(row i of index.csv describes row i of embeddings.npy).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import numpy as np
import pandas as pd

from .align import l2_normalise
from .data import KinshipDataset
from .utils import ensure_dir, get_logger, load_image_rgb

INDEX_COLUMNS = ["image", "pid", "ok", "detected", "det_score", "gender", "age", "n_faces", "pad"]


def image_list(ds: KinshipDataset) -> list[tuple[str, str]]:
    """All (relative image path, person id), in a fixed order (needed for resuming)."""
    return sorted((img, p.pid) for p in ds.persons.values() for img in p.images)


def _save_npy_atomic(path: Path, arr: np.ndarray) -> None:
    """Write to a temp file first, then rename: a crash never leaves half a file."""
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        np.save(f, arr)
    os.replace(tmp, path)


def extract_embeddings(ds: KinshipDataset, aligner, out_dir: str | Path, chunk_size: int = 2000,
                       crops_dir: str | Path | None = None, limit: int | None = None,
                       log=None) -> tuple[np.ndarray, pd.DataFrame]:
    """Run `aligner.process` on every image, chunk by chunk, skipping finished chunks."""
    from PIL import Image
    from tqdm.auto import tqdm

    log = log or get_logger()
    out_dir = ensure_dir(out_dir)
    items = image_list(ds)[:limit]
    n_chunks = max(1, math.ceil(len(items) / chunk_size))
    dim = None

    for k in range(n_chunks):
        part = items[k * chunk_size:(k + 1) * chunk_size]
        csv_path, npy_path = out_dir / f"chunk_{k:05d}.csv", out_dir / f"chunk_{k:05d}.npy"
        if npy_path.exists() and csv_path.exists():
            cached = pd.read_csv(csv_path)
            if cached["image"].tolist() == [img for img, _ in part]:
                continue                              # finished in an earlier run
            log.warning(f"Chunk {k} does not match the current image list - recomputing it.")

        rows, embs = [], []
        for rel, pid in tqdm(part, desc=f"chunk {k + 1}/{n_chunks}", mininterval=10):
            try:
                r = aligner.process(load_image_rgb(ds.image_path(rel)))
            except Exception as e:                    # unreadable / corrupt image
                log.warning(f"Failed on {rel}: {e}")
                rows.append({"image": rel, "pid": pid, "ok": False, "detected": False, "det_score": 0.0,
                             "gender": None, "age": None, "n_faces": 0, "pad": None})
                embs.append(None)
                continue
            dim = dim or r.embedding.shape[0]
            rows.append({"image": rel, "pid": pid, "ok": True, "detected": r.detected,
                         "det_score": r.det_score, "gender": r.gender, "age": r.age,
                         "n_faces": r.n_faces, "pad": r.pad})
            embs.append(r.embedding)
            if crops_dir is not None:
                dest = Path(crops_dir) / rel
                dest.parent.mkdir(parents=True, exist_ok=True)
                Image.fromarray(r.crop).save(dest.with_suffix(".jpg"), quality=95)
        dim = dim or 512
        arr = np.stack([e if e is not None else np.zeros(dim, np.float32) for e in embs]).astype(np.float32)
        pd.DataFrame(rows, columns=INDEX_COLUMNS).to_csv(csv_path, index=False)
        _save_npy_atomic(npy_path, arr)               # written last = "chunk finished" marker
        det = sum(r["detected"] for r in rows)
        log.info(f"Chunk {k + 1}/{n_chunks} done: {det}/{len(rows)} faces detected")

    emb = np.concatenate([np.load(out_dir / f"chunk_{k:05d}.npy") for k in range(n_chunks)])
    index = pd.concat([pd.read_csv(out_dir / f"chunk_{k:05d}.csv") for k in range(n_chunks)],
                      ignore_index=True)
    _save_npy_atomic(out_dir / "embeddings.npy", emb)
    index.to_csv(out_dir / "index.csv", index=False)
    return emb, index


def load_embeddings(out_dir: str | Path) -> tuple[np.ndarray, pd.DataFrame]:
    out_dir = Path(out_dir)
    if not (out_dir / "embeddings.npy").exists():
        raise FileNotFoundError(f"No embeddings in {out_dir}. Run scripts/extract_embeddings.py first.")
    return np.load(out_dir / "embeddings.npy"), pd.read_csv(out_dir / "index.csv")


def detection_report(index: pd.DataFrame) -> dict:
    """Counts for the logs / report: how well did face detection work?"""
    ok = index[index["ok"]]
    det = ok[ok["detected"]]
    return {"images": int(len(index)), "unreadable": int((~index["ok"]).sum()),
            "detected": int(len(det)), "fallback_centre_crop": int(len(ok) - len(det)),
            "detection_rate": round(len(det) / max(len(ok), 1), 4),
            "multiple_faces": int((det["n_faces"] > 1).sum()),
            "found_at_pad": {str(k): int(v) for k, v in det["pad"].value_counts().sort_index().items()}}


def person_attributes(index: pd.DataFrame) -> pd.DataFrame:
    """One gender and age per person from all their detected images.

    Gender: majority vote (InsightFace gives 1 = male, 0 = female).
    Age: median, which is robust to a few bad estimates (people are photographed
    at different ages, so this is only a rough "generation" signal).
    """
    det = index[index["ok"] & index["detected"] & index["gender"].notna()]
    g = det.groupby("pid").agg(male_frac=("gender", "mean"), age=("age", "median"),
                               n_faces=("gender", "size")).reset_index()
    g["gender"] = np.where(g["male_frac"] >= 0.5, "m", "f")
    return g[["pid", "gender", "age", "n_faces", "male_frac"]]


class EmbeddingStore:
    """Look up embeddings by person: one random image, or a template (mean of all).

    When a person has some images where a face was detected, only those are used;
    centre-crop fallbacks are used only for people with no detected face at all.
    """

    def __init__(self, emb: np.ndarray, index: pd.DataFrame):
        self.emb = l2_normalise(emb.astype(np.float32))
        self.rows: dict[str, np.ndarray] = {}
        for pid, g in index.groupby("pid"):
            ok = g[g["ok"]]
            det = ok[ok["detected"]]
            use = det if len(det) else ok
            if len(use):
                self.rows[pid] = use.index.to_numpy()
        self._templates: dict[str, np.ndarray] = {}

    def __contains__(self, pid: str) -> bool:
        return pid in self.rows

    def template(self, pid: str) -> np.ndarray:
        """Average of all a person's image embeddings, re-normalised."""
        if pid not in self._templates:
            self._templates[pid] = l2_normalise(self.emb[self.rows[pid]].mean(axis=0))
        return self._templates[pid]

    def sample(self, pid: str, rng: np.random.Generator) -> np.ndarray:
        return self.emb[rng.choice(self.rows[pid])]

    def stack(self, pids, mode: str, rng: np.random.Generator | None = None) -> np.ndarray:
        """Embeddings for a list of persons. mode = 'template' or 'image'."""
        if mode == "template":
            return np.stack([self.template(p) for p in pids])
        return np.stack([self.sample(p, rng) for p in pids])
