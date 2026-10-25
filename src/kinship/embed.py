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


def photo_id(image_path: str) -> str:
    """'F0007/MID1/P00074_face3.jpg' -> 'P00074': the photo this face was cut from.

    FIW and RFIW name face crops <photo>_face<k>.jpg. Relatives often appear in the
    SAME photo, and faces from one photo share lighting, camera and colour - a
    similarity that has nothing to do with kinship. Knowing the photo lets us
    avoid comparing two faces from the same picture.
    """
    name = str(image_path).rsplit("/", 1)[-1]
    return name.split("_face")[0] if "_face" in name else str(image_path)


class EmbeddingStore:
    """Look up embeddings by person: one random image, or a template (mean of all).

    When a person has some images where a face was detected, only those are used;
    centre-crop fallbacks are used only for people with no detected face at all.

    `pair(..., cross_photo=True)` compares two people using only images that come
    from different photos (see `disjoint_rows`).
    """

    def __init__(self, emb: np.ndarray, index: pd.DataFrame):
        self.emb = l2_normalise(emb.astype(np.float32))
        self.photo = index["image"].map(photo_id).to_numpy()
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

    def disjoint_rows(self, a: str, b: str) -> tuple[np.ndarray, np.ndarray] | None:
        """Images of a and of b such that no photo appears on both sides.

        a and b may well share photos; those images are simply not used for this
        comparison. To keep as many images as possible: take b's images from photos
        without a (or all of b's if a is in every one), then a's images from the
        remaining photos; if that leaves a with nothing, try the other way round.
        Returns (rows_a, rows_b), or None if every image of both is from one photo.
        """
        ra, rb = self.rows[a], self.rows[b]
        for first, second, swapped in ((rb, ra, False), (ra, rb, True)):
            second_photos = set(self.photo[second])
            own = first[[ph not in second_photos for ph in self.photo[first]]]
            if len(own) == 0:
                own = first
            own_photos = set(self.photo[own])
            other = second[[ph not in own_photos for ph in self.photo[second]]]
            if len(other):
                return (own, other) if swapped else (other, own)
        return None

    def triplet_rows(self, f: str, m: str, c: str) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Images of father, mother and child such that the child's images come from
        photos that neither parent's chosen images come from.

        One child image set is used against BOTH parents (and against their average
        in E2), so no comparison ever involves two faces from the same photo.
        First choice: the child's photos without either parent (parents keep all
        their images). Otherwise: a single child photo, with each parent represented
        by images from their other photos. None if impossible.
        """
        rf, rm, rc = self.rows[f], self.rows[m], self.rows[c]
        parent_photos = set(self.photo[rf]) | set(self.photo[rm])
        own = rc[[ph not in parent_photos for ph in self.photo[rc]]]
        if len(own):
            return rf, rm, own
        best = None
        for ph in sorted(set(self.photo[rc])):
            fr, mr = rf[self.photo[rf] != ph], rm[self.photo[rm] != ph]
            cr = rc[self.photo[rc] == ph]
            if len(fr) and len(mr) and (best is None or len(fr) + len(mr) > len(best[0]) + len(best[1])):
                best = (fr, mr, cr)
        return best

    def _embed(self, rows: np.ndarray, mode: str, rng) -> np.ndarray:
        if mode == "template":
            return l2_normalise(self.emb[rows].mean(axis=0))
        return self.emb[rng.choice(rows)]

    def triplet(self, f_list, m_list, c_list, mode: str, rng: np.random.Generator | None = None,
                cross_photo: bool = False):
        """Embeddings (eF, eM, eC, ok) for triplets; see `triplet_rows` for cross_photo."""
        f_list, m_list, c_list = list(f_list), list(m_list), list(c_list)
        if not cross_photo:
            eF, eM, eC = (self.stack(x, mode, rng) for x in (f_list, m_list, c_list))
            return eF, eM, eC, np.ones(len(f_list), bool)
        n, dim = len(f_list), self.emb.shape[1]
        out = [np.zeros((n, dim), np.float32) for _ in range(3)]
        ok = np.zeros(n, bool)
        for i, (f, m, c) in enumerate(zip(f_list, m_list, c_list)):
            rows = self.triplet_rows(f, m, c)
            if rows is not None:
                for k in range(3):
                    out[k][i] = self._embed(rows[k], mode, rng)
                ok[i] = True
        return out[0], out[1], out[2], ok

    def pair(self, a_list, b_list, mode: str, rng: np.random.Generator | None = None,
             cross_photo: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Embeddings for pairs (a_i, b_i). Returns (e_a, e_b, ok); ok is False where
        cross_photo is on and the pair has no images from different photos
        (those rows are zero vectors and must be dropped by the caller)."""
        a_list, b_list = list(a_list), list(b_list)
        if not cross_photo:
            return self.stack(a_list, mode, rng), self.stack(b_list, mode, rng), np.ones(len(a_list), bool)
        dim = self.emb.shape[1]
        ea, eb = np.zeros((len(a_list), dim), np.float32), np.zeros((len(a_list), dim), np.float32)
        ok = np.zeros(len(a_list), bool)
        for i, (a, b) in enumerate(zip(a_list, b_list)):
            rows = self.disjoint_rows(a, b)
            if rows is not None:
                ea[i], eb[i], ok[i] = self._embed(rows[0], mode, rng), self._embed(rows[1], mode, rng), True
        return ea, eb, ok


def build_crop_cache(index: pd.DataFrame, out_path: str | Path, size: int, crop_dirs: list,
                     fallback=None, log=None) -> np.ndarray:
    """All face crops in ONE uint8 array (row i = row i of index.csv), memory-mapped.

    Fine-tuning reads thousands of random crops per epoch; one array on disk is far
    faster than opening 26k small JPEG files each time. The crops come from the
    JPEGs saved by extract_embeddings (searched in `crop_dirs`, which may include an
    attached earlier Kaggle run). `fallback(rel_path) -> crop` is used for images
    whose crop is missing (e.g. re-run the aligner); without it they stay black.
    Built once, then reused.
    """
    from tqdm.auto import tqdm

    log = log or get_logger()
    out_path = Path(out_path)
    if out_path.exists():
        arr = np.load(out_path, mmap_mode="r")
        if arr.shape == (len(index), size, size, 3):
            return arr
    tmp = out_path.with_name(out_path.name + ".tmp")
    arr = np.lib.format.open_memmap(tmp, mode="w+", dtype=np.uint8, shape=(len(index), size, size, 3))
    found = recomputed = 0
    for i, rel in enumerate(tqdm(index["image"], desc="crop cache", mininterval=10)):
        crop = None
        for d in crop_dirs:
            path = (Path(d) / rel).with_suffix(".jpg")
            if path.exists():
                crop, found = load_image_rgb(path), found + 1
                break
        if crop is None and fallback is not None:
            crop, recomputed = fallback(rel), recomputed + 1
        if crop is not None:
            arr[i] = crop if crop.shape[:2] == (size, size) else _resize(crop, size)
    arr.flush()
    del arr
    os.replace(tmp, out_path)
    log.info(f"Crop cache {out_path.name}: {found} from saved crops, {recomputed} recomputed, "
             f"{len(index) - found - recomputed} missing (left black)")
    return np.load(out_path, mmap_mode="r")


def _resize(img: np.ndarray, size: int) -> np.ndarray:
    from .utils import center_square

    return center_square(img, size)
