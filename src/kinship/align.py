"""Shared face detection + alignment - the ONE preprocessing path used for training,
validation, test and the demo (so the model never sees differently prepared faces).

ArcFace path: InsightFace `buffalo_l` pack.
  1. The SCRFD detector finds faces and 5 landmarks (eyes, nose, mouth corners).
  2. `face_align.norm_crop` warps the face so the landmarks land on fixed positions
     in a 112x112 image - the exact input the ArcFace recognition model was trained on.
  3. The same pass also gives the ArcFace embedding and a gender/age estimate.

FIW / RFIW images are already tight face crops (~108x124 px). Detectors are trained
on faces that sit inside a larger picture, so a face that fills the whole frame is
often missed. We therefore pad the image with a border before detection and try a
few padding amounts (`align.pad_ratios`). If no face is found at all we fall back
to a centre crop and count it (`detected=False`).

The FaceNet path (MTCNN, 160 px) for fine-tuning is added in M3/M4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .utils import center_square, get_logger


@dataclass
class FaceResult:
    crop: np.ndarray          # aligned face, RGB uint8, crop_size x crop_size
    embedding: np.ndarray     # float32 (512,), L2-normalised
    detected: bool            # False = no face found, centre-crop fallback was used
    det_score: float          # detector confidence (0 when not detected)
    gender: int | None        # InsightFace convention: 1 = male, 0 = female
    age: float | None         # estimated age in years
    n_faces: int              # how many faces were found (we keep the largest)
    pad: float | None         # padding ratio at which the face was found


def l2_normalise(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Scale vectors to length 1, so a dot product is the cosine similarity."""
    return x / np.clip(np.linalg.norm(x, axis=axis, keepdims=True), 1e-12, None)


def pad_image(img: np.ndarray, ratio: float) -> np.ndarray:
    """Add a black border of `ratio * longest side` on every side."""
    if ratio <= 0:
        return img
    import cv2

    p = int(round(ratio * max(img.shape[:2])))
    return cv2.copyMakeBorder(img, p, p, p, p, cv2.BORDER_CONSTANT, value=(0, 0, 0))


def onnx_providers(log=None) -> list[str]:
    """Use ONNX Runtime's CUDA provider if it really works, else CPU with a warning."""
    log = log or get_logger()
    try:
        # Importing torch first loads its bundled CUDA/cuDNN libraries into the
        # process; onnxruntime-gpu can then find them (a common Kaggle problem).
        import torch  # noqa: F401
    except ImportError:
        pass
    import onnxruntime as ort

    if hasattr(ort, "preload_dlls"):          # onnxruntime >= 1.21
        try:
            ort.preload_dlls()
        except Exception:
            pass
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" in available:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    log.warning(f"ONNX Runtime has no CUDA provider (available: {available}). Running on CPU - slow.")
    return ["CPUExecutionProvider"]


class ArcFaceAligner:
    """InsightFace detector + aligner + ArcFace embedder + gender/age, in one call."""

    name = "arcface"

    def __init__(self, cfg: dict, log=None):
        import warnings

        from insightface.app import FaceAnalysis

        # insightface calls a scikit-image function that is deprecated (still works);
        # without this filter the warning is printed once per image.
        warnings.filterwarnings("ignore", category=FutureWarning, module=r"insightface")
        a = cfg["align"]
        self.log = log or get_logger()
        providers = onnx_providers(self.log)
        # Only the models we need: skipping the 2D/3D landmark models saves time.
        self.app = FaceAnalysis(name=a.get("model_pack", "buffalo_l"),
                                allowed_modules=["detection", "recognition", "genderage"],
                                providers=providers)
        use_gpu = "CUDAExecutionProvider" in providers
        self.app.prepare(ctx_id=0 if use_gpu else -1, det_thresh=float(a["det_thresh"]),
                         det_size=(int(a["det_size"]), int(a["det_size"])))
        self.rec = self.app.models["recognition"]
        self.crop_size = int(a["crop_size"])
        self.pad_ratios = [float(r) for r in a["pad_ratios"]]

        # Check what each model session actually runs on (CUDA can fail silently).
        in_use = {name: m.session.get_providers()[0] for name, m in self.app.models.items()}
        self.log.info(f"InsightFace models and providers: {in_use}")
        if use_gpu and any(p != "CUDAExecutionProvider" for p in in_use.values()):
            self.log.warning("CUDA was requested but some models fell back to CPU.")
        self.device = "cuda" if all(p == "CUDAExecutionProvider" for p in in_use.values()) else "cpu"

    def process(self, img_rgb: np.ndarray) -> FaceResult:
        """Detect, align and embed the (largest) face in an RGB image."""
        from insightface.utils import face_align

        bgr = np.ascontiguousarray(img_rgb[:, :, ::-1])      # InsightFace works in BGR
        for ratio in self.pad_ratios:
            img = pad_image(bgr, ratio)
            faces = self.app.get(img)
            if faces:
                # Several faces (e.g. someone in the background): keep the largest.
                face = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
                crop = face_align.norm_crop(img, face.kps, image_size=self.crop_size)
                return FaceResult(crop[:, :, ::-1].copy(), face.normed_embedding.astype(np.float32),
                                  True, float(face.det_score), int(face.gender), float(face.age),
                                  len(faces), ratio)
        # Fallback: no face found. The image is most likely already a face crop,
        # so a centre crop is a reasonable guess. Counted in the logs.
        crop = center_square(img_rgb, self.crop_size)
        emb = self.rec.get_feat([np.ascontiguousarray(crop[:, :, ::-1])])[0]
        return FaceResult(crop, l2_normalise(emb.astype(np.float32)), False, 0.0, None, None, 0, None)


class SmokeTestAligner:
    """CPU stand-in with the same interface as ArcFaceAligner, for local tests only.

    It is NOT a face model: the "embedding" is a fixed random projection of a tiny
    thumbnail. It lets us test caching, splits, metrics and the report on a laptop
    without InsightFace.
    """

    name = "smoketest"

    def __init__(self, cfg: dict, log=None, dim: int = 512, seed: int = 0):
        self.crop_size = int(cfg["align"]["crop_size"])
        self.proj = np.random.default_rng(seed).normal(size=(16 * 16 * 3, dim)).astype(np.float32)
        self.device = "cpu"

    def process(self, img_rgb: np.ndarray) -> FaceResult:
        crop = center_square(img_rgb, self.crop_size)
        thumb = center_square(img_rgb, 16).astype(np.float32).ravel() / 255.0 - 0.5
        return FaceResult(crop, l2_normalise(thumb @ self.proj), True, 1.0, None, None, 1, 0.0)


def make_aligner(cfg: dict, kind: str = "arcface", log=None):
    if kind == "arcface":
        return ArcFaceAligner(cfg, log)
    if kind == "smoketest":
        return SmokeTestAligner(cfg, log)
    raise ValueError(f"Unknown aligner: {kind}")
