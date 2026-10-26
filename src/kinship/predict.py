"""Kinship prediction for three uploaded photos - the logic behind the demo (app.py).

It uses exactly the same preprocessing as training and evaluation
(`align.ArcFaceAligner`) and the best model on the test set (E2-hardneg).

What the demo shows, and why:
  * Probability: the model's raw score is NOT a probability (on the test set it sits
    in a narrow band around 0.47-0.55). It is calibrated with a one-variable logistic
    regression (Platt scaling) fitted on the VALIDATION scores of the same protocol
    as the demo (one photo per person, different photos), so "70%" means roughly
    "7 in 10 such cases were kin" on validation data.
  * Verdict: probability >= the calibrated value of the threshold chosen on
    validation.
  * Resemblance (father X% / mother Y%): a softmax of the two face similarities
    cos(F, C) and cos(M, C) with temperature 0.1: equal similarity gives 50/50, a
    gap of 0.1 gives 73/27, a gap of 0.2 gives 88/12. If the child is not clearly
    similar to either parent, the demo says so. The attention weights of the ArcFace
    head do not follow who the child looks like (see results.md), so they are only
    shown as "how much the model relied on each parent".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .align import l2_normalise

DISCLAIMER = "Statistical estimate from face images. Not proof of parentage. Not a DNA test."
RESEMBLANCE_TEMPERATURE = 0.1   # softmax temperature for the father / mother resemblance split
WEAK_SIMILARITY = 0.05          # below this for both parents: "no clear resemblance"
ROLES = ("father", "mother", "child")


@dataclass
class Calibration:
    """p = sigmoid(a * score + b), fitted on validation scores; threshold from validation."""
    a: float
    b: float
    threshold: float

    def prob(self, score: float) -> float:
        return float(1 / (1 + np.exp(-(self.a * score + self.b))))

    @classmethod
    def fit(cls, scores: np.ndarray, labels: np.ndarray, threshold: float) -> "Calibration":
        from sklearn.linear_model import LogisticRegression

        lr = LogisticRegression(C=1e6).fit(np.asarray(scores).reshape(-1, 1), np.asarray(labels))
        return cls(float(lr.coef_[0, 0]), float(lr.intercept_[0]), float(threshold))


def find_artifacts(names: list[str], roots: list[str | Path]) -> Path | None:
    """First outputs folder (among roots) that contains every relative path in names."""
    for root in roots:
        if all((Path(root) / n).exists() for n in names):
            return Path(root)
    return None


def load_calibration(outputs: Path, model: str, protocol: str = "image_xphoto") -> Calibration:
    """Calibration from the saved validation scores + the validation threshold."""
    from .utils import load_json

    val = pd.read_csv(outputs / "scores" / f"{model}__val__{protocol}.csv")
    thr = load_json(outputs / "results" / f"{model}.json")["modes"][protocol]["val"]["threshold"]
    return Calibration.fit(val["score"].to_numpy(), val["label"].to_numpy(), thr)


class KinshipPredictor:
    """Aligner + trained head + calibration -> a result dict for three RGB images."""

    def __init__(self, aligner, scorer, calibration: Calibration):
        self.aligner, self.scorer, self.cal = aligner, scorer, calibration

    def predict(self, father: np.ndarray, mother: np.ndarray, child: np.ndarray) -> dict:
        faces, notes = {}, []
        for role, img in zip(ROLES, (father, mother, child)):
            if img is None:
                return {"error": f"Please upload a photo of the {role}."}
            r = self.aligner.process(np.asarray(img)[:, :, :3])
            if not r.detected:
                return {"error": f"No face was found in the {role}'s photo. Please use a clear, "
                                 f"front-facing photo of one person."}
            if r.n_faces > 1:
                notes.append(f"{r.n_faces} faces found in the {role}'s photo; the largest one was used.")
            faces[role] = r
        eF, eM, eC = (l2_normalise(faces[r].embedding)[None] for r in ROLES)
        score, d = self.scorer.score_details(eF, eM, eC)
        score = float(score[0])
        prob = self.cal.prob(score)
        cos_f, cos_m = float((eF * eC).sum()), float((eM * eC).sum())
        share_f = float(1 / (1 + np.exp(-(cos_f - cos_m) / RESEMBLANCE_TEMPERATURE)))
        cutoff = self.cal.prob(self.cal.threshold)
        return {"probability": prob, "likely": prob >= cutoff, "cutoff": cutoff,
                "raw_score": score, "resemblance_father": share_f, "resemblance_mother": 1 - share_f,
                "cos_father_child": cos_f, "cos_mother_child": cos_m,
                "reliance_father": float(d["alpha_F"][0]), "reliance_mother": float(d["alpha_M"][0]),
                "crops": {r: faces[r].crop for r in ROLES}, "notes": notes}


def format_result(res: dict) -> str:
    """Markdown text shown in the demo."""
    if "error" in res:
        return f"### ⚠️ {res['error']}\n\n*{DISCLAIMER}*"
    verdict = "Likely their child" if res["likely"] else "Unlikely to be their child"
    lines = [f"## {verdict}",
             f"**Kinship probability: {res['probability']:.0%}** (verdict cut-off {res['cutoff']:.0%}, chosen "
             f"on the validation set; raw model score {res['raw_score']:.3f})", ""]
    sims = (f"face similarity child-father {res['cos_father_child']:.2f}, "
            f"child-mother {res['cos_mother_child']:.2f}")
    if max(res["cos_father_child"], res["cos_mother_child"]) < WEAK_SIMILARITY:
        lines += [f"**Resemblance:** the child's face is not clearly similar to either parent ({sims})", ""]
    else:
        lines += [f"**Resemblance:** father {res['resemblance_father']:.0%} / mother "
                  f"{res['resemblance_mother']:.0%} ({sims})", ""]
    lines += [f"Model reliance (attention weights): father {res['reliance_father']:.0%} / "
              f"mother {res['reliance_mother']:.0%}"]
    lines += [f"- {n}" for n in res["notes"]]
    lines += ["", f"*{DISCLAIMER}*"]
    return "\n".join(lines)
