"""Demo logic: calibration, result fields, and graceful errors (no face found)."""
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from kinship.align import FaceResult, SmokeTestAligner  # noqa: E402
from kinship.models import TriSubjectHead  # noqa: E402
from kinship.predict import DISCLAIMER, Calibration, KinshipPredictor, format_result  # noqa: E402
from kinship.train import HeadScorer  # noqa: E402

CFG = {"align": {"crop_size": 112}}


def _img(seed):
    return np.random.default_rng(seed).integers(0, 255, size=(120, 100, 3), dtype=np.uint8)


def test_calibration_is_increasing_and_matches_threshold():
    rng = np.random.default_rng(0)
    scores = np.concatenate([rng.normal(0.52, 0.01, 200), rng.normal(0.49, 0.01, 200)])
    labels = np.array([1] * 200 + [0] * 200)
    cal = Calibration.fit(scores, labels, threshold=0.505)
    assert cal.a > 0                                         # higher score -> higher probability
    assert cal.prob(0.53) > 0.9 and cal.prob(0.48) < 0.1
    assert 0.3 < cal.prob(cal.threshold) < 0.7


def test_predict_returns_everything_the_demo_shows():
    pred = KinshipPredictor(SmokeTestAligner(CFG), HeadScorer(TriSubjectHead()), Calibration(10.0, -5.0, 0.5))
    res = pred.predict(_img(1), _img(2), _img(3))
    assert 0 <= res["probability"] <= 1
    assert abs(res["resemblance_father"] + res["resemblance_mother"] - 1) < 1e-9
    assert set(res["crops"]) == {"father", "mother", "child"}
    text = format_result(res)
    assert "Kinship probability" in text and DISCLAIMER in text


class NoFaceAligner(SmokeTestAligner):
    def process(self, img):
        r = super().process(img)
        return FaceResult(r.crop, r.embedding, False, 0.0, None, None, 0, None)


def test_no_face_gives_a_clear_error_not_a_score():
    pred = KinshipPredictor(NoFaceAligner(CFG), HeadScorer(TriSubjectHead()), Calibration(1.0, 0.0, 0.5))
    res = pred.predict(_img(1), _img(2), _img(3))
    assert "error" in res and "father" in res["error"]
    assert DISCLAIMER in format_result(res)
    assert "upload" in pred.predict(None, _img(2), _img(3))["error"]
