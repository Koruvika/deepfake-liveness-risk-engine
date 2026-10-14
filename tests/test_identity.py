"""identity.score must tolerate frames whose ArcFace embedding raises ValueError
(degenerate keypoints) instead of aborting the whole video."""
import numpy as np

from fdemo.signals import identity


class FakeDet:
    def __init__(self, kps):
        self.kps = kps


class FakeArc:
    def __init__(self, n_bad=0):
        self.n_bad = n_bad
        self.calls = 0

    def embed(self, img, kps):
        self.calls += 1
        if self.calls <= self.n_bad:
            raise ValueError("degenerate keypoints")
        return np.array([1.0, 0.0, 0.0], np.float32)


def make_ctx(n=5):
    class Ctx:
        pass

    ctx = Ctx()
    ctx.frames = np.zeros((n, 10, 10, 3), np.uint8)
    ctx.sampled = np.arange(n)
    return ctx


def test_score_skips_degenerate_frames_but_still_scores(monkeypatch):
    ctx = make_ctx(5)
    fake_arc = FakeArc(n_bad=1)
    dets = [FakeDet(np.zeros((5, 2), np.float32)) for _ in range(5)]
    monkeypatch.setattr(identity.models, "arcface", lambda: fake_arc)
    monkeypatch.setattr(identity, "sampled_detections", lambda c: dets)

    out = identity.score(ctx)
    assert np.isfinite(out["id_drift_std"])
    assert np.isfinite(out["id_drift_min"])


def test_score_returns_nan_when_too_few_good_embeddings_remain(monkeypatch):
    ctx = make_ctx(5)
    fake_arc = FakeArc(n_bad=4)  # only 1 good embedding left
    dets = [FakeDet(np.zeros((5, 2), np.float32)) for _ in range(5)]
    monkeypatch.setattr(identity.models, "arcface", lambda: fake_arc)
    monkeypatch.setattr(identity, "sampled_detections", lambda c: dets)

    out = identity.score(ctx)
    assert np.isnan(out["id_drift_std"])
    assert np.isnan(out["id_drift_min"])
