"""deepfake_sbi.score must run inference in small chunks (peak-memory bound) while producing
the exact same aggregated output as a single full-batch forward pass."""
import cv2
import numpy as np
import torch

from fdemo.signals import deepfake_sbi
from fdemo.signals.base import topk_mean


class FakeDet:
    def __init__(self, bbox):
        self.bbox = bbox


class ChunkSizeRecordingNet(torch.nn.Module):
    """Batch-size-invariant (no BatchNorm) linear net that records every batch size it sees."""

    def __init__(self):
        super().__init__()
        self.batch_sizes = []
        torch.manual_seed(0)
        self.flatten = torch.nn.Flatten()
        self.linear = torch.nn.Linear(380 * 380 * 3, 2)

    def forward(self, x):
        self.batch_sizes.append(x.shape[0])
        return self.linear(self.flatten(x))


def make_ctx(n):
    class Ctx:
        pass

    ctx = Ctx()
    ctx.frames = np.random.default_rng(0).integers(0, 255, (n, 50, 50, 3), dtype=np.uint8)
    ctx.sampled = np.arange(n)
    return ctx


def reference_crops(ctx, dets):
    crops = []
    for i, d in zip(ctx.sampled, dets):
        rgb = cv2.cvtColor(ctx.frames[i], cv2.COLOR_BGR2RGB)
        x0, y0, x1, y1 = deepfake_sbi.sbi_box(d.bbox, *rgb.shape[:2])
        crops.append(cv2.resize(rgb[y0:y1, x0:x1], (380, 380)).transpose(2, 0, 1))
    return np.stack(crops)


def test_score_chunks_batches_and_matches_full_batch_reference(monkeypatch):
    n = 9  # forces multiple chunks of 4 (4, 4, 1)
    ctx = make_ctx(n)
    dets = [FakeDet((5, 5, 40, 40)) for _ in range(n)]
    monkeypatch.setattr(deepfake_sbi, "sampled_detections", lambda c: dets)

    net = ChunkSizeRecordingNet().eval()
    monkeypatch.setattr(deepfake_sbi.models, "sbi", lambda: net)

    out = deepfake_sbi.score(ctx)

    assert max(net.batch_sizes) <= 4
    assert len(net.batch_sizes) > 1

    with torch.no_grad():
        x = torch.from_numpy(reference_crops(ctx, dets)).float() / 255.0
        p_full = net(x).softmax(1)[:, 1].numpy()

    assert np.isclose(out["sbi_mean"], p_full.mean(), atol=1e-6)
    assert np.isclose(out["sbi_top4"], topk_mean(p_full, 4), atol=1e-6)
    assert np.isclose(out["sbi_std"], p_full.std(), atol=1e-6)
