"""Baseline 2: Self-Blended Images EfficientNet-B4 (FF-c23 weights), reference test-time crop.
Only deviation from the official script: RetinaFace-MobileNet (uniface) instead of RetinaFace-ResNet50."""
import cv2
import numpy as np

from fdemo import models
from fdemo.ctx import sampled_detections
from fdemo.signals.base import nan_features, topk_mean

KEYS = ["sbi_mean", "sbi_top4", "sbi_std"]


def sbi_box(bbox, H, W):
    """Replicates SBI crop_face(..., crop_by_bbox=True, phase='test'): margins (w/4)*0.5, (h/4)*0.5."""
    x0, y0, x1, y1 = bbox
    w, h = x1 - x0, y1 - y0
    mw, mh = w / 4 * 0.5, h / 4 * 0.5
    return max(0, int(x0 - mw)), max(0, int(y0 - mh)), min(W, int(x1 + mw) + 1), min(H, int(y1 + mh) + 1)


def score(ctx) -> dict:
    import torch

    dets = sampled_detections(ctx)
    crops = []
    for i, d in zip(ctx.sampled, dets):
        if d is None:
            continue
        rgb = cv2.cvtColor(ctx.frames[i], cv2.COLOR_BGR2RGB)
        x0, y0, x1, y1 = sbi_box(d.bbox, *rgb.shape[:2])
        crops.append(cv2.resize(rgb[y0:y1, x0:x1], (380, 380)).transpose(2, 0, 1))
    if not crops:
        return nan_features(KEYS)
    with torch.no_grad():
        x = torch.from_numpy(np.stack(crops)).float() / 255.0
        net = models.sbi()
        logits = torch.cat([net(x[i:i + 4]) for i in range(0, len(x), 4)])
        p = logits.softmax(1)[:, 1].numpy()
    return {"sbi_mean": float(p.mean()), "sbi_top4": topk_mean(p, 4), "sbi_std": float(p.std())}
