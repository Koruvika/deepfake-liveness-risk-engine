"""Baseline 1: Silent-Face-Anti-Spoofing MiniFASNetV2 (scale 2.7) + V1SE (scale 4.0), reference preprocessing."""
import cv2
import numpy as np

from fdemo import models
from fdemo.ctx import sampled_detections
from fdemo.signals.base import nan_features

KEYS = ["pad_real_mean", "pad_real_min", "pad_real_low4", "pad_real_std"]


def minifas_box(bbox, H, W, scale):
    """Replicates yakhyo AntiSpoofingONNX._crop_face (xyxy -> int xywh -> scaled square-ish box)."""
    x, y = int(bbox[0]), int(bbox[1])
    bw, bh = int(bbox[2] - bbox[0]), int(bbox[3] - bbox[1])
    s = min((H - 1) / bh, (W - 1) / bw, scale)
    nw, nh = bw * s, bh * s
    cx, cy = x + bw / 2, y + bh / 2
    return (max(0, int(cx - nw / 2)), max(0, int(cy - nh / 2)),
            min(W - 1, int(cx + nw / 2)), min(H - 1, int(cy + nh / 2)))


def valid_bbox(bbox) -> bool:
    x0, y0, x1, y1 = bbox
    return (x1 - x0) >= 2 and (y1 - y0) >= 2


def real_prob(img, bbox) -> float:
    if not valid_bbox(bbox):
        return float("nan")
    H, W = img.shape[:2]
    probs = []
    for sess, scale in models.minifas():
        x0, y0, x1, y1 = minifas_box(bbox, H, W, scale)
        crop = img[y0:y1 + 1, x0:x1 + 1]
        if crop.size == 0:
            continue
        crop = cv2.resize(crop, (80, 80))
        logits = sess.run(None, {sess.get_inputs()[0].name: crop.astype(np.float32).transpose(2, 0, 1)[None]})[0]
        e = np.exp(logits - logits.max())
        probs.append(float((e / e.sum())[0, 1]))
    return float(np.mean(probs)) if probs else float("nan")


def score(ctx) -> dict:
    dets = sampled_detections(ctx)
    p = []
    for i, d in zip(ctx.sampled, dets):
        if d is None or not valid_bbox(d.bbox):
            continue
        rp = real_prob(ctx.frames[i], d.bbox)
        if np.isfinite(rp):
            p.append(rp)
    p = np.array(p)
    if p.size == 0:
        return nan_features(KEYS)
    return {
        "pad_real_mean": float(p.mean()),
        "pad_real_min": float(p.min()),
        "pad_real_low4": float(np.sort(p)[:4].mean()),
        "pad_real_std": float(p.std()),
    }
