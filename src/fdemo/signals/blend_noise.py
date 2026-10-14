"""Noise-residual inconsistency between face core, blending ring and background.

Every face swap has to blend a foreign face into the frame; the pasted region carries
noise statistics that differ from the surrounding camera noise.
"""
import cv2
import numpy as np
from scipy.stats import kurtosis

from fdemo.face.geometry import hull_mask
from fdemo.signals.base import nan_features, topk_mean

RING = np.ones((17, 17), np.uint8)   # ±8 px around the hull
BAND = np.ones((61, 61), np.uint8)   # background band outside the ring
BASE_KEYS = ["bn_ring_core", "bn_core_bg", "bn_kurt_diff", "bn_edge"]
KEYS = BASE_KEYS + [k + "_top" for k in BASE_KEYS]
MIN_STD = 0.3   # uint8 quantisation floor


def residual(gray) -> np.ndarray:
    return gray.astype(np.float32) - cv2.medianBlur(gray, 3).astype(np.float32)


def regions(mask):
    inner = cv2.erode(mask, RING)
    outer = cv2.dilate(mask, RING)
    core = inner > 0
    ring = (outer > 0) & ~core
    bg = (cv2.dilate(mask, BAND) > 0) & (outer == 0)
    return core, ring, bg


def frame_features(frame, pts):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mask = hull_mask(pts, gray.shape)
    core, ring, bg = regions(mask)
    if min(core.sum(), ring.sum(), bg.sum()) < 100:
        return None
    r = residual(gray)
    s_core, s_ring, s_bg = r[core].std(), r[ring].std(), r[bg].std()
    if min(s_core, s_ring, s_bg) < MIN_STD:
        return None
    eps = 1e-3
    mag = np.hypot(cv2.Sobel(gray, cv2.CV_32F, 1, 0), cv2.Sobel(gray, cv2.CV_32F, 0, 1))
    edge = cv2.morphologyEx(mask, cv2.MORPH_GRADIENT, np.ones((3, 3), np.uint8)) > 0
    return {
        "bn_ring_core": float(np.log(s_ring / s_core)),
        "bn_core_bg": float(np.log(s_core / s_bg)),
        "bn_kurt_diff": float(kurtosis(r[core]) - kurtosis(r[bg])),
        "bn_edge": float(np.log((mag[edge].mean() + eps) / (mag[ring].mean() + eps))),
    }


def score(ctx) -> dict:
    rows = [frame_features(ctx.frames[i], ctx.lms[i]) for i in ctx.sampled if ctx.face_ok[i]]
    rows = [r for r in rows if r is not None]
    if not rows:
        return nan_features(KEYS)
    out = {}
    for k in BASE_KEYS:
        v = np.array([r[k] for r in rows])
        out[k] = float(np.nanmean(v))
        out[k + "_top"] = topk_mean(np.abs(v), 4)
    return out
