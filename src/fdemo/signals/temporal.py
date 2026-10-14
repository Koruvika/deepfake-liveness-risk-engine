"""Stream-level cues of virtual-camera pipelines: duplicated frames, temporally correlated noise
(denoising / frame blending) and landmark jitter."""
import cv2
import numpy as np

from fdemo.face.geometry import LEFT_EYE_OUTER, RIGHT_EYE_OUTER, hull_mask
from fdemo.signals.blend_noise import residual

SMALL = (160, 90)
KEYS = ["tm_dup_rate", "tm_mad_cv", "tm_noise_corr", "tm_jitter"]


def mad_series(frames) -> np.ndarray:
    g = np.stack([
        cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), SMALL, interpolation=cv2.INTER_AREA).astype(np.float32)
        for f in frames
    ])
    return np.abs(np.diff(g, axis=0)).mean(axis=(1, 2))


def dup_features(frames, eps=0.35) -> dict:
    m = mad_series(frames)
    return {"tm_dup_rate": float((m < eps).mean()), "tm_mad_cv": float(m.std() / (m.mean() + 1e-6))}


def noise_corr(frames, bg) -> float:
    rs = [residual(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY))[bg] for f in frames]
    cs = [np.corrcoef(a, b)[0, 1] for a, b in zip(rs[:-1], rs[1:]) if a.std() > 1e-3 and b.std() > 1e-3]
    return float(np.mean(cs)) if cs else float("nan")


def landmark_jitter(lms, ok=None) -> float:
    if ok is None:
        ok = np.ones(len(lms), bool)
    ok = np.asarray(ok, bool)
    c = lms.mean(axis=1, keepdims=True)
    s = np.linalg.norm(lms[:, LEFT_EYE_OUTER] - lms[:, RIGHT_EYE_OUTER], axis=1)[:, None, None]
    p = (lms - c) / (s + 1e-6)
    d2 = p[2:] - 2 * p[1:-1] + p[:-2]
    v = ok[2:] & ok[1:-1] & ok[:-2]
    if v.sum() < 8:
        return float("nan")
    return float(np.linalg.norm(d2[v], axis=2).mean())


def background_mask(ctx):
    shape = ctx.frames.shape[1:3]
    union = np.zeros(shape, np.uint8)
    for t in np.flatnonzero(ctx.face_ok):
        union |= hull_mask(ctx.lms[t], shape)
    bg = cv2.dilate(union, np.ones((41, 41), np.uint8)) == 0
    bg[:10], bg[-10:], bg[:, :10], bg[:, -10:] = False, False, False, False
    return bg if bg.sum() >= 2000 else None


def score(ctx) -> dict:
    out = dup_features(ctx.frames)
    bg = background_mask(ctx)
    out["tm_noise_corr"] = noise_corr(ctx.frames, bg) if bg is not None else float("nan")
    out["tm_jitter"] = landmark_jitter(ctx.lms, ctx.face_ok)
    return out
